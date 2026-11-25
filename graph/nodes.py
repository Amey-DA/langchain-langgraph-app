import re
import ast
import json
from contextlib import closing
from decimal import Decimal

import pandas as pd

from app import llm, get_relevant_schema, get_sql_connection, BUSINESS_RULES_TEXT, VALUE_HINTS
from langchain_core.messages import HumanMessage, SystemMessage
from app import run_python_on_dataframe
from llm_tracking import invoke_tracked

DISPLAY_ROWS = 100  # keep equal to PREVIEW_ROWS in flask_app.py

def extract_code(raw: str) -> str:
    """
    Return only the code from an LLM reply.
 
    Handles: a fenced block anywhere in the reply (even with prose before it,
    e.g. "Here's the code:" which causes 'unterminated string literal'),
    fences without a language tag, and replies with no fences at all.
    """
    text = (raw or "").strip()
 
    match = re.search(
        r"```(?:python|py|sql|tsql)?[ \t]*\r?\n(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
 
    # Unclosed fence: ```python\ncode...
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*[ \t]*\r?\n", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
 
    return text.strip()

def validate_python(code: str) -> None:
    """Raise ValueError with a readable message if the code cannot be parsed."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        first_line = code.splitlines()[0] if code else ""
        raise ValueError(
            f"Generated code is not valid Python: {e.msg} (line {e.lineno}). "
            f"First line was: {first_line!r}. "
            f"Return ONLY executable Python code with no explanation."
        )

def generate_sql_plan(
    question: str,
    previous_query: str = "",
    previous_error: str | None = None,
) -> tuple[str, bool, str]:
    """Classify the request and generate its SQL in one model call."""

    schema = get_relevant_schema(question)

    system_prompt = f"""
You plan analytics requests for the GoSales SQL Server database.

Return one valid JSON object with exactly these three fields:

- request_mode: either "full_results" or "analysis"
- analysis_required: a JSON boolean, true or false
- sql_query: a string containing the SQL or an unsupported-data reason

Return only JSON. Do not use Markdown fences or explanatory prose.

REQUEST PLANNING

1. Use request_mode="full_results" and analysis_required=false for:
   - Listing records.
   - Showing grouped totals, such as revenue by product line.
   - A straightforward top N by one measure, such as the top 5 retailers
     by revenue. Use TOP and ORDER BY in SQL for this case.

2. Use request_mode="analysis" and analysis_required=false for a scalar
   answer SQL can provide directly, such as total sales quantity or a count.

3. Use request_mode="analysis" and analysis_required=true when Pandas
   needs to calculate something from the retrieved rows. Examples include
   percentage contribution, growth, trends, outliers, comparisons between
   groups, and ranking within each group.

4. When analysis_required=true, return all eligible rows at the grain the
   question needs, with the measures Pandas will use. Apply the user's
   filters in SQL. Leave the ranking, percentages, and top-N selection
   to Pandas.

   Example: For "calculate each retailer's percentage contribution and
   identify the top 5", return revenue for EVERY eligible retailer.
   Pandas needs the full denominator before selecting the top five.

SQL RULES

- Generate one read-only T-SQL SELECT statement. A CTE starting with
  WITH is allowed.
- Use only the tables, columns, and relationships in the supplied schema.
- Copy column names exactly. Never invent columns or use SQL comments.
- Use the business definitions below to calculate measures.

GOSALES BUSINESS RULES

- Retailer's home country: join DM_RETAILER.RETAILER_COUNTRY_CODE to
  DM_COUNTRY.COUNTRY_CODE. A fact table is unnecessary for this lookup.
- Country where a sale happened: join FCT_SALES.RETAILER_SITE_KEY to
  DM_RETAILER_SITE.RETAILER_SITE_KEY, then join
  DM_RETAILER_SITE.RETAILER_SITE_COUNTRY_CODE to DM_COUNTRY.COUNTRY_CODE.
- Never join FCT_SALES.RETAILER_KEY to DM_COUNTRY.
- Use DM_COUNTRY.COUNTRY for country names and DM_COUNTRY.REGION for
  regions. COUNTRY_NAME does not exist.
- Use ORDER_DAY_KEY when joining to DM_DATE unless the question asks
  about ship date or close date.
- Identify a retailer site using RETAILER_SITE_CODE and RETAILER_SITE_CITY.
  RETAILER_SITE_CONTACT_NAME identifies a person. When showing sites with
  sales, join DM_RETAILER through FCT_SALES.RETAILER_KEY for retailer names.
- SALES, SALES_AMOUNT, REVENUE, AMOUNT, TOTAL_SALES, PRICE, and COST are
  not source columns. Calculate supported measures and give them aliases.
- When listing entities and reporting how many types or categories they
  represent, include the category column needed for the distinct count.

MEASURE DEFINITIONS

{BUSINESS_RULES_TEXT}

- Use measures as defined above.
- When a required measure is unavailable and the user supplied no
  assumption, set sql_query to "UNSUPPORTED: <specific reason>".
  Keep that string inside the JSON object.
- If the user supplies an assumption, use that exact value.
- Never invent a cost, tax rate, discount, or placeholder.
- Ranking, percentages, and averages are supported when their underlying
  measures exist.

TERMINOLOGY

- "Customer" means retailer in DM_RETAILER.
- "Purchases" and "spend" mean revenue unless quantity is specified.
- "Sales", "sales amount", and "total sales" mean revenue.
- "Units sold" and "volume" mean quantity.

KNOWN VALUES FOR FILTERS

{VALUE_HINTS}

DATABASE SCHEMA

{schema}

If a previous attempt failed, use the supplied error to correct the plan.
"""

    user_input = {"question": question}

    if previous_error:
        user_input["previous_query"] = previous_query
        user_input["previous_error"] = previous_error

    response = invoke_tracked(
        "plan_sql",
        llm,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_input)),
        ],
    )

    plan = json.loads(response.content)

    if not isinstance(plan, dict):
        raise ValueError("Return a JSON object containing the SQL plan.")

    request_mode = plan.get("request_mode")
    analysis_required = plan.get("analysis_required")
    sql_query = plan.get("sql_query")

    if request_mode not in ("analysis", "full_results"):
        raise ValueError("request_mode must be 'analysis' or 'full_results'.")

    if type(analysis_required) is not bool:
        raise ValueError("analysis_required must be a JSON boolean.")

    if analysis_required and request_mode != "analysis":
        raise ValueError("Use analysis mode when analysis_required is true.")

    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("sql_query must be a nonempty string.")

    return request_mode, analysis_required, sql_query.strip()

def execute_sql_query(query: str) -> pd.DataFrame:
    """
    Validate and execute a read-only SQL query.
    Return the retrieved records as a DataFrame.
    """

    cleaned = query.strip()

    # 1. Allow only SELECT queries
    if not re.match(r"(?is)^(SELECT|WITH)\b", cleaned):
        raise ValueError("Only SELECT queries are allowed.")

    # 2. Remove one trailing semicolon
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()

    # 3. Block multiple SQL statements
    if ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed.")

    # 4. Block SQL comments
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise ValueError("SQL comments are not allowed.")

    # 5. Block restricted SQL operations
    forbidden = re.compile(
        r"(?i)\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|"
        r"EXEC|EXECUTE|GRANT|REVOKE|DENY|BACKUP|RESTORE|DBCC|"
        r"INTO|OPENROWSET|OPENDATASOURCE)\b"
    )

    if forbidden.search(cleaned):
        raise ValueError("The query contains a disallowed SQL operation.")

    if re.search(r"\*\s*0(?:\.0+)?(?![\d.])", cleaned):
        raise ValueError(
            "The query multiplies by a constant 0, which is not a real "
            "calculation. Use only real columns. If the data needed does "
            "not exist, reply with: UNSUPPORTED: <reason>"
        )

    # 6. Execute the validated query
    with closing(get_sql_connection()) as connection:
        with closing(connection.cursor()) as cursor:
            cursor.execute(cleaned)

            if cursor.description is None:
                raise ValueError("The query returned no result set.")

            columns = [column[0] for column in cursor.description]

            MAX_ROWS = 50000

            rows = cursor.fetchmany(MAX_ROWS + 1)

            if len(rows) > MAX_ROWS:
                raise ValueError(
                    f"The query returned more than {MAX_ROWS} rows. Please narrow the question."
                )
            
            df = pd.DataFrame.from_records(rows, columns=columns)

    # 7. Convert Decimal columns to numeric values
    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, Decimal)).any():
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

def clean_sql_query(query: str) -> str:
    """Remove Markdown fences and surrounding text from generated SQL."""
    return extract_code(query)

def sql_node(state: dict) -> dict:
    """Generate a plan, execute validated SQL, and update the graph state."""

    question = state["question"]
    last_query = ""
    last_error = None

    for attempt in range(1, 4):
        query = ""

        try:
            request_mode, analysis_required, query = generate_sql_plan(
                question=question,
                previous_query=last_query,
                previous_error=last_error,
            )

            print(
                f"\n[PLAN] mode={request_mode}, "
                f"analysis_required={analysis_required}"
            )

            if query.upper().startswith("UNSUPPORTED:"):
                reason = query.split(":", 1)[1].strip()
                print(f"[SQL UNSUPPORTED] {reason}")

                return {
                    "request_mode": request_mode,
                    "analysis_required": analysis_required,
                    "sql_query": "",
                    "sql_result": None,
                    "error": f"UNSUPPORTED: {reason}",
                }

            if not re.match(r"(?i)^(SELECT|WITH)\b", query):
                raise ValueError(
                    "sql_query must start with SELECT, WITH, or UNSUPPORTED:."
                )

            print(f"\n[GENERATED SQL, attempt {attempt}]")
            print(query)

            # This existing function validates the SQL before executing it.
            df = execute_sql_query(query)

            print(f"[SQL SUCCESS] Retrieved {len(df)} rows.")

            return {
                "request_mode": request_mode,
                "analysis_required": analysis_required,
                "sql_query": query,
                "sql_result": df,
                "error": None,
            }

        except Exception as error:
            last_query = query
            last_error = str(error)

            invalid_column = re.search(
                r"Invalid column name '([^']+)'", last_error
            )
            if invalid_column:
                column = invalid_column.group(1)
                last_error += (
                    f"\n'{column}' is not a source column. Check the schema. "
                    "If it is a SELECT alias, repeat the expression where "
                    "SQL Server requires it. If it is a measure, calculate "
                    "it using the supplied measure definitions."
                )

            print(f"[PLAN/SQL ATTEMPT {attempt} FAILED] {error}")

    return {
        "sql_query": last_query,
        "sql_result": None,
        "error": f"The SQL plan failed after 3 attempts. Last error: {last_error}",
    }

def generate_python_code(
    question: str,
    df: pd.DataFrame,
    request_mode: str,
    previous_error: str | None = None
) -> str:
    """
    Generate Pandas code based on the user's question
    and the SQL query result.
    """

    columns = list(df.columns)
    data_types = df.dtypes.astype(str).to_dict()
    sample_data = df.head(5).to_dict(orient="records")

    system_prompt = f"""
You are an expert Python and Pandas data analyst.

Your task is to process the SQL result DataFrame
according to the user's requirement.

IMPORTANT:
The SQL query has already executed.
The result is available as a DataFrame named df.

RULES:

1. Pandas is already imported as pd.
2. Do not import any libraries.
3. Do not access files, databases, or networks.
4. Do not use print statements.
5. Assign the final DataFrame to a variable named result.
6. Result must always be a Pandas DataFrame.
7. Return only executable Python code.
8. Do not include Markdown code fences.
9. Use the complete DataFrame, not just the sample.
10. Do not invent business rules or numerical thresholds.
11. Respect explicitly provided numerical thresholds.
12. Examine existing columns before performing transformations.
13. Reuse calculations and flags already present in df.
14. Do not unnecessarily repeat SQL calculations.
15. Do not modify the original df unnecessarily.
16. Use df.copy() when additional transformations are needed.
17. Your entire reply must be executable Python. The first character of your reply must be code. No sentences, no explanation, no markdown.
18. The SQL query has already applied every filter, grouping, sorting and
    join requested in the question. Never filter rows again by a value from
    the question (such as a country or product name). If no additional
    calculation is needed, use: result = df.copy()
19. For a question that asks ONLY "how many" or for a count, return a small
    summary DataFrame (one row per category with its count, or one row for a
    single number). If the question also asks to show or list the rows, keep
    the rows and do not replace them with counts. Never add a column that
    holds the same value in every row.
    
REQUEST MODE: {request_mode}

MODE-SPECIFIC INSTRUCTIONS:

For full_results mode:
- Preserve all records and columns by default.
- Perform only transformations explicitly required
  by the user's question.
- Do not perform unnecessary additional analysis.

For analysis mode:
- Perform the specific analytical operation requested.
- Reuse existing calculated columns wherever possible.
- Do not repeat transformations already completed by SQL.
- Return the DataFrame containing the required results.

DATAFRAME INFORMATION:

Columns: {columns}
Data types: {data_types}
Total rows: {len(df)}

SAMPLE DATA (first 5 rows):

{sample_data}

PREVIOUS EXECUTION ERROR:

{previous_error if previous_error else "None. This is the first attempt."}
If an error is shown, fix that specific problem.

"""

    response = invoke_tracked("pandas", llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    code = extract_code(response.content)
    validate_python(code)
    
    return code

def generate_analysis_code(
    question: str,
    df: pd.DataFrame,
    previous_error: str | None = None
) -> str:

    columns = list(df.columns)
    data_types = df.dtypes.astype(str).to_dict()
    sample_data = df.head(20).to_dict(orient="records")

    system_prompt = f"""
You are an expert Python and Pandas data analyst.

Your task is to generate Python code to perform the
requested analysis on the complete DataFrame.

IMPORTANT:
The sample data is provided only to understand the
structure and contents of the DataFrame.
Execution will happen on the COMPLETE DataFrame.

RULES:

1. The complete dataset is available as a DataFrame named df.
2. Pandas is already imported as pd.
3. Do not import any libraries.
4. Use Pandas operations only. Do not use NumPy.
5. Do not access files, databases, or networks.
6. Do not use print statements.
7. Assign the final result to a variable named result.
8. Result MUST always be a Pandas DataFrame.
9. Wrap scalar results in a DataFrame.
10. Return only executable Python code.
11. Do not include Markdown code fences.
12. Perform calculations using the complete DataFrame.
13. Never calculate results using only the sample data.
14. Do not invent business rules, thresholds, or assumptions.
15. Always respect explicitly provided numerical thresholds.
16. Do not multiply or divide a user-provided threshold
    unless explicitly requested.
17. Examine the existing DataFrame columns before
    generating additional calculations.
18. Reuse existing calculated columns whenever they
    already satisfy the user's requirement.
19. Do not unnecessarily repeat transformations
    already performed by SQL or the previous Python Node.
20. If the requested result is already available in df,
    return the relevant existing data without
    unnecessarily recalculating it.
21. Do not modify the original df unnecessarily.
    Use df.copy() when additional transformations are needed.
22. Your entire reply must be executable Python. The first character of
    your reply must be code. No sentences, no explanation, no markdown.
23. Keep `result` small. For ranking, best, worst, top, bottom or outlier
    questions, return only the relevant rows (default 10 per group, at most
    50 rows in total) with the ranking column included, sorted sensibly.
    Never return the full df unless the user explicitly asks for all rows.
24. For "within each group" questions, use groupby().rank() and filter
    on the rank instead of returning every row.
25. Name a percentage column after the measure it is a share of, for
    example share_of_total_quantity_pct.
26. Never add a column that holds the same value in every row.

ANALYSIS APPROACH:

- Understand the user's analytical requirement.
- Inspect the available columns and their data types.
- Identify calculations already completed.
- Perform only the additional analysis required.
- Ensure the final result directly answers the question.

DATAFRAME INFORMATION:

Columns: {columns}
Data types: {data_types}
Total rows: {len(df)}

SAMPLE DATA (first 20 rows):

{sample_data}

PREVIOUS EXECUTION ERROR:

{previous_error if previous_error else "No previous error. This is the first attempt."}

If a previous execution error exists, correct the
generated code to address that specific error.
"""

    response = invoke_tracked("analysis", llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    code = extract_code(response.content)
    validate_python(code)

    return code

def python_node(state: dict) -> dict:
    """
    Execute Python code using the existing sandbox.

    MODE 1: Initial Python processing using sql_result.
    MODE 2: Analysis execution using analysis_code on python_result.
    """

    try:
        # Stop if Analysis Node failed to generate code
        if state.get("error") and not state.get("analysis_code"):
            return {}

        # MODE 2: Execute analysis code
        if state.get("analysis_code"):

            df = state["python_result"]
            code = state["analysis_code"]

            if df is None:
                raise ValueError(
                    "No original processed DataFrame available for analysis."
                )

            print("\n[ANALYSIS EXECUTION]")
            print(code)

            result_df, error = run_python_on_dataframe(df, code)

            if error:
                raise RuntimeError(error)

            if not isinstance(result_df, pd.DataFrame):
                raise TypeError(
                    "Analysis execution must return a Pandas DataFrame."
                )

            print(
                f"\n[ANALYSIS SUCCESS] Generated {len(result_df)} result rows."
            )

            return {
                "analysis_result": result_df,
                "analysis_code": "",
                "error": None
            }

        # MODE 1: Initial Python processing (up to 3 attempts)
        df = state["sql_result"]

        if df is None:
            raise ValueError("No SQL result available for Python processing.")

        if not state.get("analysis_required", False):

            print("\n[PYTHON SKIPPED] No calculation needed: using the SQL result as is.")
            return {
                "python_code": "",
                "python_result": df.copy(),
                "error": None
            }

        last_error = None

        for attempt in range(1, 4):
            code = None

            try:
                code = generate_python_code(
                    question=state["question"],
                    df=df,
                    request_mode=state["request_mode"],
                    previous_error=last_error,
                )
                print(f"\n[GENERATED PYTHON CODE, attempt {attempt}]")
                print(code)

                result_df, error = run_python_on_dataframe(df, code)
                if error:
                    raise RuntimeError(error)
                if not isinstance(result_df, pd.DataFrame):
                    raise TypeError("Result must be a Pandas DataFrame.")
                
                if len(result_df) > 1:
                    constant = [
                        c for c in result_df.columns
                        if c not in df.columns
                        and result_df[c].nunique(dropna=False) == 1
                    ]
                    if constant:
                        raise RuntimeError(
                            f"Column(s) {constant} hold the same value in every row. "
                            "Remove them and keep the original rows and columns."
                        )
                if result_df.empty and not df.empty:
                    raise RuntimeError(
                        f"The code returned 0 rows from {len(df)} input rows. "
                        "The SQL query already applied every filter in the "
                        "question, so do not filter rows again. Return the "
                        "rows unchanged unless a calculation is requested."
                    )
                break

            except Exception as e:
                last_error = str(e) + (
                    f"\nThe failing code was:\n{code}" if code else ""
                )
                print(f"\n[PYTHON ATTEMPT {attempt} FAILED] {e}")
        else:
            raise RuntimeError(
                f"Python step failed after 3 attempts. Last error: {last_error}"
            )

        print(f"\n[PYTHON SUCCESS] Processed {len(result_df)} rows.")

        return {
            "python_code": code,
            "python_result": result_df,
            "error": None
        }

    except Exception as error:

        print(f"\n[PYTHON ERROR] {error}")

        return {"error": str(error)}

def analysis_node(state: dict) -> dict:

    attempt = state.get("analysis_attempts", 0) + 1

    try:
        df = state["python_result"]

        if df is None:
            raise ValueError(
                "No processed DataFrame available for analysis."
            )

        code = generate_analysis_code(
            question=state["question"],
            df=df,
            previous_error=state.get("error")
        )

        print(f"\n[ANALYSIS ATTEMPT {attempt}]")
        print("\n[GENERATED ANALYSIS CODE]")
        print(code)

        return {
            "analysis_code": code,
            "analysis_attempts": attempt,
            "analysis_result": None,
            "error": None
        }

    except Exception as error:

        print(f"\n[ANALYSIS NODE ERROR] {error}")

        return {
            "analysis_code": "",
            "analysis_attempts": attempt,
            "analysis_result": None,
            "error": str(error)
        }

def route_after_python(state: dict) -> str:
    """Route based on analysis requirement and execution status."""

    # Handle errors first
    if state.get("error"):

        attempts = state.get("analysis_attempts", 0)

        if 0 < attempts < 2:
            print(
                f"\n[ROUTER] Analysis failed. "
                f"Retrying (attempt {attempts + 1})."
            )
            return "analysis_node"

        print("\n[ROUTER] Maximum attempts reached or initial error.")
        return "final_response_node"

    # Analysis completed successfully
    if state.get("analysis_result") is not None:
        print("\n[ROUTER] Analysis completed. Moving to Final Response.")
        return "final_response_node"

    # Check initial Python processing result
    df = state.get("python_result")

    if df is None:
        return "final_response_node"

    row_threshold = 50

    if len(df) > row_threshold and (
        state.get("analysis_required", False)
        or state.get("request_mode") != "full_results"
    ):
        print(
            f"\n[ROUTER] {len(df)} rows detected. "
            "Routing to Analysis Node."
        )
        return "analysis_node"

    print(
        f"\n[ROUTER] {len(df)} rows detected. "
        "Routing directly to Final Response Node."
    )

    return "final_response_node"

# def summarize_for_llm(df: pd.DataFrame, head_rows: int = 25, tail_rows: int = 5) -> str:
#     parts = [
#         f"Total rows: {len(df)}",
#         f"Columns: {list(df.columns)}",
#     ]
 
#     numeric = df.select_dtypes("number")
#     if not numeric.empty:
#         parts.append("Numeric summary (all rows):\n" + numeric.describe().round(2).to_string())
 
#     if len(df) <= head_rows + tail_rows:
#         parts.append("All rows:\n" + df.to_string(index=False))
#     else:
#         parts.append(f"First {head_rows} rows:\n" + df.head(head_rows).to_string(index=False))
#         parts.append(f"Last {tail_rows} rows:\n" + df.tail(tail_rows).to_string(index=False))
 
#     return "\n\n".join(parts)

# ------------------------------------------------------------------

# Identifier-like columns: statistics on them are meaningless
ID_COLUMN = re.compile(r"(?i)(^|_)(key|code|id|number|rank|row_num)$")

# Columns that must not be added up (percentages, averages, ratios)
NON_ADDITIVE = re.compile(r"(?i)percent|pct|share|ratio|margin|average|avg|rate|mean|median")


def fmt_value(value) -> str:
    """Format a value for the prompt: thousands separators, at most 2 decimals."""
    if pd.isna(value):
        return "none"
    if pd.api.types.is_number(value) and not isinstance(value, bool):
        number = float(value)
        return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"
    return str(value)


# def build_facts(df: pd.DataFrame, max_measures: int = 3) -> str:
#     """Facts computed over ALL rows, so the LLM never counts, adds or compares."""

#     lines = [
#         f"Rows: {len(df)}",
#         f"Columns: {', '.join(str(c) for c in df.columns)}",
#     ]

#     number_cols = list(df.select_dtypes("number").columns)
#     measure_cols = [c for c in number_cols if not ID_COLUMN.search(str(c))]
#     text_cols = [c for c in df.columns if c not in number_cols]

#     # A readable name for every row (for example "France" or "Tents / Web")
#     if text_cols:
#         labels = df[text_cols].astype(str).agg(" / ".join, axis=1)
#     else:
#         labels = pd.Series(df.index.astype(str), index=df.index)

#     # Highest, lowest, total and top-3 share for each measure
#     for col in measure_cols[:max_measures]:
#         if df[col].nunique(dropna=True) == 1:
#             lines.append(
#                 f"{col}: every row has the same value = "
#                 f"{fmt_value(df[col].dropna().iloc[0])}"
#             )
#             continue
#         tmp = pd.DataFrame({"label": labels, "value": df[col]}).dropna(subset=["value"])
#         if tmp.empty:
#             continue

#         tmp = tmp.sort_values("value", ascending=False)

#         highest = ", ".join(
#             f"{row.label} ({fmt_value(row.value)})" for row in tmp.head(3).itertuples()
#         )
#         lowest = ", ".join(
#             f"{row.label} ({fmt_value(row.value)})"
#             for row in tmp.tail(3).iloc[::-1].itertuples()
#         )
#         lines.append(f"{col}: highest three = {highest}")
#         lines.append(f"{col}: lowest three = {lowest}")

#         if not NON_ADDITIVE.search(str(col)):
#             total = tmp["value"].sum()
#             lines.append(f"{col}: total = {fmt_value(total)}")
#             if total and len(tmp) > 3:
#                 share = tmp["value"].head(3).sum() / total * 100
#                 lines.append(f"{col}: the top three together are {share:.2f}% of the total")

#     # Counts for small category columns (for example retailer type)
#     n_rows = len(df)

#     def describe_counts(series) -> str:
#         return ", ".join(
#             f"{name} = {count} ({count / n_rows * 100:.2f}%)"
#             for name, count in series.items()
#         )

#     for col in text_cols:
#         distinct = df[col].nunique()
#         if 1 < distinct <= 15 and distinct < n_rows:
#             counts = df[col].value_counts()

#             if distinct <= 8:
#                 lines.append(
#                     f"{col}: {distinct} distinct values, all listed: "
#                     f"{describe_counts(counts)}"
#                 )
#             else:
#                 lines.append(
#                     f"{col}: {distinct} distinct values. "
#                     f"Most frequent five: {describe_counts(counts.head(5))}"
#                 )
#                 lines.append(
#                     f"{col}: least frequent two (smallest first): "
#                     f"{describe_counts(counts.tail(2).iloc[::-1])}"
#                 )
#     return "\n".join(lines)

# def final_response_node(state: dict) -> dict:
#     """Generate a short insights-only answer from facts computed in Python."""

#     error = state.get("error")
#     if error and str(error).startswith("UNSUPPORTED:"):
#         reason = str(error).split(":", 1)[1].strip()
#         return {
#             "final_response": (
#                 "I can't answer that from the available data. "
#                 f"{reason} "
#                 "If you want an estimate, include your own assumption in "
#                 "the question, for example: \"assume cost is 60% of revenue\"."
#             ),
#             "error": None,
#         }

#     try:
#         question = state["question"]

#         if state.get("error"):
#             return {
#                 "final_response": (
#                     "I couldn't complete that request. "
#                     "The technical details are shown below."
#                 )
#             }

#         analysis_df = state.get("analysis_result")

#         if analysis_df is not None:
#             df = analysis_df
#             data_source = "Analysis Result (computed from the full dataset)"
#         else:
#             df = state.get("python_result")
#             data_source = "Processed Data"

#         if df is None:
#             return {"final_response": "No processed data is available."}

#         # No data: answer without calling the LLM
#         if df.empty or df.isna().to_numpy().all():
#             return {
#                 "final_response": "No records were found for this request.",
#                 "error": None,
#             }

#         total_rows = len(df)

#         facts = build_facts(df) or "No additional facts."
#         data_preview = "FACTS (computed by the application over all rows):\n" + facts

#         if total_rows <= 50:
#             data_preview += "\n\nALL ROWS:\n" + df.round(2).to_string(index=False)

#         fact_note = (
#             f"The result has {total_rows} rows. The FACTS below were computed by "
#             "the application over all of them. Use the FACTS for every highest, "
#             "lowest, total, count and combined share. Never count, add up or "
#             "compare values yourself."
#         )

#         if state.get("request_mode") == "full_results":
#             style_note = (
#                 "The user asked to see the rows. The application shows the "
#                 "table, so write only 1 or 2 key insights and do not mention "
#                 "where the table is. If the FACTS contain nothing beyond the "
#                 "row count, write one sentence only."
#             )
#         else:
#             style_note = ""

#         if total_rows == 1:
#             style_note = (
#                 "The result is a single row. Answer in one sentence that "
#                 "states the value, and add nothing else."
#             )

#         system_prompt = """
#         You are a data analysis assistant.

#         The user already sees the full result table in a Results panel
#         under your answer, so never repeat it.

#         Write insights only:
#         - Start with one sentence that directly answers the question.
#         - Then give 2 to 4 short findings based on the FACTS: the leaders,
#           the lowest, the gap or concentration.
#         - Start each finding on a new line with a dash.
#         - Plain text only: no Markdown headings, bold text or tables.
#         - Never write a table or a row-by-row list.
#         - Use only the FACTS and the rows shown. Never invent, estimate,
#           count, add up or re-sort values.
#         - Only state what the data shows. Do not guess reasons and do not
#           comment on key, code or ID columns.
#         - Round numbers to at most 2 decimals and use thousands separators
#           (for example 15,722,077 and 17.62%).
#         - When numbers are quantities, call them "sales quantity" or "units",
#           never just "sales". If you give a percentage, say what it is a
#           share of (for example "share of total quantity").
#         - Do not add currency symbols such as $, ₹ or €.
#         - Never reply with only a record count. Always state findings.
#         - If the result is empty, write one sentence only.
#         - Do not generate SQL or Python code.
#         - Only if the question itself uses the word "customer" or "customers":
#           answer using retailers and note in one short sentence that
#           customers are represented by retailers. Otherwise never mention
#           customers.
#         - "Purchases" means revenue (quantity times unit sale price).
#         - If the question gave an assumption value (such as a tax rate or
#           cost percentage), restate it in one short sentence.
#         - State how many distinct values or categories exist only if the
#           FACTS give that number.
#         - Call an item "least common" or "lowest" only if the FACTS label
#           it that way.
#         - Use the percentages given in the FACTS. Never calculate your own.
#         - Never say that all rows are listed in the table. Say how many rows
#           the result contains (for example "The result contains 556 retailers").
#         - Refer to the data in business terms (for example "retailers"),
#           not "rows" or "columns". Never quote a column name such as
#           retailer_type_count.
#         """

#         human_prompt = f"""
#         User Question:
#         {question}

#         Data Source:
#         {data_source}

#         {fact_note}

#         {style_note}

#         {data_preview}

#         Based on the facts, provide the final response.
#         """

#         response = llm.invoke([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=human_prompt),
#         ])

#         text = response.content

#         # Safety net: remove any Markdown table lines the model wrote anyway
#         text = re.sub(r"(?m)^[ \t]*\|.*\|[ \t]*\r?\n?", "", text).strip()

#         if not text:
#             text = "The results are shown in the table below."

#         notes = []

#         if state.get("request_mode") == "full_results":
#             notes.append("The full table is listed in the Results panel below.")

#         if total_rows > DISPLAY_ROWS:
#             notes.append(
#                 f"Showing the first {DISPLAY_ROWS} of {total_rows} rows. "
#                 "Use the download link under the table to get all rows."
#             )

#         if notes:
#             text = text + "\n\n" + " ".join(notes)

#         return {"final_response": text, "error": None}

#     except Exception as e:
#         return {
#             "final_response": (
#                 "I couldn't complete that request. "
#                 "The technical details are shown below."
#             ),
#             "error": str(e),
#         }

# -----------------------------------------------------------------------
# TRIAL CODE
# -----------------------------------------------------------------------

def build_facts(df: pd.DataFrame, max_measures: int = 3) -> str:
    """Describe the rows in the final result passed to this function."""

    n_rows = len(df)
    lines = [
        f"Returned rows: {n_rows}",
        f"Columns: {', '.join(str(c) for c in df.columns)}",
        "Scope: statistics below describe the returned result.",
    ]

    if df.empty:
        return "\n".join(lines)

    number_cols = list(df.select_dtypes("number").columns)
    measure_cols = [c for c in number_cols if not ID_COLUMN.search(str(c))]
    text_cols = [c for c in df.columns if c not in number_cols]

    if text_cols:
        labels = df[text_cols].astype(str).agg(" / ".join, axis=1)
    else:
        labels = pd.Series(df.index.astype(str), index=df.index)

    def fmt_item(row) -> str:
        # Use the value already present in this column.
        return f"{row.label} ({fmt_value(row.value)})"

    for col in measure_cols[:max_measures]:
        tmp = pd.DataFrame({
            "label": labels,
            "value": df[col],
        }).dropna(subset=["value"])

        if tmp.empty:
            continue

        if tmp["value"].nunique() == 1:
            lines.append(
                f"{col}: all {len(tmp)} non-null values equal "
                f"{fmt_value(tmp['value'].iloc[0])}"
            )
            continue

        tmp = tmp.sort_values("value", ascending=False)
        count = len(tmp)
        rank_count = min(3, count)

        highest = ", ".join(
            fmt_item(row) for row in tmp.head(rank_count).itertuples()
        )
        lowest = ", ".join(
            fmt_item(row)
            for row in tmp.tail(rank_count).iloc[::-1].itertuples()
        )

        lines.append(f"{col}: highest {rank_count} in the result = {highest}")
        lines.append(f"{col}: lowest {rank_count} in the result = {lowest}")

        if not NON_ADDITIVE.search(str(col)):
            total = tmp["value"].sum()
            lines.append(
                f"{col}: sum across {count} returned rows with a value "
                f"= {fmt_value(total)}"
            )

        if count >= 3:
            mean_val = tmp["value"].mean()
            median_val = tmp["value"].median()
            above_mean = int((tmp["value"] > mean_val).sum())

            lines.append(
                f"{col}: average = {fmt_value(mean_val)}, "
                f"median = {fmt_value(median_val)} "
                f"({above_mean} of {count} above average)"
            )

        first = tmp.iloc[0]
        second = tmp.iloc[1]
        last = tmp.iloc[-1]

        if second["value"] > 0:
            ratio = first["value"] / second["value"]
            lines.append(
                f"{col}: {first['label']} is {ratio:.2f}x "
                f"second-place {second['label']} in the result"
            )

        if last["value"] > 0:
            ratio = first["value"] / last["value"]
            lines.append(
                f"{col}: {first['label']} is {ratio:.2f}x "
                f"lowest {last['label']} in the result"
            )

    def describe_counts(counts) -> str:
        return ", ".join(
            f"{name} = {count} ({count / n_rows * 100:.2f}% of returned rows)"
            for name, count in counts.items()
        )

    for col in text_cols:
        distinct = df[col].nunique()

        if 1 < distinct <= 15 and distinct < n_rows:
            counts = df[col].value_counts()

            if distinct <= 8:
                lines.append(
                    f"{col}: {distinct} distinct values, "
                    f"all listed: {describe_counts(counts)}"
                )
            else:
                lines.append(
                    f"{col}: {distinct} distinct values. "
                    f"Most frequent five: {describe_counts(counts.head(5))}"
                )
                lines.append(
                    f"{col}: least frequent two (smallest first): "
                    f"{describe_counts(counts.tail(2).iloc[::-1])}"
                )

    return "\n".join(lines)


def final_response_node(state: dict) -> dict:
    """Generate a concise, formatted insights answer from facts computed in Python."""

    error = state.get("error")
    if error and str(error).startswith("UNSUPPORTED:"):
        reason = str(error).split(":", 1)[1].strip()
        return {
            "final_response": (
                "I can't answer that from the available data. "
                f"{reason} "
                "If you want an estimate, include your own assumption in "
                "the question, for example: \"assume cost is 60% of revenue\"."
            ),
            "error": None,
        }

    try:
        question = state["question"]

        if state.get("error"):
            return {
                "final_response": (
                    "I couldn't complete that request. "
                    "The technical details are shown below."
                )
            }

        analysis_df = state.get("analysis_result")
        if analysis_df is not None:
            df = analysis_df
            data_source = "Analysis Result (computed from the full dataset)"
        else:
            df = state.get("python_result")
            data_source = "Processed Data"

        if df is None:
            return {"final_response": "No processed data is available."}

        if df.empty or df.isna().to_numpy().all():
            return {
                "final_response": "No records were found for this request.",
                "error": None,
            }

        total_rows = len(df)
        facts = build_facts(df) or "No additional facts."
        data_preview = "FACTS (computed by the application over all rows):\n" + facts

        if total_rows <= 25:
            data_preview += "\n\nALL ROWS:\n" + df.round(2).to_string(index=False)

        fact_note = (
            f"The final result contains {total_rows} rows. "
            "Statistics computed by build_facts describe only these returned rows. "
            "For a requested percentage or contribution, use the value already "
            "calculated by SQL or Pandas when it is present in the supplied facts "
            "or result rows. Do not recalculate it from the displayed subset. "
            "If the denominator is unclear, do not claim it covers all sales."
        )

        if total_rows == 1:
            style_note = (
                "The result is a single row. Answer in one clear sentence stating the value."
            )
        elif state.get("request_mode") == "full_results":
            style_note = (
                "The user asked to see the full records. Write 1 or 2 brief key takeaways "
                "without mentioning the Results panel or table."
            )
        else:
            style_note = (
                "Structure the response as:\n"
                "1. One opening sentence that directly answers the question and naturally includes "
                "the total and how many entities/items are covered (do NOT tack on 'The result contains X rows' at the end).\n"
                "2. 3 to 4 bullet points (each starting with '- ') with bold lead-in labels highlighting: "
                "the top leader and its gap/ratio vs. #2 or lowest, concentration/average context, and the lowest performers."
            )

        system_prompt = """
You are a clear, concise business data analyst.
The user already sees the full table below your answer, so never repeat it row by row.

Formatting (clean Markdown):
- Start with one crisp summary sentence, bolding the headline total and scope (e.g., "Across **21 countries**, total sales quantity is **89,237,091 units**.").
- Follow with 3 to 4 bullet points using "- " with bold lead-in labels:
  - **Top performer:** ...
  - **Concentration & spread:** ...
  - **Lowest performers:** ...
- Bold key entity names and headline numbers, without bolding entire sentences.
- Never include Markdown tables or code fences (```).

Grounding & Precision:
- Use only the FACTS and rows provided. Never invent reasons, guess causes, or comment on key/code/ID columns.
- Round numbers to at most 2 decimals with thousands separators (e.g., 15,722,077 and 17.62%).
- Call quantities "sales quantity" or "units", never just "sales". State what every percentage is a share of.
- Do not add currency symbols ($, ₹, €).
- Refer to data in business terms (e.g., "countries", "retailers"), never "rows" or "columns", and never quote raw column names.
- Only mention "customers" (noting they are represented by retailers) if the question itself uses the word "customer" or "customers".
- If the question provided an assumption (such as a tax rate or cost percentage), restate it in one short sentence.
- When a result contains a subtotal and a precomputed percentage, describe
  them separately. Call the subtotal "combined revenue of the returned
  retailers" or use the corresponding entity and measure. Preserve the
  denominator used for the percentage. If its scope is unclear, report
  the supplied percentage without attributing it to the displayed subtotal.
- Qualify highest and lowest statements with the result scope. For a top-five
  list, say "fifth among the top five" rather than "lowest-performing retailer."
        """

        human_prompt = f"""
User Question:
{question}

Data Source:
{data_source}

{fact_note}

{style_note}

{data_preview}
"""

        response = invoke_tracked("answer", llm, [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        text = response.content

        # Safety net: remove any Markdown table lines the model wrote anyway
        text = re.sub(r"(?m)^[ \t]*\|.*\|[ \t]*\r?\n?", "", text).strip()

        if not text:
            text = "The results are shown in the table below."

        notes = []
        if total_rows > DISPLAY_ROWS:
            notes.append(
                f"Showing the first {DISPLAY_ROWS} of {total_rows} rows. "
                "Use the download link under the table to get all rows."
            )

        if notes:
            text = text + "\n\n" + " ".join(notes)

        return {"final_response": text, "error": None}

    except Exception as e:
        return {
            "final_response": (
                "I couldn't complete that request. "
                "The technical details are shown below."
            ),
            "error": str(e),
        }
