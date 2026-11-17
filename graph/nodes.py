import re
import ast
from contextlib import closing
from decimal import Decimal

import pandas as pd

from app import llm, get_relevant_schema, get_sql_connection
from langchain_core.messages import HumanMessage, SystemMessage
from app import run_python_on_dataframe

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

def generate_sql(
    question: str,
    previous_query: str | None = None,
    previous_error: str | None = None,
) -> str:
    """
    Generate a SQL Server SELECT query using the user's
    question and relevant database schema.
    """

    schema = get_relevant_schema(question)

    retry_block = ""
    if previous_error:
        retry_block = f"""
    PREVIOUS ATTEMPT FAILED.
    Previous query:
    {previous_query}

    Database error:
    {previous_error}

    Fix the query. Use only column names that appear in the schema below.
    """

    system_prompt = f"""
    You are an expert SQL Server query generator.

    Generate a SQL Server (T-SQL) query based on the user's question.

    Use only the tables and columns provided in the schema.

    Rules:
    1. Generate only SELECT queries.
    2. Do not generate INSERT, UPDATE, DELETE, DROP or other
       modification statements.
    3. Use the provided relationships when joining tables.
    4. Do not assume columns that are not present in the schema.
    5. Return only the SQL query.
    6. Do not include explanations or Markdown code fences.
    7. Copy column names exactly as written in the schema. Never invent
       names such as COUNTRY_NAME. The country name column is
       DM_COUNTRY.COUNTRY and the region column is DM_COUNTRY.REGION.

    BUSINESS RULES:
    - Retailer's home country (for questions about retailers, with no sales
      measure): DM_RETAILER.RETAILER_COUNTRY_CODE = DM_COUNTRY.COUNTRY_CODE.
      No fact table is needed for this.
    - Country where a sale happened (for sales, quantity or revenue by
      country): FCT_SALES.RETAILER_SITE_KEY = DM_RETAILER_SITE.RETAILER_SITE_KEY,
      then DM_RETAILER_SITE.RETAILER_SITE_COUNTRY_CODE = DM_COUNTRY.COUNTRY_CODE.
    - Never join FCT_SALES.RETAILER_KEY to DM_COUNTRY.
    - Filter countries with DM_COUNTRY.COUNTRY (for example 'Canada').
      Column COUNTRY_NAME does not exist.

    {retry_block}

    DATABASE SCHEMA:
    {schema}
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content.strip()

def execute_sql_query(query: str) -> pd.DataFrame:
    """
    Validate and execute a read-only SQL query.
    Return the retrieved records as a DataFrame.
    """

    cleaned = query.strip()

    # 1. Allow only SELECT queries
    if not re.match(r"(?is)^SELECT\b", cleaned):
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
    """
    Generate SQL, execute it, and update the graph state.
    Retries up to 3 times, passing the database error back to the LLM.
    """

    question = state["question"]

    last_query = ""
    last_error = None

    for attempt in range(1, 4):
        query = ""

        try:
            query = clean_sql_query(
                generate_sql(question, last_query, last_error)
            )

            print(f"\n[GENERATED SQL, attempt {attempt}]")
            print(query)

            df = execute_sql_query(query)

            print(f"\n[SQL SUCCESS] Retrieved {len(df)} rows.")

            return {
                "sql_query": query,
                "sql_result": df,
                "error": None
            }

        except Exception as error:
            last_query = query
            last_error = str(error)
            print(f"\n[SQL ATTEMPT {attempt} FAILED] {error}")

    return {
        "sql_query": last_query,
        "sql_result": None,
        "error": f"The SQL query failed after 3 attempts. Last error: {last_error}"
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

    response = llm.invoke([
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

    response = llm.invoke([
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

        if (
            state.get("request_mode") == "full_results"
            and not state.get("analysis_required", False)
        ):

            print("\n[PYTHON SKIPPED] Raw-row request: using the SQL result as is.")
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

def summarize_for_llm(df: pd.DataFrame, head_rows: int = 25, tail_rows: int = 5) -> str:
    parts = [
        f"Total rows: {len(df)}",
        f"Columns: {list(df.columns)}",
    ]
 
    numeric = df.select_dtypes("number")
    if not numeric.empty:
        parts.append("Numeric summary (all rows):\n" + numeric.describe().round(2).to_string())
 
    if len(df) <= head_rows + tail_rows:
        parts.append("All rows:\n" + df.to_string(index=False))
    else:
        parts.append(f"First {head_rows} rows:\n" + df.head(head_rows).to_string(index=False))
        parts.append(f"Last {tail_rows} rows:\n" + df.tail(tail_rows).to_string(index=False))
 
    return "\n\n".join(parts)

def final_response_node(state: dict) -> dict:
    """Generate a natural-language response using the processed DataFrame."""
 
    try:
        question = state["question"]
 
        if state.get("error"):
            return {
                "final_response": (
                    "I couldn't complete that request. "
                    f"Details: {state['error']}"
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
 
        total_rows = len(df)
        is_preview = total_rows > 50
 
        if is_preview:
            data_preview = summarize_for_llm(df)
            preview_note = (
                f"The data has {total_rows} rows. Only a summary and a preview "
                "are shown below. Base conclusions on the summary statistics and "
                "the rows shown, and say clearly that you are describing a preview "
                "of a larger result. Do not claim anything about rows you cannot see."
            )
        else:
            data_preview = df.to_string(index=False)
            preview_note = "The complete result is shown below."
 
        system_prompt = """
        You are a data analysis assistant.
 
        Answer the user's question using only the processed data provided.
 
        Strict instructions:
        - Start with the direct answer to the question, then support it.
        - Use only values available in the processed data.
        - Never invent, estimate, or modify numerical values.
        - Do not add currency symbols such as $, ₹, or €.
        - Display numerical values as plain numbers.
        - Name specific items (products, countries, retailers) in your answer.
        - Never reply with only a record count. Always state findings.
        - Highlight important findings where relevant.
        - If the result is empty, clearly mention that no records were found.
        - Do not generate SQL or Python code.
        """
 
        human_prompt = f"""
        User Question:
        {question}
 
        Data Source:
        {data_source}
 
        {preview_note}
 
        Processed Data:
        {data_preview}
 
        Number of records: {total_rows}
 
        Based on the processed data, provide the final response.
        """
 
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
 
        return {"final_response": response.content, "error": None}
 
    except Exception as e:
        return {"final_response": "", "error": str(e)}




