import io
import os
import re
import sys
import json
import tempfile
import subprocess
import pyodbc
import pandas as pd

from contextlib import closing
from decimal import Decimal
from pathlib import Path
from langchain.agents.middleware import dynamic_prompt
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langchain.tools import tool
from langchain.agents import create_agent
from rag_application.retriever import search_documents


# --------------------------------------------------
# 0. Database metadata
# --------------------------------------------------

METADATA_FILE = Path(__file__).parent / "database_metadata.json"


def load_database_metadata():
    """Load database schema metadata from the JSON file."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


DATABASE_METADATA = load_database_metadata()
print(f"Loaded metadata for {len(DATABASE_METADATA.get('tables', []))} tables.")


# --------------------------------------------------
# 1. Environment
# --------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

api_key = os.getenv("WATSONX_PROJECT_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")

if not api_key or not project_id:
    raise ValueError("Missing Watsonx API key or Project ID. Check the .env file.")


# --------------------------------------------------
# 2. LLM
# --------------------------------------------------

llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    api_key=api_key,
    project_id=project_id,
    params={"temperature": 0, "max_tokens": 2000},
)
print("IBM watsonx model initialized successfully.")


# --------------------------------------------------
# 3. SQL Server connection
# --------------------------------------------------

def get_sql_connection():
    """Connect to the configured SQL Server database."""
    server = os.getenv("SQL_SERVER", "localhost")
    database = os.getenv("SQL_DATABASE", "GOSALESDW_SIM")
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")

    if not username or not password:
        raise ValueError("SQL_USERNAME or SQL_PASSWORD is missing from .env")

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};DATABASE={database};"
        f"UID={username};PWD={password};"
        "Encrypt=yes;TrustServerCertificate=yes;Connection Timeout=10;"
    )
    return pyodbc.connect(connection_string, timeout=10)


# --------------------------------------------------
# 4. Application state and display helpers
# --------------------------------------------------

LAST_SQL_RESULT = None            # full raw SQL result (dict)
LAST_TRANSFORMED_RESULT = None    # DataFrame produced by python_analysis
CURRENT_REQUEST_MODE = "analysis" # "full_results" or "analysis"

SQL_CALLS_THIS_REQUEST = 0
PY_CALLS_THIS_REQUEST = 0
MAX_SQL_CALLS = 3
MAX_PY_CALLS = 3


def requests_complete_results(question: str) -> bool:
    """Identify requests that need the actual rows displayed."""
    q = question.lower()

    summary_terms = (
        "summarize", "summary", "briefing", "brief", "insights", "analyze",
        "analysis", "trend", "explain", "what does", "which is highest",
        "top performers", "compare","top",
    )
    if any(term in q for term in summary_terms):
        return False

    full_result_terms = (
        "show", "display", "list", "return", "give me", "print",
        "all rows", "all records", "every row", "every record",
        "full table", "complete table", "raw results",
        "products", "retailers", "customers", "orders",
    )
    return any(term in q for term in full_result_terms)


def requests_analysis(question: str) -> bool:
    """Identify requests that require analytical processing."""

    q = question.lower()

    analysis_terms = (
        "analyze", "analyse", "analysis",
        "calculate", "compare", "comparison",
        "trend", "trends", "average",
        "percentage", "contribution",
        "highest", "lowest",
        "performance", "performing",
        "insights", "summarize", "summary",
        "total", "growth", "distribution",
        "correlation", "identify",
    )

    return any(term in q for term in analysis_terms)


def format_sql_result_as_table(result: dict) -> str:
    """Display rows as a markdown table without involving the LLM."""
    columns, rows = result["columns"], result["rows"]

    if not rows:
        return "SQL query completed successfully. No matching rows found."

    def cell(value):
        value = "NULL" if value is None else str(value)
        return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    out = [
        "| " + " | ".join(cell(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell(row.get(c)) for c in columns) + " |")

    out += ["", f"Rows displayed: {len(rows)}"]
    return "\n".join(out)


def format_for_display(df: pd.DataFrame) -> str:
    """Format a DataFrame as a table, rounding float columns to 2 decimals."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    return format_sql_result_as_table(
        {"columns": list(df.columns), "rows": df.to_dict("records"), "truncated": False}
    )


# --------------------------------------------------
# 5. Schema retrieval for the prompt
# --------------------------------------------------

def get_relevant_schema(question: str) -> str:
    """Retrieve relevant tables, columns, and declared/logical relationships."""
    question_lower = question.lower()

    tables = [
        t for t in DATABASE_METADATA.get("tables", [])
        if t.get("table_name", "").lower() != "sysdiagrams"
    ]

    table_keywords = {
        "DM_PRODUCT": ["product", "products", "item", "items", "category", "subcategory"],
        "FCT_SALES": ["sale", "sales", "quantity", "price", "revenue", "order", "orders", "sold"],
        "DM_DATE": ["date", "daily", "monthly", "yearly", "month", "year", "trend"],
        "DM_RETAILER": ["retailer", "retailers", "customer", "customers"],
        "DM_RETAILER_SITE": ["site", "location", "store", "stores"],
        "DM_EMPLOYEE": ["employee", "employees", "staff"],
        "DM_ORG": ["organization", "organisation", "department"],
        "DM_COUNTRY": ["country", "countries", "nation"],
        "DM_ORDER_METHOD": ["method", "methods", "channel"],
    }

    def norm(name: str) -> str:
        return name.split(".")[-1].upper()

    available = {t.get("table_name", "").upper() for t in tables}
    logical = DATABASE_METADATA.get("logical_relationships", [])
    selected = set()

    for table in tables:
        name = table.get("table_name", "")
        searchable = name.replace("DM_", "").replace("FCT_", "").replace("_", " ").lower()
        if any(k in question_lower for k in table_keywords.get(name, [])):
            selected.add(name.upper())
        elif searchable in question_lower:
            selected.add(name.upper())

    if any(t in question_lower for t in ["sale", "sales", "quantity", "price", "revenue", "sold"]):
        selected.add("FCT_SALES")

    changed = True
    while changed:
        changed = False
        for table in tables:
            name = table.get("table_name", "").upper()
            if name not in selected:
                continue

            for rel in table.get("relationships", []):
                related = norm(rel.get("referenced_table", ""))
                if related in available and related not in selected:
                    selected.add(related)
                    changed = True

            for rel in logical:
                a, b = norm(rel.get("from_table", "")), norm(rel.get("to_table", ""))
                if name == a:
                    related = b
                elif name == b:
                    related = a
                else:
                    continue
                if related in available and related not in selected:
                    selected.add(related)
                    changed = True

    chosen = [t for t in tables if t.get("table_name", "").upper() in selected]
    if not chosen:
        chosen = tables
        selected = {t.get("table_name", "").upper() for t in chosen}

    parts = []
    for table in chosen:
        parts.append(
            f"\nTable: {table.get('schema', 'dbo')}.{table.get('table_name', '')}\n"
            f"Description: {table.get('description', 'Not available')}"
        )
        parts.append("Columns:")
        for col in table.get("columns", []):
            parts.append(
                f"- {col.get('name')} ({col.get('data_type')})"
                f"{' [PRIMARY KEY]' if col.get('primary_key') else ''}"
            )
        for rel in table.get("relationships", []):
            parts.append(f"Declared foreign-key relationship: {rel}")

    for rel in logical:
        if norm(rel.get("from_table", "")) in selected and norm(rel.get("to_table", "")) in selected:
            parts.append(
                "Verified logical relationship: "
                f"{rel.get('from_table')}.{rel.get('from_column')} = "
                f"{rel.get('to_table')}.{rel.get('to_column')} "
                f"(Status: {rel.get('verification_status', 'unspecified')})"
            )

    return "\n".join(parts)


# --------------------------------------------------
# 6. Tools
# --------------------------------------------------

def build_dataframe_from_sql_result():
    """Build a DataFrame from LAST_SQL_RESULT (None if no result exists)."""
    if LAST_SQL_RESULT is None:
        return None

    df = pd.DataFrame(LAST_SQL_RESULT["rows"], columns=LAST_SQL_RESULT["columns"])

    # SQL Server Decimal values -> float so they survive JSON.
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, Decimal)).any():
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

@tool
def sql_query(query: str) -> str:
    """
    Query SQL Server for actual business records and numeric results.

    Use for products, sales, countries, retailers, employees,
    counts, totals, rankings, filtering, grouping, and comparisons.

    Only one read-only SELECT query is permitted. Do not do percentage,
    pivot, or other post-processing in SQL; use python_analysis for that.
    """

    global LAST_SQL_RESULT, SQL_CALLS_THIS_REQUEST

    SQL_CALLS_THIS_REQUEST += 1
    print(f"[SQL CALL {SQL_CALLS_THIS_REQUEST}] {query[:200]}")

    if SQL_CALLS_THIS_REQUEST > MAX_SQL_CALLS:
        return (
            "Query limit reached. The results are already available. "
            "Do not call sql_query again. Reply with one short sentence."
        )

    cleaned = query.strip()

    if not re.match(r"(?is)^SELECT\b", cleaned):
        return "Error: Only SELECT queries are allowed."

    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()

    if ";" in cleaned:
        return "Error: Only one SQL statement is allowed."

    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        return "Error: SQL comments are not allowed."

    forbidden = re.compile(
        r"(?i)\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|TRUNCATE|"
        r"EXEC|EXECUTE|GRANT|REVOKE|DENY|BACKUP|RESTORE|DBCC|"
        r"INTO|OPENROWSET|OPENDATASOURCE)\b"
    )
    if forbidden.search(cleaned):
        return "Error: The query contains a disallowed SQL operation."

    try:
        with closing(get_sql_connection()) as connection:
            cursor = connection.cursor()
            cursor.execute(cleaned)

            if cursor.description is None:
                return "Error: The query returned no result set."

            columns = [c[0] for c in cursor.description]
            raw_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            LAST_SQL_RESULT = {
                "columns": columns,
                "rows": raw_rows,
                "truncated": False,
                "query": cleaned,
            }
            print(f"[SQL DEBUG] Rows stored in application memory: {len(raw_rows)}")

            if not raw_rows:
                return "Query succeeded but returned 0 rows."

            next_step = (
                "The full result is stored. Do NOT call sql_query again. "
                "Call python_analysis ONLY if the user explicitly asked for a "
                "calculation the query did not already produce, such as percentage "
                "of total, pivot, running total or a new calculated column. "
                "A plain list, top-N or sorted result needs no python_analysis: "
                "reply with one short sentence."
            )

            # Full-results mode: the LLM gets the structure, not the rows.
            if CURRENT_REQUEST_MODE == "full_results":
                dtypes = build_dataframe_from_sql_result().dtypes
                col_info = ", ".join(f"{c} ({dtypes[c]})" for c in columns)
                return (
                    f"DONE. The query returned {len(raw_rows)} rows. "
                    f"Columns: {col_info}. {next_step}"
                )

            # Analysis mode: the LLM gets every row to interpret.
            lines = [
                "Columns: " + ", ".join(columns),
                f"Rows returned: {len(raw_rows)}",
            ]
            lines.extend(str(r) for r in raw_rows)
            lines.append(next_step)
            return "\n".join(lines)

    except Exception as exc:
        return f"SQL query failed: {exc}"


RUNNER_PATH = Path(__file__).parent / "sandbox_runner.py"
PYTHON_TIMEOUT_SECONDS = 20


def run_python_on_dataframe(df: pd.DataFrame, code: str):
    """Run code in a child process. Returns (result_df, error_message)."""
    payload = json.dumps(
        {"code": code, "df": df.to_json(orient="split", date_format="iso")}
    )

    # Clean environment: API keys and SQL passwords never reach the child.
    env = {
        k: os.environ[k]
        for k in ("SYSTEMROOT", "TEMP", "TMP", "PATH")
        if k in os.environ
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER_PATH)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=PYTHON_TIMEOUT_SECONDS,
            env=env,
            cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return None, f"Error: the code ran longer than {PYTHON_TIMEOUT_SECONDS} seconds."

    if proc.returncode != 0:
        return None, f"Error: execution failed. {proc.stderr[-400:]}"

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, "Error: the code produced no valid result."

    if not out.get("ok"):
        return None, f"Error: {out.get('error')}"

    result_df = pd.read_json(io.StringIO(out["df"]), orient="split", convert_dates=False)
    return result_df, None


@tool
def python_analysis(code: str) -> str:
    """
    Run pandas code on the latest SQL result for any transformation:
    percentage of total, pivot, ranking, growth, group-by, filtering,
    new calculated columns, statistics.

    The SQL result is available as a DataFrame named `df`; `pd` and `np`
    are available. Do not use imports, files, or network access.
    Assign the final DataFrame to a variable named `result`.
    Call sql_query first.
    """

    global LAST_TRANSFORMED_RESULT, PY_CALLS_THIS_REQUEST

    PY_CALLS_THIS_REQUEST += 1
    print(f"[PY CALL {PY_CALLS_THIS_REQUEST}]\n{code[:500]}")

    if PY_CALLS_THIS_REQUEST > MAX_PY_CALLS:
        return (
            "Python call limit reached. Do not call any tool again. "
            "Reply with one short sentence."
        )

    df = build_dataframe_from_sql_result()
    if df is None:
        return "Error: No SQL results are available. Call sql_query first."

    result_df, error = run_python_on_dataframe(df, code)

    if error:
        return f"{error}\nFix the code and call python_analysis once more."

    LAST_TRANSFORMED_RESULT = result_df

    if CURRENT_REQUEST_MODE == "full_results":
        return (
            f"DONE. Python analysis succeeded: {len(result_df)} rows, "
            f"columns: {', '.join(result_df.columns)}. The application will "
            "display the full table. No further tool calls are needed. "
            "Reply with one short sentence."
        )

    lines = [
        f"DONE. Python analysis succeeded: {len(result_df)} rows.",
        "Columns: " + ", ".join(result_df.columns),
    ]
    lines.extend(str(r) for r in result_df.head(50).to_dict("records"))
    lines.append("No further tool calls are needed. Summarize using these values.")
    return "\n".join(lines)


@tool
def knowledge_base(question: str) -> str:
    """
    Search the existing ChromaDB knowledge base for relevant
    technical documentation, concepts, architecture, implementation
    steps, and troubleshooting information.
    """

    try:
        results = search_documents(question, top_k=3)
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]

        if not documents[0]:
            return "No relevant document chunks were retrieved."

        chunks = []
        for i, document in enumerate(documents[0]):
            meta = metadatas[0][i] if i < len(metadatas[0]) else {}
            chunks.append(
                f"Source: {meta.get('source', 'Unknown source')}\n"
                f"Page: {meta.get('page_number', 'Unknown page')}\n"
                f"Content:\n{document}"
            )

        return "\n\n--- Retrieved Chunk ---\n\n".join(chunks)

    except Exception as exc:
        return f"Knowledge-base retrieval failed: {exc}"


# --------------------------------------------------
# 7. Agent instructions
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a data assistant with three tools: sql_query, knowledge_base,
and python_analysis.

CORE RULE

When you need data or documents, respond with a tool call immediately.
Do not write a plan, explanation, or code in your reply before calling
a tool. Write your final answer only after the tool results arrive.
When a tool result says DONE, do not call sql_query again.

TOOL SELECTION

- sql_query: fetch records, products, sales, counts, totals, rankings,
  filtering, grouping. One read-only SELECT per call. Fetch the raw
  or aggregated data only; do not compute percentages or reshape in SQL.
- python_analysis: any calculation or reshaping on the fetched data
  (percentage of total, pivot, ranking, growth, new columns, statistics).
  Use it only after sql_query, and only when the user asks for such a
  calculation.
- knowledge_base: concepts, definitions, documentation, architecture,
  implementation steps, troubleshooting.
- If a question needs several, use them in the order needed.

PYTHON ANALYSIS RULES

- The data is a DataFrame named df, with exactly the columns sql_query
  returned. pd and np are available. No imports, files, or network.
- Keep the original columns and add new ones, unless the user asks for
  a different shape (for example a pivot).
- Round percentages to 2 decimals.
- Assign the final DataFrame to a variable named result.
- Example: result = df.copy()
  result["PERCENTAGE_OF_TOTAL"] = (result["TOTAL"] / result["TOTAL"].sum() * 100).round(2)
- If python_analysis returns an error, fix the code and retry once.

ACCURACY

- Base answers only on tool results. Never invent records, numbers,
  or document facts.
- Do not add up or recompute numbers yourself; quote values shown in
  tool results.
- Do not add TOP or LIMIT unless the user asks for a specific number
  of rows.
- Use only table and column names from the provided schema.
- If a tool fails, retry once with a correction, or explain the
  limitation. Never fabricate results.
- For complete-row requests the application displays the stored rows
  itself, so do not reproduce them.

DATABASE

Database: GOSALESDW_SIM (SQL Server). Write SQL Server-compatible SQL.
Product table: dbo.DM_PRODUCT with columns PRODUCT_KEY,
PRODUCT_LINE_CODE, PRODUCT_LINE, PRODUCT_TYPE_CODE, PRODUCT_TYPE,
PRODUCT_NUMBER, PRODUCT_NAME, PRODUCT_COLOR, PRODUCT_BRAND.
"""


@dynamic_prompt
def dynamic_system_prompt(request) -> str:
    """Add relevant database schema to the system prompt."""

    question = ""
    for message in reversed(request.state["messages"]):
        if getattr(message, "type", "") == "human":
            question = message.content
            break

    if not question:
        return SYSTEM_PROMPT

    return f"""
{SYSTEM_PROMPT}

DATABASE SCHEMA FOR THIS QUESTION

{get_relevant_schema(question)}

JOIN RULES

- Follow declared foreign keys and the verified logical relationships
  listed above. Never join columns just because their names or types
  look compatible.
- Sales quantity by country uses this verified path:
  dbo.FCT_SALES.RETAILER_SITE_KEY = dbo.DM_RETAILER_SITE.RETAILER_SITE_KEY
  dbo.DM_RETAILER_SITE.RETAILER_SITE_COUNTRY_CODE = dbo.DM_COUNTRY.COUNTRY_CODE
  Use dbo.FCT_SALES.SALES_QUANTITY and dbo.DM_COUNTRY.COUNTRY.
  Do not join FCT_SALES.RETAILER_KEY to DM_COUNTRY.COUNTRY_CODE.
"""


# --------------------------------------------------
# 8. Debug helpers and safety checks
# --------------------------------------------------

DEBUG_TOOL_ACTIVITY = True
MAX_ATTEMPTS = 3
LEAK_MARKERS = ('"parameters"', "<|python", "<|header", "<|eot")


def print_tool_activity(messages):
    """Print requested tools and the results they returned."""
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            print(f"\nTool requested: {call['name']}")
            print(f"Arguments: {call.get('args', {})}")

        if getattr(message, "type", "") == "tool":
            print(f"\nTool executed: {message.name}")
            print("Actual tool result:")
            print(message.content)


def get_final_message(messages):
    """Last AI message that has no tool calls."""
    return next(
        (
            m for m in reversed(messages)
            if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", [])
        ),
        None,
    )


def is_text_tool_call(messages) -> bool:
    """True if the final reply contains raw tool-call text or special tokens."""
    final = get_final_message(messages)
    text = str(getattr(final, "content", "")) if final else ""
    return any(marker in text for marker in LEAK_MARKERS)


# --------------------------------------------------
# 9. Run the assistant
# --------------------------------------------------

if __name__ == "__main__":

    agent = create_agent(
        model=llm,
        tools=[sql_query, knowledge_base, python_analysis],
        middleware=[dynamic_system_prompt],
    )

    print("\nSQL + RAG assistant is ready.")
    print("Ask a question, or type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            CURRENT_REQUEST_MODE = (
                "full_results" if requests_complete_results(question) else "analysis"
            )

            messages = []

            for attempt in range(1, MAX_ATTEMPTS + 1):
                LAST_SQL_RESULT = None
                LAST_TRANSFORMED_RESULT = None
                SQL_CALLS_THIS_REQUEST = 0
                PY_CALLS_THIS_REQUEST = 0

                try:
                    response = agent.invoke(
                        {"messages": [{"role": "user", "content": question}]},
                        config={"recursion_limit": 12},
                    )
                    messages = response.get("messages", [])
                except Exception as agent_exc:
                    print(f"[AGENT STOPPED] {type(agent_exc).__name__}: {agent_exc}")
                    messages = []
                    break

                if not is_text_tool_call(messages):
                    break

                print(f"[RETRY] Attempt {attempt} returned raw tool-call text.")

            if DEBUG_TOOL_ACTIVITY:
                print("\n========== TOOL DEBUG TRACE ==========")
                print_tool_activity(messages)
                print("========== END TOOL DEBUG TRACE ======")

            print("\nAssistant:")

            if is_text_tool_call(messages):
                print("The model did not produce a valid answer. Please try again.")
                continue

            final_message = get_final_message(messages)

            if LAST_TRANSFORMED_RESULT is not None:
                print(format_for_display(LAST_TRANSFORMED_RESULT))
                if CURRENT_REQUEST_MODE == "analysis" and final_message:
                    print("\n" + str(final_message.content))

            elif CURRENT_REQUEST_MODE == "full_results" and LAST_SQL_RESULT:
                print(format_sql_result_as_table(LAST_SQL_RESULT))

            elif final_message is not None:
                print(final_message.content)

            else:
                print("No final assistant response was generated.")

        except Exception as exc:
            print(f"\nApplication error: {exc}")