import re
from contextlib import closing
from decimal import Decimal

import pandas as pd

from app import llm, get_relevant_schema, get_sql_connection
from langchain_core.messages import HumanMessage, SystemMessage

def generate_sql(question: str) -> str:
    """
    Generate a SQL Server SELECT query using the user's
    question and relevant database schema.
    """

    schema = get_relevant_schema(question)

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
            rows = cursor.fetchall()

            df = pd.DataFrame.from_records(rows, columns=columns)

    # 7. Convert Decimal columns to numeric values
    for column in df.columns:
        if df[column].map(lambda value: isinstance(value, Decimal)).any():
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

def sql_node(state: dict) -> dict:
    """
    Generate SQL, execute it, and update the graph state.
    """

    question = state["question"]

    try:
        # Step 1: Generate SQL
        query = generate_sql(question)

        print("\n[GENERATED SQL]")
        print(query)

        # Step 2: Execute SQL
        df = execute_sql_query(query)

        print(f"\n[SQL SUCCESS] Retrieved {len(df)} rows.")

        # Step 3: Return updated state
        return {
            "sql_query": query,
            "sql_result": df,
            "error": None
        }

    except Exception as error:
        print(f"\n[SQL ERROR] {error}")

        return {
            "sql_query": "",
            "sql_result": None,
            "error": str(error)
        }


