import re
from contextlib import closing
from decimal import Decimal

import pandas as pd

from app import llm, get_relevant_schema, get_sql_connection
from langchain_core.messages import HumanMessage, SystemMessage
from app import run_python_on_dataframe

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


def generate_python_code(
    question: str,
    df: pd.DataFrame,
    request_mode: str
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

    Your task is to generate Python code to process
    a DataFrame based on the user's requirement.

    RULES:
    1. The SQL result is already available as a DataFrame named df.
    2. Pandas is already imported as pd.
    3. Do not import any libraries.
    4. Do not access files, databases, or networks.
    5. Do not use print statements.
    6. Assign the final DataFrame to a variable named result.
    7. Return only executable Python code.
    8. Do not include Markdown code fences.

    REQUEST MODE: {request_mode}

    DATAFRAME INFORMATION:
    Columns: {columns}
    Data types: {data_types}
    Total rows: {len(df)}

    SAMPLE DATA:
    {sample_data}

    For full_results mode, preserve all records and columns
    unless the user's request requires a transformation.
    For analysis mode, perform the analysis requested by the user.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content.strip()


def python_node(state: dict) -> dict:
    """
    Generate Python code, execute it on the SQL DataFrame,
    and update the graph state.
    """

    try:
        # Step 1: Retrieve SQL result from graph state
        df = state["sql_result"]

        if df is None:
            raise ValueError("No SQL result available for Python analysis.")

        # Step 2: Generate Python code
        code = generate_python_code(
            question=state["question"],
            df=df,
            request_mode=state["request_mode"]
        )

        print("\n[GENERATED PYTHON CODE]")
        print(code)

        # Step 3: Execute generated code using existing sandbox
        result_df, error = run_python_on_dataframe(df, code)

        if error:
            raise RuntimeError(error)

        # Step 4: Return updated graph state
        print(f"\n[PYTHON SUCCESS] Processed {len(result_df)} rows.")

        return {
            "python_code": code,
            "python_result": result_df,
            "error": None
        }

    except Exception as error:
        print(f"\n[PYTHON ERROR] {error}")

        return {
            "python_code": "",
            "python_result": None,
            "error": str(error)
        }


def final_response_node(state: dict) -> dict:
    """Generate a natural-language response using the processed DataFrame."""

    try:
        question = state["question"]
        df = state["python_result"]

        if state.get("error"):
            return {
                "final_response": f"Unable to complete the request: {state['error']}"
            }

        if df is None:
            return {
                "final_response": "No processed data is available."
            }

        # Convert the processed DataFrame into text for the LLM
        data_preview = df.to_string(index=False)

        system_prompt = """
        

        You are a data analysis assistant.

        Answer the user's question using only the processed data provided.

        Strict instructions:
        - Provide a clear and concise explanation.
        - Use only values available in the processed data.
        - Never invent, estimate, or modify numerical values.
        - Do not add currency symbols such as $, ₹, or €.
        - Do not assume any currency unless explicitly specified in the data.
        - Display numerical values as plain numbers.
        - Do not claim that results represent all available records
        unless the data explicitly confirms this.
        - Highlight important findings where relevant.
        - If the result is empty, clearly mention that no records were found.
        - Do not generate SQL or Python code.
"""

    

        human_prompt = f"""
        User Question:
        {question}

        Processed Data:
        {data_preview}

        Number of records: {len(df)}

        Based on the processed data, provide the final response.
        """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        return {
            "final_response": response.content,
            "error": None
        }

    except Exception as e:
        return {
            "final_response": "",
            "error": str(e)
        }



