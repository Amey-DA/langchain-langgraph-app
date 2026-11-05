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

def clean_sql_query(query: str) -> str:
    """Remove Markdown SQL code fences from the generated query."""

    query = query.strip()

    lines = query.splitlines()

    # Remove opening code fence
    if lines and lines[0].strip().lower() in ("```sql", "```"):
        lines = lines[1:]

    # Remove closing code fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()

def sql_node(state: dict) -> dict:
    """
    Generate SQL, execute it, and update the graph state.
    """

    question = state["question"]

    try:
        # Step 1: Generate SQL
        query = generate_sql(question)

        # Step 2: Clean generated SQL
        query = clean_sql_query(query)

        print("\n[GENERATED SQL]")
        print(query)

        # Step 3: Execute SQL
        df = execute_sql_query(query)

        print(f"\n[SQL SUCCESS] Retrieved {len(df)} rows.")

        # Step 4: Return updated state
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

    Your task is to generate Python code to analyze the
    complete DataFrame based on the user's question.

    IMPORTANT:
    The sample data is provided only to understand the
    structure and contents of the DataFrame.
    The actual execution will happen on the COMPLETE DataFrame.

    RULES:
    1. The complete dataset is available as a DataFrame named df.
    2. Pandas is already imported as pd.
    3. Do not import any libraries.
    4. Do not access files, databases, or networks.
    5. Do not use print statements.
    6. Assign the final analysis result to a variable named result.
    7. Result MUST always be a Pandas DataFrame.
    8. Wrap scalar results in a DataFrame.
    9. Return only executable Python code.
    10. Do not include Markdown code fences.
    11. Perform calculations using the complete DataFrame.
    12. Do not calculate results using only the sample data.

    DATAFRAME INFORMATION:
    Columns: {columns}
    Data types: {data_types}
    Total rows: {len(df)}

    SAMPLE DATA (first 20 rows):
    {sample_data}

    PREVIOUS EXECUTION ERROR:
    {previous_error if previous_error else "No previous error. This is the first attempt."}

    If a previous execution error exists, correct the code
    to address that error.
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    return response.content.strip()

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

            print(
                f"\n[ANALYSIS SUCCESS] Generated {len(result_df)} result rows."
            )

            return {
                "analysis_result": result_df,
                "analysis_code": "",
                "error": None
            }

        # MODE 1: Initial Python processing
        df = state["sql_result"]

        if df is None:
            raise ValueError("No SQL result available for Python processing.")

        code = generate_python_code(
            question=state["question"],
            df=df,
            request_mode=state["request_mode"]
        )

        print("\n[GENERATED PYTHON CODE]")
        print(code)

        result_df, error = run_python_on_dataframe(df, code)

        if error:
            raise RuntimeError(error)

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
            "error": None
        }

    except Exception as error:

        print(f"\n[ANALYSIS NODE ERROR] {error}")

        return {
            "analysis_attempts": attempt,
            "error": str(error)
        }

def route_after_python(state: dict) -> str:
    """Route based on analysis requirement and execution status."""

    # Analysis completed successfully
    if state.get("analysis_result") is not None:
        print("\n[ROUTER] Analysis completed. Moving to Final Response.")
        return "final_response_node"

    # Handle errors
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

    # Check initial Python processing result
    df = state.get("python_result")

    if df is None:
        return "final_response_node"

    row_threshold = 50

    if (
        state.get("analysis_required", False)
        and len(df) > row_threshold
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


def final_response_node(state: dict) -> dict:
    """Generate a natural-language response using the processed DataFrame."""

    try:
        question = state["question"]

        if state.get("error"):
            return {
                "final_response": (
                    f"Unable to complete the request: {state['error']}"
                )
            }

        # Prefer analysis output when available
        analysis_df = state.get("analysis_result")

        if analysis_df is not None:
            df = analysis_df
            data_source = "Analysis Result"

        else:
            df = state.get("python_result")
            data_source = "Processed Data"

        if df is None:
            return {
                "final_response": "No processed data is available."
            }

        # Avoid sending large raw datasets to the LLM
        if len(df) > 50 and analysis_df is None:
            return {
                "final_response": (
                    f"Successfully retrieved {len(df)} records. "
                    "The complete processed dataset is available "
                    "separately in the results output."
                ),
                "error": None
            }

        # Convert selected DataFrame into text
        data_preview = df.to_string(index=False)

        system_prompt = """
        You are a data analysis assistant.

        Answer the user's question using only the processed data provided.

        Strict instructions:
        - Provide a clear and concise explanation.
        - Use only values available in the processed data.
        - Never invent, estimate, or modify numerical values.
        - Do not add currency symbols such as $, ₹, or €.
        - Do not assume any currency unless explicitly specified.
        - Display numerical values as plain numbers.
        - Do not claim results represent all available records
          unless the data explicitly confirms this.
        - Highlight important findings where relevant.
        - If the result is empty, clearly mention that no records were found.
        - Do not generate SQL or Python code.
        """

        human_prompt = f"""
        User Question:
        {question}

        Data Source:
        {data_source}

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



