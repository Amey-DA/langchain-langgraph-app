
from langchain_core.messages import HumanMessage, SystemMessage

from app import llm, get_relevant_schema


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
