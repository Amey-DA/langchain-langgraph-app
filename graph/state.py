
from typing import TypedDict
import pandas as pd


class AgentState(TypedDict):
    question: str
    request_mode: str
    sql_query: str
    sql_result: pd.DataFrame | None
    python_code: str
    python_result: pd.DataFrame | None
    final_response: str
    error: str | None
