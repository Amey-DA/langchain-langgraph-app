
from typing import TypedDict
import pandas as pd


class AgentState(TypedDict):
    question: str
    request_mode: str

    # SQL execution
    sql_query: str
    sql_result: pd.DataFrame | None

    # Initial Python processing
    python_code: str
    python_result: pd.DataFrame | None

    # Analysis workflow
    analysis_required: bool
    analysis_code: str
    analysis_result: pd.DataFrame | None
    analysis_attempts: int

    # Final response
    final_response: str
    error: str | None
