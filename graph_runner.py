from graph.workflow import build_graph
from llm_tracking import track_question


# Compile once and reuse the graph across questions.
graph = build_graph()


@track_question
def run_question(question: str) -> dict:
    """Run one analytics question and track all model calls for it."""

    initial_state = {
        "question": question,

        # The SQL node sets these from the validated plan.
        "request_mode": "analysis",
        "analysis_required": False,

        "sql_query": "",
        "sql_result": None,

        "python_code": "",
        "python_result": None,

        "analysis_code": "",
        "analysis_result": None,
        "analysis_attempts": 0,

        "final_response": "",
        "error": None,
    }

    return graph.invoke(initial_state)