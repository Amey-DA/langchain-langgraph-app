import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from graph.workflow import build_graph
from app import llm


# Compile the LangGraph workflow once
graph = build_graph()


CLASSIFIER_PROMPT = """
You classify questions asked to a sales data assistant.
Return ONLY a JSON object with two boolean fields:

{"wants_raw_rows": <bool>, "needs_analysis": <bool>}

needs_analysis = true when the question asks for something that must be
computed on top of the retrieved rows: ranking, best/worst, top/bottom
within groups, outliers, comparisons between groups, percentage or share of
total, growth, averages or other statistics, trends, correlations, or
"identify / find which ...".

wants_raw_rows = true when the user mainly wants records listed or displayed
and SQL alone can answer it (a list, a group-by total, "top N by X" where N
is stated). 
Both can be true: when the user asks to see rows AND wants a computation or insight on them.

Examples:
"Show total revenue by product line" -> {"wants_raw_rows": true, "needs_analysis": false}
"List all customers" -> {"wants_raw_rows": true, "needs_analysis": false}
"Show the top 10 products by revenue" -> {"wants_raw_rows": true, "needs_analysis": false}
"What is the total sales quantity?" -> {"wants_raw_rows": false, "needs_analysis": false}
"Rank retailer sites within each country" -> {"wants_raw_rows": false, "needs_analysis": true}
"Which product and order method combinations perform worst?" -> {"wants_raw_rows": false, "needs_analysis": true}
"What share of revenue comes from the top 20 customers?" -> {"wants_raw_rows": false, "needs_analysis": true}
"List all customers and identify the top 10 by purchases" -> {"wants_raw_rows": true, "needs_analysis": true}
"""

_ANALYSIS_RE = re.compile(
    r"\b(rank\w*|best|worst|top|bottom|highest|lowest|outlier\w*|compar\w*|"
    r"trend\w*|averag\w*|mean|median|percent\w*|share|contribut\w*|growth|"
    r"grow\w*|distribution|correlat\w*|identify|which|perform\w*|analy[sz]\w*|"
    r"summar\w*|insight\w*|calculate|ratio|differ\w*)\b",
    re.IGNORECASE,
)

_RAW_RE = re.compile(
    r"\b(show|display|list|return|give me|print|all rows|all records|"
    r"every row|every record|full table|complete table|raw)\b",
    re.IGNORECASE,
)


def _keyword_fallback(question: str) -> dict:
    """Used only if the LLM classifier fails."""
    needs_analysis = bool(_ANALYSIS_RE.search(question))
    wants_raw_rows = bool(data.get("wants_raw_rows", False))
    return {"wants_raw_rows": wants_raw_rows, "needs_analysis": needs_analysis}


def classify_request(question: str) -> dict:
    """Decide whether the user wants raw rows and/or analysis."""
    try:
        response = llm.invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=question),
        ])

        match = re.search(r"\{.*?\}", response.content, re.DOTALL)
        data = json.loads(match.group(0))

        needs_analysis = bool(data.get("needs_analysis", False))
        wants_raw_rows = bool(data.get("wants_raw_rows", False)) and not needs_analysis

        return {"wants_raw_rows": wants_raw_rows, "needs_analysis": needs_analysis}

    except Exception as error:
        print(f"[CLASSIFIER] LLM classification failed ({error}); using keywords.")
        return _keyword_fallback(question)


def run_question(question: str) -> dict:
    """
    Execute the complete LangGraph workflow
    for the supplied user question.
    """

    classification = classify_request(question)
    print(f"\n[CLASSIFIER] {classification}")

    analysis_required = classification["needs_analysis"]
    request_mode = "full_results" if classification["wants_raw_rows"] else "analysis"

    # Prepare initial state
    initial_state = {
        "question": question,
        "request_mode": request_mode,

        "sql_query": "",
        "sql_result": None,

        "python_code": "",
        "python_result": None,

        "analysis_required": analysis_required,
        "analysis_code": "",
        "analysis_result": None,
        "analysis_attempts": 0,

        "final_response": "",
        "error": None
    }

    # Execute LangGraph
    result = graph.invoke(initial_state)

    return result