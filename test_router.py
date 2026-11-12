import pandas as pd
from graph.nodes import route_after_python


def test_router(name, state, expected):
    result = route_after_python(state)

    status = "PASS" if result == expected else "FAIL"

    print(f"{status} | {name} | Expected: {expected} | Got: {result}")


# Test 1: Analysis not required
test_router(
    "Analysis not required",
    {
        "analysis_required": False,
        "python_result": pd.DataFrame({"sales": range(100)}),
        "analysis_result": None,
        "error": None,
        "analysis_attempts": 0
    },
    "final_response_node"
)

# Test 2: Dataset has fewer than 50 rows
test_router(
    "Small dataset",
    {
        "analysis_required": True,
        "python_result": pd.DataFrame({"sales": range(30)}),
        "analysis_result": None,
        "error": None,
        "analysis_attempts": 0
    },
    "final_response_node"
)

# Test 3: Analysis required and dataset exceeds 50 rows
test_router(
    "Large dataset requiring analysis",
    {
        "analysis_required": True,
        "python_result": pd.DataFrame({"sales": range(100)}),
        "analysis_result": None,
        "error": None,
        "analysis_attempts": 0
    },
    "analysis_node"
)

# Test 4: Analysis completed
test_router(
    "Analysis completed",
    {
        "analysis_required": True,
        "python_result": pd.DataFrame({"sales": range(100)}),
        "analysis_result": pd.DataFrame({"total": [500]}),
        "error": None,
        "analysis_attempts": 1
    },
    "final_response_node"
)

# Test 5: First analysis attempt failed
test_router(
    "Retry analysis",
    {
        "analysis_required": True,
        "python_result": pd.DataFrame({"sales": range(100)}),
        "analysis_result": None,
        "error": "Execution error",
        "analysis_attempts": 1
    },
    "analysis_node"
)

# Test 6: Maximum attempts reached
test_router(
    "Maximum attempts reached",
    {
        "analysis_required": True,
        "python_result": pd.DataFrame({"sales": range(100)}),
        "analysis_result": None,
        "error": "Execution error",
        "analysis_attempts": 2
    },
    "final_response_node"
)