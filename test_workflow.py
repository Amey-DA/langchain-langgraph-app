
from graph.workflow import build_graph

# Build the LangGraph workflow
app = build_graph()

# Initial state
initial_state = {
    "question": "Show the top 10 products by revenue",
    "request_mode": "analysis",
    "sql_query": "",
    "sql_result": None,
    "python_code": "",
    "python_result": None,
    "final_response": "",
    "error": None
}

# Execute the complete workflow
result = app.invoke(initial_state)

print("\n========== FINAL WORKFLOW OUTPUT ==========")

print("\n1. Generated SQL:")
print(result["sql_query"])

print("\n2. SQL Result:")
print(result["sql_result"])

print("\n3. Generated Python Code:")
print(result["python_code"])

print("\n4. Python Result:")
print(result["python_result"])

print("\n5. Final LLM Response:")
print(result["final_response"])

print("\n6. Error:")
print(result["error"])
