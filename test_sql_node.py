
from graph.nodes import sql_node

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

result = sql_node(initial_state)

print("\n--- SQL NODE OUTPUT ---")
print("Generated SQL:", result["sql_query"])
print("Error:", result["error"])

if result["sql_result"] is not None:
    print("\nRetrieved Data:")
    print(result["sql_result"].head(10))
    print("\nTotal rows:", len(result["sql_result"]))
