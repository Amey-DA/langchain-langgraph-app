
from graph.workflow import build_graph

# Step 1: Build and compile the graph
app = build_graph()

# Step 2: Provide initial state
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

# Step 3: Invoke the graph
result = app.invoke(initial_state)

# Step 4: Display results
print("\n--- LANGGRAPH EXECUTION RESULT ---")

print("\nGenerated SQL:")
print(result["sql_query"])

print("\nError:")
print(result["error"])

if result["sql_result"] is not None:
    print("\nSQL Result:")
    print(result["sql_result"].head(10))

    print("\nTotal Rows:", len(result["sql_result"]))
