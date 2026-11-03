
import pandas as pd

from graph.nodes import python_node

# Simulate the output received from SQL node
sample_df = pd.DataFrame({
    "PRODUCT_NAME": ["Star Lite", "Zone", "Infinity"],
    "REVENUE": [168756400.456, 157369300.789, 97109770.123]
})

initial_state = {
    "question": "Show the product revenue rounded to 2 decimal places",
    "request_mode": "analysis",
    "sql_query": "SELECT ...",
    "sql_result": sample_df,
    "python_code": "",
    "python_result": None,
    "final_response": "",
    "error": None
}

result = python_node(initial_state)

print("\n--- PYTHON NODE OUTPUT ---")
print("Generated Python Code:")
print(result["python_code"])

print("\nError:")
print(result["error"])

if result["python_result"] is not None:
    print("\nTransformed DataFrame:")
    print(
    result["python_result"].to_string(
        formatters={
            "REVENUE": lambda x: f"{x:.2f}"
        },
        index=False
    )
    )  