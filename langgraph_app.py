
import pandas as pd

from graph.workflow import build_graph
from app import requests_complete_results, requests_analysis


# Initialize the compiled LangGraph workflow
graph = build_graph()

# Configure DataFrame display
pd.set_option("display.float_format", "{:,.2f}".format)


def main():
    print("\n====================================")
    print("       LANGGRAPH DATA ASSISTANT")
    print("====================================")

    while True:
        question = input(
            "\nEnter your question (or type 'exit' to quit): "
        ).strip()

        if question.lower() in ("exit", "quit"):
            print("\nExiting LangGraph Data Assistant. Goodbye!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        print(f"\nProcessing your question: {question}")

        # Determine request mode
        is_full_result = requests_complete_results(question)
        analysis_required = requests_analysis(question)

        request_mode = "full_results" if is_full_result else "analysis"

        print(f"\nRequest Mode: {request_mode}")
        print(f"Analysis Required: {analysis_required}")

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

        print("\nExecuting LangGraph workflow...")

        # Execute the complete workflow
        try:
            result = graph.invoke(initial_state)

            print("\n========== FINAL RESPONSE ==========")
            print(result.get("final_response"))

            print("\n========== PROCESSED DATA ==========")

            if result.get("python_result") is not None:
                print(result["python_result"].to_string(index=False))
            else:
                print("No processed DataFrame returned.")

            print("\n========== ERROR ==========")
            print(result.get("error"))

        except Exception as e:
            print(
                f"\nWorkflow execution failed: "
                f"{type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()
