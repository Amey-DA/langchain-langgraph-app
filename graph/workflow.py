
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import (
    sql_node,
    python_node,
    analysis_node,
    route_after_python,
    final_response_node
)


def build_graph():

    # Initialize the graph with our state structure
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("python_node", python_node)
    workflow.add_node("analysis_node", analysis_node)
    workflow.add_node("final_response_node", final_response_node)

    # Define initial workflow connections
    workflow.add_edge(START, "sql_node")
    workflow.add_edge("sql_node", "python_node")

    # Conditional routing after Python processing
    workflow.add_conditional_edges(
        "python_node",
        route_after_python,
        {
            "analysis_node": "analysis_node",
            "final_response_node": "final_response_node"
        }
    )

    # Execute generated analysis code using the existing Python node
    workflow.add_edge("analysis_node", "python_node")

    # Complete workflow
    workflow.add_edge("final_response_node", END)

    # Compile graph
    app = workflow.compile()

    return app
