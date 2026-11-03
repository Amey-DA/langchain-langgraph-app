
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import sql_node, python_node, final_response_node


def build_graph():

    # Initialize the graph with our state structure
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("sql_node", sql_node)
    workflow.add_node("python_node", python_node)
    workflow.add_node("final_response_node", final_response_node)

    # Define workflow connections
    workflow.add_edge(START, "sql_node")
    workflow.add_edge("sql_node", "python_node")
    workflow.add_edge("python_node", "final_response_node")
    workflow.add_edge("final_response_node", END)

    # Compile graph
    app = workflow.compile()

    return app
