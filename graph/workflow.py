
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import sql_node


def build_graph():

    # Create graph using our AgentState
    workflow = StateGraph(AgentState)

    # Register SQL node
    workflow.add_node("sql_node", sql_node)

    # Define workflow connections
    workflow.add_edge(START, "sql_node")
    workflow.add_edge("sql_node", END)

    # Compile graph
    app = workflow.compile()

    return app

