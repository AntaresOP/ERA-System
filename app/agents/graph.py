from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes import AgentNodes
from app.rag.hybrid_search import HybridSearchPipeline


def build_agent_graph(hybrid_pipeline: HybridSearchPipeline):
    """
    Compiles state graph nodes into an executable agentic workflow.
    """
    nodes = AgentNodes(hybrid_pipeline)
    workflow = StateGraph(AgentState)

    # Add computational nodes
    workflow.add_node("decompose_query", nodes.decompose_query)
    workflow.add_node("execute_retrieval", nodes.execute_hybrid_retrieval)
    workflow.add_node("generate_response", nodes.generate_response)

    # Define execution graph sequence
    workflow.set_entry_point("decompose_query")
    workflow.add_edge("decompose_query", "execute_retrieval")
    workflow.add_edge("execute_retrieval", "generate_response")
    workflow.add_edge("generate_response", END)

    # Compile workflow state machine
    return workflow.compile()