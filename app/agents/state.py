from typing import List, Dict, Any, TypedDict, Optional


class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph engine.
    """
    original_query: str
    sub_queries: List[str]
    retrieved_documents: List[Dict[str, Any]]
    final_context: str
    generation: str
    is_sufficient: bool
    step_count: int