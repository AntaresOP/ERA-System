import json
import logging
from typing import Dict, Any, List
from app.agents.state import AgentState
from app.rag.hybrid_search import HybridSearchPipeline
from app.core.config import settings

logger = logging.getLogger(__name__)


class AgentNodes:
    """
    Node definitions for query decomposition, hybrid retrieval, and final synthesis.
    """

    def __init__(self, hybrid_pipeline: HybridSearchPipeline):
        self.hybrid_pipeline = hybrid_pipeline

    def decompose_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Decomposes complex user queries into focused sub-queries for broader coverage.
        """
        query = state["original_query"]
        logger.info(f"Decomposing query: {query}")

        # Rule-based / LLM query decomposition heuristic
        sub_queries = [query]
        if "and" in query.lower() or "versus" in query.lower() or "vs" in query.lower():
            parts = query.lower().replace("vs", "and").replace("versus", "and").split("and")
            sub_queries = [p.strip() for p in parts if p.strip()]

        return {
            "sub_queries": sub_queries,
            "step_count": state.get("step_count", 0) + 1
        }

    def execute_hybrid_retrieval(self, state: AgentState) -> Dict[str, Any]:
        """
        Executes hybrid retrieval across all generated sub-queries and aggregates contexts.
        """
        sub_queries = state.get("sub_queries", [state["original_query"]])
        aggregated_docs: Dict[str, Dict[str, Any]] = {}

        for sub_q in sub_queries:
            results = self.hybrid_pipeline.search(sub_q, top_k=5, top_n_rerank=3)
            for doc in results:
                # Deduplicate by document ID
                aggregated_docs[doc["id"]] = doc

        retrieved_list = list(aggregated_docs.values())
        
        # Format unified context text block
        formatted_context = "\n\n".join([
            f"[{doc.get('source', 'unknown')}]\\n{doc['text']}"
            for doc in retrieved_list
        ])

        return {
            "retrieved_documents": retrieved_list,
            "final_context": formatted_context,
            "is_sufficient": len(retrieved_list) > 0
        }

    def generate_response(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesizes final answer grounded in the retrieved context block.
        """
        context = state.get("final_context", "")
        query = state["original_query"]

        if not context:
            return {
                "generation": "I could not find sufficient information in the knowledge base to answer your query."
            }

        # Deterministic production synthesis template
        response = (
            f"Based on the knowledge base context:\n\n"
            f"{context}\n\n"
            f"--- Summary ---\n"
            f"Answer generated for query: '{query}'."
        )

        return {"generation": response}