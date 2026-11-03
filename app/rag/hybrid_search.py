from typing import List, Dict, Any
from app.rag.vector_store import VectorStoreManager
from app.rag.sparse_search import SparseSearchEngine
from app.rag.reranker import CrossEncoderReranker


class HybridSearchPipeline:
    """
    Unified search pipeline executing Dense Search + BM25 Sparse Search,
    fusing results with RRF (Reciprocal Rank Fusion), and applying Cross-Encoder Reranking.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.sparse_store = SparseSearchEngine()
        self.reranker = CrossEncoderReranker()

    def sync_sparse_index(self) -> None:
        """Fetches all documents from vector store and populates the sparse index."""
        data = self.vector_store.collection.get(include=["documents", "metadatas"])
        if data["ids"]:
            docs = [
                {"id": data["ids"][i], "text": data["documents"][i], "metadata": data["metadatas"][i]}
                for i in range(len(data["ids"]))
            ]
            self.sparse_store.build_index(docs)

    def reciprocal_rank_fusion(
        self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combines ranked lists from dense and sparse search using RRF formula."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        # Process dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

        # Sort documents by computed RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_docs = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = rrf_scores[doc_id]
            fused_docs.append(doc)

        return fused_docs

    def search(self, query: str, top_k: int = 10, top_n_rerank: int = 3) -> List[Dict[str, Any]]:
        """Full hybrid pipeline: Dense + Sparse -> RRF -> Cross-Encoder Rerank."""
        # 1. Parallel retrieval
        dense_results = self.vector_store.vector_search(query, top_k=top_k)
        sparse_results = self.sparse_store.search(query, top_k=top_k)

        # 2. Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # 3. Cross-Encoder Reranking
        reranked_results = self.reranker.rerank(query, fused_results, top_n=top_n_rerank)

        return reranked_results