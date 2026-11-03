import logging
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Local reranking using FlashRank cross-encoder models 
    with dynamic offline/fallback handling for network interruptions.
    """

    def __init__(self):
        self.ranker = None
        try:
            from flashrank import Ranker
            self.ranker = Ranker(model_name=settings.FLASHRANK_MODEL)
            logger.info("FlashRank model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load FlashRank model due to network/download constraint: {e}. "
                "Pipeline will fallback to Reciprocal Rank Fusion (RRF) scores."
            )

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Re-scores and re-ranks retrieved chunks. Fallback to input RRF ordering 
        if FlashRank model is unavailable.
        """
        if not documents:
            return []

        # Fallback path if FlashRank model failed to download/initialize
        if self.ranker is None:
            return documents[:top_n]

        try:
            from flashrank import RerankRequest
            passages = [
                {"id": doc["id"], "text": doc["text"], "meta": doc["metadata"]}
                for doc in documents
            ]

            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)

            reranked_docs = []
            for item in results[:top_n]:
                reranked_docs.append({
                    "id": str(item["id"]),
                    "text": item["text"],
                    "metadata": item["meta"],
                    "score": float(item["score"])
                })

            return reranked_docs

        except Exception as e:
            logger.error(f"Reranking execution failed: {e}. Returning RRF order.")
            return documents[:top_n]