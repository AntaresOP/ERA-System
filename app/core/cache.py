import json
import logging
import numpy as np
import redis
from typing import Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)


class SemanticCacheManager:
    """
    Redis-backed semantic cache using embedding cosine similarity
    to achieve sub-10ms query lookup times for similar intents.
    """

    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=False  # Keep raw bytes for binary vectors
        )
        self.embedding_model = SentenceTransformer(settings.DENSE_EMBEDDING_MODEL)
        self.similarity_threshold = settings.SEMANTIC_CACHE_THRESHOLD

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two normalized vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Queries Redis for semantically similar cached prompts.
        Returns cached response dict if similarity > threshold, else None.
        """
        try:
            query_vector = self.embedding_model.encode(query)
            keys = self.redis_client.keys("cache:*")

            for key in keys:
                raw_data = self.redis_client.get(key)
                if not raw_data:
                    continue

                item = json.loads(raw_data.decode("utf-8"))
                cached_vector = np.array(item["embedding"], dtype=np.float32)

                similarity = self._cosine_similarity(query_vector, cached_vector)
                if similarity >= self.similarity_threshold:
                    logger.info(f"Semantic Cache Hit! (Similarity: {similarity:.4f})")
                    return {
                        "generation": item["response"],
                        "cached": True,
                        "similarity": similarity
                    }

            return None
        except Exception as e:
            logger.warning(f"Semantic Cache Read Warning (Redis offline or connecting): {e}")
            return None

    def set_cached_response(self, query: str, response: str, ttl_seconds: int = 3600) -> None:
        """Stores query embedding and generation output in Redis with TTL expiration."""
        try:
            query_vector = self.embedding_model.encode(query).tolist()
            cache_key = f"cache:{hash(query)}"

            payload = {
                "query": query,
                "embedding": query_vector,
                "response": response
            }

            self.redis_client.setex(
                name=cache_key,
                time=ttl_seconds,
                value=json.dumps(payload)
            )
            logger.info(f"Cached query response successfully for: '{query[:30]}...'")
        except Exception as e:
            logger.warning(f"Semantic Cache Write Warning (Redis offline or connecting): {e}")