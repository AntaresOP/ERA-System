from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class SparseSearchEngine:
    """
    Manages in-memory Rank-BM25 index for lexical and exact keyword matching.
    """

    def __init__(self):
        self.bm25 = None
        self.documents: List[Dict[str, Any]] = []

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Builds a BM25 index from document text chunks.
        """
        self.documents = documents
        tokenized_corpus = [doc["text"].lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant documents based on BM25 keyword matching.
        """
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices by BM25 score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Include matching documents only
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                results.append(doc)

        return results