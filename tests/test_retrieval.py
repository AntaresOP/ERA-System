import pytest
from app.rag.vector_store import VectorStoreManager
from app.rag.sparse_search import SparseSearchEngine
from app.rag.hybrid_search import HybridSearchPipeline


def test_vector_and_sparse_search():
    vector_store = VectorStoreManager()
    pipeline = HybridSearchPipeline(vector_store)
    
    # Ingest mock test data
    test_chunks = [
        {
            "id": "test-uuid-1",
            "text": "ChromaDB stores high dimensional vector embeddings for semantic search.",
            "metadata": {"source": "unit_test", "parent_id": "p-1"}
        },
        {
            "id": "test-uuid-2",
            "text": "BM25 handles sparse keyword search and exact phrase matching.",
            "metadata": {"source": "unit_test", "parent_id": "p-2"}
        }
    ]
    
    vector_store.add_documents(test_chunks)
    pipeline.sync_sparse_index()
    
    results = pipeline.search("semantic search vector", top_k=2, top_n_rerank=2)
    
    assert len(results) > 0
    assert "text" in results[0]
    assert "metadata" in results[0]