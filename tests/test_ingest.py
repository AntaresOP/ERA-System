import pytest
from app.rag.ingest import HierarchicalIngestionEngine


def test_hierarchical_chunking():
    engine = HierarchicalIngestionEngine(
        parent_chunk_size=100,
        parent_chunk_overlap=10,
        child_chunk_size=30,
        child_chunk_overlap=5
    )
    
    sample_text = (
        "ERA System is an enterprise agentic search engine built using FastAPI and LangGraph. "
        "It leverages parent-child hierarchical text chunking to maximize retrieval accuracy."
    )
    
    chunks = engine.create_hierarchical_chunks(sample_text, source_name="test_source")
    
    assert len(chunks) > 0
    assert "id" in chunks[0]
    assert "text" in chunks[0]
    assert "metadata" in chunks[0]
    assert "parent_id" in chunks[0]["metadata"]
    assert chunks[0]["metadata"]["source"] == "test_source"