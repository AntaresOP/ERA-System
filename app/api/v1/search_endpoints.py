from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.rag.ingest import HierarchicalIngestionEngine
from app.rag.vector_store import VectorStoreManager
from app.rag.hybrid_search import HybridSearchPipeline
from app.agents.graph import build_agent_graph
from app.core.cache import SemanticCacheManager

router = APIRouter()

# Instantiate core engines globally
vector_store = VectorStoreManager()
hybrid_pipeline = HybridSearchPipeline(vector_store)
agent_app = build_agent_graph(hybrid_pipeline)
semantic_cache = SemanticCacheManager()


class SearchQueryRequest(BaseModel):
    query: str = Field(..., example="What is long term semantic memory in LLM agents?")
    use_cache: bool = True


class SearchQueryResponse(BaseModel):
    query: str
    answer: str
    cached: bool = False
    retrieved_documents: List[Dict[str, Any]] = []


class IngestRequest(BaseModel):
    file_path: str = Field(..., example="data/arxiv_ai_ml_papers.csv")
    max_rows: Optional[int] = 200


@router.post("/ingest", status_code=200)
async def ingest_documents(payload: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers asynchronous document ingestion and BM25 index synchronization.
    """
    try:
        engine = HierarchicalIngestionEngine()
        if payload.file_path.endswith(".csv"):
            papers = engine.load_arxiv_csv(payload.file_path, max_rows=payload.max_rows)
            all_chunks = []
            for p in papers:
                all_chunks.extend(engine.create_hierarchical_chunks(p["text"], p["source"]))
            vector_store.add_documents(all_chunks)
        else:
            text = engine.load_document(payload.file_path)
            chunks = engine.create_hierarchical_chunks(text, payload.file_path)
            vector_store.add_documents(chunks)

        # Sync sparse BM25 index in background task
        background_tasks.add_task(hybrid_pipeline.sync_sparse_index)
        
        return {
            "status": "success",
            "message": f"Successfully ingested documents from {payload.file_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchQueryResponse)
async def execute_agentic_search(payload: SearchQueryRequest):
    """
    Executes Agentic Search pipeline with Semantic Redis Cache lookup.
    """
    # 1. Check Semantic Redis Cache
    if payload.use_cache:
        cached_res = semantic_cache.get_cached_response(payload.query)
        if cached_res:
            return SearchQueryResponse(
                query=payload.query,
                answer=cached_res["generation"],
                cached=True,
                retrieved_documents=[]
            )

    # 2. Execute LangGraph Agent Flow
    initial_state = {
        "original_query": payload.query,
        "sub_queries": [],
        "retrieved_documents": [],
        "final_context": "",
        "generation": "",
        "is_sufficient": False,
        "step_count": 0
    }
    
    result = agent_app.invoke(initial_state)

    # 3. Store result in semantic cache if enabled
    if payload.use_cache and result.get("generation"):
        semantic_cache.set_cached_response(payload.query, result["generation"])

    return SearchQueryResponse(
        query=payload.query,
        answer=result["generation"],
        cached=False,
        retrieved_documents=result.get("retrieved_documents", [])
    )