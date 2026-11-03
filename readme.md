# ERA System: Enterprise Agentic RAG & Knowledge Graph Search Engine

ERA System is a high-performance, enterprise-grade AI search engine built using FastAPI, LangGraph, and ChromaDB. It features parent-child hierarchical chunking, hybrid retrieval (BM25 + Dense Vector), FlashRank cross-encoder reranking, and semantic Redis caching.

## Key Features

- **Hierarchical Text Chunking**: Parent (1024 tokens) and Child (256 tokens) chunking for high-precision retrieval without context loss.
- **Hybrid Retrieval & Reranking**: Combines Sparse BM25 and Dense ChromaDB vector search using Reciprocal Rank Fusion (RRF) and FlashRank cross-encoder reranking.
- **Agentic Workflows**: Multi-step query decomposition and state routing using LangGraph.
- **Semantic Caching**: Sub-10ms query lookup times backed by Redis query embedding similarity.
- **Automated RAG Evaluation**: Faithfulness, Context Recall, and Answer Relevance metrics framework using Ragas.
- **Production Ready**: Fully containerized using Docker & Docker Compose with async FastAPI endpoints.

## Architecture

1. **Ingestion**: Document parsing -> Hierarchical Chunking -> Dense Vector Embedding -> ChromaDB.
2. **Retrieval**: Dual Sparse (BM25) + Dense Vector Search -> RRF Fusion -> FlashRank Rerank.
3. **Agent Flow**: Query Decomposition -> Multi-Query Search -> Synthesis -> Semantic Caching.

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Setup Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt