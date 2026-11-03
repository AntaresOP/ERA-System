from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class VectorStoreManager:
    """
    Manages local persistent vector embeddings inside ChromaDB using 
    HuggingFace/SentenceTransformers embeddings.
    """

    def __init__(self):
        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(allow_reset=True)
        )
        
        # Load local dense embedding model
        self.embedding_model = SentenceTransformer(settings.DENSE_EMBEDDING_MODEL)
        
        # Get or create default collection
        self.collection = self.client.get_or_create_collection(
            name="enterprise_knowledge_base",
            metadata={"hnsw:space": "cosine"}  # Cosine distance for semantic similarity
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Generates dense vector embeddings and persists chunks to ChromaDB."""
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()

        # Upsert documents into vector database
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes dense vector similarity search given a text query."""
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i] # Convert distance to similarity score
                })

        return formatted_results