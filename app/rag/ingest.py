import csv
import uuid
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HierarchicalIngestionEngine:
    """
    Ingestion engine that extracts text from PDFs, TXT, and CSV files,
    implementing Parent-Child hierarchical text chunking.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1024,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 256,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> str:
        """Extract raw text from PDF or TXT files."""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def load_arxiv_csv(self, file_path: str, max_rows: int = 500) -> List[Dict[str, str]]:
        """
        Loads ArXiv research papers from Kaggle CSV and structures each row 
        into a clean searchable text block with metadata.
        """
        documents = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= max_rows:  # Limit rows during ingestion for fast testing
                    break
                
                title = row.get("title", row.get("Title", ""))
                abstract = row.get("abstract", row.get("Abstract", row.get("summary", "")))
                categories = row.get("categories", row.get("Categories", "AI/ML"))
                
                combined_text = f"Title: {title}\nCategories: {categories}\nAbstract: {abstract}"
                documents.append({
                    "text": combined_text,
                    "source": f"ArXiv Paper #{idx+1}: {title[:30]}"
                })
        return documents

    def create_hierarchical_chunks(
        self, text: str, source_name: str
    ) -> List[Dict[str, Any]]:
        """
        Splits text into parent chunks, then splits each parent into child chunks.
        """
        processed_chunks = []
        raw_parents = self.parent_splitter.split_text(text)

        for p_idx, parent_text in enumerate(raw_parents):
            parent_id = str(uuid.uuid4())
            child_texts = self.child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(child_texts):
                child_id = str(uuid.uuid4())
                processed_chunks.append({
                    "id": child_id,
                    "text": child_text,
                    "metadata": {
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "source": source_name,
                        "parent_index": p_idx,
                        "child_index": c_idx,
                    }
                })

        return processed_chunks