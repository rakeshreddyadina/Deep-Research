import json
import pickle
import os
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure for storage"""
    id: str
    title: str
    content: str
    source: str
    timestamp: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def to_dict(self):
        data = asdict(self)
        # Don't include embedding in JSON serialization
        if 'embedding' in data:
            del data['embedding']
        return data

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class DocumentStorage:
    """Handles document storage and retrieval"""

    def __init__(self, storage_path: str = "storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        self.documents_file = self.storage_path / "documents.json"
        self.embeddings_file = self.storage_path / "embeddings.pkl"
        self.metadata_file = self.storage_path / "metadata.json"

        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata = {"created": datetime.now().isoformat(), "document_count": 0}

        self._load_storage()

    def _generate_doc_id(self, title: str, content: str) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5((title + content).encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{content_hash}_{timestamp}"

    def add_document(self, title: str, content: str, source: str = "unknown",
                     metadata: Dict = None, embedding: np.ndarray = None) -> str:
        """Add a document to storage"""
        doc_id = self._generate_doc_id(title, content)

        if metadata is None:
            metadata = {}

        document = Document(
            id=doc_id,
            title=title,
            content=content,
            source=source,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
            embedding=embedding
        )

        self.documents[doc_id] = document
        if embedding is not None:
            self.embeddings[doc_id] = embedding

        self.metadata["document_count"] = len(self.documents)
        self._save_storage()

        logger.info(f"Added document: {doc_id} - {title[:50]}...")
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID"""
        return self.documents.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        return list(self.documents.values())

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]

            self.metadata["document_count"] = len(self.documents)
            self._save_storage()

            logger.info(f"Deleted document: {doc_id}")
            return True
        return False

    def search_by_similarity(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Search documents by embedding similarity"""
        if not self.embeddings:
            return []

        doc_ids = list(self.embeddings.keys())
        doc_embeddings = np.array([self.embeddings[doc_id] for doc_id in doc_ids])

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc_id = doc_ids[idx]
            document = self.documents[doc_id]
            similarity_score = similarities[idx]
            results.append((document, similarity_score))

        return results

    def _save_storage(self):
        """Save documents and embeddings to disk"""
        try:
            # Save documents (without embeddings)
            docs_data = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
            with open(self.documents_file, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)

            # Save embeddings separately
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)

            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving storage: {e}")

    def _load_storage(self):
        """Load documents and embeddings from disk"""
        try:
            if self.documents_file.exists():
                with open(self.documents_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                    self.documents = {doc_id: Document.from_dict(data)
                                      for doc_id, data in docs_data.items()}

            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)

            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)

            logger.info(f"Loaded {len(self.documents)} documents from storage")

        except Exception as e:
            logger.error(f"Error loading storage: {e}")