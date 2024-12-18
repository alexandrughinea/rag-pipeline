import os
from datetime import datetime
from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, embeddings_generator, collection_name="documents"):
        persistence_dir = os.getenv("VECTOR_STORAGE_DIR", "./storage")
        persistence_dir_path = Path(persistence_dir)
        persistence_dir_path.mkdir(exist_ok=True)

        # Convert string path to Path object and ensure it's a string when passed
        persist_dir = Path(persistence_dir_path).resolve()
        self.client = chromadb.PersistentClient(path=str(persist_dir))

        # Get embedding dimension from model
        model_dimension = len(embeddings_generator.model.encode("test"))

        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=collection_name)
            if self.collection.metadata.get("dimension") != model_dimension:
                # If dimensions don't match, recreate collection
                self.client.delete_collection(collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"dimension": model_dimension}
                )
        except Exception:
            # Create new collection with dimension metadata
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"dimension": model_dimension}
            )

    def add_embeddings(self, embeddings, document_chunks, source_info):
        """Add document chunks and their embeddings with proper metadata."""
        metadatas = [{
                "source": source_info["filename"],
                "file_type": source_info["file_type"],
                "chunk_index": i,
                "chunk_total": len(document_chunks),
                "timestamp": datetime.now().isoformat()
            } for i in range(len(document_chunks))]

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=document_chunks,
            metadatas=metadatas,
            ids=[f"{source_info['filename']}_{i}" for i in range(len(document_chunks))]
        )
        return len(document_chunks)

    def query(self, query_embedding, n_results=5):
        """Query the vector store for similar documents."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        return results