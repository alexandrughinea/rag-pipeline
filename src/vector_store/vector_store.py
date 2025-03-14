import os
from datetime import datetime
from pathlib import Path

import chromadb


class VectorStore:
    def __init__(self, embeddings_generator, collection_name="documents"):
        persistence_dir = os.getenv("VECTOR_STORAGE_DIR", "./storage")
        persistence_dir_path = Path(persistence_dir)
        persistence_dir_path.mkdir(exist_ok=True)

        persist_dir = Path(persistence_dir_path).resolve()
        self.client = chromadb.PersistentClient(path=str(persist_dir))

        model_dimension = len(embeddings_generator.model.encode("test"))

        try:
            self.collection = self.client.get_collection(name=collection_name)

            # If dimensions don't match, recreate collection:
            if self.collection.metadata.get("dimension") != model_dimension:
                self.client.delete_collection(collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"dimension": model_dimension}
                )
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"dimension": model_dimension}
            )

    def add_embeddings(self, embeddings, document_chunks, source_info, max_batch_size=5000):
        """Add document chunks and their embeddings with proper metadata in batches."""
        if len(embeddings) == 0 or len(document_chunks) == 0:
            raise ValueError("Cannot add empty embeddings or chunks to vector store")

        if len(embeddings) != len(document_chunks):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of chunks ({len(document_chunks)})")

        total_chunks = len(document_chunks)

        # Batches:
        for i in range(0, total_chunks, max_batch_size):
            batch_end = min(i + max_batch_size, total_chunks)

            batch_embeddings = embeddings[i:batch_end]
            batch_chunks = document_chunks[i:batch_end]

            batch_metadatas = [{
                "source": source_info["filename"],
                "file_type": source_info["file_type"],
                "chunk_index": j,
                "chunk_total": total_chunks,
                "batch": f"{i//max_batch_size}",
                "timestamp": datetime.now().isoformat()
            } for j in range(i, batch_end)]

            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_metadatas,
                ids=[f"{source_info['filename']}_{j}" for j in range(i, batch_end)]
            )

        return total_chunks

    def query(self, query_embedding, n_results=5):
        """Query the vector store for similar documents."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        return results