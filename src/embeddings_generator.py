import os
import textwrap

from sentence_transformers import SentenceTransformer


class EmbeddingsGenerator:
    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", "multi-qa-distilbert-cos-v1")
        self.model = SentenceTransformer(self.model_name)
        self.chunk_size = int(os.getenv("EMBEDDING_CHUNK_SIZE", "500"))

    def generate_embeddings(self, text):
        """Generate embeddings for text chunks."""
        if not text or not text.strip():
            raise ValueError("Cannot generate embeddings for empty text")

        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Text chunking resulted in no chunks")

        print(f"Generating embeddings for {len(chunks)} chunks")
        print(f"First chunk preview: {chunks[0][:100]}...")  # Debug print

        embeddings = self.model.encode(chunks)

        # Validate embeddings
        if embeddings.size == 0:
            raise ValueError("Generated embeddings are empty")

        print(f"Generated embeddings shape: {embeddings.shape}")  # Debug print

        return embeddings, chunks

    def _chunk_text(self, text):
        """Split text into chunks of approximately equal size."""
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text)}")

        chunks = textwrap.wrap(text,
                               width=self.chunk_size,
                               break_long_words=False)

        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]

        print(f"Created {len(chunks)} chunks")  # Debug print
        return chunks