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
        chunks = self._chunk_text(text)
        return self.model.encode(chunks), chunks

    def _chunk_text(self, text):
        """Split text into chunks of approximately equal size."""
        return textwrap.wrap(text,
                             width=self.chunk_size,
                             break_long_words=False)