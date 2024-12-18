class QueryEngine:
    def __init__(self, embeddings_generator, vector_store):
        self.embeddings_generator = embeddings_generator
        self.vector_store = vector_store

    def query(self, query_text, num_results=5):
        """Query the RAG system with a text query."""
        query_embedding = self.embeddings_generator.model.encode([query_text])[0]
        results = self.vector_store.query(query_embedding, num_results)
        return results