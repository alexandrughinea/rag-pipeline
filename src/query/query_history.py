import os
import sqlite3
from pathlib import Path
from typing import List


class QueryHistory:
    def __init__(self, query_history_db="./query_history.db"):
        persistence_dir = os.getenv("VECTOR_STORAGE_DIR", "./storage")
        persistence_dir_path = Path(persistence_dir)
        persistence_dir_path.mkdir(exist_ok=True)
        query_history_db = os.getenv("VECTOR_QUERY_HISTORY_DB", query_history_db)

        self.db_path = persistence_dir + "/" + query_history_db
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""                
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    num_results INTEGER,
                    query_response_time FLOAT
                )
            """)

    def add_query(self, query: str, num_results: int, query_response_time: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO queries (
                    query, 
                    num_results, 
                    query_response_time
                    ) VALUES (?, ?, ?)""",
                (query, num_results, query_response_time)
            )

    def get_recent_queries(self, limit: int = 10) -> List[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]