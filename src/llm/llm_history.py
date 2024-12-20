import os
import sqlite3
from pathlib import Path


class LLMHistory:
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
               CREATE TABLE IF NOT EXISTS conversations (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   title TEXT,
                   created_at DATETIME DEFAULT CURRENT_TIMESTAMP
               )""")

            conn.execute("""
               CREATE TABLE IF NOT EXISTS messages (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   conversation_id INTEGER,
                   role TEXT NOT NULL,
                   content TEXT NOT NULL,
                   created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                   FOREIGN KEY(conversation_id) REFERENCES conversations(id)
               )""")

    def create_conversation(self, title=None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
            return cursor.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO messages (
                    conversation_id, 
                    role, 
                    content
                ) 
                VALUES (?, ?, ?)""",
                (conversation_id, role, content)
            )

    def get_conversation(self, conversation_id: int, limit: int = 5) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT role, content FROM messages 
                WHERE conversation_id = ? 
                ORDER BY created_at DESC LIMIT ?""",
                (conversation_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()][::-1]

    def get_conversations(self, limit: int = 10) -> list:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(
                "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )]
