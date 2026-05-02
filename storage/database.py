import sqlite3
import uuid
from pathlib import Path
from config import DB_PATH


class Database:
    def __init__(self):
        Path("data").mkdir(exist_ok=True)

        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()

    # -----------------------------------
    # Initialize Schema
    # -----------------------------------
    def _initialize_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"

        with open(schema_path, "r", encoding="utf-8") as f:
            self.conn.executescript(f.read())

        self.conn.commit()

    # -----------------------------------
    # Insert Document
    # -----------------------------------
    def insert_document(self, title: str) -> str:
        document_id = str(uuid.uuid4())

        self.conn.execute(
            """
            INSERT INTO documents (id, title)
            VALUES (?, ?)
            """,
            (document_id, title),
        )

        self.conn.commit()
        return document_id

    # -----------------------------------
    # Insert Chunk
    # -----------------------------------
    def insert_chunk(
        self,
        document_id: str,
        content: str,
        page_number: int | None = None,
    ) -> int:

        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO chunks (document_id, page_number, content)
            VALUES (?, ?, ?)
            """,
            (document_id, page_number, content),
        )

        self.conn.commit()
        return cursor.lastrowid

    # -----------------------------------
    # Fetch All Chunks
    # -----------------------------------
    def get_all_chunks(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT c.id, c.content, c.page_number, d.title
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
        """)

        return cursor.fetchall()

    # -----------------------------------
    # Close
    # -----------------------------------
    def close(self):
        self.conn.close()