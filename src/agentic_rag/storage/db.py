from pathlib import Path
import psycopg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection():
    """
    Returns a psycopg3 connection object to the RAG database.
    Usage:
        with get_connection() as conn:
            # do stuff
    """
    #TODO : Make the connection string configurable via env vars or config file
    return psycopg.connect(
        "postgresql://rag:rag@localhost:5432/rag",
        autocommit=False 
    )


def ensure_schema(conn) -> None:
    """
    Executes the schema.sql file to create tables, extensions, etc.
    """
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        ddl = f.read()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit() 
