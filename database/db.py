from sqlalchemy import create_engine, MetaData, Table, Column, String, JSON, select, delete, text
from sqlalchemy.engine import URL
from sqlalchemy.dialects.postgresql import insert as pg_insert
import json
import re

# Connection details
DB_USER = "postgres"
DB_PASSWORD = "123qwe!@#"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "face_reg"

# Build connection URL
url = URL.create(
    "postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# Create engine
engine = create_engine(url, echo=False, future=True)

# -----------------------------
# Schema and utility using SQLAlchemy Core
# -----------------------------
metadata = MetaData()

face_embeddings = Table(
    "face_embeddings",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("embedding", JSON, nullable=False),
)

def ensure_table_exists():
    """Create table if not exists (SQLAlchemy-managed)"""
    metadata.create_all(engine)

def _is_embedding_column_vector():
    """Return True if the 'embedding' column uses pgvector (udt_name == 'vector')"""
    q = text("""
        SELECT udt_name
        FROM information_schema.columns
        WHERE table_name = 'face_embeddings' AND column_name = 'embedding'
        LIMIT 1
    """)
    with engine.connect() as conn:
        res = conn.execute(q).fetchone()
        if not res:
            return False
        udt = res[0]
        return str(udt).lower() == "vector"

def _parse_vector_result(val):
    """Parse DB-returned vector/json into list of floats"""
    if val is None:
        return None
    # If database driver already returned Python list/dict, just normalize
    if isinstance(val, (list, tuple)):
        return list(val)
    # If string like '[0.1,0.2,...]' try JSON
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            # Fallback: extract floats with regex
            nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", val)
            try:
                return [float(x) for x in nums]
            except Exception:
                return None
    return None

def insert_embedding(account_id: str, embedding: list):
    """Insert or update an embedding for a given account id.
    Adapts to either JSON storage or pgvector storage in the DB.
    """
    if _is_embedding_column_vector():
        # Store into pgvector column by casting a string representation to ::vector
        # Use raw DB connection to leverage psycopg2 parameter binding
        vector_str = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
        sql = """
            INSERT INTO face_embeddings (id, embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (id) DO UPDATE
            SET embedding = EXCLUDED.embedding
        """
        # Use raw connection for direct psycopg2 execution and commit
        raw_conn = engine.raw_connection()
        try:
            cur = raw_conn.cursor()
            cur.execute(sql, (account_id, vector_str))
            raw_conn.commit()
        finally:
            try:
                cur.close()
            except Exception:
                pass
            raw_conn.close()
    else:
        # Default JSON/JSONB case — use SQLAlchemy upsert
        stmt = pg_insert(face_embeddings).values(id=account_id, embedding=embedding)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={"embedding": stmt.excluded.embedding},
        )
        with engine.begin() as conn:
            conn.execute(stmt)

def get_embedding(account_id: str):
    """Fetch embedding for one account id"""
    stmt = select(face_embeddings.c.embedding).where(face_embeddings.c.id == account_id)
    with engine.connect() as conn:
        res = conn.execute(stmt).fetchone()
        if res:
            return _parse_vector_result(res[0])
    return None

def get_all_embeddings():
    """Fetch all embeddings as {id: embedding}"""
    stmt = select(face_embeddings.c.id, face_embeddings.c.embedding).order_by(face_embeddings.c.id)
    out = {}
    with engine.connect() as conn:
        res = conn.execute(stmt)
        for row in res:
            out[row[0]] = _parse_vector_result(row[1])
    return out

def delete_embedding(account_id: str):
    """Delete one embedding by id"""
    stmt = delete(face_embeddings).where(face_embeddings.c.id == account_id)
    with engine.begin() as conn:
        conn.execute(stmt)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    ensure_table_exists()

    # Example embedding vector (shortened)
    example_embedding = [0.12, 0.34, 0.56]

    # Insert
    insert_embedding("23020020", example_embedding)
    print("✅ Inserted embedding")

    # Get single
    emb = get_embedding("23020020")
    print("Fetched embedding:", emb)

    # Get all
    all_embs = get_all_embeddings()
    print("All embeddings:", all_embs)

    # Delete
    delete_embedding("23020020")
    print("✅ Deleted embedding")
