# src/core/db.py
import os
import json
import base64
import hashlib
import pathlib
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from openai import OpenAI
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase

load_dotenv()

# -----------------------------
# Environment variables
# -----------------------------
model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
api_key = os.getenv("OPENAI_API_KEY")
pg_connection = os.getenv("SQLALCHEMY_DATABASE_URL")
agentic_db_url = os.getenv("AGENTIC_RAG_DB_URL")

# -----------------------------
# OpenAI embeddings for retrieval
# -----------------------------
_embeddings = OpenAIEmbeddings(
    model=model,
    api_key=api_key
)

def get_embeddings():
    return _embeddings

def get_vector_store(collection_name: str = "RerankingRAGVectorStore"):
    return PGVector(
        collection_name=collection_name,
        connection=pg_connection,
        embeddings=get_embeddings(),
        use_jsonb=True
    )

def get_sql_database() -> SQLDatabase:
    if not agentic_db_url:
        raise ValueError("AGENTIC_RAG_DB_URL not set. Check your .env file")
    return SQLDatabase.from_uri(
        agentic_db_url,
        include_tables=["accounts","card_transactions","credit_cards","fixed_deposits","loan_accounts","transactions"],
        sample_rows_in_table_info=2,
    )

# -----------------------------
# OpenAI client & embedding config for ingestion
# -----------------------------
_openai_client = OpenAI(api_key=api_key)
_EMBED_MODEL = model
_EMBED_BATCH_SIZE = 100
_EMBED_DIMENSIONS = 1536

def _embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i:i+_EMBED_BATCH_SIZE]
        result = _openai_client.embeddings.create(
            model=_EMBED_MODEL,
            input=batch,
            dimensions=_EMBED_DIMENSIONS
        )
        embeddings.extend(item.embedding for item in result.data)
    return embeddings

# -----------------------------
# PostgreSQL connection pool
# -----------------------------
_PG_DSN = pg_connection.replace("postgresql+psycopg://", "postgresql://")
_pool: ConnectionPool | None = None

def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row}
        )
    return _pool

def get_db_conn():
    return _get_pool().connection()

# -----------------------------
# Document registry & chunks
# -----------------------------
def upsert_document(filename: str, source_path: str) -> str:
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at = now()
                RETURNING id
                """,
                (filename, source_path)
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    if not chunks:
        return 0
    all_embeddings = _embed_texts([chunk["content"] for chunk in chunks])
    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }
    rows_inserted = 0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                (doc_id,)
            )
            for chunk, embedding in zip(chunks, all_embeddings):
                meta = chunk["metadata"]
                img_b64 = meta.get("image_base64")
                image_path = None
                mime_type = "image/png" if img_b64 else None
                if img_b64:
                    image_bytes = base64.b64decode(img_b64)
                    img_dir = pathlib.Path("data/images")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                    img_file = img_dir / f"{doc_id}_{img_hash}.png"
                    img_file.write_bytes(image_bytes)
                    image_path = str(img_file)
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
                clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}
                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        doc_id, chunk_type, element_type, content,
                        image_path, mime_type,
                        page_number, section, source_file,
                        position, embedding, metadata
                    ) VALUES (
                        %s::uuid, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s::jsonb, %s::vector, %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk["content_type"],
                        meta.get("element_type"),
                        chunk["content"],
                        image_path,
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position")) if meta.get("position") else None,
                        embedding_str,
                        json.dumps(clean_meta),
                    ),
                )
                rows_inserted += 1
        conn.commit()
    return rows_inserted

def search_chunks(query: str, k: int = 20) -> list[dict]:
    """Search multimodal_chunks using cosine similarity against the query embedding."""
    query_embedding = _embed_texts([query])[0]
    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    content,
                    chunk_type,
                    element_type,
                    page_number,
                    section,
                    source_file,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM multimodal_chunks
                WHERE content IS NOT NULL AND content != ''
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, k)
            )
            rows = cur.fetchall()
    return rows


def fts_search_chunks(query: str, k: int = 20) -> list[dict]:
    """Full-text search on multimodal_chunks using PostgreSQL tsvector/tsquery.

    Best suited for:
    - Exact keyword matches (product codes, abbreviations, numeric IDs)
    - Short uppercase tokens (e.g. "IMPS", "RTGS", "KYC")
    - Long numeric strings (account/loan IDs)

    Uses plainto_tsquery so the caller need not escape special characters.
    Returns rows ordered by ts_rank descending, with a 'fts_rank' field
    appended so callers can inspect relevance scores.
    """
    sql = """
        SELECT
            content,
            chunk_type,
            element_type,
            page_number,
            section,
            source_file,
            metadata,
            ts_rank(
                to_tsvector('english', content),
                plainto_tsquery('english', %s)
            ) AS fts_rank
        FROM multimodal_chunks
        WHERE
            content IS NOT NULL
            AND content != ''
            AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY fts_rank DESC
        LIMIT %s
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, k))
            rows = cur.fetchall()
    # Normalise: add similarity_score alias so downstream code is uniform
    result = []
    for row in rows:
        r = dict(row)
        r["similarity_score"] = float(r.pop("fts_rank", 0.0))
        result.append(r)
    return result


def hybrid_search_chunks(query: str, k: int = 20) -> list[dict]:
    """Reciprocal Rank Fusion of semantic (vector) and keyword (FTS) results.

    Combines search_chunks() and fts_search_chunks() using the standard RRF
    formula  score += 1 / (60 + rank)  so that chunks appearing highly in
    BOTH lists are promoted to the top, while chunks strong in only one list
    still surface.

    Best suited for short, ambiguous, or mixed queries where neither pure
    semantic nor pure keyword search alone is reliable.

    Deduplication key: first 200 characters of content (enough to uniquely
    identify a chunk without hashing every row).
    """
    _RRF_K = 60  # standard constant; lower values reward top-rank hits more

    # ── Fetch both result sets ────────────────────────────────────────────────
    semantic_rows = search_chunks(query, k=k)
    fts_rows      = fts_search_chunks(query, k=k)

    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    def _key(row: dict) -> str:
        return (row.get("content") or "")[:200]

    # ── Score semantic results ────────────────────────────────────────────────
    for rank, row in enumerate(semantic_rows):
        key = _key(row)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_map[key]  = row

    # ── Score FTS results and merge ───────────────────────────────────────────
    for rank, row in enumerate(fts_rows):
        key = _key(row)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        # Prefer the semantic row (already has similarity_score); only add if new
        if key not in chunk_map:
            chunk_map[key] = row

    # ── Sort by accumulated RRF score and return top-k ────────────────────────
    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:k]
    result = []
    for key in ranked_keys:
        row = dict(chunk_map[key])
        row["similarity_score"] = round(rrf_scores[key], 6)  # expose final RRF score
        result.append(row)
    return result