"""
Run this from your project root:
    python debug_retrieval.py

It checks each step of the retrieval pipeline independently so you can
pinpoint exactly where things are breaking.
"""

import os
import sys
sys.path.insert(0, ".")   # make sure project root is on path

from dotenv import load_dotenv
load_dotenv()

# ── Step 1: Check DB connection & row counts ─────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Check multimodal_chunks table")
print("="*60)

try:
    from src.core.db import get_db_conn

    with get_db_conn() as conn:
        with conn.cursor() as cur:

            # Total row count
            cur.execute("SELECT COUNT(*) AS total FROM multimodal_chunks")
            row = cur.fetchone()
            print(f"  Total rows in multimodal_chunks: {row['total']}")

            if row['total'] == 0:
                print("\n  ❌ TABLE IS EMPTY — ingestion never ran or failed silently.")
                print("     Run: python -m src.ingestion.ingestion data/your_file.pdf")
                sys.exit(1)

            # Rows per source file
            cur.execute("""
                SELECT source_file, COUNT(*) AS chunks
                FROM multimodal_chunks
                GROUP BY source_file
                ORDER BY chunks DESC
            """)
            rows = cur.fetchall()
            print(f"\n  Chunks per source file:")
            for r in rows:
                print(f"    {r['source_file']}: {r['chunks']} chunks")

            # Sample content preview
            cur.execute("""
                SELECT content, source_file, page_number
                FROM multimodal_chunks
                WHERE content IS NOT NULL AND content != ''
                LIMIT 3
            """)
            samples = cur.fetchall()
            print(f"\n  Sample chunk contents:")
            for i, s in enumerate(samples):
                preview = s['content'][:120].replace('\n', ' ')
                print(f"    [{i+1}] page={s['page_number']} file={s['source_file']}")
                print(f"         \"{preview}...\"")

            # Check embeddings are actually stored
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE embedding IS NULL) AS null_embeddings,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS stored_embeddings
                FROM multimodal_chunks
            """)
            emb = cur.fetchone()
            print(f"\n  Embeddings: {emb['stored_embeddings']} stored, {emb['null_embeddings']} NULL")
            if emb['null_embeddings'] > 0:
                print("  ⚠️  Some chunks have NULL embeddings — they won't be retrieved!")

except Exception as e:
    print(f"  ❌ DB connection/query failed: {e}")
    sys.exit(1)


# ── Step 2: Test embedding generation ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Test query embedding generation")
print("="*60)

try:
    from src.core.db import _embed_texts
    test_query = "Tell me the bank name"
    embedding = _embed_texts([test_query])[0]
    print(f"  Query: \"{test_query}\"")
    print(f"  Embedding dimensions: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    print(f"  ✅ Embedding generation works")
except Exception as e:
    print(f"  ❌ Embedding failed: {e}")
    sys.exit(1)


# ── Step 3: Raw SQL similarity search ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Raw SQL cosine similarity search")
print("="*60)

try:
    from src.core.db import _embed_texts, get_db_conn

    test_query = "Tell me the bank name"
    embedding = _embed_texts([test_query])[0]
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    content,
                    source_file,
                    page_number,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM multimodal_chunks
                WHERE content IS NOT NULL AND content != ''
                ORDER BY embedding <=> %s::vector
                LIMIT 5
            """, (embedding_str, embedding_str))

            results = cur.fetchall()

    if not results:
        print("  ❌ No results returned from similarity search!")
    else:
        print(f"  ✅ Got {len(results)} results")
        print(f"\n  Top 5 similarity matches:")
        for i, r in enumerate(results):
            preview = r['content'][:100].replace('\n', ' ')
            print(f"\n    [{i+1}] score={r['similarity_score']:.4f} | page={r['page_number']} | file={r['source_file']}")
            print(f"         \"{preview}...\"")

        # Flag if scores are very low (possible embedding model mismatch)
        top_score = results[0]['similarity_score']
        if top_score < 0.2:
            print(f"\n  ⚠️  WARNING: Top similarity score is very low ({top_score:.4f})")
            print("     This often means the embedding model used during ingestion")
            print("     differs from the model used during retrieval.")
            print("     Check OPENAI_EMBEDDING_MODEL in your .env")

except Exception as e:
    print(f"  ❌ Raw search failed: {e}")
    import traceback; traceback.print_exc()


# ── Step 4: Test search_chunks() function (if it exists) ─────────────────────
print("\n" + "="*60)
print("STEP 4: Test search_chunks() from db.py")
print("="*60)

try:
    from src.core.db import search_chunks
    results = search_chunks("Tell me the bank name", k=5)
    if not results:
        print("  ❌ search_chunks() returned 0 results")
        print("     The function exists but returned nothing — check the SQL in search_chunks()")
    else:
        print(f"  ✅ search_chunks() returned {len(results)} results")
        for i, r in enumerate(results):
            preview = r['content'][:100].replace('\n', ' ')
            score = r.get('similarity_score', 'N/A')
            print(f"  [{i+1}] score={score} | {preview[:80]}...")
except ImportError:
    print("  ⚠️  search_chunks() not found in db.py — make sure you added it")
except Exception as e:
    print(f"  ❌ search_chunks() failed: {e}")
    import traceback; traceback.print_exc()


# ── Step 5: Test vector_search_node ──────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Test vector_search_node (full pipeline node)")
print("="*60)

try:
    from src.api.v1.tools.tools import vector_search_node, RAGState

    state: RAGState = {
        "query": "Tell me the bank name",
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "document",
        "generated_sql": "",
        "sql_result": "",
    }

    result_state = vector_search_node(state)
    docs = result_state["retrieved_docs"]

    if not docs:
        print("  ❌ vector_search_node returned 0 documents")
        print("     The node itself is returning nothing — check tools.py implementation")
    else:
        print(f"  ✅ vector_search_node returned {len(docs)} documents")
        for i, doc in enumerate(docs[:3]):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  [{i+1}] source={doc.metadata.get('source')} page={doc.metadata.get('page')}")
            print(f"       \"{preview}...\"")

except Exception as e:
    print(f"  ❌ vector_search_node failed: {e}")
    import traceback; traceback.print_exc()


print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("Share the output above and we'll know exactly what to fix.\n")