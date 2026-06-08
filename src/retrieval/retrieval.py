import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.core.db import get_db_conn

load_dotenv()

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RERANKER_MODEL = os.getenv("COHERE_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4")

_embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("OPENAI_API_KEY"),
)


def _embedding_to_pgvector(embedding: list[float]) -> str:
    """
    Convert Python list embedding into pgvector string format.
    Example:
        [0.1,0.2,0.3]
    """
    return "[" + ",".join(str(value) for value in embedding) + "]"


def embed_query(query: str) -> str:
    """
    Create embedding for user query and return pgvector string.
    """
    embedding = _embeddings.embed_query(query)
    return _embedding_to_pgvector(embedding)


def vector_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Vector search using PGVector cosine distance.
    This searches by meaning/semantic similarity.
    """

    query_embedding = embed_query(query)

    sql = """
        SELECT
            id,
            doc_id,
            chunk_type,
            element_type,
            content,
            page_number,
            section,
            source_file,
            document_name,
            product_category,
            image_path,
            metadata,
            embedding <=> %s::vector AS distance
        FROM multimodal_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

    results: list[dict[str, Any]] = []

    for rank, row in enumerate(rows, start=1):
        distance = float(row["distance"])

        results.append(
            {
                "id": str(row["id"]),
                "doc_id": str(row["doc_id"]),
                "rank": rank,
                "search_type": "vector",
                "score": 1 / (1 + distance),
                "distance": distance,
                "chunk_type": row["chunk_type"],
                "element_type": row["element_type"],
                "content": row["content"],
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "document_name": row["document_name"] or row["source_file"],
                "product_category": row["product_category"],
                "image_path": row["image_path"],
                "metadata": row["metadata"],
            }
        )

    return results


def fts_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Full-text search using PostgreSQL tsvector and plainto_tsquery.
    This searches by exact keywords.
    """

    sql = """
        SELECT
            id,
            doc_id,
            chunk_type,
            element_type,
            content,
            page_number,
            section,
            source_file,
            document_name,
            product_category,
            image_path,
            metadata,
            ts_rank_cd(
                to_tsvector('english', COALESCE(content, '')),
                plainto_tsquery('english', %s)
            ) AS rank_score
        FROM multimodal_chunks
        WHERE to_tsvector('english', COALESCE(content, ''))
              @@ plainto_tsquery('english', %s)
        ORDER BY rank_score DESC
        LIMIT %s;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, top_k))
            rows = cur.fetchall()

    results: list[dict[str, Any]] = []

    for rank, row in enumerate(rows, start=1):
        results.append(
            {
                "id": str(row["id"]),
                "doc_id": str(row["doc_id"]),
                "rank": rank,
                "search_type": "fts",
                "score": float(row["rank_score"]),
                "chunk_type": row["chunk_type"],
                "element_type": row["element_type"],
                "content": row["content"],
                "page_number": row["page_number"],
                "section": row["section"],
                "source_file": row["source_file"],
                "document_name": row["document_name"] or row["source_file"],
                "product_category": row["product_category"],
                "image_path": row["image_path"],
                "metadata": row["metadata"],
            }
        )

    return results


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    fts_results: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Merge vector search and FTS results using Reciprocal Rank Fusion.

    Formula:
        RRF score = 1 / (rrf_k + rank)

    If same chunk appears in both vector and FTS, its final score improves.
    """

    fused: dict[str, dict[str, Any]] = {}

    for result in vector_results:
        chunk_id = result["id"]
        rank = result["rank"]
        rrf_score = 1 / (rrf_k + rank)

        fused[chunk_id] = {
            **result,
            "rrf_score": rrf_score,
            "matched_by": ["vector"],
        }

    for result in fts_results:
        chunk_id = result["id"]
        rank = result["rank"]
        rrf_score = 1 / (rrf_k + rank)

        if chunk_id in fused:
            fused[chunk_id]["rrf_score"] += rrf_score
            fused[chunk_id]["matched_by"].append("fts")
        else:
            fused[chunk_id] = {
                **result,
                "rrf_score": rrf_score,
                "matched_by": ["fts"],
            }

    merged_results = list(fused.values())
    merged_results.sort(key=lambda item: item["rrf_score"], reverse=True)

    return merged_results[:top_k]


def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank top hybrid results using cross-encoder/ms-marco-MiniLM-L-6-v2.

    If sentence-transformers is missing, it safely skips reranking.
    """

    if not results:
        return []

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("[rerank_results] sentence-transformers not installed. Skipping reranker.")
        return results[:top_n]

    model = CrossEncoder(RERANKER_MODEL)

    pairs = [(query, result["content"]) for result in results]
    scores = model.predict(pairs)

    reranked: list[dict[str, Any]] = []

    for result, score in zip(results, scores):
        reranked.append(
            {
                **result,
                "rerank_score": float(score),
            }
        )

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    final_results = reranked[:top_n]

    for index, result in enumerate(final_results, start=1):
        result["final_rank"] = index

    return final_results


def hybrid_search(
    query: str,
    vector_top_k: int = 5,
    fts_top_k: int = 5,
    fused_top_k: int = 10,
    final_top_k: int = 5,
    use_reranker: bool = True,
) -> list[dict[str, Any]]:
    """
    Full hybrid retrieval pipeline.

    Steps:
    1. Vector search
    2. Full-text search
    3. RRF merge
    4. Optional reranking
    """

    print(f"[hybrid_search] Query: {query}")

    vector_results = vector_search(query=query, top_k=vector_top_k)
    print(f"[hybrid_search] Vector results: {len(vector_results)}")

    fts_results = fts_search(query=query, top_k=fts_top_k)
    print(f"[hybrid_search] FTS results: {len(fts_results)}")

    fused_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        fts_results=fts_results,
        top_k=fused_top_k,
    )
    print(f"[hybrid_search] Fused results: {len(fused_results)}")

    if use_reranker:
        final_results = rerank_results(
            query=query,
            results=fused_results,
            top_n=final_top_k,
        )
    else:
        final_results = fused_results[:final_top_k]

        for index, result in enumerate(final_results, start=1):
            result["final_rank"] = index

    print(f"[hybrid_search] Final results: {len(final_results)}")

    return final_results


def rewrite_query(query: str) -> list[str]:
    """
    Generate 2 alternate search phrases for retry logic.

    This is used only when the original search returns zero chunks.
    """

    try:
        llm = ChatOpenAI(
            model=OPENAI_CHAT_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

        prompt = f"""
You are a banking RAG query rewriter.

Rewrite the user's query into 2 alternate search phrases that can help retrieve
relevant banking product documents.

Rules:
- Return only 2 lines.
- Do not use numbering.
- Keep each line short.
- Preserve the user's meaning.
- Use banking/product terms where helpful.

User query:
{query}
"""

        response = llm.invoke(prompt)
        content = response.content

        if isinstance(content, list):
            content = " ".join(str(item) for item in content)

        lines = [
            line.strip("-• 1234567890. ").strip()
            for line in str(content).splitlines()
            if line.strip()
        ]

        rewritten = lines[:2]

        if len(rewritten) >= 2:
            return rewritten

    except Exception as exc:
        print(f"[rewrite_query] LLM rewrite failed: {exc}")

    # Safe fallback if LLM rewrite fails
    return [
        f"{query} banking product terms charges fees policy",
        f"{query} NorthStar Bank loan deposit credit card rules",
    ]


def retry_hybrid_search(
    query: str,
    vector_top_k: int = 5,
    fts_top_k: int = 5,
    fused_top_k: int = 10,
    final_top_k: int = 5,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """
    Retry logic for retrieval.

    attempt_1: search original query
    if no chunks:
    attempt_2: search rewritten phrase 1
    if no chunks:
    attempt_3: search rewritten phrase 2
    if still no chunks:
    return no relevant documents found
    """

    print("\n[retry_hybrid_search] Attempt 1: original query")

    results = hybrid_search(
        query=query,
        vector_top_k=vector_top_k,
        fts_top_k=fts_top_k,
        fused_top_k=fused_top_k,
        final_top_k=final_top_k,
        use_reranker=use_reranker,
    )

    if results:
        return {
            "status": "success",
            "original_query": query,
            "final_query": query,
            "retry_count": 0,
            "results": results,
            "message": "Results found using original query.",
        }

    rewritten_queries = rewrite_query(query)

    for retry_index, rewritten_query in enumerate(rewritten_queries, start=1):
        print(f"\n[retry_hybrid_search] Attempt {retry_index + 1}: {rewritten_query}")

        results = hybrid_search(
            query=rewritten_query,
            vector_top_k=vector_top_k,
            fts_top_k=fts_top_k,
            fused_top_k=fused_top_k,
            final_top_k=final_top_k,
            use_reranker=use_reranker,
        )

        if results:
            return {
                "status": "success",
                "original_query": query,
                "final_query": rewritten_query,
                "retry_count": retry_index,
                "results": results,
                "message": "Results found after query rewrite.",
            }

    return {
        "status": "not_found",
        "original_query": query,
        "final_query": query,
        "retry_count": 2,
        "results": [],
        "message": "No relevant documents found.",
    }


def print_results(results: list[dict[str, Any]]) -> None:
    """
    Pretty print retrieval results for local testing.
    """

    print("\nTop Retrieval Results:")

    for result in results:
        print("-" * 100)
        print(f"Final Rank      : {result.get('final_rank')}")
        print(f"Matched By      : {result.get('matched_by')}")
        print(f"Chunk Type      : {result.get('chunk_type')}")
        print(f"Page Number     : {result.get('page_number')}")
        print(f"Document        : {result.get('document_name')}")
        print(f"Product Category: {result.get('product_category')}")
        print(f"RRF Score       : {result.get('rrf_score')}")
        print(f"Rerank Score    : {result.get('rerank_score')}")
        print(f"Content         : {result.get('content', '')[:700]}")


if __name__ == "__main__":
    test_query = "What are the foreclosure charges for fixed rate home loans before 2022?"

    search_response = retry_hybrid_search(
        query=test_query,
        vector_top_k=5,
        fts_top_k=5,
        fused_top_k=10,
        final_top_k=5,
        use_reranker=True,
    )

    print("\nSearch Status:")
    print(f"Status       : {search_response['status']}")
    print(f"Retry Count  : {search_response['retry_count']}")
    print(f"Original Query: {search_response['original_query']}")
    print(f"Final Query  : {search_response['final_query']}")
    print(f"Message      : {search_response['message']}")

    print_results(search_response["results"])