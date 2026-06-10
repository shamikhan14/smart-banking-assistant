import os
import re
from typing import TypedDict, List, Dict, Any

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from src.core.db import search_chunks, fts_search_chunks, hybrid_search_chunks


# ── State definition (unchanged — agents.py imports this) ────────────────────

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    response: dict
    route: str
    generated_sql: str
    sql_result: str
    # Each entry: {"role": "user"|"assistant", "content": str}
    chat_history: List[Dict[str, Any]]


# ── Shared row → Document converter (unchanged) ───────────────────────────────

def _rows_to_docs(rows: list[dict]) -> list[Document]:
    """Convert raw DB rows from any search function to LangChain Documents."""
    docs = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata["source"]           = row.get("source_file", "unknown")
        metadata["page"]             = (row.get("page_number") or 1) - 1   # 0-indexed
        metadata["section"]          = row.get("section")
        metadata["similarity_score"] = row.get("similarity_score")
        docs.append(Document(page_content=row["content"], metadata=metadata))
    return docs


# ── @tool-decorated search functions ─────────────────────────────────────────
# The LLM (bound via .bind_tools()) sees these descriptions and decides which
# tool to call — exactly like the reference agent.py pattern.

@tool
def semantic_search(query: str) -> list[dict]:
    """Use for long, conversational, or natural-language questions where
    semantic meaning matters — e.g. 'What are the eligibility criteria for
    a home loan?' or 'Explain the interest rate policy for FDs.'
    Performs cosine-similarity vector search against the knowledge base.
    """
    print("🔵 [tool] semantic_search called")
    rows = search_chunks(query, k=20)
    # Return lightweight dicts; the node converts them to Documents afterwards
    return [
        {
            "content":          r["content"],
            "source_file":      r.get("source_file"),
            "page_number":      r.get("page_number"),
            "section":          r.get("section"),
            "similarity_score": r.get("similarity_score"),
            "metadata":         dict(r.get("metadata") or {}),
        }
        for r in rows
    ]


@tool
def keyword_search(query: str) -> list[dict]:
    """Use for exact-match queries: policy/ticket codes (e.g. POL-2024-HR-007),
    short uppercase abbreviations (KYC, IMPS, RTGS, NEFT), or long numeric
    identifiers like account numbers or loan IDs.
    Performs PostgreSQL full-text search (tsvector/tsquery).
    """
    print("🟡 [tool] keyword_search called")
    rows = fts_search_chunks(query, k=20)
    return [
        {
            "content":          r["content"],
            "source_file":      r.get("source_file"),
            "page_number":      r.get("page_number"),
            "section":          r.get("section"),
            "similarity_score": r.get("similarity_score"),
            "metadata":         dict(r.get("metadata") or {}),
        }
        for r in rows
    ]


@tool
def hybrid_search(query: str) -> list[dict]:
    """Use for short, isolated, or ambiguous phrases that need both keyword
    precision and semantic context — e.g. 'gold loan', 'minimum balance',
    'overdraft facility'. Combines vector search and FTS via Reciprocal Rank
    Fusion (RRF) so results strong in both lists are promoted.
    """
    print("🟢 [tool] hybrid_search called")
    rows = hybrid_search_chunks(query, k=20)
    return [
        {
            "content":          r["content"],
            "source_file":      r.get("source_file"),
            "page_number":      r.get("page_number"),
            "section":          r.get("section"),
            "similarity_score": r.get("similarity_score"),
            "metadata":         dict(r.get("metadata") or {}),
        }
        for r in rows
    ]


# ── Tool registry ─────────────────────────────────────────────────────────────

SEARCH_TOOLS = [semantic_search, keyword_search, hybrid_search]

# Map tool name → callable so we can dispatch ToolCall results
_TOOL_MAP: dict[str, Any] = {t.name: t for t in SEARCH_TOOLS}


# ── vector_search_node — LLM-driven tool-calling retrieval ───────────────────

def vector_search_node(state: RAGState) -> RAGState:
    """Retrieval node that lets the LLM decide which search tool to invoke.

    Flow:
      1. Bind the three @tool functions to an LLM.
      2. Send the query; the LLM responds with a tool_call (not plain text).
      3. Execute whichever tool the LLM chose.
      4. If the chosen tool returns 0 results, fall back to semantic_search.
      5. Convert raw rows → LangChain Documents and store in state.

    The rerank and generate_answer nodes downstream are completely unchanged
    because they only consume state["retrieved_docs"].
    """
    query = state["query"]

    # ── Step 1: Build LLM with tools bound ───────────────────────────────────
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(SEARCH_TOOLS, tool_choice="required")

    # ── Step 2: Ask the LLM to pick a tool ───────────────────────────────────
    system_msg = (
       """ You are a retrieval strategy selector for a banking knowledge base. 
        Given the user query, call EXACTLY ONE of the three search tools:
        TOOL SELECTION STRATEGY:
        - keyword_search: Matching Abbreviations, Any regular expression pattern matching the exact below pattern
            [r""[A-Z]{2,}-\d{4}-\w+"",   # policy/ticket codes: POL-2024-HR-007
            r""\b[A-Z]{2,5}\b"",         # short uppercase abbreviations: LTA, CTC, ESI
            r""\d{6,}""]                # long numeric 				
        - vector_search: Conversational scenarios, general strategy advice, matching risk appetite profiles to assets.
        - hybrid_search: Short, isolated, or ambiguous financial phrases.
        Do NOT answer the question yourself. Only call the appropriate tool."""
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": query},
    ]

    ai_message: AIMessage = llm_with_tools.invoke(messages)

    # ── Step 3: Dispatch the chosen tool call ─────────────────────────────────
    tool_calls = getattr(ai_message, "tool_calls", [])

    chosen_tool_name = "semantic_search"   # safe default
    rows: list[dict] = []

    if tool_calls:
        tc = tool_calls[0]                 # we asked for exactly one
        chosen_tool_name = tc["name"]
        tool_fn = _TOOL_MAP.get(chosen_tool_name)

        _LABELS = {
            "semantic_search": "🔵 SEMANTIC  (cosine vector search)",
            "keyword_search":  "🟡 FTS       (PostgreSQL full-text search)",
            "hybrid_search":   "🟢 HYBRID    (RRF: semantic + FTS)",
        }
        print(
            f"\n[vector_search_node] ┌─ LLM chose tool ──────────────────────────────────\n"
            f"[vector_search_node] │  {_LABELS.get(chosen_tool_name, chosen_tool_name)}\n"
            f"[vector_search_node] │  Query : {query!r}\n"
            f"[vector_search_node] └───────────────────────────────────────────────────"
        )

        if tool_fn:
            rows = tool_fn.invoke({"query": query})
    else:
        # LLM returned text instead of a tool call (shouldn't happen with
        # tool_choice="required", but guard against it gracefully)
        print("[vector_search_node] ⚠️  No tool_call in LLM response — defaulting to semantic_search")
        rows = semantic_search.invoke({"query": query})

    # ── Step 4: Fallback if chosen tool returned nothing ─────────────────────
    if not rows and chosen_tool_name != "semantic_search":
        print(
            f"[vector_search_node] ⚠️  {chosen_tool_name} returned 0 results — "
            f"falling back to 🔵 semantic_search"
        )
        chosen_tool_name = "semantic_search"
        rows = semantic_search.invoke({"query": query})

    # ── Step 5: Convert to LangChain Documents ────────────────────────────────
    docs = _rows_to_docs(rows)

    top_score_str = ""
    if docs:
        top_score = docs[0].metadata.get("similarity_score")
        if top_score is not None:
            top_score_str = f"  │  Top score : {top_score:.4f}"

    print(
        f"[vector_search_node] ┌─ Retrieval complete ───────────────────────────────\n"
        f"[vector_search_node] │  Tool used     : {chosen_tool_name}\n"
        f"[vector_search_node] │  Chunks found  : {len(docs)}"
        + (f"\n[vector_search_node] {top_score_str}" if top_score_str else "")
        + f"\n[vector_search_node] └───────────────────────────────────────────────────\n"
    )

    return {**state, "retrieved_docs": docs}