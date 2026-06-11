import json
import os
import re
from uuid import uuid4
from typing import Literal, List, Dict, Any
import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

try:
    from langsmith import Client as LangSmithClient, traceable
except Exception:
    LangSmithClient = None

    def traceable(*args, **kwargs):
        """Fallback no-op decorator when langsmith is unavailable."""
        def decorator(func):
            return func
        return decorator

from src.api.v1.schemas.query_schema import AIResponse
from src.api.v1.tools.tools import RAGState, vector_search_node
from src.core.db import get_sql_database

load_dotenv()


# ── Helper: build the OpenAI LLM ──────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ── Helper: strip any HTML that leaks into LLM answer text ────────────────────
# The structured LLM occasionally echoes HTML tags (citation cards, sql-blocks)
# into the `answer` field. Strip them so they don't appear as raw markup in the
# Streamlit chat bubble (the UI renders citations separately from metadata fields).

_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_CITATION_CARD_RE = re.compile(
    r'<div class="citation-card">.*?</div>', re.DOTALL | re.IGNORECASE
)
_SQL_BLOCK_RE = re.compile(
    r'<div class="sql-block">.*?</div>', re.DOTALL | re.IGNORECASE
)

# Lightweight PII leak checks for evaluator metadata.
# These checks do not change the final answer; they only help LangSmith review.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_INDIAN_MOBILE_RE = re.compile(r"(?<!\d)(?:\+91[-\s]?)?[6-9]\d{9}(?!\d)")
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)")

def _clean_answer(text: str) -> str:
    """Remove stray HTML from the answer field before sending to the UI."""
    text = _CITATION_CARD_RE.sub("", text)
    text = _SQL_BLOCK_RE.sub("", text)
    # Remove any remaining tags but keep the inner text (rare case)
    text = _HTML_TAG_RE.sub("", text)
    return text.strip()


# ── Helper: LangSmith metadata tracing + evaluator-ready final output ─────────

def _build_trace_config(
    *,
    query: str,
    chat_history: List[Dict[str, Any]] | None,
    session_type: str,
    run_name: str,
    extra_tags: List[str] | None = None,
) -> tuple[dict, str, dict]:
    """Build LangSmith config metadata without changing agent execution logic."""
    run_id = str(uuid4())
    base_metadata = {
        "project": os.getenv("LANGSMITH_PROJECT", "smart-banking-assistant"),
        "course_code": os.getenv("COURSE_CODE", "BFSI-ARAG-002"),
        "application": "Smart Banking Assistant",
        "environment": os.getenv("APP_ENV", "local"),
        "session_type": session_type,
        "query": query,
        "query_length": len(query or ""),
        "chat_history_count": len(chat_history or []),
        "agent_framework": "LangGraph",
        "graph_name": "smart_banking_rag_graph",
        "entry_node": "router",
        "expected_routes": ["document", "banking_data", "hybrid", "smalltalk", "memory"],
        "retrieval_strategy": "vector_search_with_rerank",
        "vector_database": "PostgreSQL_pgvector",
        "reranker": "Cohere_rerank_english_v3_0",
        "citation_required_for": ["document", "hybrid"],
        "sql_safety_required": True,
        "allowed_sql_type": "SELECT_ONLY",
        "evaluator_focus": [
            "conciseness_check",
            "hallucination_check",
            "pii_leak_check",
            "citation_presence",
            "route_correctness",
            "sql_safety",
        ],
    }

    tags = [
        "smart-banking-assistant",
        "langgraph",
        "capstone",
        "metadata-tracing",
        "evaluator-ready",
    ]
    if extra_tags:
        tags.extend(extra_tags)

    config = {
        "run_id": run_id,
        "run_name": run_name,
        "tags": tags,
        "metadata": base_metadata,
    }
    return config, run_id, base_metadata


def _build_evaluator_metadata(final_state: RAGState) -> dict:
    """Create dynamic evaluator metadata from the actual final response/state."""
    response = final_state.get("response", {}) or {}
    answer = _clean_answer(response.get("answer", "") or "")
    answer_words = answer.split()

    route = final_state.get("route") or "unknown"
    policy_citations = response.get("policy_citations", "") or ""
    document_name = response.get("document_name", "") or ""
    page_no = response.get("page_no", "") or ""
    generated_sql = final_state.get("generated_sql") or response.get("sql_query_executed") or ""
    sql_result = final_state.get("sql_result") or ""

    citation_present = bool(
        str(policy_citations).strip()
        and str(policy_citations).strip().upper() not in {"N/A", "NA", "NONE"}
    )
    sql_present = bool(str(generated_sql).strip())
    answer_word_count = len(answer_words)

    detected_pii_patterns = []
    if _EMAIL_RE.search(answer):
        detected_pii_patterns.append("email")
    if _INDIAN_MOBILE_RE.search(answer):
        detected_pii_patterns.append("mobile")
    if _PAN_RE.search(answer):
        detected_pii_patterns.append("pan")
    if _AADHAAR_RE.search(answer):
        detected_pii_patterns.append("aadhaar")

    pii_leak_detected = bool(detected_pii_patterns)

    # Lightweight deterministic checks for evaluator review.
    # Hallucination is not fully proven here; this flags risk based on grounding signals.
    citation_required = route in {"document", "hybrid"}
    hallucination_risk = "low" if (
        route in {"banking_data", "smalltalk", "memory"}
        or sql_present
        or (citation_required and citation_present)
    ) else "needs_review"

    conciseness_check = "pass" if answer_word_count <= 120 else "review_needed"

    return {
        "actual_route": route,
        "retrieved_docs_count": len(final_state.get("retrieved_docs") or []),
        "reranked_docs_count": len(final_state.get("reranked_docs") or []),
        "generated_sql_present": sql_present,
        "sql_result_present": bool(str(sql_result).strip()),
        "citation_present": citation_present,
        "citation_required": citation_required,
        "document_name": document_name,
        "page_no": page_no,
        "answer_char_count": len(answer),
        "answer_word_count": answer_word_count,
        "answer_present": bool(answer.strip()),
        "conciseness_check": conciseness_check,
        "conciseness_rule": "pass when answer_word_count <= 120",
        "hallucination_check": hallucination_risk,
        "hallucination_check_method": "heuristic grounding check using route, citation presence, and SQL presence",
        "hallucination_note": "Use LangSmith LLM-as-judge evaluator for final hallucination scoring.",
        "pii_leak_check": "needs_review" if pii_leak_detected else "pass",
        "pii_leak_detected": pii_leak_detected,
        "pii_patterns_detected": detected_pii_patterns,
        "pii_check_method": "regex check for email, Indian mobile number, PAN, and Aadhaar-like patterns in final answer",
    }


def _safe_update_langsmith_metadata(run_id: str, metadata: dict) -> None:
    """Best-effort update of completed LangSmith run metadata; never breaks API flow."""
    if LangSmithClient is None or not os.getenv("LANGSMITH_API_KEY"):
        return
    try:
        LangSmithClient().update_run(run_id, extra={"metadata": metadata})
    except Exception as exc:
        print(f"[langsmith_metadata] Could not update evaluator metadata: {exc}")


@traceable(
    name="smart_banking_final_response",
    run_type="chain",
    tags=["smart-banking-assistant", "final-response", "evaluator-ready"],
)
def _capture_final_response_for_evaluators(
    query: str,
    response: dict,
    evaluator_metadata: dict,
) -> dict:
    """Expose final assistant response as clean LangSmith output for evaluators.

    This does not change business logic. It only creates a LangSmith run whose
    outputs contain `answer`, so conciseness, hallucination, and PII leak
    evaluators can read `outputs.answer` instead of an empty graph state.
    """
    return {
        **(response or {}),
        "evaluator_metadata": evaluator_metadata,
    }


# ── Node 0: Router ────────────────────────────────────────────────────────────

class _RouteDecision(BaseModel):
    route: Literal["banking_data", "document", "smalltalk", "memory", "hybrid"]
    reason: str


def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)

    # Build a compact summary of past queries for the router so it can
    # detect memory-style questions like "what have I asked so far?"
    history = state.get("chat_history") or []
    past_questions = [
        m["content"] for m in history if m.get("role") == "user"
    ]
    history_hint = (
        f"The user has previously asked {len(past_questions)} question(s) in this session."
        if past_questions
        else "This is the first message in the session."
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a query router for a smart banking assistant.
            Classify the user's query into EXACTLY one of five routes:

            "memory"   — the query is about the current conversation history itself,
                         asking what was said, asked, or discussed in this session.
                         Examples: "what have I asked so far", "list my questions",
                         "what did we discuss", "show my conversation history",
                         "summarise our chat", "what was my last question".

            "smalltalk" — greeting, farewell, casual chat, or anything completely
                         unrelated to banking (math, geography, jokes, coding, etc.).
                         Examples: "hi", "how are you", "bye", "1+1", "capital of France".

            "banking_data"  — asks about SPECIFIC customer data answerable from the DB alone:
                        account balance, transaction history, credit card details,
                        fixed deposits, loan status, EMI amounts, outstanding dues.
                        Typically mentions an account number or "my account/balance"
                        AND does NOT also ask about policies, eligibility rules, or
                        banking_data features that require the knowledge base.

            "document" — asks about general banking knowledge alone: prod features,
                        interest rates, eligibility criteria, policies, fees, procedures,
                        terms & conditions — with NO reference to a specific customer's
                        live account data.

            "hybrid"   — the query genuinely needs BOTH live customer/account data
                        from the database AND policy/banking_data knowledge from the PDF
                        knowledge base to give a complete answer. Use this when the
                        question cross-references a specific customer's data against
                        a bank policy or banking_data rule.
                        Examples:
                          "Does James (account 1345367) qualify for a top-up home loan?"
                          "Based on my transaction history, am I eligible for a credit card upgrade?"
                          "What is the interest rate for my current loan type and when is my next EMI?"
                          "Compare my FD rate to the current best FD rates offered by the bank."
                          "Can I get a personal loan given my outstanding home loan balance?"
                          "Show my last 3 months spending and explain any applicable cashback policy."

            Priority order: memory → smalltalk → hybrid → banking_data → document.
            Choose "hybrid" whenever the answer requires fetching live DB data AND
            looking up a policy/banking_data rule; do NOT split such queries into banking_data
            or document alone.

            Context: {history_hint}

            Reply with the route and a one-sentence reason."""
        ),
        ("human", "Query: {query}")
    ])

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"], "history_hint": history_hint})
    print(f"[router_node] Route → '{decision.route}' | Reason: {decision.reason}")
    return {
        **state,
        "route": decision.route
    }


# ── Node 0a: Memory ───────────────────────────────────────────────────────────

def memory_node(state: RAGState) -> RAGState:
    """Answer questions about the current conversation from in-memory history."""
    llm = _get_llm()

    history: List[Dict[str, Any]] = state.get("chat_history") or []

    # Build a readable transcript for the LLM
    if history:
        transcript_lines = []
        q_index = 1
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                transcript_lines.append(f"Q{q_index}: {content}")
                q_index += 1
            else:
                transcript_lines.append(f"A{q_index - 1}: {content}")
        transcript = "\n".join(transcript_lines)
    else:
        transcript = "(No previous messages in this session.)"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are NorthStar Bank's virtual assistant.
            The user is asking about the current chat session.
            Use ONLY the conversation transcript below to answer.
            Be concise and accurate. Format lists clearly.
            Do NOT mention document names, page numbers, or citations."""
        ),
        (
            "human",
            "Conversation so far:\n{transcript}\n\nUser's question: {query}"
        )
    ])

    chain = prompt | llm
    result = chain.invoke({"transcript": transcript, "query": state["query"]})
    answer = _clean_answer(result.content.strip())

    print(f"[memory_node] Memory response generated ({len(history)} history items).")
    return {
        **state,
        "response": {
            "query": state["query"],
            "answer": answer,
            "policy_citations": "",
            "page_no": "",
            "document_name": "",
            "sql_query_executed": None,
        }
    }


# ── Node 0b: Small Talk ───────────────────────────────────────────────────────

def smalltalk_node(state: RAGState) -> RAGState:
    """Handle greetings and off-topic queries without touching the RAG pipeline."""
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are NorthStar Bank's friendly virtual assistant.
            Your role is strictly limited to banking topics.

            For greetings and casual messages (hi, hello, how are you, bye, thank you, etc.):
            — Respond warmly and briefly. Mention you are here to help with banking queries.

            For anything completely unrelated to banking (math, general knowledge, coding,
            science, jokes, geography, etc.):
            — Politely decline and redirect the user to banking topics.
            — Keep it short (1-2 sentences).

            Never answer non-banking questions. Never include HTML tags, citation blocks,
            or SQL in your response."""
        ),
        ("human", "{query}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": state["query"]})
    answer = _clean_answer(result.content.strip())

    print(f"[smalltalk_node] Smalltalk response generated.")
    return {
        **state,
        "response": {
            "query": state["query"],
            "answer": answer,
            "policy_citations": "",
            "page_no": "",
            "document_name": "",
            "sql_query_executed": None,
        }
    }


# ── Node 1: NL2SQL ────────────────────────────────────────────────────────────

def nl2sql_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    db = get_sql_database()

    # ── Step 1: Generate SQL ─────────────────────────────────────────────────
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a PostgreSQL expert. Given the database schema below,
            write a single valid SELECT query that answers the user's question.

            Rules:
            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
            - Use only the tables and columns present in the schema.
            - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
            - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.

            Database schema:
            {schema}"""
        ),
        ("human", "Question: {question}")
    ])

    sql_chain = sql_prompt | llm
    raw_sql = sql_chain.invoke({
        "schema": schema_info,
        "question": state["query"]
    })

    content = raw_sql.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    generated_sql = content.strip().strip("```").strip()
    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()
    print(f"[nl2sql_node] Generated SQL:\n{generated_sql}")

    # ── Step 2: Execute SQL ──────────────────────────────────────────────────
    try:
        sql_result: str = db.run(generated_sql)
    except Exception as exc:
        sql_result = f"SQL execution error: {exc}"
    print(f"[nl2sql_node] Raw result (truncated): {str(sql_result)[:200]}")

    # ── Step 3: Summarise into AIResponse ────────────────────────────────────
    structured_llm = llm.with_structured_output(AIResponse)
    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful data analyst. Answer the user's question using "
            "the SQL query results below. Be concise and format numbers/lists clearly. "
            "Set policy_citations to empty string, "
            "page_no to 'N/A', and document_name to 'agentic_rag_db'. "
            "IMPORTANT: The answer field must contain ONLY plain text — "
            "no HTML tags, no <div> blocks, no citation cards."
        ),
        (
            "human",
            "Question: {query}\n\n"
            "SQL Used:\n{sql}\n\n"
            "Query Results:\n{result}"
        )
    ])

    chain = answer_prompt | structured_llm
    answer = chain.invoke({
        "query": state["query"],
        "sql": generated_sql,
        "result": sql_result
    })
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    response["answer"] = _clean_answer(response.get("answer", ""))
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response
    }


# ── Node 2: Rerank ────────────────────────────────────────────────────────────

def rerank_node(state: RAGState) -> RAGState:
    docs = state["retrieved_docs"]

    # Guard: if vector search returned nothing, skip reranking
    if not docs:
        print("[rerank_node] No documents retrieved — skipping rerank.")
        return {**state, "reranked_docs": []}

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=10
    )

    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        print(f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}")

    return {**state, "reranked_docs": reranked_docs}


# ── Node 3: Generate Answer ───────────────────────────────────────────────────

def generate_answer_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    # Guard: no context available
    if not state["reranked_docs"]:
        print("[generate_answer_node] No reranked docs — returning fallback response.")
        return {
            **state,
            "response": {
                "query": state["query"],
                "answer": "I could not find any relevant documents to answer your question.",
                "policy_citations": "N/A",
                "page_no": "N/A",
                "document_name": "N/A",
                "sql_query_executed": None,
            }
        }

    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]\n{doc.page_content}"
        for doc in state["reranked_docs"]
    ])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the user's question using only the "
            "provided context.\n\n"
            "IMPORTANT: The context may contain chunks from MULTIPLE versions of the same "
            "document (e.g. a 2025 edition and a 2026 edition). When the answer differs "
            "across versions, do NOT pick only one. Instead:\n"
            "  - Lead with the most recent / current version's answer (highest year).\n"
            "  - Then explicitly note how earlier versions differed "
            "(e.g. 'As of the 2026 policy ...; previously, under the 2025 policy ...').\n"
            "  - If all versions agree, just give the single answer.\n\n"
            "Citation rules (fill the structured fields):\n"
            "  - document_name: comma-separated list of EVERY source document you used.\n"
            "  - page_no: comma-separated page numbers, aligned with the documents above.\n"
            "  - policy_citations: a readable citation combining each document and its page "
            "(e.g. 'KB_Smart_Banking.pdf, Page 1').\n"
            "Always cite ALL versions you drew the answer from, not just one.\n\n"
            "CRITICAL: The `answer` field must contain ONLY plain text — "
            "absolutely no HTML tags, no <div> elements, no citation cards, no SQL blocks. "
            "Citation information belongs ONLY in the structured fields above."
        ),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "query": state["query"]})

    response = result.model_dump()
    response["answer"] = _clean_answer(response.get("answer", ""))
    print(f"[generate_answer_node] Answer generated.")
    return {**state, "response": response}


# ── Node 4: Hybrid RAG + SQL Fusion ──────────────────────────────────────────
#
# This node handles queries that need BOTH:
#   • Live customer / account data  → executed via NL2SQL against the DB
#   • Policy / banking_data knowledge    → retrieved from the PDF knowledge base
#
# Pipeline:
#   1. Run NL2SQL exactly like nl2sql_node (generate + execute SQL)
#   2. Run vector_search_node to retrieve relevant PDF chunks
#   3. Rerank the retrieved docs with Cohere
#   4. Hand BOTH the SQL result AND the reranked PDF context to a fusion LLM
#      that synthesises a single, coherent, cited answer
#
# The node writes directly to state["response"] so it can wire straight to END.

def hybrid_rag_node(state: RAGState) -> RAGState:
    """Fuse live DB data (NL2SQL) with PDF knowledge-base context into one answer."""
    llm = _get_llm()
    db  = get_sql_database()

    print("[hybrid_rag_node] ── Starting hybrid RAG + SQL pipeline ──────────────")

    # ── Step 1: Generate & execute SQL ───────────────────────────────────────
    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a PostgreSQL expert. Given the database schema below,
            write a single valid SELECT query that fetches the customer/account
            data relevant to answering the user's question.

            Rules:
            - Return ONLY the raw SQL — no explanation, no markdown fences, no backticks.
            - Use only the tables and columns present in the schema.
            - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL.
            - Always add a LIMIT clause (max 50 rows) unless the question is an aggregate.
            - If no DB data is needed, return the string: NO_SQL_NEEDED

            Database schema:
            {schema}"""
        ),
        ("human", "Question: {question}")
    ])

    raw_sql_msg = (sql_prompt | llm).invoke({
        "schema": schema_info,
        "question": state["query"],
    })

    raw_content = raw_sql_msg.content
    if isinstance(raw_content, list):
        raw_content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in raw_content
        )
    generated_sql = raw_content.strip().strip("```").strip()
    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()

    print(f"[hybrid_rag_node] Generated SQL:\n{generated_sql}")

    if generated_sql.upper() == "NO_SQL_NEEDED":
        sql_result = "(No database query was needed for this question.)"
        generated_sql = ""
    else:
        try:
            sql_result = db.run(generated_sql)
        except Exception as exc:
            sql_result = f"SQL execution error: {exc}"
            print(f"[hybrid_rag_node] SQL error: {exc}")

    print(f"[hybrid_rag_node] SQL result (truncated): {str(sql_result)[:200]}")

    # ── Step 2: Vector search → retrieve PDF chunks ───────────────────────────
    # Re-use the existing vector_search_node which handles tool-calling logic
    # (semantic / keyword / hybrid search selection) internally.
    intermediate_state = vector_search_node({**state})
    retrieved_docs = intermediate_state.get("retrieved_docs", [])
    print(f"[hybrid_rag_node] Retrieved {len(retrieved_docs)} docs from vector search.")

    # ── Step 3: Rerank the PDF chunks with Cohere ─────────────────────────────
    reranked_docs: list = []
    if retrieved_docs:
        try:
            co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=state["query"],
                documents=[doc.page_content for doc in retrieved_docs],
                top_n=min(8, len(retrieved_docs)),
            )
            reranked_docs = [retrieved_docs[r.index] for r in rerank_response.results]
            print(f"[hybrid_rag_node] Reranked to top {len(reranked_docs)} PDF chunks.")
            for i, r in enumerate(rerank_response.results):
                print(f"  Rank {i+1} | score: {r.relevance_score:.4f} | idx: {r.index}")
        except Exception as exc:
            # Reranking is best-effort; fall back to raw retrieval order
            print(f"[hybrid_rag_node] Cohere rerank failed ({exc}); using raw order.")
            reranked_docs = retrieved_docs[:8]
    else:
        print("[hybrid_rag_node] No PDF chunks retrieved — SQL-only fusion.")

    # ── Step 4: Build fused context strings ──────────────────────────────────
    pdf_context = "\n\n".join([
        f"[PDF Source: {doc.metadata.get('source', 'unknown')} | "
        f"Page: {doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]\n"
        f"{doc.page_content}"
        for doc in reranked_docs
    ]) if reranked_docs else "(No relevant policy/banking_data documents found.)"

    db_context = (
        f"SQL Query Executed:\n{generated_sql}\n\nQuery Results:\n{sql_result}"
        if generated_sql
        else sql_result
    )

    # ── Step 5: Fusion LLM call ───────────────────────────────────────────────
    structured_llm = llm.with_structured_output(AIResponse)

    fusion_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are NorthStar Bank's expert assistant. You have been given TWO
            sources of information to answer the user's question:

            SOURCE A — Live Customer / Account Data (from the bank's database):
            This contains real-time facts about the specific customer: their account
            balances, transaction history, loan details, FD holdings, credit card
            status, EMI schedules, etc.

            SOURCE B — Bank Policy & banking_data Knowledge Base (from ingested PDFs):
            This contains general banking policies, eligibility rules, interest rate
            tables, banking_data features, fee schedules, terms & conditions, etc.

            Your task:
            1. Use SOURCE A facts to ground the answer in the customer's actual situation.
            2. Use SOURCE B to apply the relevant policy, eligibility rule, or banking_data
               information to those facts.
            3. Synthesise BOTH into a single coherent, accurate, helpful answer.
            4. Be explicit about which part comes from the customer's data vs. the policy.
               For example: "Your outstanding loan balance is ₹38,20,000 (from your account).
               The bank's top-up loan policy requires the outstanding to be below 80% of
               the original principal — you are currently at 84.9%, so you do not yet qualify."
            5. If SOURCE A or SOURCE B is missing or insufficient, clearly state what
               information is unavailable and answer with whatever IS available.

            Citation rules (fill the structured fields):
            - document_name : comma-separated list of PDF documents used (from SOURCE B).
              Write "agentic_rag_db" when only DB data was used, or append it when both.
            - page_no        : comma-separated page numbers from SOURCE B docs used.
            - policy_citations: readable citation, e.g. "KB_Smart_Banking.pdf, Page 5".
            - sql_query_executed: the SQL that was run (already provided below).

            CRITICAL: The `answer` field must contain ONLY plain text — no HTML tags,
            no <div> elements, no citation cards, no SQL blocks."""
        ),
        (
            "human",
            "User Question: {query}\n\n"
            "── SOURCE A: Live Database Data ──────────────────────────────\n"
            "{db_context}\n\n"
            "── SOURCE B: PDF Knowledge Base ──────────────────────────────\n"
            "{pdf_context}"
        ),
    ])

    fusion_chain = fusion_prompt | structured_llm
    result = fusion_chain.invoke({
        "query":       state["query"],
        "db_context":  db_context,
        "pdf_context": pdf_context,
    })

    response = result.model_dump()
    response["answer"]             = _clean_answer(response.get("answer", ""))
    response["sql_query_executed"] = generated_sql or None

    print(f"[hybrid_rag_node] Fusion answer generated.")
    print(f"[hybrid_rag_node] ── Hybrid pipeline complete ─────────────────────────\n")

    return {
        **state,
        "retrieved_docs":        retrieved_docs,
        "reranked_docs":         reranked_docs,
        "generated_sql":         generated_sql,
        "sql_result":            str(sql_result),
        "hybrid_rag_sql_result": str(sql_result),
        "response":              response,
    }


# ── Build the LangGraph ────────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("router",         router_node)
    graph.add_node("memory",         memory_node)
    graph.add_node("smalltalk",      smalltalk_node)
    graph.add_node("nl2sql",         nl2sql_node)
    graph.add_node("vector_search",  vector_search_node)
    graph.add_node("rerank",         rerank_node)
    graph.add_node("generate_answer",generate_answer_node)
    graph.add_node("hybrid_rag",     hybrid_rag_node)   # ← new fusion node

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "memory":    "memory",
            "smalltalk": "smalltalk",
            "banking_data":   "nl2sql",
            "document":  "vector_search",
            "hybrid":    "hybrid_rag",     # ← new route
        }
    )

    graph.add_edge("memory",         END)
    graph.add_edge("smalltalk",      END)
    graph.add_edge("nl2sql",         END)
    graph.add_edge("vector_search",  "rerank")
    graph.add_edge("rerank",         "generate_answer")
    graph.add_edge("generate_answer",END)
    graph.add_edge("hybrid_rag",     END)   # ← fusion node writes response directly

    compiled = graph.compile()

    graph_image = compiled.get_graph().draw_mermaid_png()
    with open("rag_graph.png", "wb") as f:
        f.write(graph_image)

    return compiled


# Compile once at module load — reused across all requests
rag_graph = build_rag_graph()


# ── Public entrypoint (called by query_service.py) ─────────────────────────
def run_search_agent(query: str, chat_history: List[Dict[str, Any]] = None) -> dict:
    initial_state: RAGState = {
        "query":                  query,
        "retrieved_docs":         [],
        "reranked_docs":          [],
        "response":               {},
        "route":                  "",
        "generated_sql":          "",
        "sql_result":             "",
        "hybrid_rag_sql_result":  "",   # ← new field
        "chat_history":           chat_history or [],
    }
    trace_config, run_id, base_metadata = _build_trace_config(
        query=query,
        chat_history=chat_history,
        session_type="api",
        run_name="smart_banking_agent",
        extra_tags=["api"],
    )

    final_state = rag_graph.invoke(initial_state, config=trace_config)

    evaluator_metadata = _build_evaluator_metadata(final_state)
    _safe_update_langsmith_metadata(run_id, {**base_metadata, **evaluator_metadata})

    response = _capture_final_response_for_evaluators(
        query=query,
        response=final_state["response"],
        evaluator_metadata=evaluator_metadata,
    )

    return response


async def run_search_agent_stream(query: str, chat_history: List[Dict[str, Any]] = None):
    """Run the full RAG graph and stream the final answer token by token."""
    import asyncio

    initial_state: RAGState = {
        "query":                  query,
        "retrieved_docs":         [],
        "reranked_docs":          [],
        "response":               {},
        "route":                  "",
        "generated_sql":          "",
        "sql_result":             "",
        "hybrid_rag_sql_result":  "",   # ← new field
        "chat_history":           chat_history or [],
    }

    trace_config, run_id, base_metadata = _build_trace_config(
        query=query,
        chat_history=chat_history,
        session_type="streaming_api",
        run_name="smart_banking_agent_stream",
        extra_tags=["streaming"],
    )

    final_state = rag_graph.invoke(initial_state, config=trace_config)

    evaluator_metadata = _build_evaluator_metadata(final_state)
    _safe_update_langsmith_metadata(run_id, {**base_metadata, **evaluator_metadata})

    response = _capture_final_response_for_evaluators(
        query=query,
        response=final_state.get("response", {}),
        evaluator_metadata=evaluator_metadata,
    )

    answer: str = _clean_answer(
        response.get("answer", "I'm sorry, I couldn't generate a response.")
    )

    # Stream the answer word by word for a smooth typing effect
    words = answer.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == len(words) - 1 else word + " "
        yield f"data:{json.dumps({'token': chunk})}\n\n"
        await asyncio.sleep(0.018)

    # Emit structured metadata as a single trailing event
    meta = {
        "policy_citations":   response.get("policy_citations", ""),
        "page_no":            response.get("page_no", ""),
        "document_name":      response.get("document_name", ""),
        "sql_query_executed": response.get("sql_query_executed"),
        "evaluator_metadata": evaluator_metadata,
    }
    yield f"data:{json.dumps(meta)}\n\n"
    yield "data:[DONE]\n\n"