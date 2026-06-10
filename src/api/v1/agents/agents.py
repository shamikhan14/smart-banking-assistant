import json
import os
import re
from typing import Literal, List, Dict, Any
import cohere
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

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

def _clean_answer(text: str) -> str:
    """Remove stray HTML from the answer field before sending to the UI."""
    text = _CITATION_CARD_RE.sub("", text)
    text = _SQL_BLOCK_RE.sub("", text)
    # Remove any remaining tags but keep the inner text (rare case)
    text = _HTML_TAG_RE.sub("", text)
    return text.strip()


# ── Node 0: Router ────────────────────────────────────────────────────────────

class _RouteDecision(BaseModel):
    route: Literal["product", "document", "smalltalk", "memory"]
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
            Classify the user's query into EXACTLY one of four routes:

            "memory"   — the query is about the current conversation history itself,
                         asking what was said, asked, or discussed in this session.
                         Examples: "what have I asked so far", "list my questions",
                         "what did we discuss", "show my conversation history",
                         "summarise our chat", "what was my last question".

            "smalltalk" — greeting, farewell, casual chat, or anything completely
                         unrelated to banking (math, geography, jokes, coding, etc.).
                         Examples: "hi", "how are you", "bye", "1+1", "capital of France".

            "product"  — asks about SPECIFIC customer data answerable from the DB:
                        account balance, transaction history, credit card details,
                        fixed deposits, loan status, EMI amounts, outstanding dues.
                        Typically mentions an account number or "my account/balance".

            "document" — asks about general banking knowledge: product features,
                        interest rates, eligibility, policies, fees, procedures,
                        terms & conditions.

            Priority order: memory → smalltalk → product → document.
            When in doubt between product and document, prefer document.

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


# ── Build the LangGraph ────────────────────────────────────────────────────────

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("router", router_node)
    graph.add_node("memory", memory_node)
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "memory":    "memory",
            "smalltalk": "smalltalk",
            "product":   "nl2sql",
            "document":  "vector_search",
        }
    )

    graph.add_edge("memory", END)
    graph.add_edge("smalltalk", END)
    graph.add_edge("nl2sql", END)
    graph.add_edge("vector_search", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", END)

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
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "",
        "generated_sql": "",
        "sql_result": "",
        "chat_history": chat_history or [],
    }
    final_state = rag_graph.invoke(initial_state)
    return final_state["response"]


async def run_search_agent_stream(query: str, chat_history: List[Dict[str, Any]] = None):
    """Run the full RAG graph and stream the final answer token by token."""
    import asyncio

    initial_state: RAGState = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "route": "",
        "generated_sql": "",
        "sql_result": "",
        "chat_history": chat_history or [],
    }

    final_state = rag_graph.invoke(initial_state)
    response = final_state.get("response", {})

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
        "policy_citations": response.get("policy_citations", ""),
        "page_no":          response.get("page_no", ""),
        "document_name":    response.get("document_name", ""),
        "sql_query_executed": response.get("sql_query_executed"),
    }
    yield f"data:{json.dumps(meta)}\n\n"
    yield "data:[DONE]\n\n"