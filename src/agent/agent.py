import os
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.retrieval.retrieval import retry_hybrid_search
from src.sql_agent.nl2sql import ask_database

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


class AgentState(TypedDict):
    query: str
    query_path: str
    rag_result: dict[str, Any]
    sql_result: dict[str, Any]
    final_response: dict[str, Any]


class RouteDecision(BaseModel):
    route: Literal["rag", "sql", "hybrid"]
    reason: str


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def query_classifier_node(state: AgentState) -> AgentState:
    """
    Decide whether the query should go to:
    - rag: document/product policy questions
    - sql: structured account/transaction/loan/FD/card data
    - hybrid: needs both SQL data and document explanation
    """

    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = f"""
You are a query classifier for a Smart Banking Assistant.

Classify the user query into exactly one route:

1. rag
Use this when the question asks about banking product documents, policies, fees,
charges, eligibility, interest rates, terms and conditions, disclosures, or rules.

Examples:
- What are foreclosure charges before 2022?
- What is the FD interest rate for 444 days?
- What are credit card international transaction fees?

2. sql
Use this when the question asks about structured customer/account data from database tables:
accounts, transactions, loan_accounts, fixed_deposits, credit_cards, card_transactions.

Examples:
- Give me last 3 months purchase history of account 1345367.
- What is outstanding balance for loan L-789012?
- Show active FDs for account 1345367.
- Show international transactions on card CC-881001.

3. hybrid
Use this when the question needs both:
- SQL data from customer/account tables
- RAG explanation from product/policy documents

Examples:
- Show international transactions on CC-881001 and explain international transaction fees.
- Show active FDs for account 1345367 and explain premature withdrawal policy.
- Show loan L-789012 details and explain foreclosure charges.

User query:
{state["query"]}
"""

    decision = structured_llm.invoke(prompt)

    print(f"[query_classifier_node] Route: {decision.route}")
    print(f"[query_classifier_node] Reason: {decision.reason}")

    return {
        **state,
        "query_path": decision.route,
    }


def rag_node(state: AgentState) -> AgentState:
    """
    Run RAG retrieval path.
    """

    print("[rag_node] Running RAG retrieval")

    rag_result = retry_hybrid_search(
        query=state["query"],
        vector_top_k=5,
        fts_top_k=5,
        fused_top_k=10,
        final_top_k=5,
        use_reranker=True,
    )

    return {
        **state,
        "rag_result": rag_result,
    }


def sql_node(state: AgentState) -> AgentState:
    """
    Run SQL/NL2SQL path.
    """

    print("[sql_node] Running NL2SQL")

    sql_result = ask_database(state["query"])

    return {
        **state,
        "sql_result": sql_result,
    }


def hybrid_node(state: AgentState) -> AgentState:
    """
    Run both RAG and SQL paths.
    """

    print("[hybrid_node] Running both RAG and SQL")

    rag_result = retry_hybrid_search(
        query=state["query"],
        vector_top_k=5,
        fts_top_k=5,
        fused_top_k=10,
        final_top_k=5,
        use_reranker=True,
    )

    sql_result = ask_database(state["query"])

    return {
        **state,
        "rag_result": rag_result,
        "sql_result": sql_result,
    }


def response_generator_node(state: AgentState) -> AgentState:
    """
    Create final response object.
    Later FastAPI will return this response.
    """

    query_path = state["query_path"]
    rag_result = state.get("rag_result", {})
    sql_result = state.get("sql_result", {})

    if query_path == "rag":
        top_chunks = rag_result.get("results", [])

        if not top_chunks:
            answer = rag_result.get("message", "No relevant documents found.")
        else:
            best_chunk = top_chunks[0]
            answer = best_chunk.get("content", "")

        final_response = {
            "query": state["query"],
            "query_path": "rag",
            "answer": answer,
            "citations": [
                {
                    "document_name": chunk.get("document_name"),
                    "page_number": chunk.get("page_number"),
                    "chunk_type": chunk.get("chunk_type"),
                    "matched_by": chunk.get("matched_by"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in top_chunks
            ],
            "retry_count": rag_result.get("retry_count", 0),
            "sql_query": None,
            "sql_result": None,
            "confidence_score": top_chunks[0].get("rerank_score") if top_chunks else None,
        }

    elif query_path == "sql":
        final_response = {
            "query": state["query"],
            "query_path": "sql",
            "answer": sql_result.get("answer"),
            "citations": [],
            "retry_count": 0,
            "sql_query": sql_result.get("sql_query"),
            "sql_result": sql_result.get("sql_result"),
            "confidence_score": None,
        }

    else:
        top_chunks = rag_result.get("results", [])

        rag_answer = top_chunks[0].get("content", "") if top_chunks else rag_result.get(
            "message", "No relevant documents found."
        )

        final_response = {
            "query": state["query"],
            "query_path": "hybrid",
            "answer": {
                "sql_answer": sql_result.get("answer"),
                "document_answer": rag_answer,
            },
            "citations": [
                {
                    "document_name": chunk.get("document_name"),
                    "page_number": chunk.get("page_number"),
                    "chunk_type": chunk.get("chunk_type"),
                    "matched_by": chunk.get("matched_by"),
                    "rerank_score": chunk.get("rerank_score"),
                }
                for chunk in top_chunks
            ],
            "retry_count": rag_result.get("retry_count", 0),
            "sql_query": sql_result.get("sql_query"),
            "sql_result": sql_result.get("sql_result"),
            "confidence_score": top_chunks[0].get("rerank_score") if top_chunks else None,
        }

    print("[response_generator_node] Final response generated")

    return {
        **state,
        "final_response": final_response,
    }


def route_from_classifier(state: AgentState) -> str:
    """
    Conditional edge router.
    """

    return state["query_path"]


def build_agent():
    """
    Build LangGraph workflow.
    """

    graph = StateGraph(AgentState)

    graph.add_node("query_classifier", query_classifier_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("hybrid_node", hybrid_node)
    graph.add_node("response_generator", response_generator_node)

    graph.set_entry_point("query_classifier")

    graph.add_conditional_edges(
        "query_classifier",
        route_from_classifier,
        {
            "rag": "rag_node",
            "sql": "sql_node",
            "hybrid": "hybrid_node",
        },
    )

    graph.add_edge("rag_node", "response_generator")
    graph.add_edge("sql_node", "response_generator")
    graph.add_edge("hybrid_node", "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()


smart_banking_agent = build_agent()


def run_smart_banking_agent(query: str) -> dict[str, Any]:
    """
    Public function to call from FastAPI later.
    """

    initial_state: AgentState = {
        "query": query,
        "query_path": "",
        "rag_result": {},
        "sql_result": {},
        "final_response": {},
    }

    final_state = smart_banking_agent.invoke(initial_state)

    return final_state["final_response"]


if __name__ == "__main__":
    test_questions = [
        "What are the foreclosure charges for fixed rate home loans before 2022?",
        "Show me all active FDs for account 1345367",
        "Show international transactions on credit card CC-881001 and explain international transaction fees",
    ]

    for question in test_questions:
        print("\n" + "=" * 100)
        print(f"Question: {question}")
        print("=" * 100)

        response = run_smart_banking_agent(question)

        print("\nFinal Response:")
        print(response)