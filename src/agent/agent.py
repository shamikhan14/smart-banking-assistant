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
    route: Literal["chitchat", "rag", "sql", "hybrid", "out_of_scope"]
    reason: str


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def query_classifier_node(state: AgentState) -> AgentState:
    """
    Classify user query into:
    - chitchat
    - rag
    - sql
    - hybrid
    - out_of_scope
    """

    query = state["query"]
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = f"""
You are a query classifier for a Smart Banking Assistant.

Classify the user query into exactly one route.

Routes:

1. chitchat
Use this only for light casual conversation such as:
- greetings
- thanks
- asking how the assistant is
Examples:
- hi
- hello
- how are you
- thanks

Important:
Do NOT use chitchat for math, coding, weather, news, general knowledge, jokes, or unrelated questions.

2. rag
Use this when the question asks about banking product documents, policies, fees,
charges, eligibility, interest rates, terms and conditions, disclosures, or rules.

Examples:
- What are foreclosure charges before 2022?
- What is the FD interest rate for 444 days?
- What are credit card international transaction fees?
- What is the premature withdrawal policy?

3. sql
Use this when the question asks about structured customer/account data from database tables:
accounts, transactions, loan_accounts, fixed_deposits, credit_cards, card_transactions.

Examples:
- Give me last 3 months purchase history of account 1345367.
- What is outstanding balance for loan L-789012?
- Show active FDs for account 1345367.
- Show international transactions on card CC-881001.
- List transactions above 50000 for account 1345367.

4. hybrid
Use this when the question needs both:
- SQL data from customer/account tables
- RAG explanation from product/policy documents

Examples:
- Show international transactions on CC-881001 and explain international transaction fees.
- Show active FDs for account 1345367 and explain premature withdrawal policy.
- Show loan L-789012 details and explain foreclosure charges.

5. out_of_scope
Use this for anything outside Smart Banking Assistant scope, including:
- math questions like 1+1
- coding questions
- weather/news/general knowledge
- jokes
- personal advice unrelated to banking
- random questions not related to banking

User query:
{query}

Return only the structured route and reason.
"""

    try:
        decision = structured_llm.invoke(prompt)
        route = decision.route
        reason = decision.reason
    except Exception as exc:
        print(f"[query_classifier_node] Classifier failed: {exc}")
        route = "out_of_scope"
        reason = "Classifier failed, defaulting to out_of_scope."

    print(f"[query_classifier_node] Route: {route}")
    print(f"[query_classifier_node] Reason: {reason}")

    return {
        **state,
        "query_path": route,
    }


def chitchat_node(state: AgentState) -> AgentState:
    """
    Let LLM respond naturally to light casual conversation.
    This is not hardcoded.
    """

    llm = get_llm()

    prompt = f"""
You are a friendly Smart Banking Assistant.

The user is making light casual conversation.

Reply naturally and briefly.
Do not answer math, coding, weather, news, jokes, or unrelated factual questions.
Gently guide the user to ask about banking products, accounts, transactions, loans, FDs, or credit cards.

User message:
{state["query"]}
"""

    response = llm.invoke(prompt)
    answer = response.content

    if isinstance(answer, list):
        answer = " ".join(str(item) for item in answer)

    return {
        **state,
        "final_response": {
            "query": state["query"],
            "query_path": "chitchat",
            "answer": str(answer),
            "citations": [],
            "retry_count": 0,
            "sql_query": None,
            "sql_result": None,
            "confidence_score": None,
        },
    }


def out_of_scope_node(state: AgentState) -> AgentState:
    """
    Reject non-banking questions like math, coding, weather, jokes, etc.
    """

    return {
        **state,
        "final_response": {
            "query": state["query"],
            "query_path": "out_of_scope",
            "answer": (
                "I am designed to help with banking-related questions such as "
                "product policies, account transactions, loans, fixed deposits, "
                "and credit cards. Please ask me a banking-related question."
            ),
            "citations": [],
            "retry_count": 0,
            "sql_query": None,
            "sql_result": None,
            "confidence_score": None,
        },
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
    Run SQL / NL2SQL path.
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
    FastAPI and Streamlit return this response.
    """

    query_path = state["query_path"]
    rag_result = state.get("rag_result", {})
    sql_result = state.get("sql_result", {})

    if query_path == "rag":
        top_chunks = rag_result.get("results", [])

        if not top_chunks:
            answer = rag_result.get("message", "No relevant documents found.")
        else:
            answer = top_chunks[0].get("content", "")

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
            "confidence_score": (
                top_chunks[0].get("rerank_score") if top_chunks else None
            ),
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

    elif query_path == "hybrid":
        top_chunks = rag_result.get("results", [])

        document_answer = (
            top_chunks[0].get("content", "")
            if top_chunks
            else rag_result.get("message", "No relevant documents found.")
        )

        final_response = {
            "query": state["query"],
            "query_path": "hybrid",
            "answer": {
                "sql_answer": sql_result.get("answer"),
                "document_answer": document_answer,
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
            "confidence_score": (
                top_chunks[0].get("rerank_score") if top_chunks else None
            ),
        }

    else:
        final_response = {
            "query": state["query"],
            "query_path": "out_of_scope",
            "answer": (
                "I am designed to help with banking-related questions such as "
                "product policies, account transactions, loans, fixed deposits, "
                "and credit cards. Please ask me a banking-related question."
            ),
            "citations": [],
            "retry_count": 0,
            "sql_query": None,
            "sql_result": None,
            "confidence_score": None,
        }

    print("[response_generator_node] Final response generated")

    return {
        **state,
        "final_response": final_response,
    }


def route_from_classifier(state: AgentState) -> str:
    return state["query_path"]


def build_agent():
    """
    Build LangGraph workflow.
    """

    graph = StateGraph(AgentState)

    graph.add_node("query_classifier", query_classifier_node)
    graph.add_node("chitchat_node", chitchat_node)
    graph.add_node("out_of_scope_node", out_of_scope_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("hybrid_node", hybrid_node)
    graph.add_node("response_generator", response_generator_node)

    graph.set_entry_point("query_classifier")

    graph.add_conditional_edges(
        "query_classifier",
        route_from_classifier,
        {
            "chitchat": "chitchat_node",
            "out_of_scope": "out_of_scope_node",
            "rag": "rag_node",
            "sql": "sql_node",
            "hybrid": "hybrid_node",
        },
    )

    graph.add_edge("chitchat_node", END)
    graph.add_edge("out_of_scope_node", END)

    graph.add_edge("rag_node", "response_generator")
    graph.add_edge("sql_node", "response_generator")
    graph.add_edge("hybrid_node", "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()


smart_banking_agent = build_agent()


def run_smart_banking_agent(query: str) -> dict[str, Any]:
    """
    Public function called by FastAPI and Streamlit.
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


def save_graph_image() -> None:
    """
    Save LangGraph workflow diagram as PNG.

    Output:
        docs/langgraph_agent.png
    """

    output_path = "docs/langgraph_agent.png"

    try:
        from pathlib import Path

        Path("docs").mkdir(parents=True, exist_ok=True)

        png_bytes = smart_banking_agent.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as file:
            file.write(png_bytes)

        print(f"[save_graph_image] Graph saved to: {output_path}")

    except Exception as exc:
        print(f"[save_graph_image] Could not save graph image: {exc}")


if __name__ == "__main__":
    save_graph_image()

    test_questions = [
        "hi",
        "how are you",
        "1+1",
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
