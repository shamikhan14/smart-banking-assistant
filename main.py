from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.agent import run_smart_banking_agent

app = FastAPI(
    title="Smart Banking Assistant API",
    version="1.0.0",
    description="Agentic RAG + NL2SQL API for Smart Banking Assistant",
)


class QueryRequest(BaseModel):

    query: str


class QueryResponse(BaseModel):
    query: str
    query_path: str
    answer: Any
    citations: list[dict[str, Any]]
    retry_count: int
    sql_query: str | None = None
    sql_result: Any | None = None
    confidence_score: float | None = None


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Smart Banking Assistant API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/v1/query", response_model=QueryResponse)
def query_assistant(request: QueryRequest) -> dict[str, Any]:
    """
    Main query endpoint.

    It sends the user query to LangGraph agent.
    The agent decides:
    - rag
    - sql
    - hybrid
    """

    response = run_smart_banking_agent(request.query)
    return response
