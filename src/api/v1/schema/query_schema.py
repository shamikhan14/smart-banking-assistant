from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        example="What are the foreclosure charges for fixed rate home loans before 2022?",
    )


class QueryResponse(BaseModel):
    query: str
    query_path: str
    answer: Any
    citations: list[dict[str, Any]]
    retry_count: int
    sql_query: str | None = None
    sql_result: Any | None = None
    confidence_score: float | None = None
