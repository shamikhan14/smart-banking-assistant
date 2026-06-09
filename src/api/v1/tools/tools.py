from typing import Any, Literal, TypedDict

from pydantic import BaseModel


class AgentState(TypedDict):
    query: str
    query_path: str
    rag_result: dict[str, Any]
    sql_result: dict[str, Any]
    final_response: dict[str, Any]


class RouteDecision(BaseModel):
    route: Literal["chitchat", "rag", "sql", "hybrid", "out_of_scope"]
    reason: str
