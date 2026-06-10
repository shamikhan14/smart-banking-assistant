from src.api.v1.agents.agents import run_search_agent, run_search_agent_stream
from src.api.v1.schemas.query_schema import ChatMessage
from typing import List


def query_documents(query: str, chat_history: List[ChatMessage] = None):
    history = [{"role": m.role, "content": m.content} for m in (chat_history or [])]
    return run_search_agent(query, chat_history=history)


async def query_documents_stream(query: str, chat_history: List[ChatMessage] = None):
    history = [{"role": m.role, "content": m.content} for m in (chat_history or [])]
    return run_search_agent_stream(query, chat_history=history)