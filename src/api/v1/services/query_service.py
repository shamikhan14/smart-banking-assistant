from typing import Any
from src.api.v1.agents.agent import (
    run_smart_banking_agent,
    run_smart_banking_agent_stream,
)


def query_smart_banking_assistant(query: str) -> dict[str, Any]:
    return run_smart_banking_agent(query)


async def query_smart_banking_assistant_stream(query: str):
    return run_smart_banking_agent_stream(query)
