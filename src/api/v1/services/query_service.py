from src.api.v1.agents.agents import run_search_agent, run_search_agent_stream
from src.core.guardrails import guard_input, guard_output


def query_documents(query: str):
    """
    Process a single query with Guardrails protection.

    1. guard_input(query) checks toxic input.
    2. run_search_agent(query) runs LangGraph/RAG/NL2SQL.
    3. guard_output(answer) masks PII in answer.
    4. guard_output(query) masks PII in returned query field also.
    """

    # Input guardrail: toxic language check
    guard_input(query)

    # Run Smart Banking Agent
    result = run_search_agent(query)

    # Output guardrail: mask PII before sending response to UI/API
    if isinstance(result, dict):

        # Mask PII in answer
        if result.get("answer"):
            result["answer"] = guard_output(result["answer"])

        # Mask PII in returned query also
        if result.get("query"):
            result["query"] = guard_output(result["query"])

    return result


async def query_documents_stream(query: str):
    """
    Streaming version.

    Note:
    guard_input protects the input before streaming starts.
    Output masking for streaming is not applied here unless you also
    guard each streamed token/chunk inside run_search_agent_stream.
    """

    guard_input(query)
    return run_search_agent_stream(query)