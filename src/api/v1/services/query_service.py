# query_service.py  ← replace the whole file

from src.api.v1.agents.agents import run_search_agent, run_search_agent_stream
from src.core.guardrails import guard_input, guard_output, GuardrailViolation


def query_documents(query: str, chat_history: list = None):
    guard_input(query)
    result = run_search_agent(query, chat_history=chat_history or [])
    if isinstance(result, dict):
        if result.get("answer"):
            result["answer"] = guard_output(result["answer"])
        if result.get("query"):
            result["query"] = guard_output(result["query"])
    return result


async def query_documents_stream(query: str, chat_history: list = None):
    """
    guard_input  → raises GuardrailViolation before any SSE is sent (caught by route → HTTP 400).
    guard_output → runs on the FULL assembled answer; re-streams the clean version word-by-word.
    guardrail_error SSE event emitted if output guard itself raises.
    """
    guard_input(query)   # ← raises here → HTTP 400, nothing streamed

    async def _guarded_stream():
        import json, asyncio

        raw_tokens: list[str] = []
        meta = {"policy_citations": "", "page_no": "", "document_name": "", "sql_query_executed": None}

        # 1. Collect all tokens from the agent
        async for sse_line in run_search_agent_stream(query, chat_history=chat_history or []):
            if not sse_line.startswith("data:"):
                continue
            payload_str = sse_line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if "token" in payload:
                raw_tokens.append(payload["token"])
            else:
                for field in ("policy_citations", "page_no", "document_name", "sql_query_executed"):
                    if field in payload:
                        meta[field] = payload[field]

        # 2. Output guardrail on full answer
        full_answer = "".join(raw_tokens)
        try:
            clean_answer = guard_output(full_answer)
        except Exception:
            yield 'data:' + json.dumps({
                "guardrail_error": True,
                "message": "The response was blocked by the output safety filter."
            }) + "\n\n"
            yield "data:[DONE]\n\n"
            return

        # 3. Re-stream the clean answer
        words = clean_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data:{json.dumps({'token': chunk})}\n\n"
            await asyncio.sleep(0.018)

        yield f"data:{json.dumps(meta)}\n\n"
        yield "data:[DONE]\n\n"

    return _guarded_stream()