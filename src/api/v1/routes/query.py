from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.v1.services.query_service import query_documents, query_documents_stream
from src.api.v1.schemas.query_schema import QueryRequest
from src.core.guardrails import GuardrailViolation

router = APIRouter()

@router.post("/query")
def query_endpoint(request: QueryRequest):
    """Standard query endpoint with Guardrails exception handling."""
    try:
        return query_documents(request.query)
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )


@router.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    """Endpoint returning streaming responses with Guardrails."""
    generator = await query_documents_stream(request.query)
    return StreamingResponse(generator, media_type="text/event-stream")