from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.v1.schema.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import (
    query_smart_banking_assistant,
    query_smart_banking_assistant_stream,
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    return query_smart_banking_assistant(request.query)


@router.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    generator = await query_smart_banking_assistant_stream(request.query)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
    )