
import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
from src.api.v1.services.query_service import query_documents,query_documents_stream
from src.api.v1.schemas.query_schema import QueryRequest,QueryResponse

router = APIRouter()
@router.post("/query")
def query_endpoint(request: QueryRequest):
   docs = query_documents(request.query)
   return docs

@router.post("/query/stream")
async def stream_query_endpoint(request:QueryRequest):
   """Endpoint that returns SSE stream of agent's response"""
   generator=await query_documents_stream(request.query)
   return StreamingResponse(
      generator,
      media_type="text/event-stream"
   )
   