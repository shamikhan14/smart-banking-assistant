"""from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.v1.services.query_service import query_documents, query_documents_stream
from src.api.v1.schemas.query_schema import QueryRequest
from src.core.guardrails import GuardrailViolation

router = APIRouter()

@router.post("/query")
def query_endpoint(request: QueryRequest):
    Standard query endpoint with Guardrails exception handling.
    try:
        return query_documents(request.query)
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )


@router.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    Endpoint returning streaming responses with Guardrails.
    try:
        generator = await query_documents_stream(request.query)
        return StreamingResponse(generator, media_type="text/event-stream")
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )
    """
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from src.api.v1.services.query_service import query_documents, query_documents_stream
from src.api.v1.schemas.query_schema import QueryRequest
from src.core.guardrails import GuardrailViolation
import shutil, pathlib, tempfile

router = APIRouter()

@router.post("/query")
def query_endpoint(request: QueryRequest):
    """Standard query endpoint with Guardrails exception handling."""
    try:
        return query_documents(request.query, chat_history=[m.model_dump() for m in request.chat_history])
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )


@router.post("/query/stream")
async def stream_query_endpoint(request: QueryRequest):
    """Endpoint returning streaming responses with Guardrails."""
    try:
        generator = await query_documents_stream(request.query, chat_history=[m.model_dump() for m in request.chat_history])
        return StreamingResponse(generator, media_type="text/event-stream")
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )
    

@router.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    """Accept a PDF upload and run the full ingestion pipeline on the backend."""
    suffix = pathlib.Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        from src.ingestion.ingestion import run_ingestion
        result = run_ingestion(tmp_path)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)