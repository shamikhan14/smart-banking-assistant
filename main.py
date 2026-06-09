from fastapi import FastAPI

from src.api.v1.routes.query import router as query_router

app = FastAPI(
    title="Smart Banking Assistant API",
    version="1.0.0",
    description="Agentic RAG + NL2SQL API for Smart Banking Assistant",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Smart Banking Assistant API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


app.include_router(
    query_router,
    prefix="/api/v1",
    tags=["Smart Banking Query"],
)
