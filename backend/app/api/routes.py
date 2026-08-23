from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.rag_service import service
from app.schemas import ChatRequest, ChatResponse, GraphResponse, LastResultResponse, ResetResponse, ReadinessResponse

router = APIRouter(prefix="/api", tags=["rag"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/readiness", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    status = service.is_ready()
    return ReadinessResponse(**status)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = service.run_query(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc
    return ChatResponse(**result)


@router.get("/graph", response_model=GraphResponse)
def graph() -> GraphResponse:
    return GraphResponse(**service.get_graph_payload())


@router.get("/last-result", response_model=LastResultResponse)
def last_result() -> LastResultResponse:
    return LastResultResponse(data=service.get_last_result())


@router.post("/reset", response_model=ResetResponse)
def reset() -> ResetResponse:
    service.reset_chat()
    return ResetResponse(ok=True, message="Chat state cleared")
