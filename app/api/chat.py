"""
Chat API — the primary question-answering endpoint.

POST /api/v1/chat  →  Router Agent  →  PDF | SQL | Web agent  →  answer
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.router_agent import run_rag_pipeline
from app.database.db import get_db
from app.database.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question to the multi-agent RAG system",
)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Route a natural language question to the correct specialist agent:

    - **PDF Agent** — questions about uploaded documents
    - **SQL Agent** — questions about database records / statistics
    - **Web Agent** — questions about a URL / website content

    The routing is automatic: the Router Agent classifies the query
    and dispatches it.  The answer, the agent used, and source
    references are returned in the response.

    Conversation history is persisted per `session_id`.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query must not be empty.",
        )

    logger.info(
        "Chat request received",
        session_id=payload.session_id,
        query=payload.query[:80],
    )

    try:
        result = await run_rag_pipeline(
            query=payload.query,
            session_id=payload.session_id,
            db=db,
        )
    except Exception as exc:
        logger.exception("RAG pipeline error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    # Persist to chat_history
    chat_svc = ChatService(db)
    await chat_svc.save_message(
        session_id=payload.session_id,
        user_query=payload.query,
        response=result["answer"],
        agent_used=result.get("agent_used", "unknown"),
        user_id=payload.user_id,
    )

    return ChatResponse(
        session_id=payload.session_id,
        query=payload.query,
        answer=result["answer"],
        agent_used=result.get("agent_used", "unknown"),
        sources=result.get("sources", []),
    )


@router.get(
    "/chat/history/{session_id}",
    summary="Retrieve chat history for a session",
)
async def get_chat_history(
    session_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Return the last `limit` messages for a given session, oldest first.
    """
    chat_svc = ChatService(db)
    history = await chat_svc.get_session_history(session_id, limit=limit)
    return {
        "session_id": session_id,
        "total": len(history),
        "messages": [
            {
                "id": h.id,
                "user_query": h.user_query,
                "response": h.response,
                "agent_used": h.agent_used,
                "timestamp": h.timestamp.isoformat(),
            }
            for h in history
        ],
    }