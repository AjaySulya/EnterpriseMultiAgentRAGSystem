"""
Chat Service — persists chat messages and retrieves session history.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import ChatHistory
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChatService:
    """Handles chat_history DB operations."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save_message(
        self,
        session_id: str,
        user_query: str,
        response: str,
        agent_used: str,
        user_id: str | None = None,
    ) -> ChatHistory:
        """
        Persist a Q&A exchange.

        Args:
            session_id:  Conversation session identifier.
            user_query:  The user's question.
            response:    The system's answer.
            agent_used:  Which agent produced the answer.
            user_id:     Optional authenticated user ID.

        Returns:
            Persisted ChatHistory ORM object.
        """
        entry = ChatHistory(
            session_id=session_id,
            user_id=user_id,
            user_query=user_query,
            response=response,
            agent_used=agent_used,
        )
        self._db.add(entry)
        await self._db.flush()
        logger.debug(
            "Chat message saved",
            session_id=session_id,
            agent=agent_used,
        )
        return entry

    async def get_session_history(
        self, session_id: str, limit: int = 20
    ) -> list[ChatHistory]:
        """Return recent messages for a session, oldest first."""
        result = await self._db.execute(
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.timestamp.asc())
            .limit(limit)
        )
        return list(result.scalars().all())