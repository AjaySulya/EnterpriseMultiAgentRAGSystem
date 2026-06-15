"""
SQL Agent — converts natural language questions to safe SQL and executes them
against the PostgreSQL database.
"""

import re
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

HF_API_URL = (
    f"https://api-inference.huggingface.co/models/{settings.HF_LLM_MODEL}"
)

# Schema passed to the LLM as context
DB_SCHEMA = """
Tables:
  users(id TEXT, username TEXT, email TEXT, created_at TIMESTAMPTZ)
  documents(id TEXT, filename TEXT, original_filename TEXT, upload_date TIMESTAMPTZ,
            status TEXT, chunk_count INT, error_message TEXT)
  chat_history(id TEXT, session_id TEXT, user_id TEXT, user_query TEXT,
               response TEXT, agent_used TEXT, timestamp TIMESTAMPTZ)
""".strip()

# Allowlist of safe SQL patterns (SELECT only, no DDL / DML)
_SAFE_PATTERN = re.compile(
    r"^\s*SELECT\b",
    re.IGNORECASE,
)
_DANGEROUS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|INSERT|UPDATE|ALTER|CREATE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class SQLAgent:
    """
    Translates NL questions to SQL via LLM, executes them safely,
    and returns human-readable results.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def answer(self, query: str) -> dict[str, Any]:
        """
        Process a natural language database question.

        Args:
            query: User's question about the database.

        Returns:
            Dict with 'answer', 'sql', 'rows', and 'agent'.
        """
        logger.info("SQLAgent processing query", query=query[:80])

        sql = await self._generate_sql(query)
        logger.info("Generated SQL", sql=sql)

        if not self._is_safe(sql):
            return {
                "answer": "I can only run SELECT queries for safety reasons.",
                "sql": sql,
                "rows": [],
                "agent": "sql_agent",
            }

        rows, error = await self._execute_sql(sql)
        if error:
            return {
                "answer": f"SQL execution error: {error}",
                "sql": sql,
                "rows": [],
                "agent": "sql_agent",
            }

        answer = self._format_results(query, sql, rows)
        return {
            "answer": answer,
            "sql": sql,
            "rows": rows,
            "agent": "sql_agent",
        }

    async def _generate_sql(self, query: str) -> str:
        """Ask the LLM to produce a SQL query."""
        prompt = (
            "You are a SQL expert. Given the schema below, write a single "
            "PostgreSQL SELECT statement that answers the question. "
            "Return ONLY the SQL — no explanation, no markdown.\n\n"
            f"Schema:\n{DB_SCHEMA}\n\n"
            f"Question: {query}\n\n"
            "SQL:"
        )

        headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "return_full_text": False,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(HF_API_URL, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            raw = ""
            if isinstance(data, list) and data:
                raw = data[0].get("generated_text", "")

            # Strip markdown fences if present
            sql = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip()
            sql = sql.split(";")[0].strip() + ";"  # take only first statement
            return sql

        except Exception as exc:
            logger.warning("LLM SQL generation failed, using fallback", error=str(exc))
            return "SELECT count(*) FROM documents;"

    def _is_safe(self, sql: str) -> bool:
        """Return True only for safe SELECT-only queries."""
        if not _SAFE_PATTERN.match(sql):
            return False
        if _DANGEROUS.search(sql):
            return False
        return True

    async def _execute_sql(
        self, sql: str
    ) -> tuple[list[dict], str | None]:
        """Execute SQL and return rows + optional error."""
        try:
            result = await self._db.execute(text(sql))
            keys = list(result.keys())
            rows = [dict(zip(keys, row)) for row in result.fetchall()]
            return rows, None
        except Exception as exc:
            logger.error("SQL execution error", error=str(exc))
            return [], str(exc)

    @staticmethod
    def _format_results(query: str, sql: str, rows: list[dict]) -> str:
        """Format query results as a readable string."""
        if not rows:
            return "The query returned no results."

        lines = [f"Results for: {query}", f"SQL: {sql}", ""]
        for i, row in enumerate(rows[:20], 1):  # cap display at 20 rows
            lines.append(f"{i}. " + ", ".join(f"{k}={v}" for k, v in row.items()))

        if len(rows) > 20:
            lines.append(f"... and {len(rows) - 20} more rows.")

        return "\n".join(lines)