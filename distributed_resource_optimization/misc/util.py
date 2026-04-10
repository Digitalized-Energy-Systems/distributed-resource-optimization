"""Logging and task utilities."""

from __future__ import annotations

import asyncio
import logging
import traceback

logger = logging.getLogger(__name__)


def log_exception(exc: BaseException, tb: str | None = None) -> None:
    """Log an exception with its traceback."""
    if tb is None:
        tb = traceback.format_exc()
    logger.error("Exception occurred:\n%s\n%s", exc, tb)


async def spawn_logged(coro) -> asyncio.Task:
    """Schedule a coroutine as a task; log any exceptions it raises."""
    async def _wrapped():
        try:
            return await coro
        except Exception as exc:
            log_exception(exc)
            raise

    return asyncio.create_task(_wrapped())
