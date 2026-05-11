from __future__ import annotations

import hashlib
import json
from typing import Any

import redis
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


def _cache_key(query: str, source: str) -> str:
    raw = f"{query.lower().strip()}|{source}"
    return "rag:" + hashlib.sha256(raw.encode()).hexdigest()


class QueryCache:
    def __init__(self) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl = settings.redis_ttl_seconds

    def get(self, query: str, source: str) -> dict[str, Any] | None:
        key = _cache_key(query, source)
        try:
            raw = self._redis.get(key)
            if raw:
                logger.debug("cache.hit", key=key[:16])
                return json.loads(raw)
        except Exception as exc:
            logger.warning("cache.get_error", error=str(exc))
        return None

    def set(self, query: str, source: str, value: dict[str, Any]) -> None:
        key = _cache_key(query, source)
        try:
            self._redis.setex(key, self._ttl, json.dumps(value))
            logger.debug("cache.set", key=key[:16])
        except Exception as exc:
            logger.warning("cache.set_error", error=str(exc))

    def health(self) -> bool:
        try:
            return self._redis.ping()
        except Exception:
            return False
