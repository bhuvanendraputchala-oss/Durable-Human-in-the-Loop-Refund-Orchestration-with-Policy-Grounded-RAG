from __future__ import annotations

import atexit
import os
from typing import Optional

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

_pool: Optional[ConnectionPool] = None
_checkpointer: Optional[PostgresSaver] = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        uri = os.environ.get("POSTGRES_URI")
        if not uri:
            raise RuntimeError(
                "POSTGRES_URI is not set. "
            )
        _pool = ConnectionPool(conninfo=uri, max_size=10, open=True)
        atexit.register(_pool.close)
    return _pool


def get_checkpointer() -> PostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        pool = _get_pool()
        _checkpointer = PostgresSaver(pool)
        _checkpointer.setup()
    return _checkpointer
