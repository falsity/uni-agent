"""
Env-driven configuration. Single source for DB, API base URLs, model names, and limits.
"""

import os

# API & DB
CHAT_BASE_URL = os.environ.get(
    "OPENAI_API_BASE",
    "http://192.168.0.201:9000/v1",
)
EMBED_BASE_URL = os.environ.get(
    "EMBED_BASE_URL",
    "http://192.168.0.201:9001/v1",
)
DB_URI = os.environ.get(
    "POSTGRES_URI",
    "postgresql://postgres:password@192.168.0.201:15433/postgres?sslmode=disable",
)

# LLM & embed
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-8B")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Limits
STORE_EMBEDDING_DIMS = 1024
JOB_PROMPT_MAX_CHARS = 28_000
MODEL_MAX_CONTEXT = int(os.environ.get("MODEL_MAX_CONTEXT", "40960"))
SAFE_MESSAGE_TOKENS = min(12_000, MODEL_MAX_CONTEXT // 3)
TOOL_RESULT_MAX_CHARS = 6_000

# Optional
TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "180"))


def get_store_index_config() -> dict:
    """PostgresStore index config for job vector search (embedding over job fields)."""
    from langchain_openai import OpenAIEmbeddings
    return {
        "dims": STORE_EMBEDDING_DIMS,
        "embed": OpenAIEmbeddings(
            model=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            api_key=OPENAI_API_KEY,
        ),
        "fields": ["title", "description_snippet", "company", "location"],
    }
