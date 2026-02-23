"""
Env-driven configuration. Single source for DB, API base URLs, model names, and limits.
"""

import os

# API & DB
CHAT_BASE_URL = os.environ.get(
    "OPENAI_API_BASE",
    "http://192.168.0.201:8000/v1",
)
EMBED_BASE_URL = os.environ.get(
    "EMBED_BASE_URL",
    "http://192.168.0.201:8001/v1",
)
# Default: local postgres. When using server/docker-compose, port is often 15433 (not 5432).
DB_URI = os.environ.get(
    "POSTGRES_URI",
    "postgresql://postgres:password@127.0.0.1:15433/postgres?sslmode=disable",
)

# LLM & embed
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-8B")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DIMS = int(os.environ.get("EMBED_DIMS", "1024"))

# Limits
JOB_PROMPT_MAX_CHARS = 28_000
MODEL_MAX_CONTEXT = int(os.environ.get("MODEL_MAX_CONTEXT", "40960"))
SAFE_MESSAGE_TOKENS = min(12_000, MODEL_MAX_CONTEXT // 3)
TOOL_RESULT_MAX_CHARS = 6_000

# Optional
TIMEOUT_SECONDS = int(os.environ.get("AGENT_TIMEOUT_SECONDS", "180"))

# Mem0: cross-session user memory (set MEM0_DISABLED=1 to disable)
MEM0_DISABLED = os.environ.get("MEM0_DISABLED", "").strip().lower() in ("1", "true", "yes")
MEM0_SEARCH_LIMIT = int(os.environ.get("MEM0_SEARCH_LIMIT", "8"))

# Mem0 uses .env: EMBED_MODEL, EMBED_DIMS, OPENAI_MODEL. MEM0_LOCAL=1 (Ollama): MEM0_EMBED_MODEL, MEM0_LLM_MODEL.
MEM0_LOCAL = os.environ.get("MEM0_LOCAL", "").strip().lower() in ("1", "true", "yes")
MEM0_OLLAMA_BASE_URL = os.environ.get("MEM0_OLLAMA_BASE_URL", "http://localhost:11434")
MEM0_QDRANT_HOST = os.environ.get("MEM0_QDRANT_HOST", "localhost")
MEM0_QDRANT_PORT = int(os.environ.get("MEM0_QDRANT_PORT", "6333"))
MEM0_EMBED_MODEL = os.environ.get("MEM0_EMBED_MODEL") or EMBED_MODEL

# Mem0 vector store: "qdrant" (default) or "milvus"
MEM0_VECTOR_STORE = (os.environ.get("MEM0_VECTOR_STORE") or "qdrant").strip().lower()
MEM0_MILVUS_URI = os.environ.get("MEM0_MILVUS_URI", "http://localhost:19530")
MEM0_MILVUS_COLLECTION = os.environ.get("MEM0_MILVUS_COLLECTION", "mem0")
MEM0_MILVUS_TOKEN = os.environ.get("MEM0_MILVUS_TOKEN", "").strip() or None
MEM0_MILVUS_DB_NAME = os.environ.get("MEM0_MILVUS_DB_NAME", "").strip() or None


def _mem0_vector_store_config() -> dict:
    """Build Mem0 vector_store config for qdrant or milvus based on MEM0_VECTOR_STORE."""
    if MEM0_VECTOR_STORE == "milvus":
        cfg = {
            "provider": "milvus",
            "config": {
                "url": MEM0_MILVUS_URI,
                "collection_name": MEM0_MILVUS_COLLECTION,
                "embedding_model_dims": EMBED_DIMS,
            },
        }
        if MEM0_MILVUS_TOKEN:
            cfg["config"]["token"] = MEM0_MILVUS_TOKEN
        if MEM0_MILVUS_DB_NAME:
            cfg["config"]["db_name"] = MEM0_MILVUS_DB_NAME
        return cfg
    # default: qdrant
    return {
        "provider": "qdrant",
        "config": {
            "host": MEM0_QDRANT_HOST,
            "port": MEM0_QDRANT_PORT,
            "embedding_model_dims": EMBED_DIMS,
        },
    }


def get_mem0_config() -> dict:
    """
    Mem0 config: local (Ollama + Qdrant, no API key) or cloud (OpenAI-compatible embedder).
    - MEM0_LOCAL=1: full local stack, no key required.
    - Otherwise: use project embedder (EMBED_BASE_URL + EMBED_MODEL), requires api_key.
    """
    if MEM0_LOCAL:
        # Fully local: Ollama for embedder + LLM; vector store from MEM0_VECTOR_STORE (qdrant/milvus)
        return {
            "vector_store": _mem0_vector_store_config(),
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": os.environ.get("MEM0_LLM_MODEL") or OPENAI_MODEL,
                    "temperature": 0,
                    "max_tokens": 2000,
                    "ollama_base_url": MEM0_OLLAMA_BASE_URL.rstrip("/"),
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": MEM0_EMBED_MODEL,
                    "ollama_base_url": MEM0_OLLAMA_BASE_URL.rstrip("/"),
                },
            },
        }
    # Use .env: EMBED_MODEL + EMBED_BASE_URL (embedder), OPENAI_MODEL + CHAT_BASE_URL (llm).
    # Do not pass embedding_dims to embedder: vLLM rejects dimensions param for Qwen3-Embedding.
    # vector_store must use EMBED_DIMS so store dim matches embedder output (1024 for 0.6B).
    return {
        "vector_store": _mem0_vector_store_config(),
        "embedder": {
            "provider": "openai",
            "config": {
                "model": MEM0_EMBED_MODEL,
                "api_key": OPENAI_API_KEY,
                "openai_base_url": EMBED_BASE_URL.rstrip("/"),
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": OPENAI_MODEL,
                "api_key": OPENAI_API_KEY,
                "openai_base_url": CHAT_BASE_URL.rstrip("/"),
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        },
    }


def get_store_index_config() -> dict:
    """PostgresStore index config for job vector search (embedding over job fields)."""
    from langchain_openai import OpenAIEmbeddings
    return {
        "dims": EMBED_DIMS,
        "embed": OpenAIEmbeddings(
            model=EMBED_MODEL,
            base_url=EMBED_BASE_URL,
            api_key=OPENAI_API_KEY,
        ),
        # title/description_snippet/company/location for jobs; content for user_memory RAG
        "fields": ["title", "description_snippet", "company", "location", "content"],
    }
