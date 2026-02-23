"""
LangGraph store adapter for job results and user preferences.

- Jobs: namespace ("jobs", user_id); keys "0".."N-1" (job index), "meta" (count, updated_at).
  Value per job: {title, company, description_snippet, ...}. Indexed for embedding search.
- User preferences: namespace ("user_preferences", user_id), key "summary".
  Value: {summary_text, updated_at}. Full update after classify_job_detail (no clarify).
"""

from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.store.base import BaseStore, PutOp

# Max jobs to keep per user per full update
MAX_JOBS_PER_USER = 500

JOBS_NAMESPACE_PREFIX = "jobs"
JOBS_META_KEY = "meta"

PREF_NAMESPACE_PREFIX = "user_preferences"
PREF_SUMMARY_KEY = "summary"

# User memory for chat RAG: multiple entries per user, keyed by index, value has "content" for embedding
USER_MEMORY_NAMESPACE_PREFIX = "user_memory"
USER_MEMORY_META_KEY = "meta"
MAX_USER_MEMORY_ENTRIES = 200


def _jobs_namespace(user_id: str) -> tuple[str, ...]:
    """Namespace for a user's job list: ("jobs", user_id)."""
    return (JOBS_NAMESPACE_PREFIX, user_id)


def _pref_namespace(user_id: str) -> tuple[str, ...]:
    """Namespace for a user's job preference summary: ("user_preferences", user_id)."""
    return (PREF_NAMESPACE_PREFIX, user_id)


def _user_memory_namespace(user_id: str) -> tuple[str, ...]:
    """Namespace for a user's chat memory entries: ("user_memory", user_id)."""
    return (USER_MEMORY_NAMESPACE_PREFIX, user_id)


def save_user_preference_sync(store: BaseStore, user_id: str, summary_text: str) -> None:
    """
    Full update: write user job preference summary (e.g. verification from classify_job_detail).
    Call after classify_job_detail when need_clarify is false, before mcp_jobs_tool_call.
    """
    if not user_id or not summary_text:
        return
    now = datetime.now(timezone.utc).isoformat()
    store.put(_pref_namespace(user_id), PREF_SUMMARY_KEY, {"summary_text": summary_text, "updated_at": now})


def get_user_preference(store: BaseStore, user_id: str) -> Optional[str]:
    """Read user job preference summary for job_brief / prompt injection; None if not set."""
    if not user_id:
        return None
    item = store.get(_pref_namespace(user_id), PREF_SUMMARY_KEY)
    v = getattr(item, "value", None) if item else None
    if not isinstance(v, dict):
        return None
    return (v.get("summary_text") or "").strip() or None


def save_user_memory_sync(store: BaseStore, user_id: str, content: str) -> None:
    """
    Append one memory entry for chat RAG (work preferences, etc.).
    Uses namespace (user_memory, user_id); keys "0".."N-1" (round-robin overwrite when full).
    """
    if not user_id or not (content or "").strip():
        return
    ns = _user_memory_namespace(user_id)
    meta_item = store.get(ns, USER_MEMORY_META_KEY)
    meta = meta_item.value if meta_item and isinstance(meta_item.value, dict) else {}
    next_index = int(meta.get("next_index", 0))
    key = str(next_index % MAX_USER_MEMORY_ENTRIES)
    now = datetime.now(timezone.utc).isoformat()
    store.put(ns, key, {"content": content.strip(), "updated_at": now})
    store.put(
        ns,
        USER_MEMORY_META_KEY,
        {"next_index": next_index + 1, "updated_at": now},
    )


def get_user_memories_by_search(
    store: BaseStore, user_id: str, query: str, limit: int = 5
) -> list[str]:
    """
    Semantic search over user memory for RAG. Returns list of content strings.
    """
    if not user_id:
        return []
    q = (query or "").strip() or "工作偏好 求职偏好"
    results = store.search(_user_memory_namespace(user_id), query=q, limit=limit * 2)
    out = []
    seen = set()
    for r in results:
        if not (hasattr(r, "value") and isinstance(r.value, dict)):
            continue
        content = (r.value.get("content") or "").strip()
        if content and content not in seen:
            seen.add(content)
            out.append(content)
            if len(out) >= limit:
                break
    return out


def _job_value_for_store(job: dict[str, Any], index: int) -> dict[str, Any]:
    """Single job dict for store (string keys, JSON-serializable). Include index for ordering."""
    return {
        "title": str(job.get("title") or ""),
        "company": str(job.get("company") or ""),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "link": str(job.get("link") or ""),
        "description_snippet": str(job.get("description_snippet") or "")[:8000],
        "location": str(job.get("location") or ""),
        "index": index,
    }


def _search_item_to_job_dict(item: Any) -> dict[str, Any]:
    """Extract prompt-facing job dict from SearchItem.value (drop index if needed for display)."""
    v = item.value if hasattr(item, "value") else item
    if not isinstance(v, dict):
        return {}
    return {
        "title": v.get("title") or "",
        "company": v.get("company") or "",
        "salary_min": v.get("salary_min"),
        "salary_max": v.get("salary_max"),
        "link": v.get("link") or "",
        "description_snippet": v.get("description_snippet") or "",
        "location": v.get("location") or "",
    }


def get_job_count(store: BaseStore, user_id: str) -> int:
    """Return number of jobs stored for user_id; 0 if none. Uses sync store.get."""
    if not user_id:
        return 0
    item = store.get(_jobs_namespace(user_id), JOBS_META_KEY)
    if not item or not isinstance(item.value, dict):
        return 0
    return int(item.value.get("count", 0))


def get_jobs_for_prompt(
    store: BaseStore,
    user_id: str,
    limit: int = 50,
    salary_min_filter: Optional[float] = None,
    salary_max_filter: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Read jobs for user_id for injection into prompt (no semantic query).
    Uses store.search with namespace_prefix; sorts by index, applies salary filter.
    """
    if not user_id:
        return []
    results = store.search(
        _jobs_namespace(user_id),
        limit=limit + 200,
    )
    items = [
        r
        for r in results
        if hasattr(r, "value") and isinstance(r.value, dict) and isinstance(r.value.get("index"), int)
    ]
    items.sort(key=lambda r: r.value.get("index", 0))
    out = []
    for r in items:
        v = r.value
        if salary_min_filter is not None:
            smax = v.get("salary_max")
            if smax is not None and smax < salary_min_filter:
                continue
        if salary_max_filter is not None:
            smin = v.get("salary_min")
            if smin is not None and smin > salary_max_filter:
                continue
        out.append(_search_item_to_job_dict(r))
        if len(out) >= limit:
            break
    return out


def get_jobs_by_search(
    store: BaseStore,
    user_id: str,
    query: str,
    limit: int = 20,
    salary_min_filter: Optional[float] = None,
    salary_max_filter: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over jobs for RAG: embed query, return top-k relevant jobs.
    Uses store.search with query=query. Optional salary filter applied in memory.
    """
    if not user_id or not (query or "").strip():
        return []
    results = store.search(
        _jobs_namespace(user_id),
        query=query.strip(),
        limit=limit * 2,
    )
    out = []
    for r in results:
        if not (hasattr(r, "value") and isinstance(r.value, dict)):
            continue
        v = r.value
        if salary_min_filter is not None and v.get("salary_max") is not None and v["salary_max"] < salary_min_filter:
            continue
        if salary_max_filter is not None and v.get("salary_min") is not None and v["salary_min"] > salary_max_filter:
            continue
        out.append(_search_item_to_job_dict(r))
        if len(out) >= limit:
            break
    return out


def _build_save_ops(user_id: str, jobs: list[dict[str, Any]]) -> list[PutOp]:
    """Build batch ops: delete old keys 0..MAX-1 and meta, then put each job and meta."""
    ns = _jobs_namespace(user_id)
    now = datetime.now(timezone.utc).isoformat()
    jobs = jobs[:MAX_JOBS_PER_USER]
    ops: list[PutOp] = []
    for i in range(MAX_JOBS_PER_USER):
        ops.append(PutOp(ns, str(i), None))
    ops.append(PutOp(ns, JOBS_META_KEY, None))
    for i, j in enumerate(jobs):
        ops.append(PutOp(ns, str(i), _job_value_for_store(j, i)))
    ops.append(PutOp(ns, JOBS_META_KEY, {"updated_at": now, "count": len(jobs)}))
    return ops


def save_job_results_sync(store: BaseStore, user_id: str, jobs: list[dict[str, Any]]) -> None:
    """
    Full update: delete old job keys, put one item per job (for embedding index) + meta.
    Uses sync store.batch.
    """
    if not user_id:
        return
    ops = _build_save_ops(user_id, jobs)
    store.batch(ops)


async def save_job_results_async(
    store: BaseStore, user_id: str, jobs: list[dict[str, Any]]
) -> None:
    """
    Full update: delete old job keys, put one item per job (for embedding index) + meta.
    Uses async store.abatch.
    """
    if not user_id:
        return
    ops = _build_save_ops(user_id, jobs)
    await store.abatch(ops)
