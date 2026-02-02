# Job parsing (job_store) and langgraph.store adapter (store_adapter) for long-term job results.

from uni_agent.store.job_store import MAX_JOBS_PER_USER, parse_mcp_result_to_jobs
from uni_agent.store.store_adapter import (
    get_job_count,
    get_jobs_by_search,
    get_jobs_for_prompt,
    get_user_preference,
    save_job_results_async,
    save_job_results_sync,
    save_user_preference_sync,
)

__all__ = [
    "parse_mcp_result_to_jobs",
    "MAX_JOBS_PER_USER",
    "get_job_count",
    "get_jobs_for_prompt",
    "get_jobs_by_search",
    "get_user_preference",
    "save_job_results_sync",
    "save_job_results_async",
    "save_user_preference_sync",
]
