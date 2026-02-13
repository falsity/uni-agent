"""
Parse MCP job search result into job dicts. Persistence is via langgraph.store (store_adapter).
"""

import ast
import json
import re
from typing import Any

from uni_agent.store.store_adapter import MAX_JOBS_PER_USER


def _extract_json_from_mcp_content_parts(mcp_raw: str) -> str | None:
    """
    If mcp_raw is Python repr of MCP content parts [{'type':'text','text':'...'}],
    extract the inner JSON string so we can parse jobs from it.
    """
    s = (mcp_raw or "").strip()
    if not s or not s.startswith("["):
        return None
    try:
        data = ast.literal_eval(s)
        if not isinstance(data, list):
            return None
        texts = []
        for item in data:
            if isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
        if texts:
            return "\n".join(texts)
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def _extract_jobs_json_by_brace_matching(s: str) -> str | None:
    """
    Find substring like "jobs": [...] or "jobs": [...] and extract the full JSON
    by bracket matching. Handles wrapped/escaped content (e.g. inside repr).
    """
    s = (s or "").strip()
    if not s or "jobs" not in s:
        return None
    # Find start of array after "jobs": or \"jobs\": (escaped in repr) or 'jobs':
    for marker in ('"jobs":', '"jobs" :', '\\"jobs\\":', "'jobs':", "'jobs' :"):
        i = s.find(marker)
        if i == -1:
            continue
        start = i + len(marker)
        # Skip whitespace
        while start < len(s) and s[start] in " \t\n\r":
            start += 1
        if start >= len(s):
            continue
        if s[start] == "[":
            # Find matching ]
            depth = 0
            for j in range(start, len(s)):
                if s[j] == "[":
                    depth += 1
                elif s[j] == "]":
                    depth -= 1
                    if depth == 0:
                        return "{\"jobs\":" + s[start : j + 1] + "}"
        elif s[start] == "{":
            depth = 0
            for j in range(start, len(s)):
                if s[j] in "{[":
                    depth += 1
                elif s[j] in "}]":
                    depth -= 1
                    if depth == 0:
                        return s[start : j + 1]
    return None


def _jobs_from_zhipin_items(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Extract job list from MCP zhipin format: { "items": [ { "data": { "jobInfo": [ {...}, ... ] } }, ... ] }.
    Returns flattened list of normalized job dicts, or None if structure not matched.
    """
    items = data.get("items")
    if not isinstance(items, list):
        return None
    jobs: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        data_block = item.get("data")
        if not isinstance(data_block, dict):
            continue
        job_info = data_block.get("jobInfo")
        if not isinstance(job_info, list):
            continue
        for j in job_info[:MAX_JOBS_PER_USER]:
            jobs.append(_normalize_job_item(j))
    return jobs if jobs else None


def parse_mcp_result_to_jobs(mcp_raw: str) -> list[dict[str, Any]]:
    """
    Parse MCP tool result string into list of job dicts (JobItem-like).
    Supports: (1) JSON with "jobs" or "items" (zhipin); (2) MCP content parts; (3) brace-match; (4) separator split.
    """
    if not (mcp_raw or "").strip():
        return []

    # Try JSON: "items" (zhipin format), array, or {"jobs": [...]}
    try:
        data = json.loads(mcp_raw.strip())
        if isinstance(data, dict) and "items" in data:
            out = _jobs_from_zhipin_items(data)
            if out is not None:
                return out[:MAX_JOBS_PER_USER]
        if isinstance(data, list):
            return [_normalize_job_item(j) for j in data]
        if isinstance(data, dict) and "jobs" in data:
            return [_normalize_job_item(j) for j in data["jobs"]]
        if isinstance(data, dict):
            return [_normalize_job_item(data)]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: MCP content parts (e.g. str([{'type':'text','text':'{"items":[...]}'}]))
    inner = _extract_json_from_mcp_content_parts(mcp_raw)
    if inner:
        try:
            data = json.loads(inner.strip())
            if isinstance(data, dict) and "items" in data:
                out = _jobs_from_zhipin_items(data)
                if out is not None:
                    return out[:MAX_JOBS_PER_USER]
            if isinstance(data, list):
                return [_normalize_job_item(j) for j in data]
            if isinstance(data, dict) and "jobs" in data:
                return [_normalize_job_item(j) for j in data["jobs"]]
            if isinstance(data, dict):
                return [_normalize_job_item(data)]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: find "jobs": [...] or "items": [...] in raw string by brace matching
    extracted = _extract_jobs_json_by_brace_matching(mcp_raw)
    if extracted:
        try:
            data = json.loads(extracted)
            if isinstance(data, dict) and "jobs" in data:
                return [_normalize_job_item(j) for j in data["jobs"]]
            if isinstance(data, dict) and "items" in data:
                out = _jobs_from_zhipin_items(data)
                if out is not None:
                    return out[:MAX_JOBS_PER_USER]
            if isinstance(data, list):
                return [_normalize_job_item(j) for j in data]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: split by separator, each block = one job with raw text
    separator = "\n\n---\n"
    parts = [p.strip() for p in mcp_raw.split(separator) if p.strip()]
    return [
        {
            "title": "",
            "company": "",
            "salary_min": None,
            "salary_max": None,
            "link": "",
            "description_snippet": raw[:8000] if len(raw) > 8000 else raw,
            "location": "",
            "raw": raw,
        }
        for raw in parts[:MAX_JOBS_PER_USER]
    ]


def _parse_salary_from_string(s: str) -> tuple[float | None, float | None]:
    """
    Parse zhipin-style salary string to (salary_min_k, salary_max_k) in thousands.
    E.g. "20-40K·16薪" -> (20.0, 40.0), "15-20K" -> (15.0, 20.0), "面议" -> (None, None).
    """
    if not (s and isinstance(s, str)):
        return (None, None)
    s = s.strip()
    # Match "20-40K" or "20-40K·16薪" or "20K" (single value -> min=max)
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*[Kk万]?", s)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Kk万]", s)
    if m:
        v = float(m.group(1))
        return (v, v)
    return (None, None)


def _normalize_job_item(j: Any) -> dict[str, Any]:
    """Ensure job dict has expected keys; accept various key names from MCP (incl. zhipin: jobDetail, address, salary, tags)."""
    if not isinstance(j, dict):
        return {"title": "", "company": "", "description_snippet": str(j)[:8000], "raw": str(j)[:8000]}
    salary_min = j.get("salary_min")
    salary_max = j.get("salary_max")
    if salary_min is None and salary_max is None and j.get("salary"):
        salary_min, salary_max = _parse_salary_from_string(str(j.get("salary")))
    # Fallback: parse salary from title (e.g. liepin "【北京】10-20k经验不限" or "25-45k3年以上")
    if salary_min is None and salary_max is None:
        title = str(j.get("title") or j.get("name") or "")
        salary_min, salary_max = _parse_salary_from_string(title)
    desc = str(j.get("description_snippet") or j.get("description") or "")[:8000]
    if not desc and (j.get("salary") or j.get("tags")):
        parts = []
        if j.get("salary"):
            parts.append(f"salary: {j.get('salary')}")
        if isinstance(j.get("tags"), list):
            parts.append("tags: " + ", ".join(str(t) for t in j["tags"][:20]))
        elif j.get("tags"):
            parts.append(str(j.get("tags")))
        if parts:
            desc = " | ".join(parts)
    return {
        "title": str(j.get("title") or j.get("name") or ""),
        "company": str(j.get("company") or ""),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "link": str(j.get("link") or j.get("url") or j.get("jobDetail") or ""),
        "description_snippet": desc[:8000] if desc else "",
        "location": str(j.get("location") or j.get("address") or ""),
        "raw": str(j.get("raw") or "")[:8000] or None,
    }
