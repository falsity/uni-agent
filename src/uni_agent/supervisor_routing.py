"""
Supervisor routing: lexical scoring + thresholds + optional LLM (see agent._supervisor_next_node).

Preflight (no MCP job results yet):
- score >= JOB_SCORE_THRESHOLD  -> job (classify_job_detail), skip LLM
- score <= CHAT_SCORE_THRESHOLD -> llm_call, skip LLM
- else -> LLM with optional lexical hint in prompt
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Weights chosen so clear hiring vs clear tech-survey cases separate; tune with eval/datasets + run_evaluate.py.
_JOB_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    (
        re.compile(
            r"相关职位|相关岗位|招聘职位|招聘岗位",
            re.IGNORECASE,
        ),
        4,
        "hiring_phrase",
    ),
    (
        re.compile(
            r"找工作|找.{0,12}(工作|职位|机会)|求职|投递简历|校招|社招|内推|应届生想投",
            re.IGNORECASE | re.DOTALL,
        ),
        4,
        "hiring_action",
    ),
    (
        re.compile(
            r"(搜|找|查|推荐|筛选).{0,60}(职位|岗位|工作机会)",
            re.IGNORECASE | re.DOTALL,
        ),
        4,
        "search_job_role",
    ),
    (
        re.compile(
            r"(职位|岗位).{0,25}(相关|开发|工程师|实习|应届|远程|全职|兼职)",
            re.IGNORECASE | re.DOTALL,
        ),
        3,
        "role_context",
    ),
]

# Strong signals for generic web / research (not MCP job listings)
_CHAT_PATTERNS: list[tuple[re.Pattern[str], int, str]] = [
    (
        re.compile(
            r"最新进展|行业报告|技术趋势|论文导读|文献综述|教程|"
            r"今天天气|现在几点|今日新闻|新闻速递",
            re.IGNORECASE,
        ),
        4,
        "general_web",
    ),
]

_NEGATION_RE = re.compile(
    r"(不|没|别|勿).{0,4}(找|搜|投|换).{0,16}(工作|职位|岗位|机会)",
    re.IGNORECASE | re.DOTALL,
)

# Net score thresholds (after negation penalty)
JOB_SCORE_THRESHOLD = 3
CHAT_SCORE_THRESHOLD = -3
NEGATION_PENALTY = 6


@dataclass
class IntentScore:
    """Lexical signals for supervisor hybrid routing."""

    score: int
    positive_hits: list[str] = field(default_factory=list)
    negative_hits: list[str] = field(default_factory=list)
    negation: bool = False


def score_supervisor_intent(text: str) -> IntentScore:
    """
    Sum weighted regex hits; apply negation penalty if user rejects job search.
    """
    if not (text and text.strip()):
        return IntentScore(score=0)
    t = text.strip()
    pos_hits: list[str] = []
    neg_hits: list[str] = []
    score = 0

    for pat, w, tag in _JOB_PATTERNS:
        if pat.search(t):
            score += w
            pos_hits.append(tag)

    for pat, w, tag in _CHAT_PATTERNS:
        if pat.search(t):
            score -= w
            neg_hits.append(tag)

    negation = bool(_NEGATION_RE.search(t))
    if negation:
        score -= NEGATION_PENALTY

    return IntentScore(
        score=score,
        positive_hits=pos_hits,
        negative_hits=neg_hits,
        negation=negation,
    )


def preflight_no_results_route(text: str) -> Literal["job", "chat", "llm"]:
    """
    Fast path without LLM when confidence is high; otherwise defer to LLM.
    """
    snap = score_supervisor_intent(text)
    if snap.score >= JOB_SCORE_THRESHOLD:
        return "job"
    if snap.score <= CHAT_SCORE_THRESHOLD:
        return "chat"
    return "llm"


def format_lexical_hint(snap: IntentScore) -> str:
    """Short line for supervisor prompt when LLM is used (ambiguous zone)."""
    if snap.score == 0 and not snap.positive_hits and not snap.negative_hits:
        return ""
    parts = [f"net_lexical_score={snap.score}"]
    if snap.positive_hits:
        parts.append("hiring:" + ",".join(sorted(set(snap.positive_hits))))
    if snap.negative_hits:
        parts.append("general_web:" + ",".join(sorted(set(snap.negative_hits))))
    if snap.negation:
        parts.append("negation=true")
    return "; ".join(parts) + (
        ". If hiring signals dominate, prefer job_search; if general_web dominates, prefer llm_call."
    )
