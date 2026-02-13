from typing import Annotated, Literal, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, model_validator


class JobState(MessagesState):
    job_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Set by supervisor to route to job_search (classify->mcp) or optimize_recommendations
    next_agent: Optional[str]
    # Raw MCP job search results; when store is used, can be removed; kept for backward compat
    mcp_job_results: Optional[str]
    # True when job results were written to store (mcp_jobs_tool_call); used by supervisor for routing
    has_job_results: Optional[bool]
    # Number of jobs written to store in mcp_jobs_tool_call (for E2E verification)
    jobs_stored_count: Optional[int]
    # Internal routing in job_agent: set by classify_job_detail to "mcp_jobs_tool_call" or "__end__"
    classify_goto: Optional[str]


class SupervisorDecision(BaseModel):
    """Supervisor routing: which sub-agent to invoke."""

    route: Literal["job_search", "optimize", "llm_call"] = Field(
        description="job_search: run job search (classify + MCP). optimize: user already has job results and wants to refine/optimize recommendations. llm_call: user asks a general question, greeting, or off-topic; answer directly without job search or optimization."
    )


class ClarifyJobDetail(BaseModel):
    """Schema for clarifying the job detail"""

    need_clarify: bool = Field(
        description="Whether the job detail needs to be clarified"
    )
    question: str = Field(
        default="",
        description="Question to clarify the job detail, if the job detail is not clear, ask the user for more information"
    )
    verification: str = Field(
        default="",
        description="Verification message that the necessary information for finding a position has been provided"
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_none_to_empty_str(cls, data):
        """Coerce null from LLM JSON to empty string so str fields pass validation."""
        if not isinstance(data, dict):
            return data
        for key in ("question", "verification"):
            if key in data and data[key] is None:
                data = {**data, key: ""}
        return data


class OptimizeRetrievalParams(BaseModel):
    """
    Agentic RAG: retrieval params for optimize_recommendations, parsed from user message by LLM.
    Used to query store (salary filter, sort, semantic query, limit) instead of hardcoded regex.
    """

    salary_min_k: Optional[float] = Field(
        default=None,
        description="Minimum salary in thousands (e.g. 25 = 25k). Set when user says '25k以上' or similar.",
    )
    salary_max_k: Optional[float] = Field(
        default=None,
        description="Maximum salary in thousands. Set when user gives an upper bound or range.",
    )
    sort_by_salary_desc: bool = Field(
        default=False,
        description="True when user wants '按薪资从高到低' or 'salary high to low'.",
    )
    semantic_query: Optional[str] = Field(
        default=None,
        description="Short query for vector search over jobs (e.g. keywords from user). Empty or null = use full list.",
    )
    limit: int = Field(
        default=60,
        ge=10,
        le=100,
        description="Max number of jobs to retrieve (10-100). Use larger when user asks for 'all' or 'more'.",
    )
