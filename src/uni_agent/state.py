from typing import Annotated, Literal, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class JobState(MessagesState):
    job_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Set by supervisor to route to job_search (classify->mcp) or optimize_recommendations
    next_agent: Optional[str]
    # Raw MCP job search results; stored by mcp_jobs_tool_call, used by optimize_recommendations only (not in general context)
    mcp_job_results: Optional[str]


class SupervisorDecision(BaseModel):
    """Supervisor routing: which sub-agent to invoke."""

    route: Literal["job_search", "optimize"] = Field(
        description="job_search: run job search (classify + MCP). optimize: user already has job results and wants to refine/optimize recommendations via LLM dialogue."
    )


class ClarifyJobDetail(BaseModel):
    """Schema for clarifying the job detail"""

    need_clarify: bool = Field(
        description="Whether the job detail needs to be clarified"
    )
    question: str = Field(
        description="Question to clarify the job detail, if the job detail is not clear, ask the user for more information"
    )
    verification: str = Field(
        description="Verfiy message that the necessary information for finding a position has been provided"
    )
