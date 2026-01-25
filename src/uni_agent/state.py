from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class JobState(MessagesState):
    job_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]


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
