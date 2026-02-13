import base64
import json
import os
import uuid
from typing import Literal

import httpx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily.tavily import TavilyClient

from uni_agent.prompts import SUMMARIZE_WEB_SEARCH
from uni_agent.utils import get_today_str

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
summarization_model = ChatOpenAI(
    model="Qwen/Qwen3-8B", base_url="http://192.168.0.201:9000/v1"
)


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def run_tavily_search(
    query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Run a web search using the Tavily API.
    Args:
        query: The search query.
        max_results: The maximum number of results to return.
        topic: The topic of the search.
        include_raw_content: Whether to include the raw content of the search results.
    Returns:
        A dictionary containing the search results.
    """
    result = tavily_client.search(query, max_results=max_results)
    return result


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Summary object with filename and summary
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary_and_filename = structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEB_SEARCH.format(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )

        return summary_and_filename

    except Exception:
        # Return a basic summary object on failure
        return Summary(
            filename="search_result.md",
            summary=webpage_content[:1000] + "..."
            if len(webpage_content) > 1000
            else webpage_content,
        )


def process_search_results(results: dict) -> list[dict]:
    """Process search results by summarizing content where available.

    Args:
        results: Tavily search results dictionary

    Returns:
        List of processed results with summaries
    """
    processed_results = []

    # Create a client for HTTP requests with timeout
    HTTPX_CLIENT = httpx.Client(timeout=30.0)  # Add 30 second timeout

    for result in results.get("results", []):
        # Get url
        url = result["url"]

        # Read url with timeout and error handling
        try:
            response = HTTPX_CLIENT.get(url)

            if response.status_code == 200:
                # Convert HTML to markdown
                raw_content = markdownify(response.text)
                summary_obj = summarize_webpage_content(raw_content)
            else:
                # Use Tavily's generated summary
                raw_content = result.get("raw_content", "")
                summary_obj = Summary(
                    filename="URL_error.md",
                    summary=result.get(
                        "content", "Error reading URL; try another search."
                    ),
                )
        except (httpx.TimeoutException, httpx.RequestError) as e:
            # Handle timeout or connection errors gracefully
            raw_content = result.get("raw_content", "")
            summary_obj = Summary(
                filename="connection_error.md",
                summary=result.get(
                    "content",
                    f"Could not fetch URL (timeout/connection error). Try another search.",
                ),
            )

        # uniquify file names
        uid = (
            base64.urlsafe_b64encode(uuid.uuid4().bytes)
            .rstrip(b"=")
            .decode("ascii")[:8]
        )
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext}"

        processed_results.append(
            {
                "url": result["url"],
                "title": result["title"],
                "summary": summary_obj.summary,
                "filename": summary_obj.filename,
                "raw_content": raw_content,
            }
        )

    return processed_results


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Run a web search using the Tavily API. Use when the user needs up-to-date or external information.

    Args:
        query (str): Search query. Must reflect only the current user request (same intent and language).
        max_results (int): Maximum number of results. Defaults to 1.
        topic (Literal["general", "news", "finance"]): Topic of the search. Defaults to "general".
        include_raw_content (bool): Whether to include raw content. Defaults to True.

    Returns:
        dict: Search results.
    """
    result = tavily_client.search(query, max_results=max_results)
    processed_results = process_search_results(result)
    return json.dumps(processed_results, indent=4)
