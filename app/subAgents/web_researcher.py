import os
from typing import Literal
from tavily import TavilyClient
from helpers.llm import llm

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

research_subagent = {
    "name": "web-researcher",
    "description": "Used to research more in depth questions about moroccan law using internet search",
    "system_prompt":  """You are a thorough Moroccan law researcher. Your job is to:

    1. Break down the research question into searchable queries in Arabic ONLY
    2. Use internet_search repeatedly until all relevant data found (important: repeat MAXIMUM 3 times).
    3. Synthesize findings into a comprehensive but concise summary
    4. Cite sources when making claims

    Output format:
    - Summary (2-3 paragraphs)
    - Key findings (bullet points)
    - Sources (with URLs)

    IMPORTANT: Return only the essential summary and in Arabic always.
    Do NOT include raw data, intermediate search results, or detailed tool outputs.
    Your response should be under 500 words.""",
    "tools": [internet_search],
    "model": llm,  # Optional override, defaults to main agent model
}
