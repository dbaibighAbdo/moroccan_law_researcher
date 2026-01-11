from helpers.llm import embeddings, llm
from helpers.graph import graph
import os
from langchain_neo4j import Neo4jVector


neo4jvector = Neo4jVector.from_existing_index(
    embedding=embeddings,
    graph=graph,
    index_name="legal_vector_index",
    embedding_dimension=1536
)

retriever = neo4jvector.as_retriever(search_type="similarity", k=3)


def get_related_context(query: str) -> str:
    """
    Retrieve related contextual passages from the Neo4j vector index.
    Returns a clean, well-structured string in Arabic.
    """
    docs = retriever.invoke(query)

    if not docs:
        return ""

    clean_docs = []
    for doc in docs:
        # Remove unnecessary newlines and extra spaces
        text = doc.page_content.replace("\n", " ").strip()
        # Optional: collapse multiple spaces into one
        text = " ".join(text.split())
        clean_docs.append(text)

    # Use explicit separator for multiple docs to improve readability
    return "\n---\n".join(clean_docs)

vectorDB_subagent = {
    "name": "vectorDB-researcher",
    "description": "Used to retrieve more in depth information about moroccan law using a Neo4j vector database",
    "system_prompt": """You are a thorough Moroccan law Neo4j vector database researcher. Your job is to:

    1. Break down the research question into searchable queries in Arabic ONLY
    2. Use get_related_context repeatedly until all relevant data found (important: repeat MAXIMUM 3 times).
    3. Synthesize findings into a comprehensive but concise summary

    Output format:
    - Summary (2-3 paragraphs)
    - Key findings (bullet points)

    IMPORTANT: Return only the essential summary and in Arabic always.
    Do NOT include raw data, intermediate search results, or detailed tool outputs.
    Your response should be under 500 words.""",
    "tools": [get_related_context],
    "model": llm,  # Optional override, defaults to main agent model
}