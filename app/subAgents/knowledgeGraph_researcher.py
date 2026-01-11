
from helpers.llm import llm
from helpers.graph import graph

from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about Moroccan Labor Laws.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.
Do not return ANYTHING "" if the answer is not found in the graph.

Schema:
{schema}

Question:
{query}

Cypher Query:
"""


cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=ChatPromptTemplate([CYPHER_GENERATION_TEMPLATE]),
    allow_dangerous_requests=True
)


def cypher_retriever(query: str) -> str:
    """Retrieve information from the knowledge graph."""
    try:
        graph_result = cypher_qa.invoke({"query": query})
        graph_answer = graph_result.get("result", "").strip()
        if not graph_answer:
            graph_answer = ""
    except Exception:
        graph_answer = ""

    return graph_answer


kg_subagent = {
    "name": "knowledge-graph-researcher",
    "description": "Used to retrieve more in depth information about moroccan law using a (Neo4j) knowledge graph database",
    "system_prompt": """You are a thorough Moroccan law Neo4j knowledge graph database researcher. Your job is to:

    1. Break down the research question into searchable queries in Arabic ONLY
    2. Use cypher_retriever repeatedly until all relevant data found (important: repeat MAXIMUM 3 times).
    3. Synthesize findings into a comprehensive but concise summary

    Output format:
    - Summary (2-3 paragraphs)
    - Key findings (bullet points)

    IMPORTANT: Return only the essential summary and in Arabic always.
    Do NOT include raw data, intermediate search results, or detailed tool outputs.
    Your response should be under 500 words.""",
    "tools": [cypher_retriever],
    "model": llm,  # Optional override, defaults to main agent model
}





