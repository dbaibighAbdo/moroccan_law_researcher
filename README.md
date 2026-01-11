# moroccan-law-researcher

**DeepLawExpert** is an AI research assistant focused on Moroccan law. It composes a main DeepLawExpert agent that orchestrates three specialized subagents (vector DB researcher, knowledge-graph researcher, and web researcher) to provide structured, cited legal research in Arabic.

**Table of contents**
- **Overview**: What the project does and high-level features
- **Quickstart**: Install, configure environment, run the app
- **Architecture**: Components and subagents
- **Key files**: Important source files and purpose
- **Environment variables**: Required `.env` keys
- **Dependencies**: Major Python packages used
- **Development**: Contributing and notes

**Overview**
- **Purpose**: Provide an AI-powered research assistant specialized in Moroccan law. The agent is designed to analyze legal questions (in Arabic), query multiple information sources, and synthesize answers with citations.
- **Primary outputs**: Structured Arabic responses including main findings, sources, and concise analyses.

**Quickstart**

Prerequisites
- Python >= 3.12
- A Neo4j instance (optional but recommended for knowledge graph features)
- OpenAI API key (or other LLM provider keys used in `app/helpers/llm.py`)

Install
```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
; pip install --upgrade pip
; pip install -r requirements.txt  # or install via pyproject.toml tooling
```

Environment
- Copy `.env.example` to `.env` and set secrets:
  - `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
  - Optional tracing & API keys: `LANGSMITH_API_KEY`, `TAVILY_API_KEY`, etc.

Run the Streamlit app (chat UI)
```powershell
; streamlit run app/bot.py
```

Run quick script
```powershell
; python main.py
```

**Architecture**
- `deepagents` main agent (`app/agent.py`) coordinates three subagents for every query (use of all three is enforced by system prompt):
  - `vectorDB-researcher` (`app/subAgents/vectorDB_researcher.py`) — Neo4j vector retrieval (RAG)
  - `knowledge-graph-researcher` (`app/subAgents/knowledgeGraph_researcher.py`) — Cypher/graph queries over a Neo4j knowledge graph
  - `web-researcher` (`app/subAgents/web_reseacher.py`) — Web searches (via Tavily client) for recent sources
- `app/helpers/llm.py` defines LLM and embeddings configuration
- `knowledgeGraph_construction/kg_construtor.py` contains a pipeline for building the Neo4j knowledge graph from PDFs
- `app/bot.py` provides a Streamlit front-end chat UI supporting Arabic rendering and session management

**Key files**
- `main.py` — trivial script entrypoint / example runner
- `app/agent.py` — main DeepLawExpert agent configuration and `generate_response()` helper
- `app/bot.py` — Streamlit-based chat UI and interaction handler
- `app/helpers/llm.py` — LLM and embeddings initialization
- `app/helpers/utils.py` — UI helpers and session utilities
- `app/subAgents/*` — subagents implementations:
  - `vectorDB_researcher.py` — Neo4j vector retrieval and summarization tool
  - `knowledgeGraph_researcher.py` — Cypher generation and QA chain
  - `web_reseacher.py` — Web search wrapper and aggregator
- `knowledgeGraph_construction/kg_construtor.py` — tools and pipeline to ingest PDFs and populate Neo4j

**Environment variables** (see `.env.example`)
- `OPENAI_API_KEY` — OpenAI key for LLMs and embeddings
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` — Neo4j connection details
- `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — optional tracing via LangSmith
- `TAVILY_API_KEY` — required by the web researcher for web searches

**Dependencies** (from `pyproject.toml`)
- `deepagents` — agent orchestration
- `dotenv` / `python-dotenv` — environment variable loader
- `langchain`, `langchain-neo4j`, `langchain-openai` — LLM + graph integrations
- `streamlit` (+ `streamlit-arabic-support-wrapper`) — front-end chat UI
- `tavily-python` — web search client

Notes on usage & behavior
- The main agent enforces the use of all three subagents for every query (vector DB, knowledge graph, web).
- Subagents typically return concise Arabic summaries; the main agent synthesizes final output and cites sources.
- Some modules expect specific environment variables (Neo4j, Tavily, OpenAI). Without them, corresponding features will fail.

**Development**
- To extend the knowledge base, add PDF/text sources and run `knowledgeGraph_construction/kg_construtor.py` to populate the Neo4j graph and vector index.
- To add or tune subagents, edit `app/subAgents/*` and adjust the `system_prompt` and `tools` definitions.

**Contributing**
- Fork, create a feature branch, and open a pull request. Provide tests or a short demo for major changes.

**License**
- Add a license file or badge here. (No license specified in repo — add `LICENSE` if you want to open-source.)

**Contact**
- Repository owner: `dbaibighAbdo` (see the repo for contact details)

---

If you'd like, I can also:
- generate a `requirements.txt` based on `pyproject.toml`,
- add a short developer `Makefile` or PowerShell script to streamline setup, or
- run a quick local smoke-check to confirm the Streamlit app starts (if you provide API keys).
# moroccan_law_researcher
