# RAG Workshop: Smart Course Advisor

A hands-on workshop for building RAG (Retrieval-Augmented Generation) applications using Python, LangChain, and Google Gemini.

## What You'll Build

A **Smart Course Advisor** that can:
- Answer questions about courses, prerequisites, and learning paths
- Show its reasoning process (Agentic RAG)
- Stream responses in a clean chat interface

## Workshop Structure

| Part | Duration | Topics |
|------|----------|--------|
| **1. Fundamentals** | ~25 min | Chunking, embeddings, vector search |
| **2. RAG Pipeline** | ~20 min | Build end-to-end RAG system |
| **3. Agentic RAG** | ~15 min | Agent reasoning demo |
| **4. Hands-On** | ~45 min | Run RAG on your college data |

---

## Quick Start

### Prerequisites

1. **Miniconda** - [Download here](https://docs.conda.io/en/latest/miniconda.html)
2. **Google Gemini API Key** - [Get free key](https://aistudio.google.com/apikey)

### Setup (One-Time)

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/rag-workshop
cd rag-workshop

# 2. Create conda environment
conda env create -f environment.yml
conda activate rag-workshop

# 3. Install dependencies with uv
uv sync

# 4. Set up API key
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Running the Workshop

**Option 1: Jupyter Notebooks** (for learning)
```bash
uv run jupyter notebook
```
Open the notebooks in order:
1. `notebooks/01_fundamentals.ipynb`
2. `notebooks/02_rag_pipeline.ipynb`
3. `notebooks/03_agentic_rag.ipynb`

**Option 2: Streamlit App** (for demos)
```bash
uv run streamlit run app.py
```

---

## Using Your Own College Data

### Step 1: Prepare Your Data

Create markdown files in `data/my_college/` folder. Each file should contain:

```markdown
# CS101: Introduction to Programming

## Course Information
- **Credits:** 3
- **Instructor:** Dr. Smith
- **Prerequisites:** None

## Description
This course introduces fundamental programming concepts...

## Topics Covered
- Variables and data types
- Control structures
- Functions
```

**Options for data collection:**
- **Manual entry:** Use the interactive script: `python data/scraper_template.py`
- **Web scraping:** Customize `data/scraper_template.py` for your college website

### Step 2: Index Your Data

```bash
uv run python src/index_data.py --folder my_college
```

### Step 3: Run the App

```bash
uv run streamlit run app.py
```

Select "my_college" from the data source dropdown in the sidebar.

---

## Project Structure

```
rag-workshop/
├── app.py                    # Streamlit chat UI
├── data/
│   ├── syllabi/              # Sample course data (8 courses)
│   ├── my_college/           # Your college data goes here
│   ├── courses.json          # Structured course data for agent
│   └── scraper_template.py   # Template for scraping
├── notebooks/
│   ├── 01_fundamentals.ipynb # Chunking, embeddings demos
│   ├── 02_rag_pipeline.ipynb # Full RAG pipeline
│   └── 03_agentic_rag.ipynb  # Agent demo
├── src/
│   ├── rag_chain.py          # Core RAG logic
│   ├── index_data.py         # Document indexing script
│   └── agent.py              # Agentic RAG tools
├── environment.yml           # Conda environment
├── pyproject.toml            # Python dependencies
└── .env.example              # API key template
```

---

## Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| LLM | Google Gemini | Free tier: 5 RPM, paid ~$0.10/1M tokens |
| Embeddings | Sentence Transformers | Local, no API cost |
| Vector Store | ChromaDB | In-memory or persistent |
| Orchestration | LangChain | Chains and agents |
| UI | Streamlit | Chat interface with streaming |

---

## Library Reference

### Complete Library Overview

| Library | Type | API Needed? | Purpose |
|---------|------|-------------|---------|
| `sentence-transformers` | Direct | No (local) | Embedding models |
| `langchain-huggingface` | LangChain wrapper | No (local) | Embeddings for LangChain |
| `chromadb` | Direct | No (local) | Vector database |
| `langchain-chroma` | LangChain wrapper | No (local) | Vector DB for LangChain |
| `langchain-google-genai` | LangChain wrapper | **Yes (Gemini)** | LLM integration |
| `langchain-text-splitters` | LangChain | No | Document chunking |
| `langchain-community` | LangChain | No | Document loaders |
| `langchain` | Core | No | Prompts, agents, chains |
| `streamlit` | Direct | No | Web UI |

### Import Reference

```python
# Embeddings (local - no API needed)
from sentence_transformers import SentenceTransformer          # Direct usage
from langchain_huggingface import HuggingFaceEmbeddings        # LangChain wrapper

# Vector Store (local - no API needed)
import chromadb                                                 # Direct usage
from langchain_chroma import Chroma                            # LangChain wrapper

# Document Processing (local - no API needed)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# LLM (requires GOOGLE_API_KEY)
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Core
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  langchain   │    │  langchain-  │    │  langchain-  │   │
│  │    (core)    │    │   huggingface│    │    chroma    │   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘   │
│         │                   │                   │            │
│         │                   ▼                   ▼            │
│         │           ┌──────────────┐    ┌──────────────┐    │
│         │           │  sentence-   │    │   chromadb   │    │
│         │           │ transformers │    │              │    │
│         │           └──────────────┘    └──────────────┘    │
│         │                   │                   │            │
│         │                   ▼                   ▼            │
│         │           ┌──────────────┐    ┌──────────────┐    │
│         │           │ HuggingFace  │    │   SQLite/    │    │
│         │           │   Models     │    │   DuckDB     │    │
│         │           │   (LOCAL)    │    │   (LOCAL)    │    │
│         │           └──────────────┘    └──────────────┘    │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │  langchain-  │───────────▶  Google Gemini API (CLOUD)    │
│  │ google-genai │                                           │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** Only `langchain-google-genai` requires an API key. Everything else runs locally!

---

## LangChain Ecosystem

```
┌─────────────────────────────────────────────────────────┐
│              LANGCHAIN ECOSYSTEM (2026)                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  langchain-core     → Base abstractions (LLMs, prompts) │
│  langchain          → Chains, RAG helpers, utilities    │
│  langchain.agents   → Agent framework (what we use!)    │ ← THIS WORKSHOP
│  langgraph          → Stateful graphs, production agents│ ← NEXT STEP
│  langsmith          → Observability, evals, debugging   │ ← PRODUCTION
│  langserve          → Deploy as REST API                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## AI-Assisted Coding with Gemini CLI

Want to modify the code? Use Gemini CLI for AI-assisted development:

```bash
# Install Gemini CLI
npm install -g @google/gemini-cli

# Run in project folder
gemini

# Example prompts:
# "Add a new tool to search by instructor name"
# "Change chunk size to 1000 tokens"
# "Add a sidebar filter for course level"
```

---

## Troubleshooting

### "GOOGLE_API_KEY not found"
```bash
# Make sure .env file exists with your key
cat .env
# Should show: GOOGLE_API_KEY=your_key_here
```

### "No markdown files found"
```bash
# Check data folder has .md files
ls data/syllabi/
# Or for your data:
ls data/my_college/
```

### ChromaDB issues
```bash
# Clear the database and re-index
rm -rf chroma_db/
uv run python src/index_data.py
```

### Conda environment issues
```bash
# Remove and recreate
conda deactivate
conda env remove -n rag-workshop
conda env create -f environment.yml
conda activate rag-workshop
uv sync
```

---

## Resources for Further Learning

### LangChain
- [LangChain Docs](https://python.langchain.com/docs)
- [LangChain Agents](https://python.langchain.com/docs/how_to/#agents)
- [LangGraph](https://langchain-ai.github.io/langgraph/) - For more control over agent flows
- [LangSmith](https://smith.langchain.com) - Observability for production

### Vector Databases
- [ChromaDB Docs](https://docs.trychroma.com)
- [Pinecone](https://www.pinecone.io/) - Managed vector DB

### Evaluation
- [RAGAS](https://docs.ragas.io) - RAG evaluation metrics

### Graph RAG (Bonus)
- [Neo4j GraphAcademy](https://graphacademy.neo4j.com) - Free courses
- [LangChain + Neo4j](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher/)

### AI Coding Assistants
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [Claude Code](https://claude.ai/code)

---

## Sample Queries to Try

**Simple factual:**
- "What topics does CS301 cover?"
- "Who teaches Linear Algebra?"

**Prerequisites:**
- "What are the prerequisites for Deep Learning?"
- "Can I take CS401 if I've only completed CS101?"

**Learning paths:**
- "I've completed CS101 and MATH101. What should I take next for AI?"
- "What's the complete path to become an NLP specialist?"

**Comparison:**
- "What's the difference between CS301 and CS401?"
- "Which courses cover neural networks?"

---

## License

MIT License - Feel free to use and modify for your workshops!

---

## Acknowledgments

Built for academic faculty learning to integrate AI into their institutions.

**Workshop Author:** [Your Name]
**Contact:** [Your Email]
