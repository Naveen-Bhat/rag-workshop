## The Vibe Coding Prompt for Gemini CLI

```
I have an existing RAG-based course advisor application. I want to add my college's course data to it.

## CRITICAL INSTRUCTIONS - READ FIRST:
- DO NOT modify any existing files (app.py, src/*.py, etc.)
- ONLY create NEW files in the locations specified below
- The existing app will automatically detect new data in `data/my_college/`

## Current Project Structure:
- `data/syllabi/` - Sample course syllabi (8 markdown files) - DO NOT TOUCH
- `data/my_college/` - Empty folder for custom data - ADD FILES HERE
- `app.py` - Streamlit UI (already supports my_college folder) - DO NOT MODIFY
- `src/rag_chain.py` - RAG pipeline - DO NOT MODIFY
- `src/agent.py` - Agent tools - DO NOT MODIFY

## What I Need You To Do:

### Step 1: Create sample syllabus files
Create 2 course syllabus files in `data/my_college/` folder:
- `data/my_college/BCA101.md` - Introduction to Programming (Python basics)
- `data/my_college/BCA201.md` - Database Management Systems

Each file should be 50-80 lines, following this structure:
```markdown
# Course Code - Course Name
## Course Information
- Credits, Prerequisites, Instructor, etc.
## Course Description
## Learning Objectives
## Module breakdown (4-5 modules with topics)
## Textbooks
## Assessment
```

### Step 2: Create a standalone test script
Create a NEW file `test_my_college_rag.py` in the project root that:
1. Loads documents from `data/my_college/`
2. Chunks them (chunk_size=500, chunk_overlap=50)
3. Creates embeddings using `all-MiniLM-L6-v2`
4. Stores in ChromaDB (in-memory is fine for demo)
5. Tests retrieval with query: "What programming language is taught in BCA101?"
6. Uses Gemini to generate an answer from retrieved context

### Technical Stack (already installed):
- langchain>=0.3.0
- langchain-google-genai>=2.0.0
- langchain-chroma>=0.2.0
- langchain-huggingface>=0.1.0
- sentence-transformers>=3.0.0
- chromadb>=0.5.0

### Important Guidelines:
- Use `HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2` (local, no API)
- Use `ChatGoogleGenerativeAI` with model `gemini-2.0-flash` for LLM
- Load API key from .env using `python-dotenv`
- Explain each RAG step as you write the code
- The test script should be runnable with: `uv run python test_my_college_rag.py`

## Summary of files to create:
1. `data/my_college/BCA101.md` (NEW)
2. `data/my_college/BCA201.md` (NEW)
3. `test_my_college_rag.py` (NEW)

DO NOT create or modify any other files.
```