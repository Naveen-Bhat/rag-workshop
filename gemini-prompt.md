I have an existing RAG-based course advisor application. I want to add my college's course data to it.

## Current Setup:
- Sample syllabi are in `data/syllabi/` folder (8 markdown files)
- Empty folder `data/my_college/` exists for custom data
- `src/index_data.py` handles document indexing
- `src/rag_chain.py` has CourseAdvisorRAG class
- `app.py` is the Streamlit interface

## What I Need:
1. First, create 2-3 sample course syllabus files in `data/my_college/` folder in markdown format. Make them for a fictional "ABC College" with courses like:
   - BCA101 - Introduction to Programming (Python basics)
   - BCA201 - Database Management Systems
   Keep them realistic but simple (about 50-80 lines each).

2. Then show me how to:
   - Load these documents using LangChain
   - Chunk them appropriately (explain chunk_size and chunk_overlap)
   - Create embeddings using sentence-transformers
   - Store in ChromaDB vector database
   - Test retrieval with a sample query

## Technical Stack (already installed):
- langchain>=0.3.0
- langchain-google-genai>=2.0.0
- langchain-chroma>=0.2.0
- langchain-huggingface>=0.1.0
- sentence-transformers>=3.0.0
- chromadb>=0.5.0

## Important:
- Use `all-MiniLM-L6-v2` for embeddings (runs locally, no API needed)
- Use Google Gemini for the LLM via langchain-google-genai
- Explain each RAG step as you write the code so I understand what's happening
- At the end, show a working query against the new college data
