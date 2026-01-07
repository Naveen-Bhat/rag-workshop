"""
RAG Chain Module - Core RAG functionality for the Course Advisor.

This module provides the CourseAdvisorRAG class for building RAG pipelines.
It supports both in-memory and persistent vector stores.

NOTE:
-----
Set USE_PERSISTENT_STORE = True to enable persistent storage.
When enabled:
- First run: Creates and saves vector store to disk
- Subsequent runs: Loads from disk (faster startup)
- force_reindex=True: Rebuilds the store from source files
"""

import shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
# Set to True to enable persistent storage (saves to chroma_db_agent/ folder)
USE_PERSISTENT_STORE = True


class CourseAdvisorRAG:
    """RAG system for the Course Advisor.

    This class handles document loading, chunking, embedding, and retrieval
    for the course advisor application. Used by the Agent for search_courses tool.
    """

    def __init__(
        self,
        data_folder: str = "syllabi",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int = 4,
        model_name: str = "gemini-2.0-flash",
        force_reindex: bool = False
    ):
        """Initialize the RAG system.

        Args:
            data_folder: Folder under data/ containing markdown files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
            model_name: Gemini model to use
            force_reindex: If True, rebuild vector store from source files
        """
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.model_name = model_name
        self.force_reindex = force_reindex

        # Will be initialized lazily
        self._vectorstore = None
        self._retriever = None
        self._chain = None

    @property
    def vectorstore(self):
        """Get or create the vector store."""
        if self._vectorstore is None:
            self._load_vectorstore()
        return self._vectorstore

    @property
    def retriever(self):
        """Get the retriever."""
        if self._retriever is None:
            self._retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k}
            )
        return self._retriever

    @property
    def chain(self):
        """Get or create the RAG chain."""
        if self._chain is None:
            self._create_chain()
        return self._chain

    def _load_vectorstore(self):
        """Load documents and create vector store.

        Supports two modes:
        - In-memory (default): Creates fresh vector store each time
        - Persistent (USE_PERSISTENT_STORE=True): Saves/loads from disk
        """
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / self.data_folder

        # Create embeddings (needed for both modes)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # =====================================================================
        # PERSISTENT STORE MODE
        # =====================================================================
        if USE_PERSISTENT_STORE:
            persist_path = project_root / "chroma_db_agent" / self.data_folder

            # Force reindex: delete existing store
            if self.force_reindex and persist_path.exists():
                shutil.rmtree(persist_path)
                print(f"[Agent RAG] Deleted existing index at {persist_path}")

            # Load from existing persistent store if available
            if persist_path.exists() and not self.force_reindex:
                self._vectorstore = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=embeddings,
                    collection_name=f"agent_rag_{self.data_folder}"
                )
                print(f"[Agent RAG] Loaded from persistent store: {persist_path}")
                return

            # Create persistent store (first run or after reindex)
            if not data_path.exists():
                raise FileNotFoundError(f"Data folder not found: {data_path}")

            documents = self._load_documents(data_path)
            chunks = self._chunk_documents(documents)

            persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(persist_path),
                collection_name=f"agent_rag_{self.data_folder}"
            )
            print(f"[Agent RAG] Created persistent index: {len(chunks)} chunks at {persist_path}")
            return

        # =====================================================================
        # IN-MEMORY STORE MODE (default)
        # =====================================================================
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_path}")

        documents = self._load_documents(data_path)
        chunks = self._chunk_documents(documents)

        self._vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="course_advisor"
        )

    def _load_documents(self, data_path: Path):
        """Load markdown documents from the data folder."""
        loader = DirectoryLoader(
            str(data_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        if not documents:
            raise ValueError(f"No markdown files found in {data_path}")

        return documents

    def _chunk_documents(self, documents):
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_documents(documents)

    def _create_chain(self):
        """Create the RAG chain."""
        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0,
            streaming=True
        )

        # RAG prompt
        template = """You are a helpful course advisor for Fictional University.
Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."
Be concise but helpful.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Build chain
        self._chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs) -> str:
        """Format retrieved documents for the prompt."""
        return "\n\n---\n\n".join(
            f"[Source: {Path(doc.metadata['source']).stem}]\n{doc.page_content}"
            for doc in docs
        )

    def ask(self, question: str) -> str:
        """Ask a question and get an answer.

        Args:
            question: The question to ask

        Returns:
            The answer string
        """
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """Ask a question and get answer with sources.

        Args:
            question: The question to ask

        Returns:
            Dict with 'answer' and 'sources' keys
        """
        # Get relevant documents
        docs = self.retriever.invoke(question)

        # Get answer
        answer = self.chain.invoke(question)

        # Extract sources
        sources = list(set(
            Path(doc.metadata["source"]).stem
            for doc in docs
        ))

        return {
            "answer": answer,
            "sources": sources
        }

    def stream(self, question: str):
        """Stream the response for a question.

        Args:
            question: The question to ask

        Yields:
            Response chunks
        """
        for chunk in self.chain.stream(question):
            yield chunk

    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results (uses default if not specified)

        Returns:
            List of dicts with 'content' and 'source' keys
        """
        if k is None:
            k = self.k

        docs = self.vectorstore.similarity_search(query, k=k)

        return [
            {
                "content": doc.page_content,
                "source": Path(doc.metadata["source"]).stem
            }
            for doc in docs
        ]


# Convenience function
def create_rag(data_folder: str = "syllabi", **kwargs) -> CourseAdvisorRAG:
    """Create a CourseAdvisorRAG instance.

    Args:
        data_folder: Folder under data/ containing markdown files
        **kwargs: Additional arguments for CourseAdvisorRAG

    Returns:
        Configured CourseAdvisorRAG instance
    """
    return CourseAdvisorRAG(data_folder=data_folder, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create RAG system
    rag = create_rag()

    # Test queries
    questions = [
        "What topics does CS301 cover?",
        "Who teaches Linear Algebra?",
        "What are the prerequisites for Deep Learning?"
    ]

    for question in questions:
        print(f"\nQ: {question}")
        result = rag.ask_with_sources(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
