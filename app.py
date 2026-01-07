"""
RAG Workshop - Course Advisor Streamlit App
Enhanced chat interface with RAG and Agentic modes.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

PAGE_CONFIG = {
    "page_title": "Course Advisor",
    "page_icon": "🎓",
    "layout": "wide"
}

EXAMPLE_QUESTIONS = {
    "rag": [
        "What topics does CS301 cover?",
        "Who teaches Linear Algebra?",
        "What are the prerequisites for Deep Learning?",
    ],
    "agentic": [
        "I've completed CS101 and MATH101. What's my path to AI research?",
        "Can I take CS401 if I only completed CS101?",
        "What courses cover neural networks and who teaches them?",
    ]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_course_code(query: str) -> str | None:
    """Extract course code like CS301, MATH201 from query."""
    match = re.search(r'\b([A-Z]{2,4}\d{3})\b', query.upper())
    return match.group(1) if match else None


def format_chat_history(messages: list, max_turns: int = 5) -> str:
    """Format recent chat history for context."""
    if not messages:
        return ""
    recent = messages[-(max_turns * 2):]
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


# =============================================================================
# RAG MODE FUNCTIONS
# =============================================================================

@st.cache_resource
def get_embeddings():
    """Get cached embeddings model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def load_vector_store(data_folder: str, force_reindex: bool = False):
    """Load documents and create vector store.

    NOTE:
    --------------------
    This function re-indexes documents into an IN-MEMORY vector store on every app startup.
    This is fine for small datasets (like the sample syllabi) but inefficient for large datasets.

    For production or large datasets, you can:
    1. Set USE_PERSISTENT_STORE = True below
    2. On first run, it will auto-index and save to disk
    3. Subsequent runs will load from disk (fast)
    4. Use "Reindex" button in UI to rebuild the index

    See src/index_data.py for CLI indexing utility.
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    import shutil

    data_path = Path(__file__).parent / "data" / data_folder

    # =========================================================================
    # OPTION: PERSISTENT STORE (faster startup for large datasets)
    # =========================================================================
    # Set to True to enable persistent storage (auto-indexes on first run)
    USE_PERSISTENT_STORE = False

    if USE_PERSISTENT_STORE:
        persist_path = Path(__file__).parent / "chroma_db" / data_folder

        # Force reindex: delete existing store
        if force_reindex and persist_path.exists():
            shutil.rmtree(persist_path)
            st.info(f"Deleted existing index at {persist_path}")

        # Load from existing persistent store if available
        if persist_path.exists() and not force_reindex:
            embeddings = get_embeddings()
            vectorstore = Chroma(
                persist_directory=str(persist_path),
                embedding_function=embeddings,
                collection_name=f"course_advisor_{data_folder}"
            )
            st.sidebar.success("Loaded from persistent store")
            return vectorstore, 0, ["(loaded from disk)"]

        # Create persistent store (first run or after reindex)
        if not data_path.exists():
            st.error(f"Data path does not exist: {data_path}")
            return None, 0, []

        loader = DirectoryLoader(
            str(data_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents = loader.load()

        if not documents:
            st.error(f"No documents found in {data_path}")
            return None, 0, []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            source_name = Path(chunk.metadata['source']).stem
            chunk.page_content = f"[{source_name}]\n{chunk.page_content}"
            chunk.metadata['course_code'] = source_name

        embeddings = get_embeddings()
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_path),
            collection_name=f"course_advisor_{data_folder}"
        )

        doc_names = [Path(d.metadata['source']).stem for d in documents]
        st.sidebar.success(f"Created persistent index: {len(chunks)} chunks")
        return vectorstore, len(chunks), doc_names
    # =========================================================================

    # DEFAULT: In-memory store (re-indexes every startup)
    if not data_path.exists():
        st.error(f"Data path does not exist: {data_path}")
        return None, 0, []

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        st.error(f"No documents found in {data_path}")
        return None, 0, []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        source_name = Path(chunk.metadata['source']).stem
        chunk.page_content = f"[{source_name}]\n{chunk.page_content}"
        chunk.metadata['course_code'] = source_name

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"course_advisor_{data_folder}"
    )

    doc_names = [Path(d.metadata['source']).stem for d in documents]
    return vectorstore, len(chunks), doc_names


def get_llm():
    """Get the LLM instance for RAG mode."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        streaming=True
    )


def generate_rag_response(question: str, vectorstore, chat_history: str, search_mode: str):
    """Generate a response using RAG with chat history."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    course_code = extract_course_code(question)

    if search_mode == "smart" and course_code:
        docs = vectorstore.similarity_search(question, k=6, filter={"course_code": course_code})
        filter_info = f" (filtered: {course_code})"
    else:
        docs = vectorstore.similarity_search(question, k=6)
        filter_info = " (no filter)" if search_mode == "smart" else " (standard mode)"

    st.sidebar.write(f"🔍 Retrieved {len(docs)} docs{filter_info}")

    context = "\n\n---\n\n".join(
        f"[Source: {Path(doc.metadata['source']).stem}]\n{doc.page_content}"
        for doc in docs
    )

    sources = list(set(Path(doc.metadata['source']).stem for doc in docs))

    template = """You are a helpful and friendly course advisor for Fictional University.
You help students understand courses, prerequisites, and plan their academic journey.

## Retrieved Course Information:
{context}

## Recent Conversation:
{chat_history}

## Current Question:
{question}

## Instructions:
- Use the retrieved course information to answer the question
- Be helpful, concise, and conversational
- If information is not in the context, say so but try to be helpful
- Reference specific courses by their codes (e.g., CS301, MATH201)
- For prerequisite questions, explain the chain of courses needed

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "context": context,
        "chat_history": chat_history if chat_history else "No previous conversation.",
        "question": question
    }), sources, docs


# =============================================================================
# AGENTIC MODE FUNCTIONS
# =============================================================================

def get_agent(data_folder: str, force_reindex: bool = False):
    """Get agent instance.

    Note: Not using @st.cache_resource here because we need to support force_reindex.
    The agent is cached in st.session_state instead.
    """
    from src.agent import CourseAdvisorAgent
    return CourseAdvisorAgent(data_folder=data_folder, force_reindex=force_reindex)


def display_agent_step(step: dict, container):
    """Display a single agent reasoning step."""
    if step["type"] == "thought":
        container.markdown(f"🤔 **Thought:** Using tool `{step['tool']}`")
        container.code(str(step["args"]), language="json")
    elif step["type"] == "observation":
        content = step["content"]
        truncated = content[:500] + "..." if len(content) > 500 else content
        container.markdown(f"📋 **Observation:**")
        container.text(truncated)
    elif step["type"] == "answer":
        container.markdown("---")
        container.markdown(f"**Answer:** {step['content']}")


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with all controls."""
    with st.sidebar:
        st.title("🎓 Course Advisor")
        st.markdown("---")

        # Mode selection
        st.markdown("### 🔧 Mode")
        mode = st.radio(
            "Select Mode",
            options=["rag", "agentic"],
            format_func=lambda x: "RAG (Fast)" if x == "rag" else "Agentic (Reasoning)",
            help="RAG: Simple retrieval + generation\nAgentic: Multi-step reasoning with tools"
        )

        if mode == "rag":
            st.info("📚 Retrieves relevant docs and generates answer")
        else:
            st.info("🤖 Agent reasons through complex questions using tools")

        st.markdown("---")

        # Data folder selector
        data_folder = st.selectbox(
            "📁 Data Source",
            options=["syllabi", "my_college"],
            index=0,
            help="Choose which course data to use"
        )

        st.markdown("---")

        # Mode-specific settings
        reindex_rag = False
        reindex_agent = False

        if mode == "rag":
            st.markdown("### 🔬 RAG Settings")
            search_mode = st.radio(
                "Search Strategy",
                options=["smart", "standard"],
                index=0,
                format_func=lambda x: "Smart (Recommended)" if x == "smart" else "Standard",
                help="Smart: Uses metadata filtering when course code detected"
            )
            if search_mode == "smart":
                st.success("✅ Metadata filtering enabled")
            else:
                st.warning("⚠️ Pure semantic search")

            # RAG-specific reindex button
            reindex_rag = st.button("📥 Reindex RAG Store", use_container_width=True,
                                    help="Rebuild RAG vector index (chunk_size=1000)")
        else:
            search_mode = "smart"  # Not used in agentic mode
            st.markdown("### 🛠️ Agent Tools")
            st.markdown("""
            - `search_courses` - Find courses by topic
            - `get_course_info` - Get course details
            - `check_prerequisites` - View prereq chains
            - `find_learning_path` - Plan your path
            - `list_all_courses` - See all courses
            """)

            # Agent-specific reindex button
            reindex_agent = st.button("📥 Reindex Agent Store", use_container_width=True,
                                      help="Rebuild Agent vector index (chunk_size=500)")

        st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        with col2:
            reload_data = st.button("🔄 Reload", use_container_width=True)

        st.markdown("---")

        # Example questions
        st.markdown("### 💡 Try these:")
        questions = EXAMPLE_QUESTIONS.get(mode, EXAMPLE_QUESTIONS["rag"])
        for q in questions:
            if st.button(q, key=f"btn_{hash(q)}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        return mode, data_folder, search_mode, clear_chat, reload_data, reindex_rag, reindex_agent


def render_chat_history():
    """Render the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for RAG mode responses
            if message["role"] == "assistant":
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources used"):
                        for source in message["sources"]:
                            st.markdown(f"- `{source}`")

                # Show reasoning for agentic mode responses
                if "reasoning" in message and message["reasoning"]:
                    with st.expander("🤔 Agent Reasoning"):
                        for step in message["reasoning"]:
                            if step["type"] == "thought":
                                st.markdown(f"**Tool:** `{step['tool']}`")
                                st.code(str(step["args"]), language="json")
                            elif step["type"] == "observation":
                                content = step["content"][:300] + "..." if len(step["content"]) > 300 else step["content"]
                                st.text(content)
                            st.markdown("---")


def handle_rag_response(prompt: str, vectorstore, chat_history: str, search_mode: str):
    """Handle response generation in RAG mode."""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            response_stream, sources, docs = generate_rag_response(
                prompt, vectorstore, chat_history, search_mode
            )

            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            if sources:
                with st.expander("📚 Sources used"):
                    for source in sources:
                        st.markdown(f"- `{source}`")

        except Exception as e:
            full_response = f"Sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(full_response)
            sources = []

    return full_response, sources


def handle_agentic_response(prompt: str, agent):
    """Handle response generation in Agentic mode."""
    with st.chat_message("assistant"):
        reasoning_container = st.container()
        answer_placeholder = st.empty()

        full_response = ""
        reasoning_steps = []

        try:
            with reasoning_container:
                st.markdown("**Agent Reasoning:**")
                steps_container = st.container()

            for step in agent.stream(prompt):
                reasoning_steps.append(step)

                with steps_container:
                    if step["type"] == "thought":
                        st.markdown(f"🤔 Using tool `{step['tool']}`")
                    elif step["type"] == "observation":
                        content = step["content"][:200] + "..." if len(step["content"]) > 200 else step["content"]
                        st.text(f"📋 {content}")
                    elif step["type"] == "answer":
                        full_response = step["content"]

            answer_placeholder.markdown(f"---\n**Answer:** {full_response}")

        except Exception as e:
            full_response = f"Sorry, I encountered an error: {str(e)}"
            answer_placeholder.markdown(full_response)
            reasoning_steps = []

    return full_response, reasoning_steps


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Page configuration
    st.set_page_config(**PAGE_CONFIG)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found. Please add it to your .env file.")
        st.stop()

    # Render sidebar and get settings
    mode, data_folder, search_mode, clear_chat, reload_data, reindex_rag, reindex_agent = render_sidebar()

    # Handle clear/reload actions
    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    if reload_data:
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "agent" in st.session_state:
            del st.session_state.agent
        if "loaded_folder" in st.session_state:
            del st.session_state.loaded_folder
        st.cache_resource.clear()
        st.rerun()

    # Handle RAG reindex action (force rebuild RAG vector store)
    if reindex_rag:
        st.session_state.force_reindex_rag = True
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        st.cache_resource.clear()
        st.rerun()

    # Handle Agent reindex action (force rebuild Agent vector store)
    if reindex_agent:
        st.session_state.force_reindex_agent = True
        if "agent" in st.session_state:
            del st.session_state.agent
        st.cache_resource.clear()
        st.rerun()

    # Handle mode/data folder changes
    if st.session_state.get("current_mode") != mode:
        st.session_state.messages = []
        st.session_state.current_mode = mode

    if st.session_state.get("data_folder") != data_folder:
        st.session_state.data_folder = data_folder
        st.session_state.messages = []
        if "vectorstore" in st.session_state:
            del st.session_state.vectorstore
        if "agent" in st.session_state:
            del st.session_state.agent
        st.cache_resource.clear()
        st.rerun()

    # Main content header
    mode_label = "RAG" if mode == "rag" else "Agentic"
    st.title(f"🎓 Course Advisor ({mode_label} Mode)")
    st.markdown("Ask me anything about courses, prerequisites, and learning paths!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Check if reindex was requested (separate flags for RAG and Agent)
    force_reindex_rag = st.session_state.pop("force_reindex_rag", False)
    force_reindex_agent = st.session_state.pop("force_reindex_agent", False)

    # Load resources based on mode
    if mode == "rag":
        # Load vector store for RAG mode
        if "vectorstore" not in st.session_state or st.session_state.get("loaded_folder") != data_folder or force_reindex_rag:
            with st.spinner("Reindexing RAG store..." if force_reindex_rag else "Loading course data..."):
                vectorstore, chunk_count, doc_names = load_vector_store(data_folder, force_reindex=force_reindex_rag)
                if vectorstore is not None:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunk_count = chunk_count
                    st.session_state.doc_names = doc_names
                    st.session_state.loaded_folder = data_folder

        vectorstore = st.session_state.get("vectorstore")
        if vectorstore is None:
            st.error("Failed to load course data.")
            return

        chunk_count = st.session_state.get("chunk_count", 0)
        doc_names = st.session_state.get("doc_names", [])
        st.caption(f"📊 Loaded {chunk_count} chunks from {len(doc_names)} documents")

    else:
        # Load agent for Agentic mode
        if "agent" not in st.session_state or st.session_state.get("agent_folder") != data_folder or force_reindex_agent:
            with st.spinner("Reindexing Agent store..." if force_reindex_agent else "Loading agent..."):
                try:
                    agent = get_agent(data_folder, force_reindex=force_reindex_agent)
                    st.session_state.agent = agent
                    st.session_state.agent_folder = data_folder
                except Exception as e:
                    st.error(f"Failed to load agent: {str(e)}")
                    return

        agent = st.session_state.get("agent")
        if agent is None:
            st.error("Failed to load agent.")
            return

        st.caption(f"🤖 Agent ready with {len(agent._tools)} tools")

    # Render chat history
    render_chat_history()

    # Handle pending question from sidebar
    if "pending_question" in st.session_state:
        prompt = st.session_state.pending_question
        del st.session_state.pending_question
    else:
        prompt = st.chat_input("Ask about courses...")

    # Process user input
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response based on mode
        if mode == "rag":
            chat_history = format_chat_history(st.session_state.messages[:-1])
            full_response, sources = handle_rag_response(
                prompt, vectorstore, chat_history, search_mode
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
        else:
            full_response, reasoning = handle_agentic_response(prompt, agent)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "reasoning": reasoning
            })


if __name__ == "__main__":
    main()
