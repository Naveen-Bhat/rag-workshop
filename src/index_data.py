"""
Index course data into ChromaDB vector store.

Usage:
    uv run python src/index_data.py
    uv run python src/index_data.py --data-folder my_college
"""

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()


def index_documents(data_folder: str = "syllabi", persist_dir: str = "./chroma_db"):
    """Index documents from the specified folder into ChromaDB."""

    # Resolve paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / data_folder
    persist_path = project_root / persist_dir

    print(f"📂 Loading documents from: {data_path}")

    if not data_path.exists():
        print(f"❌ Error: Data folder not found: {data_path}")
        sys.exit(1)

    # Load documents
    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()

    if not documents:
        print(f"⚠️ No markdown files found in {data_path}")
        sys.exit(1)

    print(f"📄 Loaded {len(documents)} documents")

    # Show document names
    for doc in documents:
        name = Path(doc.metadata["source"]).name
        print(f"   - {name}: {len(doc.page_content)} chars")

    # Chunk documents
    print("\n🔪 Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Create embeddings
    print("\n🧮 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("   Model loaded: all-MiniLM-L6-v2")

    # Create vector store
    print(f"\n💾 Creating vector store at: {persist_path}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name="course_advisor"
    )

    print(f"\n✅ Successfully indexed {len(chunks)} chunks!")
    print(f"   Vector store saved to: {persist_path}")

    # Test a sample query
    print("\n🔍 Testing retrieval...")
    results = vectorstore.similarity_search("machine learning", k=2)
    print(f"   Query: 'machine learning'")
    print(f"   Found {len(results)} results:")
    for i, doc in enumerate(results):
        source = Path(doc.metadata["source"]).stem
        print(f"   [{i+1}] {source}: {doc.page_content[:80]}...")

    print("\n🎉 Indexing complete!")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(
        description="Index course documents into ChromaDB"
    )
    parser.add_argument(
        "--data-folder",
        default="syllabi",
        help="Folder name under data/ to index (default: syllabi)"
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="Directory to persist vector store (default: ./chroma_db)"
    )

    args = parser.parse_args()
    index_documents(args.data_folder, args.persist_dir)


if __name__ == "__main__":
    main()
