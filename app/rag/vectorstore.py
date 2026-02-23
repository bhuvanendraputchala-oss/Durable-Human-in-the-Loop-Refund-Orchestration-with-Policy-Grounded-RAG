from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

logger = logging.getLogger(__name__)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERSIST_DIR = os.path.join(_ROOT, "vector_db")
POLICY_DIR = os.path.join(_ROOT, "mock_data", "policies")

_vectorstore: Chroma | None = None


def _build_index(vs: Chroma) -> None:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    if not os.path.isdir(POLICY_DIR):
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}. "
            "Create mock_data/policies/ and populate it with Markdown files."
        )

    md_files = sorted(f for f in os.listdir(POLICY_DIR) if f.endswith(".md"))
    if not md_files:
        raise FileNotFoundError(
            f"No Markdown policy files found in {POLICY_DIR}. "
            "Add *.md files to enable RAG retrieval."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    texts: list[str] = []
    metadatas: list[dict] = []

    for filename in md_files:
        with open(os.path.join(POLICY_DIR, filename), encoding="utf-8") as fh:
            content = fh.read()
        for idx, doc in enumerate(splitter.create_documents([content])):
            texts.append(doc.page_content)
            metadatas.append({
                "source": filename,
                "chunk_id": f"{filename}:chunk_{idx}",
                "start_char": doc.metadata.get("start_index", 0),
            })

    vs.add_texts(texts, metadatas=metadatas)
    logger.info("Auto-indexed %d chunks from %d policy files into %s.", len(texts), len(md_files), PERSIST_DIR)


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env")

    embeddings = OpenAIEmbeddings()
    vs = Chroma(
        collection_name="policy_kb",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    count = vs._collection.count()
    if count == 0:
        logger.warning("Vector store is empty — auto-indexing policy documents from %s", POLICY_DIR)
        _build_index(vs)
    else:
        logger.info("Vector store ready: %d chunks in 'policy_kb'.", count)

    _vectorstore = vs
    return _vectorstore
