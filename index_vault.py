"""
Local RAG Indexer for Obsidian Vaults
======================================

Reads all .md notes from an Obsidian vault, splits them into chunks,
generates local embeddings with Ollama (nomic-embed-text), and stores
them in ChromaDB (persisted on disk, no cloud dependency).

100% local: no data leaves your machine. Ollama runs the embedding and
chat models locally, ChromaDB persists to a local folder.

Usage:
    python index_vault.py index              # (re)index the whole vault
    python index_vault.py query "question"    # show the most relevant raw chunks
    python index_vault.py chat "question"     # full RAG: retrieval + generated answer
"""

import os
import sys
import re
from pathlib import Path

from langchain_text_splitters import MarkdownTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# CONFIGURATION — adjust via environment variables or edit directly
# ---------------------------------------------------------------------------
VAULT_PATH = Path(os.environ.get("VAULT_PATH", "./vault"))
CHROMA_DIR = "./chroma_db"          # created next to this script
COLLECTION_NAME = "second_brain"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b"

# Folders/files to ignore during indexing
IGNORE_DIRS = {".obsidian", ".git", ".trash", "_attachments", ".vscode"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def extract_frontmatter_tags(text: str) -> list[str]:
    """Extract simple tags from YAML frontmatter (tags: [a, b, c])."""
    match = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return []
    fm = match.group(1)
    tag_match = re.search(r"tags:\s*\[(.*?)\]", fm)
    if tag_match:
        return [t.strip() for t in tag_match.group(1).split(",") if t.strip()]
    return []


def load_vault_documents(vault_path: Path) -> list[Document]:
    """Walk the vault and load each .md file as a Document with metadata."""
    docs = []
    for md_file in vault_path.rglob("*.md"):
        if any(part in IGNORE_DIRS for part in md_file.parts):
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  [!] Could not read {md_file}: {e}")
            continue

        rel_path = md_file.relative_to(vault_path).as_posix()
        tags = extract_frontmatter_tags(text)

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": rel_path,
                    "folder": rel_path.split("/")[0],
                    "tags": ", ".join(tags),
                },
            )
        )
    return docs


def index_vault():
    import time

    print(f"Reading notes from: {VAULT_PATH}")
    documents = load_vault_documents(VAULT_PATH)
    print(f"  -> {len(documents)} notes found")

    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    print(f"  -> {len(chunks)} chunks generated")

    print(f"Generating embeddings with '{EMBED_MODEL}' (Ollama, local)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    BATCH_SIZE = 25          # chunks per Ollama call (small, to avoid overloading the runner)
    MAX_RETRIES = 3
    RETRY_WAIT_SECONDS = 5

    total = len(chunks)
    failed_batches = []

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start: start + BATCH_SIZE]
        batch_num = start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.add_documents(batch)
                print(f"  Batch {batch_num}/{total_batches} OK ({start + len(batch)}/{total} chunks)")
                break
            except Exception as e:
                print(f"  [!] Batch {batch_num}/{total_batches} failed attempt {attempt}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    failed_batches.append(batch_num)

    print("\nIndexing complete.")
    if failed_batches:
        print(f"  [!] {len(failed_batches)} batch(es) failed permanently: {failed_batches}")
        print("  You can re-run 'index' -- ChromaDB is cumulative, nothing already saved is lost.")
    else:
        print("  All batches saved without errors.")


def query_vault(question: str, k: int = 4):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    results = db.similarity_search(question, k=k)

    if not results:
        print("No results. Did you run 'index' first?")
        return

    print(f"\nTop {len(results)} results for: \"{question}\"\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ({doc.metadata.get('source')}) ---")
        preview = doc.page_content.strip().replace("\n", " ")[:300]
        print(preview + ("..." if len(doc.page_content) > 300 else ""))
        print()


RAG_PROMPT = """You are an assistant that answers questions using ONLY the \
context from the personal notes given below. Do not invent information that \
is not in the context. If the context is not enough to answer, say so \
clearly. Answer concisely and directly.

Context (vault notes):
{context}

Question: {question}

Answer:"""


def rag_chat(question: str, k: int = 4):
    """Retrieval + generation: finds the most relevant chunks and asks the
    local LLM to write an answer grounded in them, citing sources."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    results = db.similarity_search(question, k=k)

    if not results:
        print("No results in the vault. Did you run 'index' first?")
        return

    context_parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content.strip()}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = RAG_PROMPT.format(context=context, question=question)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)
    print(f"\nGenerating answer with '{CHAT_MODEL}'...\n")
    response = llm.invoke(prompt)

    print("=== Answer ===\n")
    print(response.content)

    print("\n=== Sources consulted ===")
    seen = set()
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            print(f"  - {source}")
            seen.add(source)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_vault.py [index|query \"question\"|chat \"question\"]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "index":
        index_vault()
    elif command == "query":
        if len(sys.argv) < 3:
            print("Missing question. Ex: python index_vault.py query \"what is RAG\"")
            sys.exit(1)
        query_vault(sys.argv[2])
    elif command == "chat":
        if len(sys.argv) < 3:
            print("Missing question. Ex: python index_vault.py chat \"what is RAG\"")
            sys.exit(1)
        rag_chat(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print("Usage: python index_vault.py [index|query \"question\"|chat \"question\"]")
