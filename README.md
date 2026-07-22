# Local RAG for Obsidian — Second Brain

A fully local Retrieval-Augmented Generation (RAG) pipeline that turns an Obsidian vault into a queryable knowledge base — no cloud APIs, no data leaving the machine.

## Why

I keep my technical notes, lab writeups, and certification study material in Obsidian. This project lets me ask questions against that knowledge base instead of manually searching through hundreds of markdown files, while keeping everything private and offline.

## Architecture

```
Obsidian Vault (.md files)
        |
        v
 Markdown-aware chunking (LangChain MarkdownTextSplitter)
        |
        v
 Local embeddings (Ollama, nomic-embed-text)
        |
        v
 ChromaDB (persistent local vector store)
        |
        v
 Retrieval (similarity search) --> Local LLM (Ollama, qwen2.5:7b) --> Answer + cited sources
```

## Stack

| Layer | Tool | Why |
|---|---|---|
| Notes source | Obsidian vault (markdown) | Existing knowledge base, no migration needed |
| Chunking | LangChain MarkdownTextSplitter | Respects markdown structure instead of blind character splitting |
| Embeddings | Ollama (nomic-embed-text) | Runs locally, no API cost, no data leaves the machine |
| Vector store | ChromaDB (local, persistent) | Lightweight, no server to manage |
| Generation | Ollama (qwen2.5:7b) | Local inference, good quality/resource trade-off |

## Features

- Fully local / offline: embeddings and generation both run through Ollama; nothing is sent to a third-party API.
- Frontmatter-aware: extracts YAML tags from each note and keeps them as metadata for filtering.
- Incremental and resilient indexing: batched embedding calls with retries; a failed batch doesn't lose previously indexed content, since ChromaDB persists cumulatively.
- Source-cited answers: every generated answer lists which notes it drew from.

## Usage

```bash
pip install langchain-text-splitters langchain-ollama langchain-chroma langchain-core

export VAULT_PATH="/path/to/your/obsidian/vault"

# Index the vault
python index_vault.py index

# Raw retrieval only (see which chunks match)
python index_vault.py query "what did I learn about MITRE ATT&CK?"

# Full RAG: retrieval + generated answer with cited sources
python index_vault.py chat "what did I learn about MITRE ATT&CK?"
```

Requires Ollama running locally with the nomic-embed-text and qwen2.5:7b models pulled.

## Notes / possible extensions

- Swap qwen2.5:7b for any other Ollama-served model depending on hardware.
- The retrieval layer is decoupled from generation, so it's straightforward to plug in a different LLM or add a reranking step.
- Next planned step: expose this as an MCP server so it can be queried directly from Claude Desktop / Claude Code.

---

Part of a broader "second brain" system combining Obsidian, local LLMs, and automation (n8n) for personal knowledge management.
