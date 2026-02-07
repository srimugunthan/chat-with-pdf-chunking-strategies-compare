# Session Log — Chunking Strategy Compare Tool

**Date:** 2026-02-04

---

## 1. What We're Building

A Streamlit app that lets users upload a PDF and chat with it. The key differentiator: every query runs against **3 different chunking strategies simultaneously**, and results are displayed in 3 side-by-side panes. This lets users instantly compare which chunking approach produces better retrieval and answers for their specific document.

**Tech decisions made:**
- **LLM:** OpenAI GPT-4o
- **Embeddings:** OpenAI text-embedding-ada-002
- **Vector Store:** ChromaDB (persistent by default, configurable)
- **PDF Parsing:** PyMuPDF (fitz)
- **3 Chunking Strategies:**
  1. Fixed-size — naive sliding window by character count with overlap
  2. Recursive character splitting — LangChain's `RecursiveCharacterTextSplitter`
  3. Semantic — custom implementation using embedding-based topic boundary detection

---

## 2. Architecture Diagram

### Ingestion Pipeline (on PDF upload)

```
User uploads PDF
       │
       ▼
 pdf_parser.extract_text()
       │
       ▼
   full_text: str
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
 fixed_size.py          recursive.py          semantic.py
 chunk_fixed_size()     chunk_recursive()     chunk_semantic()
       │                      │                      │
       ▼                      ▼                      ▼
 embeddings.py          embeddings.py          embeddings.py
 embed_texts()          embed_texts()          embed_texts()
       │                      │                      │
       ▼                      ▼                      ▼
 vector_store.py        vector_store.py        vector_store.py
 ChromaDB               ChromaDB               ChromaDB
 "chunking_tool_       "chunking_tool_        "chunking_tool_
  fixed_size"           recursive"              semantic"
```

All 3 branches run in parallel via `ThreadPoolExecutor`.

### Query Pipeline (on each chat message)

```
User types question
       │
       ▼
 embeddings.embed_texts([query])  ← single API call, shared across strategies
       │
       ├──────────────────────┬──────────────────────┐
       ▼                      ▼                      ▼
 query_collection()     query_collection()     query_collection()
 (top_k chunks)         (top_k chunks)         (top_k chunks)
       │                      │                      │
       ▼                      ▼                      ▼
 generate_answer()      generate_answer()      generate_answer()
 (GPT-4o)               (GPT-4o)               (GPT-4o)
       │                      │                      │
       ▼                      ▼                      ▼
 ┌──────────────┬──────────────┬──────────────┐
 │  Column 1    │  Column 2    │  Column 3    │
 │  Fixed-Size  │  Recursive   │  Semantic    │
 │              │              │              │
 │  LLM Answer  │  LLM Answer  │  LLM Answer  │
 │  ▶ Chunks    │  ▶ Chunks    │  ▶ Chunks    │
 └──────────────┴──────────────┴──────────────┘
```

### UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  SIDEBAR                         MAIN AREA                      │
│  ┌────────────┐  ┌────────────────────────────────────────────┐ │
│  │ API Key    │  │  Chat history                              │ │
│  │ [********] │  │  ┌────────────┬────────────┬────────────┐  │ │
│  │            │  │  │ Fixed-Size │ Recursive  │ Semantic   │  │ │
│  │ Upload PDF │  │  │ Answer     │ Answer     │ Answer     │  │ │
│  │ [Browse]   │  │  │ ▶ Chunks   │ ▶ Chunks   │ ▶ Chunks   │  │ │
│  │            │  │  └────────────┴────────────┴────────────┘  │ │
│  │ chunk_size │  │  ... more Q&A pairs ...                    │ │
│  │ [===o====] │  │                                            │ │
│  │ overlap    │  ├────────────────────────────────────────────┤ │
│  │ [==o=====] │  │  [Type your question here...]   [Send]     │ │
│  │ threshold  │  └────────────────────────────────────────────┘ │
│  │ [===o====] │                                                 │
│  │ top_k      │                                                 │
│  │ [===o====] │                                                 │
│  │            │                                                 │
│  │ Status:    │                                                 │
│  │ "PDF loaded│                                                 │
│  │  Fixed: 42 │                                                 │
│  │  Recur: 38 │                                                 │
│  │  Seman: 29"│                                                 │
│  └────────────┘                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Current File Structure

```
chunking-strategy-compare-tool/
├── config.py           ✅ DONE — model names, chunking defaults, ChromaDB settings, strategy labels, system prompt
├── requirements.txt    ✅ DONE — 7 dependencies (streamlit, openai, chromadb, langchain-text-splitters, PyMuPDF, numpy, python-dotenv)
├── README.md           ✅ DONE — setup instructions (uv install, ChromaDB modes, API key, run command)
├── plan.md             ✅ DONE — full implementation plan with module details and function signatures
├── SESSION_LOG.md      ✅ DONE — this file
│
│   --- NOT YET CREATED ---
│
├── app.py              ⬜ TODO — Streamlit entry point
├── pdf_parser.py       ⬜ TODO — PDF text extraction
├── chunkers/
│   ├── __init__.py     ⬜ TODO — chunker registry
│   ├── fixed_size.py   ⬜ TODO — strategy 1
│   ├── recursive.py    ⬜ TODO — strategy 2
│   └── semantic.py     ⬜ TODO — strategy 3
├── embeddings.py       ⬜ TODO — OpenAI embedding wrapper
├── vector_store.py     ⬜ TODO — ChromaDB operations
├── rag_pipeline.py     ⬜ TODO — retrieval + generation
├── ui_components.py    ⬜ TODO — Streamlit UI helpers
└── .env.example        ⬜ TODO — API key template
```

---

## 4. Exact Next Steps

### Step 1: Install dependencies

```bash
cd /Users/srimugunthan/chunking-strategy-compare-tool
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 2: Create `pdf_parser.py`

```python
import fitz  # PyMuPDF


def extract_text(uploaded_file) -> str:
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
```

### Step 3: Create `chunkers/fixed_size.py`

```python
def chunk_fixed_size(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - chunk_overlap
    return [c for c in chunks if c]
```

### Step 4: Create `chunkers/recursive.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_recursive(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
```

### Step 5: Create `embeddings.py`

Needed before semantic chunker since it depends on embedding sentences.

```python
from openai import OpenAI

def embed_texts(texts: list[str], api_key: str, model: str = "text-embedding-ada-002", batch_size: int = 100) -> list[list[float]]:
    client = OpenAI(api_key=api_key)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings
```

### Step 6: Create `chunkers/semantic.py`

Most complex module — sentence splitting, embedding, cosine similarity breakpoint detection, grouping.

### Step 7: Create `chunkers/__init__.py`

```python
from chunkers.fixed_size import chunk_fixed_size
from chunkers.recursive import chunk_recursive
from chunkers.semantic import chunk_semantic

CHUNKERS = {
    "fixed_size": chunk_fixed_size,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
}
```

### Step 8: Create `vector_store.py`

ChromaDB collection creation, chunk insertion, similarity search.

### Step 9: Create `rag_pipeline.py`

`RAGResult` dataclass + `generate_answer()` + `query_single_strategy()`.

### Step 10: Create `ui_components.py`

Sidebar rendering, 3-column result display, expandable chunk sections.

### Step 11: Create `app.py`

Main orchestration — session state init, PDF processing pipeline, parallel query dispatch, chat UI.

### Step 12: Create `.env.example`

```
OPENAI_API_KEY=sk-...
```

---

## 5. Terminal Commands to Verify Current State

```bash
# Check project directory exists and see what files are present
ls -la /Users/srimugunthan/chunking-strategy-compare-tool/

# Verify config.py has the expected constants
python3 -c "import sys; sys.path.insert(0, '/Users/srimugunthan/chunking-strategy-compare-tool'); import config; print('Strategies:', config.STRATEGY_NAMES); print('ChromaDB dir:', config.CHROMA_PERSIST_DIR); print('Model:', config.CHAT_MODEL)"

# Verify requirements.txt is well-formed
cat /Users/srimugunthan/chunking-strategy-compare-tool/requirements.txt

# Check that no venv exists yet (dependencies not installed)
ls /Users/srimugunthan/chunking-strategy-compare-tool/.venv 2>&1 || echo "No venv yet — run 'uv venv' to create one"
```

---

## What We Accomplished Today

1. **Chose the full tech stack** — OpenAI GPT-4o + ada-002 embeddings, ChromaDB, PyMuPDF, Streamlit
2. **Designed the architecture** — module breakdown, data flow, UI layout, session state management
3. **Created `config.py`** — all constants including model settings, chunking defaults, ChromaDB config, strategy labels, and system prompt
4. **Created `requirements.txt`** — 7 pinned dependencies
5. **Created `README.md`** — setup instructions with uv, ChromaDB modes (persistent/in-memory/server), and run command
6. **Created `plan.md`** — comprehensive implementation plan with function signatures for all 12 files

## Important Decisions and Trade-offs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | ChromaDB (persistent default) | Lightweight, no server needed, survives restarts. Can switch to in-memory via config. |
| Parallelism | `ThreadPoolExecutor` | All heavy work is I/O-bound (OpenAI API calls). Threads are simpler than async and sufficient here. |
| Semantic chunking threshold | Percentile-based (25th) | More robust than absolute cosine similarity threshold. Adapts to document characteristics. |
| Query embedding | Computed once, shared | Same query embedding is reused across all 3 strategy retrievals to avoid redundant API calls. |
| Session state for chat | `list[dict]` with RAGResult | Stores full retrieval context (chunks + distances) so expanding chunks works on historical messages. |
| ChromaDB collections | 3 separate collections | One per strategy. No write contention during parallel ingestion. Clean separation. |
| PDF re-upload | Detect by filename change | Creates fresh ChromaDB client, clears chat history. Simple and reliable. |
| Chunk display | `st.expander` (collapsed) | Keeps UI clean by default. Users opt-in to inspect retrieved chunks per strategy. |
| Embedding batching | 100 per batch | Stays within OpenAI API limits. Handles large PDFs without hitting payload size errors. |
| Semantic chunking sentence cap | 1000 sentences | Prevents excessive embedding costs on very large documents. Shows warning to user. |
