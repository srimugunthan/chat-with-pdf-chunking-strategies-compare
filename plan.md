# Chunking Strategy Compare Tool - Implementation Plan

## Overview
A Streamlit app that lets users upload a PDF, chat with it using 3 different chunking strategies simultaneously, and compare responses side-by-side to understand which strategy works best.

## Tech Stack
- **UI**: Streamlit (wide layout)
- **LLM**: OpenAI GPT-4o
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: ChromaDB (in-memory, ephemeral)
- **PDF Parsing**: PyMuPDF (fitz)
- **Chunking**: Fixed-size, Recursive (LangChain), Semantic (custom)

## Project Structure

```
chunking-strategy-compare-tool/
├── app.py                  # Streamlit entry point, wires everything together
├── config.py               # Constants: model names, defaults, prompt template
├── pdf_parser.py           # PDF text extraction using PyMuPDF
├── chunkers/
│   ├── __init__.py         # Exports chunker registry dict
│   ├── fixed_size.py       # Strategy 1: sliding window by char count + overlap
│   ├── recursive.py        # Strategy 2: LangChain RecursiveCharacterTextSplitter
│   └── semantic.py         # Strategy 3: embed sentences, detect topic boundaries
├── embeddings.py           # OpenAI embedding wrapper with batching + retry
├── vector_store.py         # ChromaDB: create collections, add chunks, query
├── rag_pipeline.py         # Retrieve chunks + generate answer, returns RAGResult
├── ui_components.py        # Sidebar, 3-column result display, chunk expanders
├── requirements.txt
└── .env.example
```

## Data Flow

```
Upload PDF -> extract_text() -> full_text
                                    |
          ┌─────────────────────────┼─────────────────────────┐
          v                         v                         v
   fixed_size_chunk()       recursive_chunk()       semantic_chunk()
          |                         |                         |
          v                         v                         v
   embed_texts()             embed_texts()            embed_texts()
          |                         |                         |
          v                         v                         v
   ChromaDB collection_1    collection_2             collection_3
          └─────────────────────────┼─────────────────────────┘
                                    |
                             User asks question
                                    |
                             embed query (once)
                                    |
          ┌─────────────────────────┼─────────────────────────┐
          v                         v                         v
   query collection_1       query collection_2       query collection_3
          |                         |                         |
          v                         v                         v
   generate_answer()        generate_answer()        generate_answer()
          |                         |                         |
          v                         v                         v
   ┌──────────────┬──────────────┬──────────────┐
   │  Column 1    │  Column 2    │  Column 3    │
   │  Fixed-Size  │  Recursive   │  Semantic    │
   │  Answer      │  Answer      │  Answer      │
   │  > Chunks    │  > Chunks    │  > Chunks    │
   └──────────────┴──────────────┴──────────────┘
```

Parallelism via `concurrent.futures.ThreadPoolExecutor` for both ingestion and querying (I/O-bound work).

## UI Layout

- **Sidebar**: API key input, PDF uploader, sliders (chunk_size, overlap, semantic threshold percentile, top_k), processing status with chunk counts per strategy
- **Main area**: Chat messages top-down. User messages span full width. Each assistant response is a row of `st.columns(3)` — one per strategy with the answer + an `st.expander` showing retrieved chunks with color-coded similarity scores
- **Chat input**: `st.chat_input` at bottom, disabled until PDF is processed

```
┌─────────────────────────────────────────────────────────────────────┐
│  SIDEBAR                              MAIN AREA                     │
│  ┌─────────────┐   ┌─────────────────────────────────────────────┐  │
│  │ App Title   │   │  Chat history (scrollable)                  │  │
│  │             │   │  ┌──────────┬──────────┬──────────┐         │  │
│  │ API Key     │   │  │ Fixed    │ Recursive│ Semantic │         │  │
│  │ [********]  │   │  │          │          │          │         │  │
│  │             │   │  │ Answer   │ Answer   │ Answer   │         │  │
│  │ Upload PDF  │   │  │          │          │          │         │  │
│  │ [Browse]    │   │  │ ▶ Chunks │ ▶ Chunks │ ▶ Chunks │         │  │
│  │             │   │  └──────────┴──────────┴──────────┘         │  │
│  │ Settings    │   │                                             │  │
│  │ chunk_size  │   │  ... more Q&A pairs ...                     │  │
│  │ [===o====]  │   │                                             │  │
│  │ overlap     │   ├─────────────────────────────────────────────┤  │
│  │ [==o=====]  │   │  [Type your question here...]    [Send]     │  │
│  │ top_k       │   └─────────────────────────────────────────────┘  │
│  │ [===o====]  │                                                    │
│  │             │                                                    │
│  │ Status info │                                                    │
│  │ "PDF loaded │                                                    │
│  │  42 pages"  │                                                    │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Three Chunking Strategies

### 1. Fixed-Size (`chunkers/fixed_size.py`)
Simple sliding window: take `chunk_size` characters, advance by `chunk_size - overlap`. Pure Python.

### 2. Recursive Character Splitting (`chunkers/recursive.py`)
Wraps `langchain_text_splitters.RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`. Prefers natural text boundaries.

### 3. Semantic Chunking (`chunkers/semantic.py`)
Most complex strategy:
1. Split text into sentences (regex-based, handles abbreviations)
2. Embed all sentences in one batched API call
3. Compute cosine similarity between each consecutive pair of sentence embeddings
4. Find breakpoints where similarity drops below a percentile threshold (configurable, default 25th percentile)
5. Group sentences between breakpoints into chunks
6. Post-process: merge chunks < min_size, split chunks > max_size

**Detailed algorithm:**
- Sentence splitting handles abbreviations (Mr., Dr., etc.) to avoid false splits
- Cosine similarity: `cos_sim = dot(e_i, e_{i+1}) / (norm(e_i) * norm(e_{i+1}))`
- Percentile-based threshold is more robust than absolute threshold
- Lower percentile = more breakpoints = smaller chunks

## Key Modules

### `config.py`
Module-level constants: `EMBEDDING_MODEL`, `CHAT_MODEL`, `DEFAULT_CHUNK_SIZE` (1000), `DEFAULT_CHUNK_OVERLAP` (200), `DEFAULT_TOP_K` (5), `STRATEGY_NAMES`, `STRATEGY_LABELS`, `SYSTEM_PROMPT` template with `{context}` and `{question}` placeholders.

### `pdf_parser.py`
- `extract_text(uploaded_file) -> str`: Read PDF bytes via `fitz.open(stream=bytes, filetype="pdf")`, iterate pages, concatenate `page.get_text()`

### `embeddings.py`
- `embed_texts(texts, api_key, batch_size=100) -> list[list[float]]`: Batched embedding with retry on `RateLimitError` (exponential backoff, 3 attempts)

### `vector_store.py`
- `get_chroma_client() -> chromadb.Client()`: Ephemeral in-memory client
- `create_collection(client, name) -> Collection`: Delete-if-exists + create with `{"hnsw:space": "cosine"}`
- `add_chunks(collection, chunks, embeddings)`: Bulk insert with IDs `"chunk_0"`, `"chunk_1"`, ...
- `query_collection(collection, query_embedding, top_k) -> (docs, distances)`

### `rag_pipeline.py`
- `RAGResult` dataclass: `strategy_name`, `answer`, `retrieved_chunks`, `distances`
- `generate_answer(query, chunks, api_key) -> str`: Format prompt with context, call GPT-4o (temperature=0.2, max_tokens=1024)
- `query_single_strategy(query, query_embedding, collection, strategy_name, api_key, top_k) -> RAGResult`: Full retrieve + generate pipeline for one strategy

### `ui_components.py`
- `render_sidebar() -> dict`: Returns all settings (api_key, uploaded_file, chunk_size, overlap, threshold, top_k)
- `render_results_row(results: list[RAGResult])`: 3-column layout with answers and expandable chunks with color-coded similarity scores (green > 0.85, orange > 0.75, red otherwise)

### `app.py`
- `init_session_state()`: Initialize keys (chroma_client, collections, chunk_counts, pdf_processed, pdf_name, chat_history)
- `process_pdf(full_text, settings)`: Parallel chunk + embed + store for all 3 strategies via ThreadPoolExecutor
- `handle_query(query, settings) -> list[RAGResult]`: Embed query once, parallel retrieve + generate for all 3, sort by strategy order
- `main()`: Page config (wide layout), sidebar, PDF upload handling with re-upload detection, chat history rendering, query handling

## Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `chroma_client` | `chromadb.ClientAPI` | Ephemeral ChromaDB client |
| `collections` | `dict[str, Collection]` | Strategy name -> ChromaDB collection |
| `chunk_counts` | `dict[str, int]` | Chunk count per strategy (display in sidebar) |
| `pdf_processed` | `bool` | Whether a PDF has been processed |
| `pdf_name` | `str` | Current PDF filename (detect re-uploads) |
| `chat_history` | `list[dict]` | User messages (str) and assistant messages (list[RAGResult]) |

## Error Handling
- API key validation (must start with `sk-`)
- Empty PDF detection (no extractable text or < 50 chars)
- Empty chunk lists after splitting (raise ValueError)
- Embedding API retry with exponential backoff (3 attempts on RateLimitError)
- Large PDF cap for semantic chunking (max 1000 sentences with warning)
- Chat input disabled until PDF is processed
- Thread safety: 3 strategies write to 3 separate ChromaDB collections (no write contention)
- Re-upload creates fresh ChromaDB client (old one garbage collected)

## Implementation Order

1. `config.py` — constants
2. `requirements.txt` — dependencies
3. `pdf_parser.py` — PDF text extraction
4. `chunkers/fixed_size.py` — simplest chunker
5. `chunkers/recursive.py` — LangChain wrapper
6. `embeddings.py` — embedding wrapper (needed by semantic chunker)
7. `chunkers/semantic.py` — most complex chunker
8. `chunkers/__init__.py` — registry
9. `vector_store.py` — ChromaDB operations
10. `rag_pipeline.py` — RAG orchestration
11. `ui_components.py` — UI rendering functions
12. `app.py` — main app, wire everything together

## Dependencies (`requirements.txt`)
```
streamlit>=1.30.0
openai>=1.0.0
chromadb>=0.4.0
langchain-text-splitters>=0.1.0
PyMuPDF>=1.23.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

## Verification
1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Enter OpenAI API key in sidebar
4. Upload a multi-page PDF with diverse content
5. Verify sidebar shows chunk counts for all 3 strategies (they should differ)
6. Ask a factual question about the PDF content
7. Confirm 3 columns appear with different answers
8. Expand "Retrieved Chunks" in each column — verify chunks differ across strategies
9. Ask a follow-up question — verify chat history persists
10. Upload a different PDF — verify collections reset and chat history clears
