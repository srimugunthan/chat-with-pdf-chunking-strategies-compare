import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from config import (
    STRATEGY_NAMES,
    STRATEGY_LABELS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_THRESHOLD_PERCENTILE,
    DEFAULT_TOP_K,
)
from pdf_parser import extract_text, get_page_count
from chunkers import chunk_fixed_size, chunk_recursive, chunk_semantic
from embeddings import embed_texts
from vector_store import get_chroma_client, create_collection, add_chunks, reset_client
from rag_pipeline import query_single_strategy, RAGResult


def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        "chat_history": [],
        "pdf_processed": False,
        "pdf_name": None,
        "pdf_text": "",
        "pdf_page_count": 0,
        "chunks": {},  # {strategy_name: list[str]}
        "chunk_counts": {},  # {strategy_name: int}
        "collections": {},  # {strategy_name: ChromaDB Collection}
        "api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_single_strategy(
    strategy_name: str,
    text: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    threshold_pct: int,
    chroma_client,
) -> tuple[str, list[str], int]:
    """
    Process a single chunking strategy: chunk, embed, store.
    Returns (strategy_name, chunks, chunk_count).
    """
    # Create embed function for this API key
    def embed_fn(texts):
        return embed_texts(texts, api_key)

    # Chunk based on strategy
    if strategy_name == "fixed_size":
        chunks = chunk_fixed_size(text, chunk_size, chunk_overlap)
    elif strategy_name == "recursive":
        chunks = chunk_recursive(text, chunk_size, chunk_overlap)
    elif strategy_name == "semantic":
        chunks = chunk_semantic(
            text,
            embed_fn=embed_fn,
            threshold_percentile=threshold_pct,
        )
    else:
        chunks = []

    # Embed chunks
    embeddings = embed_texts(chunks, api_key)

    # Store in ChromaDB
    collection = create_collection(chroma_client, strategy_name)
    add_chunks(collection, chunks, embeddings)

    return strategy_name, chunks, len(chunks), collection


def process_pdf(
    uploaded_file,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    threshold_pct: int,
):
    """
    Process PDF: extract text, run all 3 chunking strategies in parallel,
    embed chunks, and store in ChromaDB.
    """
    # Validate API key
    if not api_key or not api_key.strip():
        st.error("Please enter your OpenAI API key in the sidebar.")
        return False

    if not api_key.startswith("sk-"):
        st.error("Invalid API key format. OpenAI API keys start with 'sk-'.")
        return False

    # Extract text
    text = extract_text(uploaded_file)
    page_count = get_page_count(uploaded_file)

    if not text or len(text.strip()) < 50:
        st.error("PDF appears to be empty or contains no extractable text.")
        return False

    st.session_state.pdf_text = text
    st.session_state.pdf_page_count = page_count

    # Reset ChromaDB client for fresh collections
    reset_client()
    chroma_client = get_chroma_client()

    # Process all 3 strategies in parallel
    chunks = {}
    chunk_counts = {}
    collections = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for strategy_name in STRATEGY_NAMES:
            future = executor.submit(
                process_single_strategy,
                strategy_name,
                text,
                api_key,
                chunk_size,
                chunk_overlap,
                threshold_pct,
                chroma_client,
            )
            futures[future] = strategy_name

        for future in as_completed(futures):
            try:
                name, strategy_chunks, count, collection = future.result()
                chunks[name] = strategy_chunks
                chunk_counts[name] = count
                collections[name] = collection
            except Exception as e:
                st.error(f"Error processing {futures[future]}: {str(e)}")
                return False

    st.session_state.chunks = chunks
    st.session_state.chunk_counts = chunk_counts
    st.session_state.collections = collections
    st.session_state.pdf_processed = True
    st.session_state.pdf_name = uploaded_file.name
    st.session_state.api_key = api_key

    return True


def query_all_strategies(query: str, api_key: str, top_k: int) -> list[RAGResult]:
    """
    Query all 3 strategies in parallel and return results.
    """
    # Embed query once (shared across all strategies)
    query_embedding = embed_texts([query], api_key)[0]

    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for strategy_name in STRATEGY_NAMES:
            collection = st.session_state.collections.get(strategy_name)
            if collection is None:
                continue

            future = executor.submit(
                query_single_strategy,
                query,
                query_embedding,
                collection,
                strategy_name,
                api_key,
                top_k,
            )
            futures[future] = strategy_name

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Create error result
                strategy_name = futures[future]
                results.append(
                    RAGResult(
                        strategy_name=strategy_name,
                        answer=f"Error: {str(e)}",
                        retrieved_chunks=[],
                        distances=[],
                    )
                )

    # Sort results to maintain consistent column order
    order = {name: i for i, name in enumerate(STRATEGY_NAMES)}
    results.sort(key=lambda r: order.get(r.strategy_name, 99))

    return results


def render_results_row(results: list[RAGResult]):
    """Render a row of 3 columns with RAG results."""
    cols = st.columns(3)

    for col, result in zip(cols, results):
        with col:
            st.markdown(f"**{STRATEGY_LABELS.get(result.strategy_name, result.strategy_name)}**")
            st.markdown(result.answer)

            # Expandable section for retrieved chunks
            if result.retrieved_chunks:
                with st.expander(f"Retrieved Chunks ({len(result.retrieved_chunks)})"):
                    for i, (chunk, dist) in enumerate(
                        zip(result.retrieved_chunks, result.distances)
                    ):
                        # Convert distance to similarity (cosine distance to similarity)
                        similarity = 1.0 - dist

                        # Color code based on similarity
                        if similarity > 0.85:
                            color = "green"
                        elif similarity > 0.75:
                            color = "orange"
                        else:
                            color = "red"

                        st.markdown(
                            f"**Chunk {i + 1}** · :{color}[Similarity: {similarity:.3f}]"
                        )

                        # Show truncated chunk text
                        display_text = chunk[:500] + "..." if len(chunk) > 500 else chunk
                        st.code(display_text, language=None)

                        if i < len(result.retrieved_chunks) - 1:
                            st.divider()


def main():
    st.set_page_config(
        page_title="Chunking Strategy Compare Tool",
        layout="wide",
    )
    st.title("Chunking Strategy Compare Tool")

    init_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            help="Required for embeddings and chat completions",
        )

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        st.divider()
        st.subheader("Chunking Parameters")

        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=DEFAULT_CHUNK_OVERLAP,
            step=50,
        )

        threshold_pct = st.slider(
            "Semantic Threshold Percentile",
            min_value=10,
            max_value=50,
            value=DEFAULT_SEMANTIC_THRESHOLD_PERCENTILE,
            step=5,
            help="Lower = more splits. Breakpoints occur where similarity drops below this percentile.",
        )

        top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=1,
            max_value=20,
            value=DEFAULT_TOP_K,
            step=1,
        )

        st.divider()

        # Process PDF if uploaded
        if uploaded_file is not None:
            # Check if this is a new file
            is_new_file = st.session_state.pdf_name != uploaded_file.name

            if is_new_file or not st.session_state.pdf_processed:
                with st.spinner("Processing PDF with 3 chunking strategies..."):
                    success = process_pdf(
                        uploaded_file, api_key, chunk_size, chunk_overlap, threshold_pct
                    )
                    if success:
                        st.session_state.chat_history = []
                        st.rerun()

        # Display PDF info and chunk counts
        if st.session_state.pdf_processed:
            st.success(f"**PDF loaded:** {st.session_state.pdf_name}")
            st.caption(
                f"Pages: {st.session_state.pdf_page_count} | "
                f"Characters: {len(st.session_state.pdf_text):,}"
            )

            st.divider()
            st.subheader("Chunk Counts")

            for name in STRATEGY_NAMES:
                count = st.session_state.chunk_counts.get(name, 0)
                st.metric(
                    label=STRATEGY_LABELS[name],
                    value=count,
                )

            # Reprocess button
            if st.button("Reprocess with current settings"):
                with st.spinner("Reprocessing..."):
                    success = process_pdf(
                        uploaded_file, api_key, chunk_size, chunk_overlap, threshold_pct
                    )
                    if success:
                        st.session_state.chat_history = []
                        st.rerun()
        else:
            st.info(
                "**Chunking Strategy Compare Tool**\n\n"
                "1. Enter your OpenAI API key\n"
                "2. Upload a PDF\n"
                "3. Ask questions and compare strategies"
            )

    # Main area - Chat interface

    # Show instruction if no PDF
    if not st.session_state.pdf_processed:
        st.info("Enter your OpenAI API key and upload a PDF in the sidebar to get started.")

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                render_results_row(msg["content"])

    # Chat input (enabled only after PDF is processed)
    if query := st.chat_input(
        "Ask a question about the PDF...",
        disabled=not st.session_state.pdf_processed,
    ):
        # Show user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Query all strategies and display results
        with st.chat_message("assistant"):
            with st.spinner("Querying all 3 strategies..."):
                results = query_all_strategies(query, api_key, top_k)
            render_results_row(results)

        st.session_state.chat_history.append({"role": "assistant", "content": results})


if __name__ == "__main__":
    main()
