import re
import numpy as np
from typing import Callable


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling common abbreviations.

    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []

    # Protect common abbreviations from being split
    protected = text
    abbreviations = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs", "etc", "Inc", "Ltd", "Corp", "Fig", "Vol", "No"]
    for abbr in abbreviations:
        protected = re.sub(rf"\b{abbr}\.", f"{abbr}<DOT>", protected)

    # Also protect decimal numbers (e.g., "3.14")
    protected = re.sub(r"(\d)\.", r"\1<DOT>", protected)

    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", protected)

    # Restore protected dots and clean up
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences]

    return [s for s in sentences if s]


def compute_breakpoints(
    embeddings: list[list[float]],
    threshold_percentile: int = 25,
) -> list[int]:
    """
    Find indices where semantic similarity drops below threshold.

    Computes cosine similarity between consecutive embeddings and marks
    breakpoints where similarity drops below the given percentile.

    Args:
        embeddings: List of embedding vectors
        threshold_percentile: Percentile threshold (lower = more splits)

    Returns:
        List of breakpoint indices (where new chunks should start)
    """
    if len(embeddings) < 2:
        return []

    # Compute cosine similarities between consecutive embeddings
    similarities = []
    for i in range(len(embeddings) - 1):
        a = np.array(embeddings[i])
        b = np.array(embeddings[i + 1])

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            sim = 0.0
        else:
            sim = np.dot(a, b) / (norm_a * norm_b)
        similarities.append(sim)

    if not similarities:
        return []

    # Find breakpoints below percentile threshold
    threshold = np.percentile(similarities, threshold_percentile)
    breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

    return breakpoints


def chunk_semantic(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    threshold_percentile: int = 25,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
) -> list[str]:
    """
    Semantic chunking: split by topic boundaries detected via embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Embed all sentences
    3. Compute cosine similarity between consecutive sentences
    4. Find breakpoints where similarity drops below percentile threshold
    5. Group sentences between breakpoints into chunks
    6. Post-process: merge tiny chunks, split oversized chunks

    Args:
        text: Input text to chunk
        embed_fn: Function that takes list[str] and returns list[list[float]]
        threshold_percentile: Lower = more splits (default 25)
        min_chunk_size: Merge chunks smaller than this
        max_chunk_size: Split chunks larger than this

    Returns:
        List of chunk strings
    """
    sentences = split_into_sentences(text)

    if len(sentences) == 0:
        return []

    if len(sentences) == 1:
        return [text.strip()] if text.strip() else []

    # Cap sentences to avoid excessive API costs
    max_sentences = 1000
    truncated = False
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
        truncated = True

    # Embed all sentences
    embeddings = embed_fn(sentences)

    # Find breakpoints
    breakpoints = compute_breakpoints(embeddings, threshold_percentile)

    # Group sentences into chunks
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk = " ".join(sentences[start:bp])
        if chunk:
            chunks.append(chunk)
        start = bp

    # Last chunk
    if start < len(sentences):
        chunk = " ".join(sentences[start:])
        if chunk:
            chunks.append(chunk)

    # If no breakpoints found, return all sentences as one chunk
    if not chunks:
        chunks = [" ".join(sentences)]

    # Post-process: merge tiny chunks
    merged = []
    buffer = ""
    for chunk in chunks:
        if buffer:
            buffer = buffer + " " + chunk
        else:
            buffer = chunk

        if len(buffer) >= min_chunk_size:
            merged.append(buffer)
            buffer = ""

    # Handle remaining buffer
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    # Split oversized chunks
    final = []
    for chunk in merged:
        if len(chunk) <= max_chunk_size:
            final.append(chunk)
        else:
            # Simple split by max_chunk_size
            for i in range(0, len(chunk), max_chunk_size):
                part = chunk[i:i + max_chunk_size].strip()
                if part:
                    final.append(part)

    return final
