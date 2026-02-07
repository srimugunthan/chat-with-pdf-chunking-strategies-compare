def chunk_fixed_size(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: Input text to chunk
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunk strings
    """
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap)
        start += chunk_size - chunk_overlap
        # Prevent infinite loop if overlap >= chunk_size
        if chunk_size <= chunk_overlap:
            start += 1

    return chunks
