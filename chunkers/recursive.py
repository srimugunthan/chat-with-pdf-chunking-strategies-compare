from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_recursive(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text using LangChain's recursive character splitter.

    Tries separators in order: paragraphs, newlines, sentences, words.
    Prefers splitting at natural text boundaries.

    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunk strings
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)
