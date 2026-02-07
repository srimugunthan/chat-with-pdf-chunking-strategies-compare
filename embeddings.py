import time
from openai import OpenAI, RateLimitError
from config import EMBEDDING_MODEL


_client: OpenAI | None = None


def get_client(api_key: str) -> OpenAI:
    """Return a cached OpenAI client."""
    global _client
    if _client is None or _client.api_key != api_key:
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(
    texts: list[str],
    api_key: str,
    model: str = EMBEDDING_MODEL,
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Embed texts using OpenAI API with batching and retry.

    Args:
        texts: List of text strings to embed
        api_key: OpenAI API key
        model: Embedding model name
        batch_size: Number of texts per API call (max ~8191 tokens per text)

    Returns:
        List of embedding vectors (list of floats)

    Raises:
        RuntimeError: If API fails after 3 retries
    """
    if not texts:
        return []

    client = get_client(api_key)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Retry logic with exponential backoff
        for attempt in range(3):
            try:
                response = client.embeddings.create(input=batch, model=model)
                all_embeddings.extend([item.embedding for item in response.data])
                break
            except RateLimitError:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        else:
            raise RuntimeError(
                f"Embedding API failed after 3 retries for batch starting at index {i}"
            )

    return all_embeddings
