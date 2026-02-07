import chromadb
from chromadb.api.models.Collection import Collection
from config import CHROMA_PERSIST_DIR, CHROMA_DISTANCE_FN, CHROMA_COLLECTION_PREFIX


_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """
    Return a ChromaDB client (persistent or in-memory based on config).
    Client is cached for reuse.
    """
    global _client
    if _client is None:
        if CHROMA_PERSIST_DIR:
            _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        else:
            _client = chromadb.Client()
    return _client


def reset_client():
    """Reset the cached client (useful when switching PDFs)."""
    global _client
    _client = None


def create_collection(client: chromadb.ClientAPI, strategy_name: str) -> Collection:
    """
    Create or replace a collection for a chunking strategy.

    Args:
        client: ChromaDB client
        strategy_name: Name of the chunking strategy (used in collection name)

    Returns:
        ChromaDB Collection object
    """
    name = f"{CHROMA_COLLECTION_PREFIX}_{strategy_name}"

    # Delete existing collection if it exists
    try:
        client.delete_collection(name)
    except Exception:
        pass

    # Create new collection with cosine distance
    return client.create_collection(
        name=name,
        metadata={"hnsw:space": CHROMA_DISTANCE_FN},
    )


def add_chunks(
    collection: Collection,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """
    Add chunks with their embeddings to a collection.

    Args:
        collection: ChromaDB collection
        chunks: List of text chunks
        embeddings: List of embedding vectors (must match chunks length)
    """
    if not chunks:
        return

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
    )


def query_collection(
    collection: Collection,
    query_embedding: list[float],
    top_k: int = 5,
) -> tuple[list[str], list[float]]:
    """
    Query collection with an embedding vector.

    Args:
        collection: ChromaDB collection to query
        query_embedding: Query embedding vector
        top_k: Number of results to return

    Returns:
        Tuple of (documents, distances)
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    return documents, distances
