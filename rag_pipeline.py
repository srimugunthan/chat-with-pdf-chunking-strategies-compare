from dataclasses import dataclass
from openai import OpenAI
from config import CHAT_MODEL, SYSTEM_PROMPT
from vector_store import query_collection


@dataclass
class RAGResult:
    """Holds the output of a single RAG query against one strategy."""

    strategy_name: str
    answer: str
    retrieved_chunks: list[str]
    distances: list[float]


def generate_answer(
    query: str,
    retrieved_chunks: list[str],
    api_key: str,
    model: str = CHAT_MODEL,
) -> str:
    """
    Generate an answer using retrieved context.

    Args:
        query: User's question
        retrieved_chunks: List of relevant text chunks
        api_key: OpenAI API key
        model: Chat model to use

    Returns:
        Generated answer string
    """
    # Format context from retrieved chunks
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Build prompt using system template
    prompt = SYSTEM_PROMPT.format(context=context, question=query)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content


def query_single_strategy(
    query: str,
    query_embedding: list[float],
    collection,
    strategy_name: str,
    api_key: str,
    top_k: int = 5,
) -> RAGResult:
    """
    Full RAG pipeline for one strategy.

    1. Retrieve top-k chunks from the collection
    2. Generate answer using retrieved context
    3. Return RAGResult with answer and retrieved chunks

    Args:
        query: User's question
        query_embedding: Embedded query vector
        collection: ChromaDB collection to query
        strategy_name: Name of the chunking strategy
        api_key: OpenAI API key
        top_k: Number of chunks to retrieve

    Returns:
        RAGResult containing answer and retrieved chunks
    """
    # Retrieve chunks
    documents, distances = query_collection(collection, query_embedding, top_k)

    # Generate answer
    answer = generate_answer(query, documents, api_key)

    return RAGResult(
        strategy_name=strategy_name,
        answer=answer,
        retrieved_chunks=documents,
        distances=distances,
    )
