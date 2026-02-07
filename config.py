# Model settings
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 200  # characters
DEFAULT_SEMANTIC_THRESHOLD_PERCENTILE = 25  # lower = more splits

# Retrieval
DEFAULT_TOP_K = 5

# ChromaDB settings
CHROMA_PERSIST_DIR = ".chroma_data"  # directory for persistent storage (set to None for in-memory)
CHROMA_DISTANCE_FN = "cosine"  # distance function: "cosine", "l2", or "ip" (inner product)
CHROMA_COLLECTION_PREFIX = "chunking_tool"  # prefix for collection names
CHROMA_BATCH_SIZE = 5000  # max documents per add() call (ChromaDB limit is 5461)

# Strategy names (used as collection name suffixes and display labels)
STRATEGY_NAMES = ["fixed_size", "recursive", "semantic"]
STRATEGY_LABELS = {
    "fixed_size": "Fixed-Size Chunking",
    "recursive": "Recursive Character Splitting",
    "semantic": "Semantic Chunking",
}

# System prompt template
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question based ONLY on the following context extracted from a PDF document. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
