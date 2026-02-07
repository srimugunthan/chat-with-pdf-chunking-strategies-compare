# Chunking Strategy Compare Tool

Upload a PDF and chat with it using 3 different chunking strategies side-by-side. Compare Fixed-Size, Recursive, and Semantic chunking to see which retrieval approach works best for your document.

## Setup

### Install dependencies with uv

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### ChromaDB setup

The app uses ChromaDB as the vector store. By default it runs in **persistent mode**, storing data in a `.chroma_data/` directory inside the project. No external server is needed.

**Persistent mode (default):**

No extra setup required. Collections are saved to disk and survive app restarts. To change the storage directory, edit `CHROMA_PERSIST_DIR` in `config.py`.

**In-memory mode:**

Set `CHROMA_PERSIST_DIR = None` in `config.py`. Collections live only for the duration of the Streamlit session and are discarded on restart.

**Client-server mode (optional):**

If you prefer running ChromaDB as a standalone server:

```bash
# Install the server package
uv pip install chromadb[server]

# Start the server
chroma run --path ./chroma_server_data --port 8000
```

Then update `config.py`:

```python
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
```

And use `chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)` instead of the default client in `vector_store.py`.

### Set your OpenAI API key

You can either enter it in the app sidebar or export it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

## Run the app

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.
