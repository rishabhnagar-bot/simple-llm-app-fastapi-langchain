# app.py
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ── Load .env ───────────────────────────────────────────────────────
load_dotenv()

# ── Config ──────────────────────────────────────────────────────────
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")

# ── Globals ─────────────────────────────────────────────────────────
vector_store: FAISS | None = None
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def get_llm():
    return ChatOpenAI(
        model="grok-3-mini-fast",
        api_key=GROK_API_KEY,
        base_url=GROK_BASE_URL,
        temperature=0.3,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        model="embedding-gecko-001",       # placeholder – see note below
        api_key=GROK_API_KEY,
        base_url=GROK_BASE_URL,
    )
    # NOTE: If Grok doesn't expose an embeddings endpoint yet,
    # swap this for a free local model instead:
    #
    # from langchain_huggingface import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Lifespan (load existing index on startup) ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, get_embeddings(), allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded FAISS index from {FAISS_INDEX_PATH}")
    else:
        print("ℹ️  No existing FAISS index found. Ingest documents first via POST /ingest")
    yield


app = FastAPI(title="RAG API (Grok + FAISS)", lifespan=lifespan)


# ── Schemas ─────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    texts: list[str]          # list of raw text chunks / documents
    chunk_size: int = 500
    chunk_overlap: int = 50

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health", summary="Health check")
async def health():
    return {"status": "healthy"}


# ── Routes ──────────────────────────────────────────────────────────
@app.post("/ingest", summary="Ingest documents into FAISS")
async def ingest(req: IngestRequest):
    global vector_store

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap
    )
    docs = splitter.create_documents(req.texts)

    embeddings = get_embeddings()
    if vector_store is None:
        vector_store = FAISS.from_documents(docs, embeddings)
    else:
        vector_store.add_documents(docs)

    vector_store.save_local(FAISS_INDEX_PATH)
    return {"status": "ok", "chunks_stored": len(docs)}


@app.post("/chat", response_model=ChatResponse, summary="Ask a question (RAG)")
async def chat(req: ChatRequest):
    if vector_store is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet. POST /ingest first.")

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
    )

    result = chain.invoke({"question": req.question})
    sources = [doc.page_content[:200] for doc in result.get("source_documents", [])]
    return ChatResponse(answer=result["answer"], sources=sources)


@app.post("/ask", summary="Plain LLM chat (no RAG)")
async def ask(req: ChatRequest):
    """Talk directly to Grok without retrieval."""
    llm = get_llm()
    response = llm.invoke(req.question)
    return {"answer": response.content}


@app.delete("/reset", summary="Clear vector store & memory")
async def reset():
    global vector_store, memory
    vector_store = None
    memory.clear()
    if os.path.exists(FAISS_INDEX_PATH):
        import shutil
        shutil.rmtree(FAISS_INDEX_PATH)
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)