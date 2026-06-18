import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Cross Encoder for Reranking
from sentence_transformers import CrossEncoder

load_dotenv()

app = FastAPI(title="Production RAG System API")

# Setup models and vector store
VECTOR_STORE_DIR = "faiss_index"

# 1. Embeddings model: BGE-M3 (State of the art multilingual/dense)
print("Loading BGE-M3 Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 2. Reranker: BGE Reranker
print("Loading BGE Cross-Encoder Reranker...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# 3. LLM Generator
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

vectorstore = None

# Sample data initialization
def initialize_vector_store():
    global vectorstore
    if os.path.exists(VECTOR_STORE_DIR):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index from sample data...")
        # Create a dummy knowledge base file
        sample_kb_path = "knowledge_base.txt"
        if not os.path.exists(sample_kb_path):
            with open(sample_kb_path, "w") as f:
                f.write("Machine learning is a subset of AI that involves training models on data.\n")
                f.write("LangChain is a framework for developing applications powered by language models.\n")
                f.write("FAISS is a library for efficient similarity search and clustering of dense vectors.\n")
                f.write("RAG stands for Retrieval-Augmented Generation, combining retrieval and generation.\n")
                f.write("Cross-encoder reranking significantly reduces irrelevant context retrieval.\n")
        
        loader = TextLoader(sample_kb_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        splits = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(VECTOR_STORE_DIR)
        print("FAISS index created and saved.")

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_vector_store()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank_top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    start_time = time.time()
    
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    
    # 1. Retrieve initial broad set of documents (dense retrieval)
    retriever = vectorstore.as_retriever(search_kwargs={"k": request.top_k})
    initial_docs = retriever.invoke(request.query)
    
    # 2. Rerank using Cross-Encoder
    pairs = [[request.query, doc.page_content] for doc in initial_docs]
    scores = reranker.predict(pairs)
    
    # Sort docs by reranker score descending
    scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in scored_docs[:request.rerank_top_k]]
    
    contexts = [doc.page_content for doc in reranked_docs]
    context_str = "\n\n".join(contexts)
    
    # 3. Generate Answer
    prompt_template = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )
    
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"context": context_str, "question": request.query})
    
    latency = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=answer,
        contexts=contexts,
        latency_ms=latency
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
