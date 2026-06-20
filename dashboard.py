import streamlit as st
import requests
import pandas as pd
import os
from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not set. RAGAS evaluation will fail.")

# Gemini LLM + embeddings for RAGAS
gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
ragas_llm = LangchainLLMWrapper(gemini_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

st.set_page_config(page_title="RAGAS Evaluation Dashboard", layout="wide")
st.title("📊 Production RAG System — RAGAS Dashboard")
st.markdown("Monitoring **Faithfulness** and **Answer Relevancy** via Gemini-powered RAGAS evaluation.")

API_URL = "http://localhost:8000/query"

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your query:", "What is LangChain?")
col1, col2 = st.columns([1, 1])

if st.button("Run Pipeline & Evaluate"):
    with st.spinner("Querying FastAPI Backend..."):
        try:
            response = requests.post(API_URL, json={"query": query, "top_k": 5, "rerank_top_k": 3})
            response.raise_for_status()
            data = response.json()
            answer = data["answer"]
            contexts = data["contexts"]
            latency = data["latency_ms"]
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()

    with col1:
        st.subheader("RAG Output")
        st.info(answer)
        st.write(f"**Latency (P95 target <1800ms):** {latency:.2f} ms")
        for i, ctx in enumerate(contexts):
            st.caption(f"Context {i+1}: {ctx}")

    with col2:
        st.subheader("RAGAS Metrics (Gemini Judge)")
        with st.spinner("Evaluating with RAGAS + Gemini..."):
            try:
                eval_data = {"question": [query], "answer": [answer], "contexts": [contexts]}
                dataset = Dataset.from_dict(eval_data)
                result = evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                )
                eval_scores = result.to_pandas().iloc[0]
                f_score = eval_scores.get("faithfulness", 0.0)
                ar_score = eval_scores.get("answer_relevancy", 0.0)
                st.metric("Faithfulness (Target: 0.91)", f"{f_score:.2f}")
                st.metric("Answer Relevancy (Target: 0.88)", f"{ar_score:.2f}")
                st.session_state.history.append({
                    "Query": query, "Faithfulness": f_score,
                    "Relevancy": ar_score, "Latency (ms)": latency
                })
            except Exception as e:
                st.error(f"RAGAS error: {e}")

st.divider()
st.subheader("Query History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
else:
    st.info("No queries evaluated yet.")
