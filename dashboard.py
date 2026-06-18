import streamlit as st
import requests
import pandas as pd
import time
from datasets import Dataset
import os
from dotenv import load_dotenv

# RAGAS metrics
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

load_dotenv()

# We need an OpenAI API key for RAGAS LLM-as-judge
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY not found in environment. RAGAS evaluation may fail.")

st.set_page_config(page_title="RAGAS Evaluation Dashboard", layout="wide")

st.title("📊 Production RAG System Dashboard")
st.markdown("Monitor Retrieval-Augmented Generation quality with **RAGAS** (Faithfulness & Answer Relevancy).")

API_URL = "http://localhost:8000/query"

# State to store history
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your query:", "What is LangChain?")

col1, col2 = st.columns([1, 1])

if st.button("Run Pipeline & Evaluate"):
    with st.spinner("Querying FastAPI Backend..."):
        try:
            # Call backend
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
        st.write("**Answer:**")
        st.info(answer)
        st.write(f"**Latency (P95 target <1800ms):** {latency:.2f} ms")
        
        st.write("**Retrieved & Reranked Contexts:**")
        for i, ctx in enumerate(contexts):
            st.caption(f"Context {i+1}: {ctx}")
            
    with col2:
        st.subheader("RAGAS Evaluation Metrics")
        with st.spinner("Evaluating using RAGAS (LLM-as-a-judge)..."):
            try:
                # Prepare dataset for RAGAS
                eval_data = {
                    "question": [query],
                    "answer": [answer],
                    "contexts": [contexts]
                }
                dataset = Dataset.from_dict(eval_data)
                
                # Run evaluation
                # Note: We only run 2 metrics here for speed, can add context_precision/recall later
                result = evaluate(
                    dataset,
                    metrics=[faithfulness, answer_relevancy],
                )
                
                eval_scores = result.to_pandas().iloc[0]
                
                f_score = eval_scores.get('faithfulness', 0.0)
                ar_score = eval_scores.get('answer_relevancy', 0.0)
                
                st.metric(label="Faithfulness (Target: 0.91)", value=f"{f_score:.2f}")
                st.metric(label="Answer Relevancy (Target: 0.88)", value=f"{ar_score:.2f}")
                
                # Save to history
                st.session_state.history.append({
                    "Query": query,
                    "Faithfulness": f_score,
                    "Relevancy": ar_score,
                    "Latency (ms)": latency
                })
                
            except Exception as e:
                st.error(f"Evaluation error (Check OpenAI API Key): {e}")

st.divider()
st.subheader("Query Evaluation History")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("No queries evaluated yet.")
