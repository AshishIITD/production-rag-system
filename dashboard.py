"""
Streamlit RAGAS Evaluation Dashboard
Run: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import httpx
import json
from evaluator import load_eval_history

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="🔍",
    layout="wide",
)

API_URL = "http://localhost:8000"

# ── Header ──────────────────────────────────────────────────────────────────
st.title("🔍 Production RAG — Evaluation Dashboard")
st.caption("Live RAGAS metrics | Built by Ashish Singh")

# ── Sidebar: Query Interface ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Ask a Question")
    question = st.text_area("Question", placeholder="What is RAG?", height=100)
    run_eval = st.checkbox("Run RAGAS evaluation", value=False)
    top_k = st.slider("Top-K chunks", 1, 8, 4)
    submit = st.button("Submit", type="primary", use_container_width=True)

    if submit and question.strip():
        with st.spinner("Retrieving and generating..."):
            try:
                resp = httpx.post(f"{API_URL}/query", json={
                    "question": question,
                    "run_evaluation": run_eval,
                    "top_k": top_k,
                }, timeout=60)
                data = resp.json()
                st.success(f"✅ {data['latency_ms']:.0f}ms | {data['model_used']}")
                st.markdown("**Answer:**")
                st.write(data["answer"])
                if data.get("ragas_scores"):
                    sc = data["ragas_scores"]
                    cols = st.columns(2)
                    cols[0].metric("Faithfulness", f"{sc.get('faithfulness', 'N/A'):.2f}" if sc.get('faithfulness') else "N/A")
                    cols[0].metric("Answer Relevancy", f"{sc.get('answer_relevancy', 'N/A'):.2f}" if sc.get('answer_relevancy') else "N/A")
                    cols[1].metric("Context Precision", f"{sc.get('context_precision', 'N/A'):.2f}" if sc.get('context_precision') else "N/A")
                    cols[1].metric("Context Recall", f"{sc.get('context_recall', 'N/A'):.2f}" if sc.get('context_recall') else "N/A")
                with st.expander("Sources"):
                    for src in data.get("sources", []):
                        st.markdown(f"**{src['source']}** (score: {src['score']:.3f})")
                        st.text(src["content"][:300] + "...")
            except Exception as e:
                st.error(f"API error: {e}")

# ── Metrics ───────────────────────────────────────────────────────────────────
history = load_eval_history()

if not history:
    st.info("No evaluation history yet. Submit a query with 'Run RAGAS evaluation' checked.")
    st.stop()

df = pd.DataFrame(history)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Queries", len(df))
c2.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.3f}" if "faithfulness" in df else "N/A")
c3.metric("Avg Ans. Relevancy", f"{df['answer_relevancy'].mean():.3f}" if "answer_relevancy" in df else "N/A")
c4.metric("Avg Latency", f"{df['latency_ms'].mean():.0f}ms")
c5.metric("P95 Latency", f"{df['latency_ms'].quantile(0.95):.0f}ms")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("RAGAS Scores Over Time")
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
    if available:
        fig = px.line(df, x="timestamp", y=available, title="",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(yaxis_range=[0, 1], legend_title="Metric", height=350)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average RAGAS Scores")
    means = {m: df[m].mean() for m in available if m in df.columns}
    fig2 = go.Figure(go.Bar(
        x=list(means.keys()),
        y=list(means.values()),
        marker_color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"],
        text=[f"{v:.3f}" for v in means.values()],
        textposition="auto",
    ))
    fig2.update_layout(yaxis_range=[0, 1], height=350)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Latency Distribution")
fig3 = px.histogram(df, x="latency_ms", nbins=20, color_discrete_sequence=["#3498db"])
fig3.update_layout(height=280)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Query Log")
display_cols = ["timestamp", "question", "latency_ms"] + available
st.dataframe(df[display_cols].sort_values("timestamp", ascending=False).head(50), use_container_width=True)
