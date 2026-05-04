"""
Streamlit chat UI for the Local Wikipedia RAG Assistant.

Run:
    streamlit run app_streamlit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src import embedder, vectorstore
from src.rag import answer

st.set_page_config(page_title="Local Wiki RAG", page_icon="📚", layout="wide")
st.title("📚 Local Wikipedia RAG Assistant")
st.caption("Runs entirely on your machine via Ollama. No external APIs.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 5)
    show_sources = st.checkbox("Show retrieved sources", value=True)
    st.divider()

    if embedder.ping():
        st.success("Ollama is running")
    else:
        st.error("Ollama not reachable on " + embedder.OLLAMA_HOST)

    try:
        stats = vectorstore.stats()
        st.metric("Total chunks", stats["total_chunks"])
        st.write("By type:", stats["by_type"])
    except Exception as e:
        st.warning(f"Index not ready: {e}")

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state["history"] = []
        st.rerun()
    if st.button("⚠️ Reset vector store"):
        vectorstore.reset()
        st.session_state["history"] = []
        st.success("Vector store wiped. Re-run ingest + build_index.")

# --- Chat history ---
if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for i, c in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {c['type']} — {c['title']}**  _(distance={c['distance']:.3f})_"
                    )
                    st.write(c["text"])
        if msg.get("meta"):
            st.caption(msg["meta"])

q = st.chat_input("Ask about a famous person or place…")
if q:
    st.session_state["history"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                res = answer(q, top_k=top_k, show_sources=show_sources)
            except Exception as e:
                st.error(str(e))
                st.stop()
        st.markdown(res.answer)
        meta = (
            f"route: {res.route.types} · "
            f"latency: {res.latency_ms['total']} ms "
            f"(embed {res.latency_ms['embed']} / retrieve {res.latency_ms['retrieve']} / "
            f"generate {res.latency_ms['generate']})"
        )
        st.caption(meta)
        if show_sources and res.contexts:
            with st.expander("Sources"):
                for i, c in enumerate(res.contexts, 1):
                    st.markdown(
                        f"**[{i}] {c['type']} — {c['title']}**  _(distance={c['distance']:.3f})_"
                    )
                    st.write(c["text"])
        st.session_state["history"].append(
            {
                "role": "assistant",
                "content": res.answer,
                "sources": res.contexts if show_sources else [],
                "meta": meta,
            }
        )
