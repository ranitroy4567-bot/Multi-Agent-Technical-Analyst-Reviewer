"""Streamlit frontend for the CrewAI + Ollama multi-agent analyst."""

import json
import logging
from pathlib import Path

import streamlit as st

from analyst.workflow import run_workflow

logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

st.set_page_config(page_title="CrewAI + Ollama Analyst", layout="wide")
st.title("Multi-Agent Technical Analyst & Reviewer")
st.caption("Local CrewAI + Ollama pipeline with custom tools, Pydantic handoffs, and bounded Critic review.")

with st.sidebar:
    st.header("Run Configuration")
    query = st.text_area(
        "Research Topic or Question",
        value="Compare Redis vs Valkey architecture and licensing",
        height=100,
    )
    model = st.selectbox(
        "Ollama Model",
        options=["ollama/qwen3:8b", "ollama/qwen2.5:7b", "ollama/qwen2.5-coder:7b", "ollama/llama3.1:8b"],
        index=0,
    )
    max_revisions = st.slider("Max Critic Revisions", min_value=0, max_value=3, value=2)
    workspace_dir = st.text_input("Workspace Directory", value=".")
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run_btn:
    if not query.strip():
        st.warning("Please enter a research topic or question.")
    else:
        with st.spinner("Running Researcher -> Drafter <-> Critic workflow..."):
            result = run_workflow(
                query=query.strip(),
                model=model,
                workspace_dir=workspace_dir,
                max_revisions=max_revisions,
            )
            st.session_state.last_result = result

result = st.session_state.last_result

tab_report, tab_trace, tab_history = st.tabs(["Report", "Execution Trace", "Past Reports"])

with tab_report:
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Status", result["status"].upper())
        c2.metric("Critic Revisions", result["revisions"])
        report_path_str = result["save_message"].replace("Saved report to ", "").strip()
        report_path = Path(report_path_str)
        c3.metric("Output File", report_path.name)

        if report_path.is_file():
            md_text = report_path.read_text(encoding="utf-8")
            st.download_button(
                "Download Markdown Report",
                data=md_text,
                file_name=report_path.name,
                mime="text/markdown",
            )
            st.markdown(md_text)
        else:
            st.info(result["save_message"])
    else:
        st.info("Configure your query in the sidebar and click **Run Analysis** to generate a report.")

with tab_trace:
    if result and Path(result["trace_path"]).is_file():
        lines = Path(result["trace_path"]).read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines if line.strip()]
        st.dataframe(events, use_container_width=True)
    else:
        st.info("Trace events will appear here after a run completes.")

with tab_history:
    outputs_dir = Path("outputs")
    if outputs_dir.is_dir():
        files = sorted(outputs_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            chosen = st.selectbox("Select a saved report", options=files, format_func=lambda p: p.name)
            if chosen:
                st.markdown(chosen.read_text(encoding="utf-8"))
        else:
            st.info("No saved reports in `outputs/` yet.")
    else:
        st.info("No `outputs/` directory found yet.")
