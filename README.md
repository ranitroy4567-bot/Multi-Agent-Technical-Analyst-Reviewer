# CrewAI + Ollama Multi-Agent Analyst

A local-first multi-agent technical analyst and reviewer implementing:
1. **Researcher Agent** with custom Pydantic-typed tools (`web_docs_search`, `read_local_file`).
2. **Structured Handoff** via a strict `ResearchBrief` Pydantic schema.
3. **Drafter Agent** that synthesizes Markdown reports and patches flagged sections.
4. **Critic Agent** with a deterministic, bounded revision loop (`max_revisions=2`) via `CritiqueResult`.
5. **JSONL Observability** logging every phase, handoff, and duration to `traces/`.

## Quickstart

1. Install and start [Ollama](https://ollama.com), then pull a tool-calling model:
   ```bash
   ollama pull qwen2.5:7b
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the unit tests (no LLM required):
   ```bash
   pytest
   ```

4. Run the workflow:
   ```bash
   PYTHONPATH=src python -m analyst.cli "Compare Redis vs Valkey architecture and licensing" --workspace .

5. streamlit run src/ui.py  

   ```
