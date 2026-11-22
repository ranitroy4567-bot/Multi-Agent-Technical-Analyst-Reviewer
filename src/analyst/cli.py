"""CLI entrypoint for the CrewAI + Ollama multi-agent analyst."""

import argparse
import json

from analyst.workflow import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CrewAI + Ollama multi-agent analyst.")
    parser.add_argument("query", help="Technical topic or question to analyze.")
    parser.add_argument("--model", default="ollama/qwen3:8b", help="Ollama model name.")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL.")
    parser.add_argument("--workspace", default=".", help="Workspace directory for LocalFileReaderTool.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for final Markdown reports.")
    parser.add_argument("--trace-dir", default="traces", help="Directory for JSONL trace logs.")
    parser.add_argument("--max-revisions", type=int, default=2, help="Max Critic revision loops.")
    args = parser.parse_args()

    result = run_workflow(
        query=args.query,
        model=args.model,
        base_url=args.base_url,
        workspace_dir=args.workspace,
        output_dir=args.output_dir,
        trace_dir=args.trace_dir,
        max_revisions=args.max_revisions,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
