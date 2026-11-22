"""Custom CrewAI tools with Pydantic input schemas and safe error handling."""

from pathlib import Path
from typing import Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Concise search query for Wikipedia/DuckDuckGo summary lookup.")


class WebDocsSearchTool(BaseTool):
    name: str = "web_docs_search"
    description: str = (
        "Fetch concise factual summaries and reference URLs for a technical topic "
        "using the DuckDuckGo Instant Answer API."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            resp = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            abstract = data.get("AbstractText") or ""
            abstract_url = data.get("AbstractURL") or ""
            related = [
                f"- {t.get('Text', '')} ({t.get('FirstURL', '')})"
                for t in data.get("RelatedTopics", [])[:5]
                if isinstance(t, dict) and t.get("Text")
            ]
            if not abstract and not related:
                return f"No instant summary found for '{query}'. Try broader keywords."
            parts = []
            if abstract:
                parts.append(f"Summary ({abstract_url}): {abstract}")
            if related:
                parts.append("Related:\n" + "\n".join(related))
            return "\n\n".join(parts)
        except Exception as exc:
            return f"Tool error in web_docs_search: {exc}"


class ReadFileInput(BaseModel):
    relative_path: str = Field(..., description="Path relative to the configured workspace root.")
    start_line: int = Field(default=1, ge=1, description="1-indexed start line.")
    max_lines: int = Field(default=120, ge=1, le=300, description="Maximum lines to read.")


class LocalFileReaderTool(BaseTool):
    name: str = "read_local_file"
    description: str = (
        "Safely read line-numbered text from a file inside the allowed workspace directory."
    )
    args_schema: Type[BaseModel] = ReadFileInput
    workspace_root: Path = Path(".")

    def _run(self, relative_path: str, start_line: int = 1, max_lines: int = 120) -> str:
        try:
            root = self.workspace_root.resolve()
            target = (root / relative_path).resolve()
            if not target.is_relative_to(root):
                return f"Error: access denied outside workspace for '{relative_path}'."
            if not target.is_file():
                return f"Error: file not found at '{relative_path}'."
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            end_line = min(len(lines), start_line - 1 + max_lines)
            snippet = [
                f"{idx}: {lines[idx - 1]}"
                for idx in range(start_line, end_line + 1)
            ]
            if not snippet:
                return f"File '{relative_path}' has {len(lines)} lines; start_line={start_line} is past EOF."
            return f"File: {relative_path} (lines {start_line}-{end_line} of {len(lines)})\n" + "\n".join(snippet)
        except Exception as exc:
            return f"Tool error in read_local_file: {exc}"


class SaveReportInput(BaseModel):
    filename: str = Field(..., description="Output Markdown filename (e.g., 'report.md').")
    content: str = Field(..., description="Complete Markdown report content.")


class SaveMarkdownReportTool(BaseTool):
    name: str = "save_markdown_report"
    description: str = "Save a validated Markdown report to the outputs directory."
    args_schema: Type[BaseModel] = SaveReportInput
    output_dir: Path = Path("outputs")

    def _run(self, filename: str, content: str) -> str:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(filename).name
            if not safe_name.endswith(".md"):
                safe_name += ".md"
            dest = (self.output_dir / safe_name).resolve()
            dest.write_text(content, encoding="utf-8")
            return f"Saved report to {dest}"
        except Exception as exc:
            return f"Tool error in save_markdown_report: {exc}"
