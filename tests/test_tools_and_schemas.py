"""Unit tests for schemas and custom tools (runs offline without Ollama)."""

import tempfile
import unittest
from pathlib import Path

from analyst.schemas import CritiqueResult, Finding, ResearchBrief
from analyst.tools import LocalFileReaderTool, SaveMarkdownReportTool


class TestSchemasAndTools(unittest.TestCase):
    def test_research_brief_schema(self) -> None:
        brief = ResearchBrief(
            topic="Test Topic",
            findings=[Finding(claim="Fact A", source="README.md:1-5", confidence="high")],
            open_gaps=[],
        )
        self.assertEqual(len(brief.findings), 1)
        self.assertEqual(brief.findings[0].confidence, "high")

    def test_critique_result_bounds(self) -> None:
        res = CritiqueResult(approved=True, score=9, required_fixes=[])
        self.assertTrue(res.approved)

    def test_local_file_reader_blocks_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.txt").write_text("line1\nline2\n", encoding="utf-8")
            tool = LocalFileReaderTool(workspace_root=root)

            ok = tool._run("sample.txt", start_line=1, max_lines=5)
            self.assertIn("1: line1", ok)

            blocked = tool._run("../etc/passwd")
            self.assertIn("access denied", blocked)

    def test_save_markdown_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs"
            tool = SaveMarkdownReportTool(output_dir=out_dir)
            msg = tool._run("demo_report", "# Demo\nContent")
            self.assertIn("Saved report to", msg)
            self.assertTrue((out_dir / "demo_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
