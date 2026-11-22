"""Deterministic multi-agent workflow with bounded Critic revision loop."""

import json
import re
from pathlib import Path

from crewai import Crew, Process, Task

from analyst.agents import (
    build_ollama_llm,
    create_critic,
    create_drafter,
    create_researcher,
)
from analyst.schemas import CritiqueResult, ResearchBrief
from analyst.tools import LocalFileReaderTool, SaveMarkdownReportTool, WebDocsSearchTool
from analyst.trace import TraceLogger


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug[:48] or "technical_report") + ".md"


def run_workflow(
    query: str,
    model: str = "ollama/qwen3:8b",
    base_url: str = "http://localhost:11434",
    workspace_dir: str = ".",
    output_dir: str = "outputs",
    trace_dir: str = "traces",
    max_revisions: int = 2,
) -> dict:
    logger = TraceLogger(trace_dir=trace_dir)
    logger.log("init", "started", query=query, model=model, max_revisions=max_revisions)

    # LLMs: low temperature for tool calling & critique, slightly warmer for drafting
    strict_llm = build_ollama_llm(model=model, base_url=base_url, temperature=0.1)
    draft_llm = build_ollama_llm(model=model, base_url=base_url, temperature=0.3)

    # Tools
    search_tool = WebDocsSearchTool()
    file_tool = LocalFileReaderTool(workspace_root=Path(workspace_dir))
    save_tool = SaveMarkdownReportTool(output_dir=Path(output_dir))

    # Agents
    researcher = create_researcher(strict_llm, tools=[search_tool, file_tool])
    drafter = create_drafter(draft_llm)
    critic = create_critic(strict_llm)

    # Phase 2: Researcher
    research_task = Task(
        description=(
            f"Research the following request using your tools: '{query}'. "
            "Collect concrete evidence with exact source URLs or file paths, and list any open gaps."
        ),
        expected_output="A valid ResearchBrief JSON object.",
        agent=researcher,
        output_pydantic=ResearchBrief,
    )
    research_crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )
    research_out = research_crew.kickoff()
    brief: ResearchBrief = research_out.pydantic or ResearchBrief(
        topic=query,
        findings=[],
        open_gaps=["Researcher did not return structured JSON; raw output used."],
    )
    logger.log("researcher", "completed", findings=len(brief.findings), open_gaps=len(brief.open_gaps))

    # Phase 3 & 4: Bounded Drafter <-> Critic loop
    revision_count = 0
    draft_markdown = ""
    last_critique: CritiqueResult | None = None

    while True:
        if revision_count == 0:
            draft_desc = (
                "Write a structured Markdown report answering the user's request strictly from this ResearchBrief:\n"
                f"{brief.model_dump_json(indent=2)}\n\n"
                "Include a brief Executive Summary, Key Findings (with inline sources), and Open Gaps."
            )
        else:
            fixes = json.dumps(last_critique.required_fixes if last_critique else [], indent=2)
            draft_desc = (
                "Revise the draft below to resolve every required fix from the Critic without inventing facts.\n\n"
                f"ResearchBrief:\n{brief.model_dump_json(indent=2)}\n\n"
                f"Current Draft:\n{draft_markdown}\n\n"
                f"Required Fixes:\n{fixes}"
            )

        draft_task = Task(
            description=draft_desc,
            expected_output="A clean, well-structured Markdown report.",
            agent=drafter,
        )
        draft_out = Crew(agents=[drafter], tasks=[draft_task], verbose=True).kickoff()
        draft_markdown = str(draft_out).strip()
        logger.log("drafter", "completed", revision=revision_count, chars=len(draft_markdown))

        critique_task = Task(
            description=(
                "Audit the Draft Report against the ResearchBrief on three criteria: "
                "1) Grounding (every claim cites a ResearchBrief source), "
                "2) Completeness (open gaps are acknowledged), "
                "3) Structure (valid Markdown headings).\n\n"
                f"ResearchBrief:\n{brief.model_dump_json(indent=2)}\n\n"
                f"Draft Report:\n{draft_markdown}"
            ),
            expected_output="A valid CritiqueResult JSON object.",
            agent=critic,
            output_pydantic=CritiqueResult,
        )
        critique_out = Crew(agents=[critic], tasks=[critique_task], verbose=True).kickoff()
        last_critique = critique_out.pydantic or CritiqueResult(
            approved=False,
            score=5,
            required_fixes=["Critic failed to return structured JSON; verify formatting and citations."],
        )
        logger.log(
            "critic",
            "evaluated",
            revision=revision_count,
            approved=last_critique.approved,
            score=last_critique.score,
        )

        if last_critique.approved:
            final_status = "approved"
            break
        if revision_count >= max_revisions:
            final_status = "capped"
            unresolved = "\n".join(f"- {fix}" for fix in last_critique.required_fixes)
            draft_markdown += f"\n\n## Needs Human Review\n{unresolved}\n"
            break
        revision_count += 1

    # Phase 5: Save report & finalize trace
    filename = _slugify(query)
    save_msg = save_tool._run(filename=filename, content=draft_markdown)
    logger.log("finalizer", final_status, revisions=revision_count, report=save_msg)

    return {
        "status": final_status,
        "revisions": revision_count,
        "save_message": save_msg,
        "trace_path": str(logger.trace_path),
    }
