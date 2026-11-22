"""CrewAI agent factories configured for local Ollama models."""

from crewai import Agent, LLM


def build_ollama_llm(
    model: str = "ollama/qwen3:8b ",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
) -> LLM:
    return LLM(model=model, base_url=base_url, temperature=temperature)


def create_researcher(llm: LLM, tools: list) -> Agent:
    return Agent(
        role="Technical Researcher",
        goal="Gather verified evidence using tools and return a strict ResearchBrief.",
        backstory=(
            "You are a meticulous technical analyst. You never guess facts. "
            "Every finding must cite a real URL or local file path returned by a tool."
        ),
        tools=tools,
        llm=llm,
        max_iter=6,
        verbose=True,
        allow_delegation=False,
    )


def create_drafter(llm: LLM) -> Agent:
    return Agent(
        role="Technical Drafter",
        goal="Synthesize a clear, well-structured Markdown report strictly from the ResearchBrief.",
        backstory=(
            "You write concise, accurate engineering briefs. You cite sources inline "
            "and patch only flagged sections when given Critic feedback."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_critic(llm: LLM) -> Agent:
    return Agent(
        role="Technical Critic",
        goal="Audit the draft against the ResearchBrief on grounding, completeness, and structure.",
        backstory=(
            "You are a strict reviewer. You reject any claim not backed by the ResearchBrief "
            "and list exact, actionable fixes."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
