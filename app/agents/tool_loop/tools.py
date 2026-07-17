from app.agents.helpers import chunks_to_sources
from app.core.guardrails import AGENT_PYTHON_TOOL
from app.rag.pipeline import get_default_pipeline
from app.services.prompts import format_context
from app.rag.types import RetrievedChunk

SEARCH_NOTES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": "Search project notes/knowledge base. Use when you need factual context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of chunks", "default": 5},
            },
            "required": ["query"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute a small Python expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information via Google.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


async def search_notes(question: str, *, top_k: int = 5, source_prefixes: tuple[str, ...] | None = None) -> tuple[str, list[RetrievedChunk]]:
    chunks = await get_default_pipeline().retrieve_chunks(question, top_k=top_k, source_prefixes=source_prefixes)
    text = format_context(chunks) if chunks else "(no context provided)"
    return text, chunks


async def execute_tool(name: str, args: dict, source_prefixes: tuple[str, ...] | None = None) -> tuple[str, dict]:
    if name == "search_notes":
        text, sources = await search_notes(args["query"], top_k=args.get("top_k", 5), source_prefixes = source_prefixes)
        return text, {"sources": chunks_to_sources(sources)}
    if name == "web_search":
        from app.agents.adk.search_agent import adk_agent
        result = await adk_agent.run(args["query"])
        return result.answer, {"sources": result.sources}
    if name == "run_python":
        if not AGENT_PYTHON_TOOL:
            return "run_python disabled", {"disabled": True}
        result = str(eval(args["code"], {"__builtins__": {}}, {}))
        return result, {"code": args["code"]}
    raise ValueError(f"Unknown tool: {name}")
