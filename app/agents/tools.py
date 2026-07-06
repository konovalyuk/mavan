from app.core.guardrails import AGENT_PYTHON_TOOL
from app.rag.types import RetrievedChunk
from app.services.rag_service import retrieve_chunks

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


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant notes found."
    lines = []
    for chunk in chunks:
        label = chunk.source or "unknown"
        lines.append(f"[{label} | score={chunk.score:.2f}] {chunk.text}")
    return "\n\n".join(lines)


async def search_notes(question: str, *, top_k: int = 5) -> tuple[str, list[RetrievedChunk]]:
    chunks = await retrieve_chunks(question, top_k=top_k)
    return format_chunks(chunks), chunks


async def execute_tool(name: str, args: dict) -> tuple[str, dict]:
    if name == "search_notes":
        text, sources = await search_notes(args["query"], top_k=args.get("top_k", 5))
        return text, {"sources": [c.source for c in sources]}
    if name == "run_python":
        if not AGENT_PYTHON_TOOL:
            return "run_python disabled", {"disabled": True}
        result = str(eval(args["code"], {"__builtins__": {}}, {}))
        return result, {"code": args["code"]}
    raise ValueError(f"Unknown tool: {name}")
