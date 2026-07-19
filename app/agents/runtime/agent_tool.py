from __future__ import annotations

from typing import Any

from app.agents.runtime.loop import run_agent
from app.agents.runtime.types import Agent, ToolExecutor


def tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def agent_as_tool_schema(agent_name: str, description: str) -> dict[str, Any]:
    """Expose a specialist LlmAgent as a tool for an LLM coordinator."""
    return tool_schema(
        agent_name,
        description,
        {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Concrete task for this specialist agent",
                },
            },
            "required": ["task"],
        },
    )


def make_agent_tool(agent: Agent, specialist_execute: ToolExecutor) -> ToolExecutor:
    """When name==agent.name, run nested LlmAgent loop with that agent's provider/model."""

    async def execute(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if name != agent.name:
            raise KeyError(name)
        task = str(args.get("task") or "").strip() or "Proceed toward your goal."
        result = await run_agent(agent, task, specialist_execute)
        return result.answer or "(no text)", {
            "agent": agent.name,
            "provider": agent.provider,
            "rounds": result.rounds,
            "tool_log": result.tool_log,
        }

    return execute
