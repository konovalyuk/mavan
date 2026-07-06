#!/usr/bin/env python3
"""MCP server exposing domain decision tools. Run: python scripts/run_mcp_server.py"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from app.quality.assessor import assess
from app.decision.forecast import run_forecast
from app.domain_model.inference import predict

mcp = FastMCP("mavan")


@mcp.tool()
def query_domain_model(domain_id: str, state: str, action: str) -> str:
    """Predict terminal outcome state probabilities for (state, action)."""
    return json.dumps(predict(domain_id, state, action))


@mcp.tool()
async def assess_quality(text: str) -> str:
    """Run multi-LLM quality assessment on text."""
    return json.dumps(await assess(text))


@mcp.tool()
async def run_forecast_tool(domain_id: str, context_state: str, actions_json: str) -> str:
    """Forecast outcomes. actions_json: JSON list of candidate actions."""
    actions = json.loads(actions_json)
    return json.dumps(await run_forecast(domain_id, context_state, actions))


if __name__ == "__main__":
    mcp.run()
