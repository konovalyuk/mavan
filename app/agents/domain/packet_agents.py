"""Stage-1 LlmAgents + PacketCoordinator (atomic packet)."""
from __future__ import annotations

import json

from app.agents.domain.extract import extract_transitions
from app.agents.domain.tools import fetch_url_text, google_search
from app.agents.runtime import Agent, agent_as_tool_schema, make_agent_tool, resolve_provider, tool_schema
from app.agents.runtime.types import SessionState
from app.domain_model.train import train_step
from app.quality.assessor import run_quality_team
from config import domain_settings

RESEARCH, EXTRACT, TRAIN, COORDINATOR = (
    "ResearchAgent", "ExtractAgent", "TrainAgent", "PacketCoordinator",
)

_RESEARCH_TOOLS = [
    tool_schema("google_search", "Search for domain historical state→action→final-state sources.",
                {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    tool_schema("ingest_into_packet", "Fetch URL into the ephemeral atomic packet.",
                {"type": "object", "properties": {"url": {"type": "string"}, "title": {"type": "string"}},
                 "required": ["url"]}),
]

_EXTRACT_TOOLS = [
    tool_schema("extract_sao", "Extract final-outcome S-A-O transitions from the packet.",
                {"type": "object", "properties": {}}),
    tool_schema("reject_with_feedback", "Reject packet; write feedback for the coordinator.",
                {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}),
]

_TRAIN_TOOLS = [
    tool_schema("train_on_micro_batch", "Run online train_step on current S-A-O batch.",
                {"type": "object", "properties": {}}),
]


def _prov(state: SessionState | None, *fallbacks: str | None) -> str:
    return resolve_provider(state.provider_override if state else None, *fallbacks)


def research_agent(state: SessionState | None = None) -> Agent:
    fb = f" Feedback: {state.feedback}" if state and state.feedback else ""
    return Agent(
        name=RESEARCH,
        goal="Build one atomic domain packet with state→action→final-state (enough for quality+train).",
        instruction=(
            "Use google_search and ingest_into_packet until the packet is coherent, then stop."
            f"{fb} Do not assess or train."
        ),
        tools=_RESEARCH_TOOLS,
        provider=_prov(state, domain_settings.RESEARCH_PROVIDER),
        max_rounds=domain_settings.SPECIALIST_AGENT_MAX_ROUNDS,
    )


async def execute_research(state: SessionState, name: str, args: dict) -> tuple[str, dict]:
    if name == "google_search":
        return await google_search(str(args["query"]))
    if name == "ingest_into_packet":
        url = str(args["url"]).strip()
        if url in state.seen_urls:
            return json.dumps({"ok": False, "error": "duplicate"}), {"ok": False}
        text = fetch_url_text(url)
        if not text:
            return json.dumps({"ok": False, "error": "empty"}), {"ok": False}
        state.seen_urls.add(url)
        state.packet_urls.append(url)
        sep = "\n\n---\n\n" if state.packet_text else ""
        state.packet_text = (state.packet_text or "") + sep + text
        return json.dumps({"ok": True, "url": url, "chars": len(state.packet_text)}), {"ok": True}
    raise ValueError(name)


def extract_agent(state: SessionState | None = None) -> Agent:
    return Agent(
        name=EXTRACT,
        goal="Produce a valid micro S-A-O batch (final outcomes) for training.",
        instruction="Call extract_sao. On failure call reject_with_feedback. Prefer final/stable outcomes.",
        tools=_EXTRACT_TOOLS,
        provider=_prov(state, domain_settings.EXTRACT_PROVIDER),
        max_rounds=domain_settings.SPECIALIST_AGENT_MAX_ROUNDS,
    )


def _sao_ok(samples: list[dict]) -> bool:
    return bool(samples) and all(
        s.get("state") and s.get("action") and s.get("outcome_states") for s in samples
    )


async def execute_extract(state: SessionState, name: str, args: dict) -> tuple[str, dict]:
    if name == "extract_sao":
        if not state.packet_text:
            return json.dumps({"ok": False, "error": "empty_packet"}), {"ok": False}
        samples = await extract_transitions(
            state.packet_text,
            float((state.quality or {}).get("avg") or 0),
            provider=_prov(state, domain_settings.EXTRACT_PROVIDER),
        )
        if not _sao_ok(samples):
            state.sao = []
            state.feedback = "extract produced no valid state-action-outcome rows"
            state.metrics["extract_fail"] += 1
            return json.dumps({"ok": False, "count": 0}), {"ok": False}
        state.sao = samples
        state.metrics["extract_ok"] += 1
        return json.dumps({"ok": True, "count": len(samples)}), {"ok": True}
    if name == "reject_with_feedback":
        state.feedback = str(args.get("reason") or "extract failed")
        state.sao = []
        state.metrics["extract_fail"] += 1
        return json.dumps({"ok": True, "feedback": state.feedback}), {"ok": True}
    raise ValueError(name)


def train_agent(state: SessionState | None = None) -> Agent:
    return Agent(
        name=TRAIN,
        goal="Update domain model weights on the accepted S-A-O micro-batch.",
        instruction="Call train_on_micro_batch once, then stop.",
        tools=_TRAIN_TOOLS,
        provider=_prov(state, domain_settings.TRAIN_AGENT_PROVIDER),
        max_rounds=3,
    )


async def execute_train(state: SessionState, name: str, args: dict) -> tuple[str, dict]:
    if name != "train_on_micro_batch":
        raise ValueError(name)
    if not state.sao:
        return json.dumps({"ok": False, "error": "no_sao"}), {"ok": False}
    result = await train_step(state.domain_id, state.sao)
    state.train_result = result
    if result.get("ok"):
        state.metrics["train_ok"] += 1
        state.packet_done = True
    else:
        state.metrics["train_fail"] += 1
    return json.dumps(result), {"ok": bool(result.get("ok"))}


def coordinator_agent(state: SessionState | None = None) -> Agent:
    return Agent(
        name=COORDINATOR,
        goal="One trained micro-update: research → quality → extract → train; refine on feedback.",
        instruction=(
            "Call ResearchAgent, run_quality_team, ExtractAgent, TrainAgent. "
            "On quality/extract failure, re-call ResearchAgent using feedback. "
            "Use get_packet_status as needed. Stop when train succeeds."
        ),
        tools=[
            agent_as_tool_schema(RESEARCH, "Gather/refine atomic domain packet."),
            tool_schema("run_quality_team", "Parallel QualityJudge LlmAgents + code aggregate.",
                        {"type": "object", "properties": {}}),
            agent_as_tool_schema(EXTRACT, "Extract/validate S-A-O from passed packet."),
            agent_as_tool_schema(TRAIN, "Online train_step on S-A-O."),
            tool_schema("get_packet_status", "Observe packet/feedback/quality/sao/train.",
                        {"type": "object", "properties": {}}),
        ],
        provider=_prov(state, domain_settings.COORDINATOR_PROVIDER),
        max_rounds=domain_settings.PACKET_REFINE_MAX,
    )


def reset_packet(state: SessionState) -> None:
    state.packet_text = ""
    state.packet_urls = []
    state.feedback = ""
    state.quality = {}
    state.sao = []
    state.train_result = {}
    state.packet_done = False


async def build_coordinator_execute(state: SessionState):
    extract = make_agent_tool(extract_agent(state), lambda n, a: execute_extract(state, n, a))
    train = make_agent_tool(train_agent(state), lambda n, a: execute_train(state, n, a))

    async def execute(name: str, args: dict) -> tuple[str, dict]:
        if name == RESEARCH:
            state.metrics["research_calls"] += 1
            tool = make_agent_tool(research_agent(state), lambda n, a: execute_research(state, n, a))
            return await tool(name, args)
        if name == "run_quality_team":
            if not state.packet_text.strip():
                return json.dumps({"ok": False, "error": "empty_packet"}), {"ok": False}
            qr = await run_quality_team(state.packet_text, fallback_provider=_prov(state, domain_settings.COORDINATOR_PROVIDER))
            state.quality = qr
            if qr.get("passed"):
                state.metrics["quality_pass"] += 1
                state.feedback = ""
            else:
                state.metrics["quality_fail"] += 1
                state.feedback = qr.get("feedback") or "quality failed"
                state.packet_text = ""
                state.packet_urls = []
            return json.dumps(qr), {"passed": bool(qr.get("passed"))}
        if name == EXTRACT:
            return await extract(name, args)
        if name == TRAIN:
            return await train(name, args)
        if name == "get_packet_status":
            payload = {
                "packet_chars": len(state.packet_text or ""),
                "urls": state.packet_urls,
                "feedback": state.feedback,
                "quality_passed": (state.quality or {}).get("passed"),
                "sao_count": len(state.sao or []),
                "packet_done": state.packet_done,
                "train_result": state.train_result,
            }
            return json.dumps(payload), payload
        raise ValueError(name)

    return execute
