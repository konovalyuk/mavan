# MAVAN — Kaggle Capstone (Agents for Business)

**5-Day AI Agents: Intensive Vibe Coding Course With Google (2026)**

> **Work in progress.** Core Decision Intelligence flows are implemented; domain file upload and production hardening are still evolving.

## Problem

Organizations and public institutions need **decision memory**: not generic chat, but domain-specific forecasts of **final outcomes** after actions. Generic LLMs lack persistent domain transition knowledge and calibrated scenario probabilities.

## Solution

**AI-Powered Decision Intelligence Platform** in two stages:

1. **Stage 1** — Multi-agent pipeline: discover sources → HITL approve → quality gate (parallel Gemini/LLMs) → extract State-Action-Outcome → train lightweight domain transition model → `core_memory`.
2. **Stage 2** — Fuse domain model prior + parallel LLM ensemble → decision recommendation API.

## Architecture

```
Stage 1: POST /domains → discovery → HITL sources → ingest (RAM) → quality → S-A-O extract → train → core_memory
Stage 2: POST /decisions/recommend → predict(domain) + LLM ensemble → fuse → analyzer
```

See [architecture.md](architecture.md) for RAG/agent modes.

## Agentic features (capstone checklist)

| Feature | Implementation |
|---------|----------------|
| Tool use / function calling | `app/agents/loop.py`, `google_search`, `search_notes` |
| Multi-agent orchestration | `app/agents/multi/`, `app/agents/domain/pipeline.py` |
| Agentic loops | `MAX_TOOL_ROUNDS`, `QUALITY_CYCLE_MAX` retry |
| Human-in-the-loop | `POST .../sources/approve` |
| Long-term memory | Mongo `core_memory` + replay in train |
| RAG | Hybrid retrieval (`app/rag/`) — secondary demo |
| Agent evaluation | `scripts/evaluate_*.py`, [evaluation.md](evaluation.md) |
| MCP (optional) | `app/mcp/server.py` — `pip install mcp` |

## Gemini setup

```env
LLM_CHAT_PROVIDER=gemini
GOOGLE_API_KEY=your-key
QUALITY_PROVIDERS=gemini
FORECAST_PROVIDERS=gemini
GOOGLE_SEARCH_API_KEY=your-serper-key
```

## Demo / video

See [KAGGLE_SUBMISSION.md](KAGGLE_SUBMISSION.md) — voiceover script, screen recording, asset generation.

For a quick local rehearsal, lower training thresholds in `.env` (see [setup.md](setup.md#demo-rehearsal-env)).

## Security (threat model)

- Ingested web text passes **multi-LLM quality gate** before training.
- `fetch_url_text` strips script tags; URLs must be http(s).
- `run_python` tool disabled when `AGENT_PYTHON_TOOL=false`.
- API guardrails: max context length, action count caps (`app/core/guardrails.py`).

## Eval

See [evaluation.md](evaluation.md) — retrieval, RAG, agent, LLM approaches.

## Known gaps (WIP)

- Domain training from **uploaded files** (generic file API exists; not wired to domain pipeline yet).
- Default `TRAIN_MIN_*` thresholds require substantial approved sources for `status: ready`.

## Kaggle submission

See [KAGGLE_SUBMISSION.md](KAGGLE_SUBMISSION.md) — writeup track **Agents for Business**, YouTube, GitHub, deadline.
