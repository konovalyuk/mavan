# Architecture

## Decision Intelligence (primary — capstone)

Two-stage flow for domain-specific decision forecasting.

```
Stage 1 — Build domain memory
  POST /api/v1/domains
  → discovery agent (Gemini + google_search)
  → HITL: POST .../sources/approve
  → ingest in RAM (trafilatura)
  → quality gate (parallel LLMs, 6 metrics)
  → extract State–Action–Outcome
  → train DomainTransitionModel (SentencePiece + transformer)
  → Mongo core_memory (+ checkpoint on disk)

Stage 2 — Forecast & recommend
  POST /api/v1/decisions/forecast | /recommend
  → domain model prior per (state, action)
  → parallel LLM ensemble (Gemini)
  → fuse (confidence-weighted alpha)
  → analyzer → recommended_action + rationale
```

Key modules: `app/agents/domain/`, `app/quality/`, `app/domain_model/`, `app/decision/`, `app/domain/store.py`.

Mongo collections for domains: `domains`, `source_registry`, `core_memory` (no persistent raw web storage).

## RAG & agents (secondary demo)

| Mode | Entry | When to use |
|------|-------|-------------|
| Fixed RAG pipeline | `POST /api/v1/rag/query`, `use_rag=true` in chat | Predictable retrieve → answer |
| RAG stream | `POST /api/v1/rag/query/stream` | Same, streaming |
| Agent | `POST /api/v1/agents/rag` | LLM calls `search_notes` via tool loop |
| Multi-agent | `POST /api/v1/agents/multi` | Researcher (tools) → writer |
| Plain chat | `--no-rag`, `use_rag=false` | No retrieval |

## Flow (fixed RAG)

```
question → retrieve (keyword / vector / hybrid RRF) → optional rerank → LLM + context → answer
```

## Security

- Ingest: `validate_url`, `sanitize_text` (`app/core/guardrails.py`)
- Forecast API: input length and action count limits
- Agent: `run_python` disabled when `AGENT_PYTHON_TOOL=false`

See [CAPSTONE.md](CAPSTONE.md), [setup.md](setup.md).
