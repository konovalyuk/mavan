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
  POST /api/v1/decisions/recommend
  → domain model prior per (state, action)
  → parallel LLM ensemble (Gemini)
  → fuse (confidence-weighted alpha)
  → analyzer → recommended_action + rationale
```

Key modules: `app/agents/domain/`, `app/quality/`, `app/domain_model/`, `app/decision/`, `app/domain/store.py`.

## Intelligence API (unified)

Single entry for chat and agents:

```
POST /api/v1/chat/completions
  → prepare_chat_turn (persist + build LLM messages) → execute → update_chat
```

| mode | Dispatch |
|------|----------|
| `ask` | If `chat.attachment_ids`: retrieve on current user text → one system (task + RAG context) + history + user; else system + history + user |
| `agent` | `run_tool_loop(messages, tools)` — RAG via `search_notes` with `source_prefixes` from chat attachments |

History: loaded automatically when `parent_message_id` is set (`chat_id` required). History is user/assistant only (no past system messages).

RAG: no `use_rag` flag. Scoped retrieve uses `chat.attachment_ids` after persist. Attachments must be `indexed` before attach. Unindex happens on chat soft-delete, not on standalone file delete of indexed attachments.

Streaming: `ask` — token stream via provider; `agent` — `thinking` event, then full answer (single tool_loop, no double LLM on final).

Multi-agent (`researcher` → `writer`) — CLI/scripts only (`run_multi_agent.py`), not HTTP API.

## RAG core (single)

| Function | Role |
|----------|------|
| `prepare_chat_turn()` | Persist message, optional retrieve, build one system + history + user |
| `build_system_content()` | Task/custom system + optional RAG grounding/context |
| `retrieve_chunks()` | Search only (ask path, agent tools, eval) |
| `upsert_chunks()` | Index/unindex path for notes and attachments |
| `format_context()` | Agent tool output formatting (`search_notes`) |

Metadata lives on `messages` (`mode`, `sources`, `tool_log`).

## RAG patterns

| Pattern | Where | Context injection |
|---------|-------|-------------------|
| Fixed RAG | `mode=ask` when chat has attachments | `build_system_content(chunks=...)` in `prepare_chat_turn` |
| Agent RAG | `search_notes` tool | `format_context` in tool |
| Eval / CLI | `build_system_content` + `complete`/`stream` | yes |

## Security

- Ingest: `validate_url`, `sanitize_text` (`app/core/guardrails.py`)
- Forecast API: input length and action count limits
- Agent: `run_python` disabled when `AGENT_PYTHON_TOOL=false`

See [CAPSTONE.md](CAPSTONE.md), [setup.md](setup.md).
