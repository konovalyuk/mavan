# MAVAN — Kaggle Capstone (Agents for Business)

**5-Day AI Agents: Intensive Vibe Coding Course With Google (2026)**

> **Work in progress.** Honest LlmAgent Stage 1 conveyor and decision recommend are implemented.

## Problem

Organizations need **decision memory**: domain-specific forecasts of final outcomes after actions.

## Solution

1. **Stage 1** — continuous conveyor until stop: **PacketCoordinator** (LlmAgent) orchestrates Research / parallel QualityJudges / Extract / Train on **atomic packets**. Feedback flows through the coordinator. Weights + capped S-A-O replay (no raw web store).
2. **Stage 2** — DecisionCoordinator → Forecast + Decision LlmAgents → recommendation.

## Agentic features

| Feature | Implementation |
|---------|----------------|
| Real agents | `LlmAgent` with `provider` + `model` + tools + goal + tool loop |
| Orchestration | `conveyor.py` stop-loop (not fake *Agent types) |
| Multi-model quality | Parallel `QualityJudge_*` LlmAgents + code aggregation |
| Feedback | Coordinator reads quality/extract feedback and re-calls Research |
| Memory | Capped S-A-O replay + checkpoint |
| Chat RAG | Secondary demo |

## Env (excerpt)

```env
CHAT_PROVIDER=gemini
COORDINATOR_PROVIDER=gemini
RESEARCH_PROVIDER=gemini
EXTRACT_PROVIDER=gemini
TRAIN_AGENT_PROVIDER=gemini
QUALITY_PROVIDERS=gemini,openai,mistral
PACKET_REFINE_MAX=8
CORE_MEMORY_MAX_SAMPLES=2000
```

## Known gaps

- Uploaded files not wired into Stage 1 conveyor yet.
- Soft `TRAIN_MIN_*` → `model_trusted` only.

See [architecture.md](architecture.md), [KAGGLE_SUBMISSION.md](KAGGLE_SUBMISSION.md).
