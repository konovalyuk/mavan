# Architecture

## LlmAgent vs orchestration

| Concept | Role |
|---------|------|
| **LlmAgent** | Real agent: `provider` + `model` + goal + instruction + tools + observe→reason→act (`app/agents/runtime/`) |
| **Domain conveyor** | Explicit orchestration (not an agent) — `while not stop` packet cycles in `conveyor.py` |
| **Code fan-out** | e.g. parallel QualityJudge LlmAgents via `asyncio.gather` + score aggregation |

We do **not** use SequentialAgent / ParallelAgent / LoopAgent as agent types.

## Decision Intelligence

### Stage 1 — Domain conveyor + PacketCoordinator

```
POST /api/v1/domains
POST /api/v1/domains/{id}/pipeline/start   → conveyor (background, status=running)
POST /api/v1/domains/{id}/pipeline/stop

Conveyor (until stop)
  → PacketCoordinator (LlmAgent) each packet cycle:
       ResearchAgent → run_quality_team → ExtractAgent → TrainAgent
       feedback from quality/extract returns to coordinator → Research again
```

- **Atomic packet**: domain material with state→action→final-state, sized for one quality + train step.
- **Quality**: one `QualityJudge_{provider}` LlmAgent per `QUALITY_PROVIDERS` entry (parallel) + **code** mean/`passes`.
- Raw web text is ephemeral; **capped S-A-O replay** + weight checkpoint persist.
- Collection `domains` = passport (status, metrics, checkpoint).

### Stage 2 — Forecast & recommend

```
POST /api/v1/decisions/recommend
  → DecisionCoordinator (LlmAgent)
       → ForecastAgent / DecisionAgent (each with provider/model)
```

## Chat / RAG

`POST /api/v1/chat/completions` unchanged (`ask` / `agent` tool loop). ADK demo stays in `app/agents/adk/`.

## Security

- URL validate + text sanitize on ingest
- Forecast input caps
- `AGENT_PYTHON_TOOL=false` disables run_python in chat agent mode

See [CAPSTONE.md](CAPSTONE.md), [setup.md](setup.md).
