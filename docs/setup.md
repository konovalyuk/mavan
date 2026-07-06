# Setup

> **Work in progress.** See [CAPSTONE.md](CAPSTONE.md) for scope and known gaps.

## Prerequisites

- Python 3.10+
- MongoDB (local or Docker)
- Google AI Studio API key (Gemini) — required for capstone demo
- Serper API key for domain discovery (`GOOGLE_SEARCH_API_KEY`)

Optional: `pip install mcp` for MCP server ([run_mcp_server.py](../scripts/run_mcp_server.py)).

## Local run

```bash
cd mavan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install Decision Intelligence extras if not already present:

```bash
pip install torch sentencepiece trafilatura
```

## Capstone / Gemini

```env
LLM_CHAT_PROVIDER=gemini
GOOGLE_API_KEY=your-gemini-key
QUALITY_PROVIDERS=gemini
FORECAST_PROVIDERS=gemini
GOOGLE_SEARCH_API_KEY=your-serper-key
AGENT_PYTHON_TOOL=false
```

## Demo rehearsal env

Default training thresholds (`TRAIN_MIN_SAMPLES=50`, etc.) need many quality-approved sources. For capstone rehearsal, optionally add:

```env
TRAIN_MIN_SAMPLES=10
TRAIN_MIN_UNIQUE_STATES=3
TRAIN_MIN_UNIQUE_ACTIONS=2
TRAIN_MIN_AVG_QUALITY=60
```

## Start services

```bash
docker run -d -p 27017:27017 mongo:7   # if needed
python main.py
# http://127.0.0.1:8000/docs
```

## Decision Intelligence demo

CLI scripts bypass HTTP auth and call services directly:

```bash
python scripts/create_domain.py --name energy --description "Energy policy"
# Approve sources: POST /api/v1/domains/{id}/sources/approve (Swagger or curl)
python scripts/run_domain_pipeline.py --domain-id <id> --provider gemini
python scripts/run_forecast.py --domain-id <id> --state "Oil prices rising" \
  --actions "Increase subsidy,Remove subsidy" --recommend
```

Via API (requires auth token): same paths under `/api/v1/domains`, `/api/v1/decisions/recommend`.

## MCP server (optional)

```bash
pip install mcp
python scripts/run_mcp_server.py
```

## Eval

```bash
python scripts/evaluate_agent.py
python scripts/evaluate_retrieval.py
EVAL_LLM_SLEEP_SEC=6 python scripts/evaluate_llm_approaches.py
```

Paste metrics into the Kaggle writeup — see [evaluation.md](evaluation.md).

## Kaggle submit

Checklist: [KAGGLE_SUBMISSION.md](KAGGLE_SUBMISSION.md) — track **Agents for Business**, YouTube, GitHub tag `capstone-v1`.
