# MAVAN — Decision Intelligence Platform

> **Work in progress.** This project is under active development. APIs, training thresholds, and docs may change before the capstone release. Current scope: [docs/CAPSTONE.md](docs/CAPSTONE.md).

**Kaggle Capstone:** [Agents for Business](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project) track — [docs/KAGGLE_SUBMISSION.md](docs/KAGGLE_SUBMISSION.md)

AI-Powered Decision Intelligence: multi-agent pipeline builds a **domain transition model** (State→Action→Outcome), then fuses it with Gemini/LLM ensemble for probabilistic decision forecasting.

- **Stage 1:** discovery → HITL sources → quality gate → train domain model → `core_memory`
- **Stage 2:** `POST /api/v1/decisions/recommend` — forecast + recommended action

Also includes RAG + tool-calling agents for notes ([docs/evaluation.md](docs/evaluation.md)).

FastAPI (`main.py`) + Flask UI (`run_flask.py`). Config via `.env`.

## Problem

Organizations lack **decision memory**. MAVAN collects domain data via agents, trains a lightweight outcome model, and recommends actions from scenario probabilities — not generic chat.

**Secondary demo:** personal notes RAG/agent (`RAG_DATA_DIR`).

## Requirements

- Python 3.10+
- pip

## Quick start

### 1. Clone and create a virtual environment

```bash
cd mavan
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` as needed. The `.env` file is not committed to git.

### 4. Run the services

**Flask (UI):**

```bash
python run_flask.py
# or
flask --app run_flask run
```

Default: http://127.0.0.1:5000

**FastAPI (API):**

```bash
python main.py
# or
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Default: http://127.0.0.1:8000

Interactive API docs: http://127.0.0.1:8000/docs

Both services can run at the same time — they listen on different ports.

## Project structure

```
mavan/
├── main.py                  # FastAPI entry point (uvicorn main:app)
├── run_flask.py             # Flask entry point
├── config.py                # Environment settings (.env)
├── config/
│   ├── settings.py          # Re-exports from config.py
│   └── model_config.yaml    # Model configuration (placeholder)
├── app/                     # FastAPI application
│   ├── main.py              # FastAPI app factory
│   ├── api/
│   │   ├── endpoints.py     # API routes (placeholder for chat/RAG)
│   │   └── schemas.py       # Pydantic schemas (placeholder)
│   ├── services/            # LLM, embeddings, summarizer (placeholders)
│   └── templates/
├── ui/                      # Flask application
│   └── __init__.py          # create_app() factory
├── frontend/                # React frontend scaffold (placeholder)
├── training/                # Model training scripts (placeholder)
├── notebooks/               # EDA and prototypes
├── scripts/                 # Utility scripts
├── deployment/              # Kubernetes and Nginx configs
├── monitoring/              # Prometheus and Grafana configs
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture

```
┌─────────────┐     ┌──────────────┐
│  run_flask  │     │    main.py   │
│  (Flask UI) │     │  (FastAPI)   │
└──────┬──────┘     └──────┬───────┘
       │                   │
       ▼                   ▼
   ui/create_app()    app/main.py
       │                   │
       └─────────┬─────────┘
                 ▼
            config.py  ←  .env
```

- `main.py` is a thin entry point that re-exports `app` from `app.main` — keeps PyCharm/uvicorn config as `uvicorn main:app`.
- `run_flask.py` imports the Flask app from `ui.create_app()`.

## Configuration

Settings are defined via environment variables in `.env`:

| Variable      | Description           | Default             |
|---------------|-----------------------|---------------------|
| `FLASK_HOST`  | Flask host            | `127.0.0.1`         |
| `FLASK_PORT`  | Flask port            | `5000`              |
| `FLASK_DEBUG` | Flask debug mode      | `1` (enabled)       |
| `FLASK_APP`   | Flask app module      | `run_flask`         |
| `SECRET_KEY`  | Flask secret key      | `dev-secret-key`    |
| `API_HOST`    | FastAPI host          | `127.0.0.1`         |
| `API_PORT`    | FastAPI port          | `8000`              |
| `API_RELOAD`  | API auto-reload       | `1` (enabled)       |
| `ENV`         | Environment           | `development`       |

Boolean values (`FLASK_DEBUG`, `API_RELOAD`): `1`, `true`, `yes`, `on` — enabled; anything else — disabled.

## API endpoints (FastAPI)

### Intelligence API (group 2)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat/completions` | Unified chat: `mode=ask\|agent`, optional `stream` |
| POST | `/api/v1/decisions/recommend` | Forecast + recommended action |

`POST /api/v1/chat/completions` fields:

| Field | Description |
|-------|-------------|
| `mode` | `ask` (default), `agent` |
| `task_type` | Optional Mavan preset (`translate`, `summarize`, `transcribe`, `qa`). Freeform instructions go in `messages` with `role=system` + `content` (OpenAI-style). |
| `attachment_ids` | Indexed attachments to attach to the chat; RAG runs automatically when the chat has attachments |
| `agent_tools` | Preset (`notes`, `web`, `python`, `default`, `full`) or tool list |
| `chat_id` / `parent_message_id` | Continue a chat; history loads automatically when `parent_message_id` is set |
| `stream` | SSE: `start` → `partial` (tokens) or `thinking` (agent modes) → `completed` |

Response (non-stream): `{chat_id, message_id, model, text, mode, sources, tool_calls}`.

### Domain API (Stage 1)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/domains` | Create decision domain |
| POST | `/api/v1/domains/{id}/pipeline/start` | Run domain pipeline |

### Other

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |

Interactive docs: http://127.0.0.1:8000/docs

### Example (ask + RAG via chat attachments)

Upload → `POST /api/v1/files/index` → pass `attachment_ids` on the first turn (and/or continue with `chat_id` + `parent_message_id`; RAG scopes to `chat.attachment_ids`).

Indexed attachments attached to a chat are removed from the RAG index when the chat is deleted. `POST /api/v1/files/unindex` remains as a manual escape hatch (e.g. indexed file never attached to a chat).

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "ask",
    "attachment_ids": ["ATTACHMENT_ID"],
    "stream": false,
    "messages": [{"role": "user", "content": "What is in this document?"}]
  }'
```

### Example (agent)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "agent",
    "agent_tools": "full",
    "messages": [{"role": "user", "content": "Summarize my notes"}]
  }'
```

## Dependencies

Main packages:

- Flask 3.1
- FastAPI 0.138
- Uvicorn 0.49
- python-dotenv 1.1

Full list with pinned versions is in `requirements.txt`.

Optional AI stack (commented out in `requirements.txt`):

- `openai`
- `faiss-cpu`

## Development

- Do not commit `.env` — only `.env.example` belongs in the repository.
- For production, change `SECRET_KEY` and disable debug/reload.
- Decision Intelligence demo: [docs/setup.md](docs/setup.md), [docs/CAPSTONE.md](docs/CAPSTONE.md).

## Deployment

Kubernetes manifests and Nginx config are in `deployment/`. Monitoring configs (Prometheus, Grafana) are in `monitoring/`. These are templates — adapt them before use in production.


## RAG (secondary demo — notes search)

1. Put `.txt`/`.md` files into `data/notes/` (or set `RAG_DATA_DIR`).
2. Copy env: `cp .env.example .env` — set providers and `RAG_RETRIEVER=hybrid`.
3. Build index (or upload `.txt`/`.md` via `POST /api/v1/files/upload` — background index append):

   ```bash
   python scripts/index_rag_vectors.py
   ```

4. CLI:

   ```bash
   python scripts/run_inference.py mock "your question"
   python scripts/run_agent.py mock "your question"
   python scripts/run_inference.py --no-rag mock "hello"
   ```

5. API: `POST /api/v1/chat/completions` with `mode=ask` (RAG if chat has indexed attachments) or `mode=agent` (auth token required).

## Docker

```bash
cp .env.example .env
docker compose up --build -d
docker compose exec api python scripts/index_rag_vectors.py
curl http://localhost:8000/health
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/CAPSTONE.md](docs/CAPSTONE.md) | Problem, architecture, agentic features, Gemini |
| [docs/setup.md](docs/setup.md) | Mongo, Gemini, domain demo, rehearsal env |
| [docs/evaluation.md](docs/evaluation.md) | Eval scripts and metrics |
| [docs/architecture.md](docs/architecture.md) | Decision Intelligence + RAG flows |
| [docs/KAGGLE_SUBMISSION.md](docs/KAGGLE_SUBMISSION.md) | Writeup, YouTube, GitHub checklist |

## Environment (RAG + eval)

```env
RAG_DATA_DIR=/path/to/notes
RAG_VECTOR_INDEX_PATH=/path/to/vector_index
RAG_EVAL_DIR=/path/to/eval
RAG_RETRIEVER=hybrid
```

## Capstone checklist (repo)

- [ ] Decision domain pipeline → `ready` → `/decisions/recommend`
- [ ] Gemini configured (`LLM_CHAT_PROVIDER=gemini`)
- [ ] Eval scripts run; results in writeup
- [ ] Kaggle writeup (track **Agents for Business**) + YouTube + GitHub tag `capstone-v1`

See [docs/KAGGLE_SUBMISSION.md](docs/KAGGLE_SUBMISSION.md).
