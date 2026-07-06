# MAVAN

A Python project with two independent web services and an AI-ready project scaffold:

- **Flask** (`run_flask.py`, `ui/`) — UI service
- **FastAPI** (`main.py`, `app/`) — API service

Both services share configuration from `.env` via `config.py`.

## Problem

MAVAN helps you query **your own notes** (markdown/text corpus) via:
- **Fixed RAG** — retrieve → context → LLM
- **Agent** — LLM calls `search_notes` when needed
- **Hybrid retrieval** — keyword + vector (RRF)

**Audience:** developer learning production-style RAG/agents (not course FAQ dataset).

**Dataset:** external directory `RAG_DATA_DIR` (see [docs/setup.md](docs/setup.md)).


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

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/rag/query` | RAG answer (JSON) |
| POST | `/api/v1/rag/query/stream` | RAG answer (SSE) |
| POST | `/api/v1/chat/completions` | Chat (+ optional `use_rag`) |
| POST | `/api/v1/agents/rag` | Agent with tool calling |
| POST | `/api/v1/agents/multi` | Multi-agent |

Interactive docs: http://127.0.0.1:8000/docs

### Example (RAG)

      ```bash
      curl -X POST http://127.0.0.1:8000/api/v1/rag/query \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"question":"What is in my notes?"}'

Additional routes will be added in `app/api/endpoints.py` (chat, files, RAG, etc.).

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
- IDE files (`.idea/`) are partially ignored via `.gitignore`.
- `app/services/`, `training/`, `frontend/`, and `deployment/` contain scaffold placeholders for future AI features.

## Deployment

Kubernetes manifests and Nginx config are in `deployment/`. Monitoring configs (Prometheus, Grafana) are in `monitoring/`. These are templates — adapt them before use in production.


## RAG (notes search)

1. Put `.txt`/`.md` files into `data/notes/`.
2. Copy env: `cp .env.example .env` — set `LLM_EMBED_PROVIDER=mock`, `LLM_CHAT_PROVIDER=mock`, `RAG_RETRIEVER=hybrid`.
3. Build index:
   ```bash
   python scripts/index_rag.py
   python scripts/index_rag_vectors.py --from-json
4. CLI:
    ```bash
    python scripts/run_inference.py mock "your question"      # RAG + chat
    python scripts/run_agent.py mock "your question"        # agent + RAG tool
    python scripts/run_inference.py --no-rag mock "hello"   # chat only

5. API: POST /api/v1/rag/query, POST /api/v1/agents/rag (with auth token).

## Docker
   
      ```bash
      cp .env.example .env
      docker compose up --build -d
      docker compose exec api python scripts/index_rag.py
      docker compose exec api python scripts/index_rag_vectors.py --from-json
      curl http://localhost:8000/health


### Problem

MAVAN is a personal knowledge assistant: ingest markdown notes, index them (keyword + vector + hybrid), answer questions via RAG or agent with tool calling.

### Dataset

Own corpus (not course FAQ). Stored externally:

- Notes: `RAG_DATA_DIR` (e.g. `/home/maksym/my/python/rag-data/data/notes`)
- Eval: `RAG_EVAL_DIR/ground_truth.jsonl`

### Rubric mapping

| Criterion | Where in repo |
|-----------|---------------|
| Retrieval flow | `app/rag/pipeline.py`, `POST /api/v1/rag/query` |
| Hybrid search | `app/rag/retrievers/hybrid.py`, `docs/evaluation.md` |
| Agent + tools | `app/agents/loop.py`, `POST /api/v1/agents/rag` |
| Multi-agent | `app/agents/multi/`, `POST /api/v1/agents/multi` |
| Retrieval eval | `scripts/evaluate_retrieval.py` |
| LLM eval | `scripts/evaluate_rag_answers.py` |
| Interface | FastAPI `/docs`, CLI scripts |
| Ingestion | `scripts/index_rag.py`, `scripts/index_rag_vectors.py` |
| Monitoring | `app/services/monitoring/`, `scripts/dashboard.py` |
| Docker | `Dockerfile`, `docker-compose.yml` |

See also: [docs/setup.md](docs/setup.md), [docs/architecture.md](docs/architecture.md), [docs/evaluation.md](docs/evaluation.md).


## Environment (RAG + eval)
      ```env
      RAG_DATA_DIR=/home/maksym/my/python/rag-data/data/notes
      RAG_INDEX_PATH=/home/maksym/my/python/rag-data/.rag/chunks.json
      RAG_VECTOR_INDEX_PATH=/home/maksym/my/python/rag-data/.rag/vector_index
      RAG_EVAL_DIR=/home/maksym/my/python/rag-data/eval
      RAG_RETRIEVER=hybrid

### Peer review checklist

- [ ] Problem statement clear
- [ ] Own dataset (not course FAQ)
- [ ] Hybrid retrieval with eval table
- [ ] Agent with tool calling demo
- [ ] `docker compose up` works
- [ ] Monitoring dashboard screenshot
