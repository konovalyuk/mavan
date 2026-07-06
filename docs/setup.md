# Setup

## Prerequisites

- Python 3.10+
- MongoDB (local or Docker)
- Optional: API keys for Gemini/OpenAI

## Local run

```bash
cd mavan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env