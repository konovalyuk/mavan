# Architecture

## Modes

| Mode | Entry | When to use |
|------|-------|-------------|
| Fixed RAG pipeline | `POST /api/v1/rag/query`, `use_rag=true` in chat | Predictable retrieve → answer |
| RAG stream | `POST /api/v1/rag/query/stream` | Same, but streaming tokens |
| Agent | `POST /api/v1/agents/rag` | LLM decides if/when to call `search_notes` |
| Multi-agent | `POST /api/v1/agents/multi` | Research agent (tools) → writer agent |
| Plain chat | `--no-rag`, `use_rag=false` | No retrieval |

## Flow (fixed RAG)
