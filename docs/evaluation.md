# Evaluation

Metrics for the Kaggle writeup and local regression checks. Small dev corpus (`n=2` for notes RAG) — expand later for stronger comparative claims.

## Capstone metrics (paste into writeup)

| Script | Metric | Command |
|--------|--------|---------|
| Retrieval | Hit Rate@5, MRR@5 | `python scripts/evaluate_retrieval.py` |
| RAG answers | LLM-as-judge avg | `python scripts/evaluate_rag_answers.py` |
| Agent | tool_call_rate, answer_rate | `python scripts/evaluate_agent.py` |
| LLM approaches | no_rag vs rag vs agent | `EVAL_LLM_SLEEP_SEC=6 python scripts/evaluate_llm_approaches.py` |

Run with Gemini configured or mock provider for offline smoke tests. Use `EVAL_LLM_SLEEP_SEC=6` to reduce Gemini 429 errors on `evaluate_llm_approaches.py`.

Output file: `RAG_EVAL_DIR/llm_approaches_eval.json`.

## Dataset

- Notes corpus: `RAG_DATA_DIR`
- Ground truth: `RAG_EVAL_DIR/ground_truth.jsonl`

Decision Intelligence pipeline is evaluated qualitatively via demo (`pipeline → ready → recommend`) until a domain eval set exists.

## Retrieval evaluation (n=2)

| Retriever | Hit Rate@5 | MRR@5 |
|-----------|------------|-------|
| file | 1.000 | 1.000 |
| vector | 1.000 | 1.000 |
| hybrid | 1.000 | 1.000 |

## RAG answer quality

Script: `scripts/evaluate_rag_answers.py` — **Avg score: 1.000 (n=2)**

## Agent evaluation

Script: `scripts/evaluate_agent.py` — **Tool call rate: 1.000 | Answer rate: 1.000 (n=2)**

## LLM approach comparison

Script: `scripts/evaluate_llm_approaches.py`

Run before submit and record results in the writeup:

```bash
EVAL_LLM_SLEEP_SEC=6 python scripts/evaluate_llm_approaches.py
```
