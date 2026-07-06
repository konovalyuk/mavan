# Evaluation

## Dataset

- Notes corpus: external dir via `RAG_DATA_DIR` (not course FAQ)
- Ground truth: `RAG_EVAL_DIR/ground_truth.jsonl` (LLM-generated questions per chunk)
- Index: `RAG_INDEX_PATH`, `RAG_VECTOR_INDEX_PATH`

## Retrieval evaluation (n=2)

| Retriever        | Hit Rate@5 | MRR@5 |
|------------------|------------|-------|
| file             | 1.000      | 1.000 |
| vector           | 1.000      | 1.000 |
| hybrid           | 1.000      | 1.000 |
| hybrid + rerank  | 1.000      | 1.000 |

## RAG answer quality (LLM-as-judge)

Script: `scripts/evaluate_rag_answers.py`  
**Avg score: 1.000 (n=2)**

## Agent evaluation

Script: `scripts/evaluate_agent.py`  
**Tool call rate: 1.000 | Answer rate: 1.000 (n=2)**

## LLM approach comparison

Script: `scripts/evaluate_llm_approaches.py`

| Approach | Judge avg |
|----------|-----------|
| no_rag   | TBD (run after rate-limit cooldown) |
| rag      | TBD |
| agent    | TBD |

> Small dev corpus (2 chunks). Metrics validate pipeline; expand corpus for comparative uplift.