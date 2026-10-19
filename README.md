# RAG: Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

Everything is in **`rag_rerank.ipynb`** (one short notebook, executed outputs included).

- **Document:** FIBA *Official Basketball Rules Interpretations 2024* (142 pages), `data/fiba_obri_2024.pdf`
  ([source](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-obri-v10a.pdf))
- **Chunking:** one chunk per numbered rule item (583 chunks)
- **Pipeline 1:** `Qwen/Qwen3-Embedding-4B` top-10 by cosine → top-3 → LLM
- **Pipeline 2:** same top-10 → `BAAI/bge-reranker-v2-m3` → top-3 → LLM
- **LLM:** `Qwen/Qwen3.8-27B` served locally with vLLM, temperature 0, in non-thinking and thinking mode
- **Test set:** 20 questions with a gold item id and gold answer

## Outputs (`results/`)

| file | content |
|---|---|
| `results.csv` | per question: gold ranks in both pipelines, status, top-3 ids, all 4 answers, timings |
| `latency.csv` | mean latency per stage |

The report is in `REPORT.md`.
