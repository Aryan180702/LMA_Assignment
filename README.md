# RAG: Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

Everything is in **`rag_rerank.ipynb`** (one short notebook, executed outputs included).

- **Document:** FIBA *Official Basketball Rules Interpretations 2024* (142 pages), `data/fiba_obri_2024.pdf`
  ([source](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-obri-v10a.pdf))
- **Chunking:** one chunk per numbered rule item (583 chunks)
- **Pipeline 1:** `Qwen/Qwen3-Embedding-4B` top-10 by cosine → top-3 → LLM
- **Pipeline 2:** same top-10 → `BAAI/bge-reranker-v2-m3` → top-3 → LLM
- **LLM:** `Qwen/Qwen3.8-27B` served locally with vLLM, temperature 0, in non-thinking and thinking mode
- **Test set:** 20 questions with a gold item id and gold answer

## Run

```bash
pip install -r requirements.txt

# LLM server (vLLM 0.30 docker; 8 GPUs = 4 data-parallel x 2 tensor-parallel)
docker run -d --gpus all --ipc=host --network host -v <hf_model_dir>:/hf/q38:ro vllm/vllm-openai:v0.30.0 \
  /hf/q38/snapshots/<hash> --served-model-name qwen --tensor-parallel-size 2 --data-parallel-size 4 \
  --port 8011 --max-model-len 16384 --reasoning-parser qwen3

LLM_URL=http://<server>:8011/v1 jupyter nbconvert --to notebook --execute --inplace rag_rerank.ipynb
```

## Outputs (`results/`)

| file | content |
|---|---|
| `results.csv` | per question: gold ranks in both pipelines, status, top-3 ids, all 4 answers, timings |
| `latency.csv` | mean latency per stage |

The report is in `REPORT.md`.
