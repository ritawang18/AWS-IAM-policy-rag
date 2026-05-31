# AWS IAM Policy Synthesis via RAG

> Translating natural language into secure, least-privilege AWS IAM policies using Retrieval-Augmented Generation.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Manually authoring AWS IAM policies from natural language intent is error-prone: the AWS action space spans **4,500+ granular permissions**, implicit service dependencies are rarely obvious, and vanilla LLMs frequently hallucinate non-existent action names or over-generate wildcard `*` permissions.

This project introduces a RAG pipeline engineered specifically for IAM policy synthesis. It combines a **dual-source retrieval corpus** (IAM action descriptions + managed policy documents) with a **GPT-4o-driven query decomposer** that surfaces both explicit and implicit service requirements before retrieval.

### Key results (evaluated on the Evaluation Dataset):

Results running on Llama-3.1-8B as the generator model

| Metric | No RAG | RAG | Δ |
|---|---|---|---|
| Action Recall | 0.600 | 0.688 | **+14.7%** |
| Hallucination Rate | 0.255 | 0.048 | **−81.2%** |
| Resource Scoping Accuracy | 0.827 | 0.973 | **+17.7%** |

Results running on GPT-3.5-turbo as the generator model

| Metric | No RAG | RAG | Δ |
|---|---|---|---|
| Action Recall | 0.650 | 0.655 | **+0.77%** |
| Hallucination Rate | 0.062 | 0.018 | **−71.0%** |
| Resource Scoping Accuracy | 0.880 | 0.969 | **+10.1%** |

---

## Pipeline Architecture

```
User Query (natural language)
        │
        ▼
┌───────────────────┐
│  GPT-4o Decomposer │  ← breaks query into explicit + implicit sub-queries
│  (query_decomposer)│    e.g. "Lambda reads S3" → EC2 sub-query + implicit
└────────┬──────────┘    CloudWatch Logs sub-query
         │  sub-queries
         ▼
┌───────────────────────────────────────┐
│         Dual-Source Corpus            │
│  ┌──────────────┐  ┌───────────────┐  │
│  │ 4,505 IAM    │  │  470 AWS      │  │
│  │ Action Chunks│  │ Managed Policy│  │
│  │ (voyage-code-│  │    Chunks     │  │
│  │     3+FAISS) │  │               │  │
│  └──────────────┘  └───────────────┘  │
└────────┬──────────────────────────────┘
         │  retrieved context
         ▼
┌───────────────────┐
│   Generator LLM   │  ← GPT-3.5-turbo or Llama-3.1-8B
└────────┬──────────┘
         │
         ▼
  IAM Policy JSON
  (precise actions, correct resource scoping)
```

---

## Repository Structure

```
aws-iam-rag/
├── README.md
├── ARCHITECTURE.md             ← Detailed design decisions
├── requirements.txt
├── .env.example                ← API key template
│
├── src/                        ← Core logic extracted into reusable modules (refactored from notebooks for clarity)
│   ├── collector.py            ← Scrape AWS Service Auth + Managed Policy refs
│   ├── chunker.py              ← Entity-level chunking for actions + policies
│   ├── embedder.py             ← Dense (Voyage), sparse (SPLADE), BM25 indexing
│   ├── retriever.py            ← DenseRetriever, SparseRetriever, BM25Retriever + RRF
│   ├── decomposer.py           ← GPT-4o query decomposition module
│   ├── generator.py            ← RAG generation (GPT-3.5 / Llama)
│   ├── evaluator.py            ← 4-layer evaluation pipeline
│   └── ground_truth_builder.py ← managed policy selection, GPT-4o query gen, corpus decontamination
│
├── notebooks/                  ← Step-by-step exploratory notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embedding.ipynb
│   ├── 04_ground_truth_construction.ipynb
│   ├── 05_retrieval_pipeline.ipynb
│   └── 06_evaluation.ipynb
│
├── configs/
│   └── config.yaml             ← All tunable hyperparameters
│
├── data/
│   ├── raw/                    ← Sample scraped data (2-3 files per source)
│   │   ├── sample_actions.json
│   │   └── sample_policies.json
│   ├── processed/              ← Chunked data schema examples
│   │   ├── sample_action_chunks.json
│   │   └── sample_policy_chunks.json
│   └── evaluation/             ← Ground truth dataset
│       └── README.md           ← Groud truth dataset construction methodology
│
├── eval/
│   └── analyze_results.ipynb   ← Result analysis and visualization
│
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    └── test_evaluator.py
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/yourusername/aws-iam-rag.git
cd aws-iam-rag
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, VOYAGE_API_KEY, ANTHROPIC_API_KEY
```

### 3. Run the pipeline end-to-end

Follow the numbered notebooks in order:

```
notebooks/01_data_collection.ipynb      → scrape AWS docs
notebooks/02_chunking.ipynb             → build chunks
notebooks/03_embedding.ipynb            → embed + build FAISS index
notebooks/04_ground_truth_construction  → build eval dataset
notebooks/05_retrieval_pipeline.ipynb   → run retrieval
notebooks/06_evaluation.ipynb           → evaluate results
```

Or run individual `src/` modules directly — see `ARCHITECTURE.md` for the programmatic API.

---

## Corpus Details

| Component | Count | Source |
|---|---|---|
| IAM Action Chunks | 4,505 | AWS Service Authorization Reference (56 services) |
| Managed Policy Chunks | 470 | AWS Managed Policy Reference |
| **Total Corpus** | **4,975** | Dual-source retrieval index |

---

### Evaluation Dataset

No existing benchmark covers IAM policy generation from natural language, so we constructed one from scratch. The dataset contains **75 real-world IAM policies paired with GPT-4o-generated natural language queries**, sourced from two places:

- **45 policies** parsed from CloudFormation templates in public AWS GitHub repositories (`aws-samples`, `awslabs`), filtered to 2–25 specific non-wildcard actions covering at least one of the 56 core services
- **30 policies** sampled from the AWS Managed Policy Reference using a diversity-weighted scoring strategy to ensure broad service coverage, then held out from the retrieval corpus to prevent data leakage

Queries were generated by GPT-4o under strict constraints: no AWS service names, no IAM action names, task-level descriptions only. This follows the Gorilla methodology — queries describe developer intent, not API calls.

| | GitHub | Managed | Total |
|---|---|---|---|
| Policies | 45 | 30 | **75** |
| Avg actions/policy | 8.2 | 9.1 | 8.6 |
| Action range | 2–25 | 3–20 | 2–25 |
| Services covered | 18 | 15 | **23** |
| Multi-service policies | 31 | 19 | 50 |

---

### Evaluation Pipeline

Since standard RAG metrics (Recall@K, MRR) are insufficient for IAM policy synthesis — a single hallucinated action name causes immediate deployment failure or a security gap — we designed a **four-layer domain-specific evaluation pipeline**:

- **Layer 1 — Syntactic Validity**: checks that output is valid JSON conforming to IAM schema (Version, Statement, Effect, Action, Resource fields present)
- **Layer 2 — Hallucination Detection**: compares every generated action against a ground-truth database of all 4,505 valid IAM actions; any action not in the database is flagged as a hallucination, with wildcard expansion via `fnmatch`
- **Layer 3 — Semantic Action Matching**: computes precision, recall, and F1 between generated and ground-truth action sets, plus resource namespace compatibility (e.g. `s3:GetObject` should reference `arn:aws:s3`, not `arn:aws:lambda`)
- **Layer 4 — LLM-as-Judge (Claude Sonnet)**: scores functional correctness, security correctness, and resource scoping on a 0–10 scale and gives a binary acceptability verdict; also explicitly evaluates condition block adequacy when the ground-truth policy contains conditions

---

## Retrieval Strategy Ablation

| Strategy | Recall | Precision | F1 | Perfect-Recall | Zero-Recall |
|---|---|---|---|---|
| **Decomposed (selected)** | **0.574** | **0.426** | **0.489** | 16/100 | 4/100 |
| Decomposed Hybrid | 0.556 | 0.404 | 0.468 | 16/100 | 6/100 |
| Dense | 0.538 | 0.341 | 0.417 | 18/100 | 11/100 |
| RRF Hybrid (Dense+BM25) | 0.353 | 0.228 | 0.277 | 6/100 | 13/100 |

---

## Tech Stack

| Component | Technology |
|---|---|
| Embedding model | `voyage-code-3` (Voyage AI) |
| Vector index | FAISS `IndexFlatIP` (cosine similarity) |
| Sparse retrieval | SPLADE-v3 |
| Lexical retrieval | BM25 Okapi |
| Query decomposer | GPT-4o |
| Generator models | GPT-3.5-turbo, Llama-3.1-8B (via Groq) |
| LLM-as-Judge | Claude Sonnet (Anthropic) |

---

## Citation

This project was developed as a final project for WashU CSE 4061 Text Mining. If you use this work, please cite:

```
Wang, R., Tian, A., & Li, Q. (2024). Retrieval-Augmented Generation for AWS IAM Policy Synthesis.
CSE 4061 Final Project Report.
```

---

## License

MIT License — see [LICENSE](LICENSE).
