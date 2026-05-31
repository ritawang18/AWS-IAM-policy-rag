# Architecture

Detailed design decisions for the AWS IAM RAG pipeline.

---

## System Overview

```
                        ┌─────────────────────────────────────────────────┐
                        │                 INDEXING PHASE                  │
                        │                 (run once)                      │
                        │                                                 │
  AWS Service Auth Ref ─┤──► ActionCollector ──► ActionChunker           │
                        │         4,505 actions     entity-level chunks   │
                        │                               │                 │
  AWS Managed Policy  ──┤──► PolicyCollector ──► PolicyChunker           │
  Reference               │         500 policies      full-text chunks   │
                        │                               │                 │
                        │                               ▼                 │
                        │                    ┌──────────────────┐         │
                        │                    │   DenseEmbedder  │         │
                        │                    │  voyage-code-3   │         │
                        │                    │  FAISS IndexFlatIP│        │
                        │                    └────────┬─────────┘         │
                        │                    ┌────────┴─────────┐         │
                        │                    │  SparseEmbedder  │         │
                        │                    │   SPLADE-v3      │         │
                        │                    └────────┬─────────┘         │
                        │                    ┌────────┴─────────┐         │
                        │                    │   BM25Indexer    │         │
                        │                    │  custom tokenizer│         │
                        │                    └──────────────────┘         │
                        └─────────────────────────────────────────────────┘

                        ┌─────────────────────────────────────────────────┐
                        │                  QUERY PHASE                    │
                        │                  (per query)                    │
                        │                                                 │
  User Query ───────────┼──► DecomposedRetriever                         │
                        │         │                                       │
                        │         ▼                                       │
                        │    GPT-4o Decomposer                            │
                        │    "explicit + implicit sub-queries"            │
                        │         │                                       │
                        │         ▼  (per sub-query)                     │
                        │    DenseRetriever                               │
                        │    k=4 actions + k=2 policies                  │
                        │         │                                       │
                        │         ▼                                       │
                        │    CompanionRuleAugmenter                       │
                        │    (deterministic companion action lookup)      │
                        │         │                                       │
                        │         ▼                                       │
                        │    Retrieved Context                            │
                        │    (action vocab + policy structure examples)   │
                        │         │                                       │
                        │         ▼                                       │
                        │    PolicyGenerator                              │
                        │    (GPT-3.5-turbo or Llama-3.1-8B)             │
                        │         │                                       │
                        │         ▼                                       │
                        │    IAM Policy JSON                              │
                        └─────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Dual-Source Corpus

Two complementary sources are used rather than one:

**IAM Action chunks** — provide a precise vocabulary of 4,505 individual actions.
Each action's `full_text` field includes the action name, description, access level,
resource types, and condition keys. This grounds the generator in real action names
and prevents hallucination.

**Managed Policy chunks** — provide structural examples of how actions are combined
for real tasks, including resource ARNs and conditions. These teach the generator
correct policy structure that a vanilla LLM often gets wrong (e.g., resource scoping
to `arn:aws:s3:::bucket-name` vs `*`).

### 2. Asymmetric Dense Embedding

Voyage Code-3 is used with `input_type="document"` during indexing and
`input_type="query"` during retrieval. Voyage trains separate embedding spaces
for each, optimizing documents to be found and queries to find. Using `"document"`
for both degrades retrieval performance.

Code-3 was chosen over general-purpose models because IAM documentation uses the
rigid lexical register of technical schemas (colon-separated identifiers, ARN
patterns, JSON policy syntax). Code-optimized embeddings handle this better.

### 3. Why Query Decomposition?

The fundamental challenge: queries are task-level ("allow Lambda to process S3
events"), but IAM policies are action-level ("lambda:InvokeFunction",
"s3:GetObject", "s3:ListBucket", "logs:CreateLogGroup", ...).

A single embedding of the whole query cannot simultaneously point to all the
action-level vectors it needs. Decomposing into sub-queries (one per service,
action-bundle phrasing) lets each sub-query independently search for its target
service's actions.

**Ablation result:** Decomposed Dense achieved Recall=0.574 vs Dense-only Recall=0.538,
and cut zero-recall cases from 11/100 to 4/100.

### 4. Why Companion Rules?

Retrieved context often misses implicit companion actions — actions like
`iam:PassRole` or `ec2:CreateTags` that are always required alongside primary
actions but whose AWS documentation only describes their own primary function.

No embedding model can surface these from the corpus because the semantic signal
is absent from the documentation text. A deterministic rule table bridges this gap.

This is the known limitation discussed in the paper (retrieval recall bottleneck
at ~0.574 across all strategies).

### 5. BM25 and SPLADE Not Used in Final Strategy

Both are included in the codebase but were excluded from the final strategy
after ablation:

- **BM25**: Natural language queries (e.g. "allow a function to upload files")
  share almost no exact tokens with IAM action names (e.g. "s3:PutObject").
  BM25 retrieves irrelevant chunks, hurting precision.

- **SPLADE**: Better than BM25 but still adds noise when combined with dense
  via RRF. Dense retrieval alone was more reliable.

### 6. Four-Layer Evaluation

Standard RAG metrics (Recall@K, MRR) are insufficient for IAM policy synthesis:

- **Layer 1 (Syntax)**: A policy with one missing comma is completely invalid in
  AWS — syntactic checks must come first.
- **Layer 2 (Hallucination)**: A hallucinated action name causes immediate IAM
  validation failure or a security gap. We maintain a full ground truth action DB.
- **Layer 3 (Semantic)**: Standard precision/recall, but with wildcard expansion
  and resource namespace compatibility checking.
- **Layer 4 (LLM Judge)**: Claude Sonnet evaluates security correctness and
  whether the policy would be accepted by a human IAM engineer — capturing
  dimensions that metrics alone miss.

---

## File Ownership

| File | What it owns |
|---|---|
| `src/collector.py` | AWS scraping logic |
| `src/chunker.py` | Chunking strategy and full_text construction |
| `src/embedder.py` | Dense, sparse, BM25 index construction |
| `src/retriever.py` | All retrieval implementations + RRF |
| `src/decomposer.py` | GPT-4o decomposition + companion rules |
| `src/generator.py` | RAG and No-RAG generation |
| `src/evaluator.py` | All 4 evaluation layers |
| `configs/config.yaml` | All hyperparameters (k values, model names, paths) |
| `notebooks/` | Exploratory step-by-step execution of each module |
