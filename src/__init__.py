"""
aws-iam-rag
===========
RAG pipeline for AWS IAM policy synthesis from natural language.

Module responsibilities
-----------------------
collector             → corpus building only
  ActionCollector       scrape IAM actions from AWS Service Authorization Reference
  PolicyCollector       scrape managed policies from AWS Managed Policy Reference

ground_truth_builder  → evaluation dataset only
  GitHubCollector       scrape real-world IAM policies from GitHub CloudFormation templates
  GroundTruthBuilder    select held-out managed policies, generate GPT-4o queries,
                        decontaminate retrieval corpus

chunker               → convert raw documents into retrieval chunks
  ActionChunker         entity-level chunking for IAM actions
  PolicyChunker         full-text chunking for managed policies

embedder              → build retrieval indexes from chunks
  DenseEmbedder         Voyage Code-3 embeddings + FAISS IndexFlatIP
  SparseEmbedder        SPLADE-v3 sparse vectors
  BM25Indexer           BM25 Okapi with IAM-aware tokenization

retriever             → query the retrieval indexes
  DenseRetriever        cosine similarity via FAISS
  SparseRetriever       dot-product over SPLADE sparse vectors
  BM25Retriever         BM25 keyword retrieval
  HybridRetriever       weighted RRF over any combination of the above

decomposer            → query decomposition and companion action augmentation
  DecomposedRetriever   GPT-4o decomposition + per-subquery dense retrieval
  CompanionRuleAugmenter  deterministic companion action lookup

generator             → IAM policy generation
  PolicyGenerator       RAG generation and No-RAG baseline

evaluator             → 4-layer evaluation pipeline
  PolicyEvaluator       Layer 1 (syntax), Layer 2 (hallucination),
                        Layer 3 (semantic), Layer 4 (LLM-as-judge)
"""
