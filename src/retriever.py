"""
retriever.py
============
All retrieval strategies over the dual-source IAM corpus.

Classes
-------
  DenseRetriever       → cosine similarity via Voyage Code-3 + FAISS
  SparseRetriever      → dot-product similarity via SPLADE-v3
  BM25Retriever        → BM25 Okapi with IAM-aware tokenization
  HybridRetriever      → weighted RRF over any combination of the above
  CompanionRuleAugmenter → post-retrieval deterministic companion action injection
  DecomposedRetriever  → final selected strategy: GPT-4o decomposition +
                         per-subquery DenseRetriever + CompanionRuleAugmenter

All retrievers return results in a standard format:
  [{"text": ..., "metadata": ..., "score": ...}, ...]

Ablation results (on 100-policy development set):
  DecomposedRetriever  → Recall 0.574, Zero-recall 4/100   ← selected
  Decomposed Hybrid    → Recall 0.556, Zero-recall 6/100
  DenseRetriever       → Recall 0.538, Zero-recall 11/100
  HybridRetriever      → Recall 0.353, Zero-recall 13/100  ← BM25 adds noise

Usage
-----
    from src.retriever import DecomposedRetriever, DenseRetriever
    from src.decomposer import QueryDecomposer

    # Final selected strategy
    dense = DenseRetriever(
        chunks_path="indexes/dense/combined_chunks.json",
        faiss_index_path="indexes/dense/combined_faiss.index",
        voyage_api_key="...",
    )
    retriever = DecomposedRetriever(
        dense_retriever=dense,
        decomposer=QueryDecomposer(openai_api_key="..."),
    )
    results = retriever.retrieve("Allow Lambda to read from S3 and write logs")

    # Baseline dense-only retrieval
    results = dense.retrieve("Allow Lambda to read from S3", k_actions=8, k_policies=3)
"""

import json
import pickle
import re
from collections import defaultdict

import faiss
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def tokenize_for_bm25(text: str) -> list[str]:
    """
    IAM-aware BM25 tokenizer.
    Preserves colons (s3:GetObject) and hyphens (access-analyzer) so BM25
    can match exact IAM action names rather than splitting them into fragments.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s:\-]", " ", text)
    return text.split()


def format_results(
    chunks: list[dict], scores: list[float], indices: list[int]
) -> list[dict]:
    """Standard result format used by all retrievers."""
    return [
        {"text": chunks[i]["text"], "metadata": chunks[i]["metadata"], "score": float(s)}
        for s, i in zip(scores, indices)
        if i >= 0  # FAISS returns -1 for unfilled slots
    ]


def weighted_rrf(
    ranked_lists: list[list[int]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Weighted Reciprocal Rank Fusion.
    Combines multiple ranked lists, weighting each retriever's contribution.

    Args:
        ranked_lists: each element is an ordered list of chunk indices
        weights:      contribution weight per retriever (should sum to 1)
        k:            RRF smoothing constant (default 60)

    Returns:
        list of (chunk_index, rrf_score) sorted descending
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, idx in enumerate(ranked):
            scores[idx] += weight * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def split_by_type(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split mixed results into (action_results, policy_results)."""
    actions  = [r for r in results if r["metadata"]["type"] == "iam_action"]
    policies = [r for r in results if r["metadata"]["type"] == "iam_policy"]
    return actions, policies


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Cosine-similarity retrieval using Voyage Code-3 embeddings + FAISS.

    Action and policy chunks are retrieved with independent k values because
    they serve different roles: action chunks provide exact permission names,
    policy chunks provide structural examples of how permissions are combined.

    Asymmetric embedding: documents were indexed with input_type="document",
    queries use input_type="query" — Voyage trains separate spaces for each.
    """

    MODEL = "voyage-code-3"

    def __init__(self, chunks_path: str, faiss_index_path: str, voyage_api_key: str):
        import voyageai
        print("Loading Dense Retriever...")
        with open(chunks_path) as f:
            self.chunks = json.load(f)
        self.index  = faiss.read_index(faiss_index_path)
        self.client = voyageai.Client(api_key=voyage_api_key)
        print(f"  ✓ {len(self.chunks)} chunks, {self.index.ntotal} vectors, dim={self.index.d}")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query with Voyage Code-3, normalized for cosine similarity."""
        response = self.client.embed(
            texts=[query], model=self.MODEL, input_type="query"
        )
        vec = np.array([response.embeddings[0]], dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, k_actions: int = 8, k_policies: int = 3) -> list[dict]:
        """
        Retrieve top-k action and policy chunks by cosine similarity.
        Action and policy results are ranked independently (type-aware).
        """
        query_vec = self.embed_query(query)
        distances, indices = self.index.search(query_vec, (k_actions + k_policies) * 3)
        all_results = format_results(self.chunks, distances[0], indices[0])
        actions, policies = split_by_type(all_results)
        return actions[:k_actions] + policies[:k_policies]

    def retrieve_ranked(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        """Return raw (index, score) pairs for RRF fusion."""
        query_vec = self.embed_query(query)
        distances, indices = self.index.search(query_vec, k)
        return list(zip(indices[0].tolist(), distances[0].tolist()))


# ---------------------------------------------------------------------------
# Sparse Retriever (SPLADE)
# ---------------------------------------------------------------------------

class SparseRetriever:
    """
    Dot-product similarity over SPLADE-v3 sparse vectors.

    SPLADE expands vocabulary with semantically related terms, handling
    query-document mismatch better than BM25 — but still underperforms
    dense retrieval for this domain (see ablation results).
    """

    MODEL_ID = "naver/splade-v3"

    def __init__(self, chunks_path: str, hf_token: str):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        print("Loading Sparse (SPLADE) Retriever...")
        with open(chunks_path) as f:
            self.chunks = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, token=hf_token)
        self.model     = AutoModelForMaskedLM.from_pretrained(self.MODEL_ID, token=hf_token)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ {len(self.chunks)} chunks, SPLADE on {self.device}")

    def encode_query(self, query: str) -> dict[str, float]:
        """Encode query as SPLADE sparse vector {token_id_str: weight}."""
        inputs = self.tokenizer(
            [query], return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            logits      = self.model(**inputs).logits
            logits_relu = torch.log1p(torch.relu(logits))
            token_max, _= torch.max(logits_relu * inputs.attention_mask.unsqueeze(-1), dim=1)
        nonzero_idx = torch.nonzero(token_max[0]).squeeze(-1).cpu().tolist()
        nonzero_w   = token_max[0][nonzero_idx].cpu().tolist()
        return {str(i): w for i, w in zip(nonzero_idx, nonzero_w)}

    @staticmethod
    def sparse_dot_product(query_vec: dict[str, float], doc_vec: dict[str, float]) -> float:
        """Dot product over shared keys only (sparse representation)."""
        return sum(query_vec[k] * doc_vec.get(k, 0.0) for k in query_vec)

    def retrieve(self, query: str, k_actions: int = 8, k_policies: int = 3) -> list[dict]:
        """Retrieve by SPLADE dot-product similarity. Type-aware k values."""
        query_vec = self.encode_query(query)
        scores    = [self.sparse_dot_product(query_vec, c.get("sparse_embedding", {}))
                     for c in self.chunks]
        action_results, policy_results = [], []
        for idx, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
            chunk = self.chunks[idx]
            r = {"text": chunk["text"], "metadata": chunk["metadata"], "score": score}
            if chunk["metadata"]["type"] == "iam_action":
                if len(action_results) < k_actions:
                    action_results.append(r)
            elif len(policy_results) < k_policies:
                policy_results.append(r)
            if len(action_results) >= k_actions and len(policy_results) >= k_policies:
                break
        return action_results + policy_results

    def retrieve_ranked(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        """Return raw (index, score) pairs for RRF."""
        query_vec = self.encode_query(query)
        scores    = [self.sparse_dot_product(query_vec, c.get("sparse_embedding", {}))
                     for c in self.chunks]
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    BM25 Okapi retrieval with IAM-aware tokenization.

    Weakest strategy in ablation: natural-language queries share almost no
    exact tokens with IAM action identifiers. Included for completeness and
    as a fusion component in HybridRetriever.
    """

    def __init__(self, index_path: str, chunks_path: str):
        print("Loading BM25 Retriever...")
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(chunks_path) as f:
            self.chunks = json.load(f)
        print(f"  ✓ {len(self.chunks)} chunks loaded")

    def retrieve(self, query: str, k_actions: int = 8, k_policies: int = 3) -> list[dict]:
        """Retrieve by BM25 score. Type-aware k values."""
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        action_results, policy_results = [], []
        for idx, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True):
            chunk = self.chunks[idx]
            r = {"text": chunk["text"], "metadata": chunk["metadata"], "score": float(score)}
            if chunk["metadata"]["type"] == "iam_action":
                if len(action_results) < k_actions:
                    action_results.append(r)
            elif len(policy_results) < k_policies:
                policy_results.append(r)
            if len(action_results) >= k_actions and len(policy_results) >= k_policies:
                break
        return action_results + policy_results

    def retrieve_ranked(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        """Return raw (index, score) pairs for RRF."""
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Hybrid Retriever (RRF)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Weighted RRF combination of multiple retrievers.

    Ablation results showed both Dense+BM25 and Dense+SPLADE hybrids
    underperform pure dense retrieval — sparse methods add noise for IAM
    because queries and action names have low lexical overlap.
    Retained here for experimentation.
    """

    def __init__(self, retrievers: list, weights: list[float], chunks: list[dict]):
        assert len(retrievers) == len(weights), "Each retriever needs a weight"
        self.retrievers = retrievers
        self.weights    = weights
        self.chunks     = chunks

    def retrieve(self, query: str, k_actions: int = 8, k_policies: int = 3) -> list[dict]:
        """Retrieve using weighted RRF. Type-aware k values."""
        ranked_lists = [
            [idx for idx, _ in r.retrieve_ranked(query, k=50)]
            for r in self.retrievers
        ]
        fused = weighted_rrf(ranked_lists, self.weights)
        action_results, policy_results = [], []
        for idx, score in fused:
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            r = {"text": chunk["text"], "metadata": chunk["metadata"], "score": score}
            if chunk["metadata"]["type"] == "iam_action":
                if len(action_results) < k_actions:
                    action_results.append(r)
            elif len(policy_results) < k_policies:
                policy_results.append(r)
            if len(action_results) >= k_actions and len(policy_results) >= k_policies:
                break
        return action_results + policy_results


# ---------------------------------------------------------------------------
# Companion Rule Augmenter
# ---------------------------------------------------------------------------

# Actions that are always implicitly required alongside primary actions but
# whose AWS documentation only describes their own primary function —
# making them semantically invisible to any retrieval strategy.
COMPANION_RULES: dict[str, list[str]] = {
    # EC2 resource creation
    "ec2:RunInstances":                    ["ec2:CreateTags", "ec2:DescribeInstances", "ec2:DescribeInstanceStatus", "ec2:DescribeImages", "ec2:DescribeSubnets", "ec2:DescribeVpcs", "ec2:DescribeSecurityGroups"],
    "ec2:CreateNetworkInterface":          ["ec2:CreateTags", "ec2:DescribeSubnets", "ec2:DescribeVpcs", "ec2:DescribeNetworkInterfaces", "ec2:DescribeSecurityGroups"],
    "ec2:CreateSecurityGroup":             ["ec2:CreateTags", "ec2:DescribeVpcs", "ec2:DescribeSecurityGroups"],
    "ec2:StartInstances":                  ["ec2:StopInstances", "ec2:DescribeInstances", "ec2:DescribeInstanceStatus"],
    # ECS
    "ecs:RunTask":                         ["ec2:CreateTags", "ec2:DescribeNetworkInterfaces", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups", "iam:PassRole"],
    "ecs:CreateService":                   ["ec2:CreateTags", "ec2:DescribeSubnets", "ec2:DescribeVpcs", "ec2:DescribeSecurityGroups", "iam:PassRole"],
    "ecs:RegisterTaskDefinition":          ["iam:PassRole"],
    "ecs:UpdateService":                   ["iam:PassRole"],
    # IAM role passing
    "codepipeline:StartPipelineExecution": ["iam:PassRole"],
    "codepipeline:CreatePipeline":         ["iam:PassRole"],
    "cloudformation:CreateStack":          ["iam:PassRole", "iam:CreateRole"],
    "cloudformation:ExecuteChangeSet":     ["iam:PassRole"],
    "cloudformation:CreateChangeSet":      ["iam:PassRole"],
    "codebuild:StartBuild":                ["iam:PassRole"],
    "glue:StartJobRun":                    ["iam:PassRole"],
    "lambda:CreateFunction":               ["iam:PassRole"],
    "states:StartExecution":               ["iam:PassRole"],
    # CloudWatch Logs
    "logs:CreateLogStream":                ["logs:CreateLogGroup", "logs:PutLogEvents"],
    "logs:PutLogEvents":                   ["logs:CreateLogGroup", "logs:CreateLogStream"],
    "logs:CreateLogGroup":                 ["logs:CreateLogStream", "logs:PutLogEvents"],
    # ECR pulling
    "ecr:BatchGetImage":                   ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer"],
    "ecr:GetDownloadUrlForLayer":          ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"],
    "ecr:BatchCheckLayerAvailability":     ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
    # ECR pushing
    "ecr:PutImage":                        ["ecr:GetAuthorizationToken", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:BatchCheckLayerAvailability"],
    "ecr:InitiateLayerUpload":             ["ecr:GetAuthorizationToken", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage"],
    # DynamoDB streams
    "dynamodb:GetRecords":                 ["dynamodb:GetShardIterator", "dynamodb:DescribeStream", "dynamodb:ListStreams"],
    "dynamodb:GetShardIterator":           ["dynamodb:GetRecords", "dynamodb:DescribeStream", "dynamodb:ListStreams"],
    # SQS
    "sqs:ReceiveMessage":                  ["sqs:DeleteMessage", "sqs:GetQueueAttributes"],
    "sqs:DeleteMessage":                   ["sqs:ReceiveMessage", "sqs:GetQueueAttributes"],
    # Kinesis
    "kinesis:GetRecords":                  ["kinesis:GetShardIterator", "kinesis:DescribeStream", "kinesis:ListStreams"],
    "kinesis:PutRecord":                   ["kinesis:PutRecords", "kinesis:DescribeStream"],
    # S3
    "s3:GetObject":                        ["s3:ListBucket"],
    "s3:PutObject":                        ["s3:GetObject", "s3:ListBucket"],
    "s3:ListBucket":                       ["s3:GetObject", "s3:GetBucketLocation"],
    # KMS
    "kms:Decrypt":                         ["kms:DescribeKey", "kms:GenerateDataKey"],
    "kms:GenerateDataKey":                 ["kms:Decrypt", "kms:DescribeKey"],
    "kms:Encrypt":                         ["kms:DescribeKey", "kms:GenerateDataKey"],
    # Secrets Manager
    "secretsmanager:GetSecretValue":       ["secretsmanager:DescribeSecret", "kms:Decrypt"],
    # CodeDeploy
    "codedeploy:CreateDeployment":         ["codedeploy:GetDeployment", "codedeploy:GetDeploymentConfig", "codedeploy:GetDeploymentGroup"],
    # Step Functions
    "states:DescribeExecution":            ["states:GetExecutionHistory", "states:ListExecutions"],
    # RDS
    "rds:CreateDBInstance":                ["rds:DescribeDBInstances", "rds:DescribeDBSubnetGroups", "ec2:DescribeVpcs", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups"],
    # CloudWatch
    "cloudwatch:PutMetricData":            ["cloudwatch:GetMetricStatistics", "cloudwatch:ListMetrics"],
    "cloudwatch:PutMetricAlarm":           ["cloudwatch:DescribeAlarms", "cloudwatch:GetMetricData"],
}


class CompanionRuleAugmenter:
    """
    Post-retrieval augmentation using deterministic companion rules.

    Addresses the fundamental corpus gap: actions like iam:PassRole and
    ec2:CreateTags are always implicitly required alongside certain primary
    actions, but their AWS documentation never describes this dependency.
    No embedding-based retrieval can surface them — hence deterministic rules.

    Applied as the final step in DecomposedRetriever.retrieve().
    """

    @staticmethod
    def _normalize(action: str) -> str:
        """Strip suffixes like '[permission only]' and lowercase."""
        return action.split(" [")[0].strip().lower()

    def augment(self, retrieved_chunks: list[dict], dense_retriever: DenseRetriever) -> list[dict]:
        """
        Find companion actions missing from retrieved results and inject them.

        Args:
            retrieved_chunks: current retrieval results
            dense_retriever:  used to look up companion action chunks by name

        Returns:
            original chunks + any missing companion action chunks
        """
        retrieved_actions = {
            self._normalize(c["metadata"].get("action", ""))
            for c in retrieved_chunks
            if c["metadata"]["type"] == "iam_action"
        }

        rules_lower = {
            k.lower(): [v.lower() for v in vals]
            for k, vals in COMPANION_RULES.items()
        }

        needed = {
            companion
            for action in retrieved_actions
            for companion in rules_lower.get(action, [])
            if companion not in retrieved_actions
        }

        if not needed:
            return retrieved_chunks

        companion_chunks = []
        for chunk in dense_retriever.chunks:
            if chunk["metadata"]["type"] != "iam_action":
                continue
            action_norm = self._normalize(chunk["metadata"].get("action", ""))
            if action_norm in needed:
                companion_chunk         = dict(chunk)
                companion_chunk["score"]  = 1.0   # deterministic — not similarity-ranked
                companion_chunk["source"] = "companion_rule"
                companion_chunks.append(companion_chunk)
                needed.discard(action_norm)
            if not needed:
                break

        if companion_chunks:
            print(f"  + {len(companion_chunks)} companion action(s) added by rules")

        return retrieved_chunks + companion_chunks


# ---------------------------------------------------------------------------
# Decomposed Retriever  ←  final selected strategy
# ---------------------------------------------------------------------------

class DecomposedRetriever:
    """
    Final selected retrieval strategy (Recall=0.574, Zero-recall=4/100).

    Combines three components:
      1. QueryDecomposer (src/decomposer.py) — GPT-4o breaks the query into
         N service-specific sub-queries
      2. DenseRetriever — runs independently for each sub-query, retrieving
         k_action_per_subquery actions and k_policy_per_subquery policies;
         results are deduplicated across sub-queries
      3. CompanionRuleAugmenter — appends deterministic companion actions
         that no retrieval model can surface from the corpus

    Optimal k values from k-parameter sweep on 100-policy development set:
      k_action_per_subquery = 4
      k_policy_per_subquery = 2
    """

    def __init__(self, dense_retriever: DenseRetriever, decomposer):
        """
        Args:
            dense_retriever: initialized DenseRetriever instance
            decomposer:      initialized QueryDecomposer instance
                             (from src/decomposer.py)
        """
        self.dense     = dense_retriever
        self.decomposer = decomposer
        self.augmenter  = CompanionRuleAugmenter()

    def retrieve(
        self,
        query: str,
        k_action_per_subquery: int = 4,
        k_policy_per_subquery: int = 2,
        apply_companion_rules: bool = True,
    ) -> list[dict]:
        """
        Decompose query, retrieve per sub-query, merge, deduplicate, augment.

        Args:
            query:                 natural language task description
            k_action_per_subquery: action chunks per sub-query (optimal: 4)
            k_policy_per_subquery: policy chunks per sub-query (optimal: 2)
            apply_companion_rules: whether to run CompanionRuleAugmenter

        Returns:
            deduplicated list of retrieved chunks, companion-augmented
        """
        sub_queries = self.decomposer.decompose(query)
        print(f"  Decomposed into {len(sub_queries)} sub-queries")

        seen_actions  : set[str] = set()
        seen_policies : set[str] = set()
        all_results   : list[dict] = []

        for sq in sub_queries:
            for r in self.dense.retrieve(sq, k_action_per_subquery, k_policy_per_subquery):
                meta = r["metadata"]
                key  = meta.get("action") if meta["type"] == "iam_action" else meta.get("policy_name")
                seen = seen_actions if meta["type"] == "iam_action" else seen_policies
                if key and key not in seen:
                    seen.add(key)
                    all_results.append(r)

        if apply_companion_rules:
            all_results = self.augmenter.augment(all_results, self.dense)

        return all_results
