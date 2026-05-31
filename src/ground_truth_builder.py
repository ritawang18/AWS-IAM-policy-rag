"""
ground_truth_builder.py
=======================
Builds the evaluation ground truth dataset from two sources and generates
natural language queries for each policy using GPT-4o.

Sources
-------
  1. GitHub CloudFormation templates  (→ 45 policies via GitHubCollector below)
  2. AWS Managed Policy Reference     (→ 30 policies selected here)

Pipeline
--------
  Step 1 — Select 30 held-out managed policies
    Scores policies in the collected corpus and picks a quality subset
    that (a) covers services underrepresented in GitHub policies,
    (b) has 3–20 specific non-wildcard actions, and (c) is not too
    niche (penalizes services with few corpus entries).

  Step 2 — GPT-4o query generation
    For each policy (GitHub + managed), calls GPT-4o to write a
    task-level natural language description. The system prompt enforces:
      - No AWS service names (no "S3", "Lambda", "DynamoDB", ...)
      - No IAM action names (no "GetObject", "PutItem", ...)
      - Task-level statements, not API-level descriptions
      - Multi-service policies describe all distinct functional areas
    This methodology follows Gorilla (Patil et al.) — queries describe
    what the developer wants to do, not which API to call.

  Step 3 — Corpus decontamination
    The 30 selected managed policies are removed from the FAISS index
    and combined_chunks.json to prevent data leakage during evaluation.

Usage
-----
    from src.ground_truth_builder import GroundTruthBuilder

    builder = GroundTruthBuilder(
        openai_api_key="...",
        voyage_api_key="...",   # only needed for decontamination step
    )

    # Step 1 — select managed policies for eval (pass already-collected corpus)
    managed_policies = builder.select_managed_policies(
        corpus_policies_path="data/processed/managed_policies.json",
        n=30,
    )

    # Step 2 — combine with GitHub policies and generate queries
    github_policies = json.load(open("data/processed/github_iam_policies_clean.json"))
    ground_truth = builder.build_ground_truth(
        github_policies=github_policies,
        managed_policies=managed_policies,
        output_path="data/evaluation/evaluation_ground_truth.json",
    )

    # Step 3 — remove eval policies from the retrieval corpus
    builder.decontaminate_corpus(
        ground_truth_path="data/evaluation/evaluation_ground_truth.json",
        dense_index_dir="indexes/dense/",
    )
"""

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
import re
import yaml
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ---------------------------------------------------------------------------
# GPT-4o system prompt for query generation
#
# Enforces task-level descriptions with no service/action name leakage.
# Follows Gorilla methodology: query describes intent, not API calls.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GitHubCollector
#
# Scrapes CloudFormation templates from aws-samples and awslabs to collect
# real-world IAM policies for the evaluation ground truth dataset.
# These policies are NEVER part of the retrieval corpus — eval only.
# ---------------------------------------------------------------------------

class GitHubCollector:
    """
    Collects real-world IAM policies from GitHub CloudFormation templates.
    Used exclusively for building the evaluation ground truth dataset (45 of
    the 75 evaluation policies come from here).

    Target orgs: aws-samples, awslabs
    Target repos: serverless/lambda/ecs/fargate/dynamodb/sqs/eventbridge topics
    """

    API_BASE = "https://api.github.com"

    TARGET_ORGS = ["aws-samples", "awslabs"]
    TARGET_TOPICS = [
        "serverless", "lambda", "ecs", "fargate",
        "api-gateway", "dynamodb", "sqs", "eventbridge",
    ]

    def __init__(self, github_token: str):
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def collect(
        self,
        target_orgs: list[str] | None = None,
        target_topics: list[str] | None = None,
        min_actions: int = 2,
        max_actions: int = 25,
    ) -> list[dict]:
        """
        Scrape CloudFormation templates and extract IAM policies.

        Quality filters applied:
          - 2–25 specific non-wildcard actions
          - No service-level wildcards (s3:*)
          - Must touch at least one of the 56 core AWS services
          - No unresolved CloudFormation intrinsic functions in action fields

        Returns:
            list of policy dicts with fields:
              policy_name, repo, source_url, source, actions,
              services, policy_document
        """
        if target_orgs is None:
            target_orgs = self.TARGET_ORGS
        if target_topics is None:
            target_topics = self.TARGET_TOPICS

        repos = self._collect_repos(target_orgs, target_topics)
        policies = []

        for repo in tqdm(repos, desc="Scanning GitHub repos for IAM policies"):
            cfn_files = self._find_cfn_files(repo["owner"], repo["name"])
            for f in cfn_files:
                content = self._fetch_file_content(repo["owner"], repo["name"], f["path"])
                if content:
                    for p in self._extract_policies(content, f["html_url"]):
                        if self._is_useful_policy(p["policy_document"], min_actions, max_actions):
                            actions = self._extract_actions(p["policy_document"])
                            services = sorted({a.split(":")[0].lower() for a in actions if ":" in a})
                            policies.append({
                                "policy_name":     p["resource_name"],
                                "repo":            f"{repo['owner']}/{repo['name']}",
                                "source_url":      p["source_url"],
                                "source":          "github",
                                "actions":         actions,
                                "services":        services,
                                "policy_document": p["policy_document"],
                            })
            time.sleep(0.5)

        # Remove near-duplicates (same action set from different repos)
        seen_action_sets: set = set()
        unique = []
        for p in policies:
            key = frozenset(p["actions"])
            if key not in seen_action_sets:
                seen_action_sets.add(key)
                unique.append(p)

        print(f"Collected {len(unique)} unique GitHub IAM policies "
              f"({len(policies) - len(unique)} duplicates removed)")
        return unique

    def _collect_repos(self, orgs: list[str], topics: list[str]) -> list[dict]:
        """Search repos by org + topic, deduplicating by repo ID."""
        seen_ids: set = set()
        repos = []
        for org in orgs:
            for topic in topics:
                r = requests.get(
                    f"{self.API_BASE}/search/repositories",
                    headers=self.headers,
                    params={"q": f"org:{org} topic:{topic}", "sort": "stars", "per_page": 20},
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                for item in r.json().get("items", []):
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        repos.append({"owner": item["owner"]["login"], "name": item["name"]})
                time.sleep(0.5)
        return repos

    def _find_cfn_files(self, owner: str, repo: str) -> list[dict]:
        """Traverse repo file tree to find CloudFormation template files."""
        r = requests.get(f"{self.API_BASE}/repos/{owner}/{repo}", headers=self.headers, timeout=10)
        if r.status_code != 200:
            return []
        branch = r.json().get("default_branch", "main")

        r = requests.get(
            f"{self.API_BASE}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1",
            headers=self.headers, timeout=15,
        )
        if r.status_code != 200:
            return []

        skip_dirs = ["node_modules", ".github", "test/", "tests/", "dist/", "build/", ".aws-sam/"]
        files = []
        for item in r.json().get("tree", []):
            if item.get("type") != "blob":
                continue
            path = item.get("path", "").lower()
            if not (path.endswith(".yaml") or path.endswith(".yml") or path.endswith(".json")):
                continue
            if any(s in path for s in skip_dirs):
                continue
            files.append({
                "path":     item["path"],
                "html_url": f"https://github.com/{owner}/{repo}/blob/{branch}/{item['path']}",
            })
        return files[:10]  # cap per repo to avoid very large repos dominating

    def _fetch_file_content(self, owner: str, repo: str, path: str) -> str | None:
        """Fetch raw file content from GitHub."""
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
        r = requests.get(url, headers=self.headers, timeout=10)
        return r.text if r.status_code == 200 else None

    def _extract_policies(self, content: str, source_url: str) -> list[dict]:
        """Parse a CloudFormation template and extract inline IAM policies."""
        template = None
        try:
            template = json.loads(content)
        except json.JSONDecodeError:
            try:
                template = yaml.safe_load(content)
            except Exception:
                return []

        if not isinstance(template, dict) or "Resources" not in template:
            return []

        policies = []
        for name, resource in template.get("Resources", {}).items():
            if not isinstance(resource, dict):
                continue
            rtype = resource.get("Type", "")
            props = resource.get("Properties", {}) or {}

            if rtype in ("AWS::IAM::Policy", "AWS::IAM::ManagedPolicy"):
                doc = props.get("PolicyDocument", {})
                if isinstance(doc, dict) and "Statement" in doc:
                    policies.append({"resource_name": name, "policy_document": doc, "source_url": source_url})

            elif rtype == "AWS::IAM::Role":
                # Merge all inline policies on the role into one document
                all_actions: list = []
                for inline in props.get("Policies", []):
                    doc = inline.get("PolicyDocument", {}) if isinstance(inline, dict) else {}
                    all_actions.extend(self._extract_actions(doc))
                if all_actions:
                    merged = {
                        "Version": "2012-10-17",
                        "Statement": [{"Effect": "Allow", "Action": list(set(all_actions)), "Resource": "*"}],
                    }
                    policies.append({"resource_name": name, "policy_document": merged, "source_url": source_url})

        return policies

    def _is_useful_policy(self, policy_doc: dict, min_actions: int, max_actions: int) -> bool:
        """
        Filter for policies that make good ground-truth evaluation entries.
        Rejects admin policies, wildcard-heavy policies, unresolved CF
        intrinsics, and policies outside the target action count range.
        """
        actions = self._extract_actions(policy_doc)
        if not actions or "*" in actions:
            return False
        if any(a.endswith(":*") for a in actions):
            return False
        if any(not isinstance(a, str) or ":" not in a for a in actions):
            return False  # unresolved CF intrinsic functions

        target_prefixes = {
            "s3", "lambda", "dynamodb", "sqs", "sns", "logs", "cloudwatch",
            "events", "ecs", "ecr", "ssm", "kms", "secretsmanager", "sts",
            "ec2", "iam", "xray", "cloudformation", "codebuild", "codepipeline",
        }
        prefixes = {a.split(":")[0].lower() for a in actions}
        if not prefixes & target_prefixes:
            return False

        return min_actions <= len(actions) <= max_actions

    @staticmethod
    def _extract_actions(policy_doc: dict) -> list[str]:
        """Flatten all Allow actions from a policy document."""
        actions = []
        for stmt in policy_doc.get("Statement", []):
            if not isinstance(stmt, dict):
                continue
            raw = stmt.get("Action", [])
            if isinstance(raw, str):
                raw = [raw]
            actions.extend(raw)
        return actions

    def save(self, policies: list[dict], output_path: str) -> None:
        """Save collected GitHub policies to a JSON file."""
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(policies, f, indent=2)
        print(f"Saved {len(policies)} GitHub ground-truth policies to {output_path}")



QUERY_GENERATION_SYSTEM_PROMPT = """\
You are a developer writing a task description that will be used to generate
an AWS IAM policy. Given an IAM policy, write a clear, specific task description
that captures exactly what permissions are needed and why.

RULES:
- Do NOT mention specific AWS service names (e.g. do not say "S3", "Lambda",
  "DynamoDB", "SQS", "ECR", "CloudWatch", "EventBridge", "SSM", "KMS", etc.)
- Do NOT mention specific IAM action names (e.g. do not say "GetObject",
  "PutItem", "InvokeFunction", "CreateLogGroup", etc.)
- Use a descriptive subject that reflects the role's purpose
  (e.g. 'Allow a container build process to...' not just 'Allow a function to...')
  but do not use AWS service names as the subject.
- Focus only on primary functional tasks. Do not describe supporting
  infrastructure permissions such as network interface management,
  resource tagging, or identity assumption as separate tasks unless
  they are the main purpose of the policy.
- If the policy includes role-passing permissions, do not mention them
  as a separate task — they are implicit supporting permissions.
  Only describe the primary functional tasks.
- Write as a statement, not a question
- Do not wrap the query in quotation marks
- Be specific about WHAT the service/role needs to do, not just that it needs access
- Mention the different functional areas covered if the policy spans multiple tasks
- Keep to 1-3 sentences

GOOD EXAMPLES:

Policy actions: s3:GetObject, s3:ListBucket, s3:GetBucketLocation
Good query: Allow read-only access to retrieve objects and list contents of a storage bucket

Policy actions: sqs:ReceiveMessage, sqs:DeleteMessage, sqs:GetQueueAttributes, dynamodb:PutItem, logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
Good query: Allow a function to consume messages from a queue, store processed results in a database table, and write execution logs

Policy actions: ecr:GetAuthorizationToken, ecr:BatchGetImage, ecr:GetDownloadUrlForLayer, ecr:BatchCheckLayerAvailability, logs:CreateLogStream, logs:PutLogEvents
Good query: Allow a container runtime to authenticate and pull container images from a private registry and write task execution logs

Policy actions: codedeploy:CreateDeployment, codedeploy:GetDeployment, codedeploy:GetDeploymentConfig, ecs:DescribeServices, ecs:UpdateService
Good query: Allow a deployment service to create and monitor deployments and update the target container service during rollout

BAD EXAMPLES:

Too vague:
  Allow the ability to trigger specific functions related to account inspection
  → Does not describe the actual task clearly enough

Mentions service names:
  Allow Lambda to read from SQS and write to DynamoDB
  → SQS, Lambda, DynamoDB are service names

Question format:
  How can I enable read access to my storage bucket?
  → Should be a statement not a question

Too generic:
  Allow full access to manage files in a storage bucket
  → "Full access" is vague — specify the operations (read, write, delete, list)

IMPORTANT: The query must be specific enough that someone reading it would know
exactly which functional areas need permissions. If the policy covers multiple
services doing different things, mention each distinct task.\
"""


class GroundTruthBuilder:
    """
    Builds the evaluation ground truth dataset.

    Covers three responsibilities:
      1. Selecting 30 managed policies for the held-out evaluation set
      2. Generating GPT-4o natural language queries for all 75 policies
      3. Decontaminating the retrieval corpus by removing evaluation policies
    """

    def __init__(self, openai_api_key: str, voyage_api_key: str | None = None):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.voyage_api_key = voyage_api_key

    # -------------------------------------------------------------------------
    # Step 1 — Select 30 managed policies for the held-out evaluation set
    # -------------------------------------------------------------------------

    def select_managed_policies(
        self,
        corpus_policies_path: str,
        n: int = 30,
        min_actions: int = 3,
        max_actions: int = 20,
        core_services: set[str] | None = None,
    ) -> list[dict]:
        """
        Select n managed policies from the collected corpus for use as
        held-out evaluation ground truth.

        Selection criteria:
          - 3–20 specific non-wildcard actions (broad enough to be interesting,
            narrow enough to evaluate meaningfully)
          - At least one of the 56 core services covered
          - Penalizes niche services underrepresented in the corpus
          - Balanced service coverage (avoids e.g. all-EC2 eval set)

        These selected policies are REMOVED from the retrieval corpus in Step 3
        to prevent data leakage during evaluation.

        Args:
            corpus_policies_path: path to data/processed/managed_policies.json
            n:                    number of policies to select (default: 30)
            min_actions:          minimum non-wildcard actions per policy
            max_actions:          maximum actions per policy
            core_services:        IAM prefixes considered "core" for scoring

        Returns:
            list of n selected policy dicts formatted for evaluation ground truth
        """
        if core_services is None:
            core_services = {
                "s3", "lambda", "dynamodb", "ec2", "iam", "logs",
                "cloudwatch", "events", "ecs", "ecr", "ssm", "kms",
                "rds", "sqs", "sns", "secretsmanager", "xray",
                "cloudformation", "codebuild", "codepipeline", "states",
            }

        with open(corpus_policies_path) as f:
            all_policies = json.load(f)

        print(f"Scoring {len(all_policies)} managed policies for eval selection...")

        scored = []
        for policy in all_policies:
            doc     = policy.get("policy_document", {})
            actions = self._extract_actions(doc)

            # Basic filters
            specific = [a for a in actions if isinstance(a, str) and ":" in a and "*" not in a]
            if len(specific) < min_actions or len(specific) > max_actions:
                continue
            if any("*" in a for a in actions):
                continue

            services = {a.split(":")[0].lower() for a in specific if ":" in a}
            if not services & core_services:
                continue

            # Score: reward core service coverage, penalize niche services
            core_coverage = len(services & core_services)
            niche_penalty = len(services - core_services) * 0.5
            n_actions_score = len(specific) / max_actions  # reward more specific actions

            score = core_coverage + n_actions_score - niche_penalty

            scored.append({
                "policy": policy,
                "score":  score,
                "n_specific_actions": len(specific),
                "services": list(services),
            })

        # Sort by score and sample top candidates for diversity
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored[:200]

        # Greedy service-diversity sampling from top candidates
        selected = self._diversity_sample(top_candidates, n)

        print(f"Selected {len(selected)} policies for evaluation set")
        for i, entry in enumerate(selected[:5]):
            print(f"  [{i}] {entry['policy']['policy_name']}: "
                  f"{entry['n_specific_actions']} actions, "
                  f"services: {entry['services']}")

        return [
            {
                "policy_name":     e["policy"]["policy_name"],
                "url":             e["policy"].get("url", ""),
                "source":          "aws_managed",
                "actions":         self._extract_actions(e["policy"]["policy_document"]),
                "services":        e["services"],
                "policy_document": e["policy"]["policy_document"],
            }
            for e in selected
        ]

    @staticmethod
    def _diversity_sample(candidates: list[dict], n: int) -> list[dict]:
        """
        Greedy diversity sampling: at each step pick the highest-scoring
        candidate that adds at least one new service not already in the
        selected set.
        """
        selected = []
        covered_services: set[str] = set()

        # First pass: add diversity
        remaining = list(candidates)
        while len(selected) < n and remaining:
            best_idx = None
            best_score = -1
            for i, entry in enumerate(remaining):
                new_services = set(entry["services"]) - covered_services
                diversity_bonus = len(new_services) * 0.2
                adjusted = entry["score"] + diversity_bonus
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = i
            if best_idx is None:
                break
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            covered_services.update(chosen["services"])

        # Fill remaining slots with highest-scoring if not enough unique services
        if len(selected) < n:
            for entry in candidates:
                if entry not in selected:
                    selected.append(entry)
                if len(selected) >= n:
                    break

        return selected[:n]

    # -------------------------------------------------------------------------
    # Step 2 — GPT-4o query generation
    # -------------------------------------------------------------------------

    def build_ground_truth(
        self,
        github_policies: list[dict],
        managed_policies: list[dict],
        output_path: str,
        checkpoint_every: int = 10,
    ) -> list[dict]:
        """
        Combine GitHub and managed policies, generate GPT-4o queries for each,
        and save the complete evaluation ground truth dataset.

        Args:
            github_policies:   45 policies from GitHubCollector
            managed_policies:  30 policies selected by select_managed_policies()
            output_path:       where to save evaluation_ground_truth.json
            checkpoint_every:  save progress every N policies

        Returns:
            list of 75 ground truth entries with "query" field added
        """
        all_policies = github_policies + managed_policies
        print(f"Generating queries for {len(all_policies)} policies "
              f"({len(github_policies)} GitHub + {len(managed_policies)} managed)...")

        failed = []
        for i, entry in enumerate(all_policies):
            policy_name = entry.get("policy_name", f"policy_{i}")
            source      = entry.get("source", "unknown")
            print(f"[{i+1:02d}/{len(all_policies)}] {policy_name} ({source})")

            query = self._generate_query(entry)
            if query:
                entry["query"] = query
                print(f"  → {query[:100]}{'...' if len(query) > 100 else ''}")
            else:
                failed.append(i)
                print("  → FAILED")

            time.sleep(0.5)  # respect OpenAI rate limit (~500 RPM on tier 1)

            if (i + 1) % checkpoint_every == 0:
                self._save(all_policies, output_path)
                print(f"  [checkpoint saved at {i+1} policies]")

        self._save(all_policies, output_path)

        print(f"\n{'='*55}")
        print(f"Query generation complete")
        print(f"  Success: {len(all_policies) - len(failed)} / {len(all_policies)}")
        if failed:
            print(f"  Failed indices: {failed}")
        print(f"{'='*55}")

        return all_policies

    def _generate_query(self, policy_entry: dict) -> str | None:
        """
        Generate a natural language query for a single policy entry.

        Passes the full policy document, action list, source context (GitHub
        resource name / repo, or managed policy name), and a service-count
        hint for multi-service policies to the GPT-4o model.

        Returns:
            query string, or None if API call failed
        """
        policy_doc = policy_entry["policy_document"]
        actions    = policy_entry.get("actions", self._extract_actions(policy_doc))
        source     = policy_entry.get("source", "unknown")

        context_parts = []

        if source == "github":
            context_parts.append(
                f"CloudFormation resource name: {policy_entry.get('policy_name', '')}"
            )
            context_parts.append(
                f"Source repository: {policy_entry.get('repo', '')}"
            )
        else:
            context_parts.append(
                f"AWS managed policy name: {policy_entry.get('policy_name', '')}"
            )

        context_parts.append(
            f"\nIAM Policy Document:\n{json.dumps(policy_doc, indent=2)}"
        )
        context_parts.append(
            f"\nSpecific actions granted: {', '.join(actions)}"
        )

        services = policy_entry.get("services", [])
        if len(services) > 2:
            context_parts.append(
                f"\nThis policy covers {len(services)} different service areas: "
                f"{', '.join(services)}. Make sure the query describes all "
                f"distinct functional tasks covered."
            )

        context_parts.append(
            "\n\nWrite a task description for this policy following the rules above."
        )

        user_message = "\n".join(context_parts)

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.2,  # low for consistency across runs
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Query generation error: {e}")
            return None

    # -------------------------------------------------------------------------
    # Step 3 — Corpus decontamination
    # -------------------------------------------------------------------------

    def decontaminate_corpus(
        self,
        ground_truth_path: str,
        dense_index_dir: str,
    ) -> None:
        """
        Remove evaluation managed policies from the FAISS index and
        combined_chunks.json to prevent data leakage during evaluation.

        Only AWS managed policies need to be removed (they were in the corpus).
        GitHub policies were never in the corpus — they are external ground truth.

        Overwrites:
          {dense_index_dir}/combined_chunks.json    ← filtered chunk list
          {dense_index_dir}/combined_faiss.index    ← rebuilt FAISS index

        Also saves pre-decontamination backups:
          {dense_index_dir}/combined_chunks_pre_decontam.json
          {dense_index_dir}/combined_faiss_pre_decontam.index

        Args:
            ground_truth_path: path to evaluation_ground_truth.json
            dense_index_dir:   directory containing combined_chunks.json
                               and combined_faiss.index
        """
        import voyageai  # only needed here

        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Collect names of managed policies to remove
        held_out_names = {
            p["policy_name"]
            for p in ground_truth
            if p.get("source") == "aws_managed"
        }
        print(f"Removing {len(held_out_names)} evaluation policies from corpus:")
        for name in sorted(held_out_names):
            print(f"  - {name}")

        chunks_path = os.path.join(dense_index_dir, "combined_chunks.json")
        index_path  = os.path.join(dense_index_dir, "combined_faiss.index")

        with open(chunks_path) as f:
            all_chunks = json.load(f)

        print(f"\nChunks before decontamination: {len(all_chunks)}")

        # Filter out evaluation policy chunks
        clean_chunks = [
            c for c in all_chunks
            if not (
                c["metadata"]["type"] == "iam_policy"
                and c["metadata"].get("policy_name") in held_out_names
            )
        ]

        removed = len(all_chunks) - len(clean_chunks)
        print(f"Chunks removed:               {removed}")
        print(f"Chunks remaining:             {len(clean_chunks)}")

        # Backup originals
        import shutil
        shutil.copy(chunks_path, chunks_path.replace(".json", "_pre_decontam.json"))
        shutil.copy(index_path,  index_path.replace(".index", "_pre_decontam.index"))
        print("\nOriginals backed up with _pre_decontam suffix")

        # Save cleaned chunks
        with open(chunks_path, "w") as f:
            json.dump(clean_chunks, f)

        # Rebuild FAISS index from remaining chunks
        # Chunks that have embeddings: re-use them
        # Chunks without embeddings: re-embed (shouldn't happen if built correctly)
        vectors_list = []
        missing_embeddings = 0

        for chunk in clean_chunks:
            emb = chunk.get("embedding")
            if emb is not None:
                vectors_list.append(emb)
            else:
                missing_embeddings += 1
                vectors_list.append([0.0] * 1024)  # placeholder

        if missing_embeddings:
            print(f"Warning: {missing_embeddings} chunks missing embeddings — "
                  f"re-run embedder.py to fix")

        vectors = np.array(vectors_list, dtype="float32")
        faiss.normalize_L2(vectors)

        dim   = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss.write_index(index, index_path)

        print(f"\n✓ Decontamination complete")
        print(f"  Cleaned chunks saved: {chunks_path}")
        print(f"  Rebuilt FAISS index:  {index_path} ({index.ntotal} vectors)")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_actions(policy_doc: dict) -> list[str]:
        """Flatten all Allow actions from a policy document."""
        actions = []
        for stmt in policy_doc.get("Statement", []):
            if not isinstance(stmt, dict):
                continue
            raw = stmt.get("Action", [])
            if isinstance(raw, str):
                raw = [raw]
            actions.extend(raw)
        return actions

    @staticmethod
    def _save(data: list[dict], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
