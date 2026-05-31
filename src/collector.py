"""
collector.py
============
Scrapes two AWS documentation sources to build the RAG retrieval corpus.
This module is exclusively for corpus construction — nothing here is used
for evaluation.

  ActionCollector  → AWS Service Authorization Reference → IAM action documents
  PolicyCollector  → AWS Managed Policy Reference        → managed policy documents

For evaluation ground truth collection (GitHub CloudFormation templates,
held-out managed policy selection, GPT-4o query generation), see:
  src/ground_truth_builder.py

Usage
-----
    from src.collector import ActionCollector, PolicyCollector, SERVICES_CONFIG

    # Collect IAM actions for all 56 services
    actions = ActionCollector().collect_all(SERVICES_CONFIG)
    ActionCollector().save(actions, "data/processed/action_documents.json")

    # Collect 500 managed policies (relevance-sampled across core services)
    policies = PolicyCollector().collect(n=500)
    PolicyCollector().save(policies, "data/processed/managed_policies.json")
"""

import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Service configuration — edit to add/remove AWS services from the corpus
# ---------------------------------------------------------------------------

SERVICES_CONFIG = [
    {"url_suffix": "list_amazons3",                          "prefix": "s3",             "display_name": "Amazon S3"},
    {"url_suffix": "list_awslambda",                         "prefix": "lambda",         "display_name": "AWS Lambda"},
    {"url_suffix": "list_amazondynamodb",                    "prefix": "dynamodb",       "display_name": "Amazon DynamoDB"},
    {"url_suffix": "list_amazonec2",                         "prefix": "ec2",            "display_name": "Amazon EC2"},
    {"url_suffix": "list_awsidentityandaccessmanagementiam", "prefix": "iam",            "display_name": "AWS IAM"},
    {"url_suffix": "list_awsiamidentitycenter",              "prefix": "sso",            "display_name": "AWS IAM Identity Center"},
    {"url_suffix": "list_amazonrds",                         "prefix": "rds",            "display_name": "Amazon RDS"},
    {"url_suffix": "list_amazonsns",                         "prefix": "sns",            "display_name": "Amazon SNS"},
    {"url_suffix": "list_amazonsqs",                         "prefix": "sqs",            "display_name": "Amazon SQS"},
    {"url_suffix": "list_amazoncloudwatch",                  "prefix": "cloudwatch",     "display_name": "Amazon CloudWatch"},
    {"url_suffix": "list_amazoncloudwatchlogs",              "prefix": "logs",           "display_name": "Amazon CloudWatch Logs"},
    {"url_suffix": "list_awscloudformation",                 "prefix": "cloudformation", "display_name": "AWS CloudFormation"},
    {"url_suffix": "list_amazonelasticcontainerregistry",    "prefix": "ecr",            "display_name": "Amazon ECR"},
    {"url_suffix": "list_amazonelasticcontainerservice",     "prefix": "ecs",            "display_name": "Amazon ECS"},
    {"url_suffix": "list_awssystemsmanager",                 "prefix": "ssm",            "display_name": "AWS Systems Manager"},
    {"url_suffix": "list_awssecuritytokenservice",           "prefix": "sts",            "display_name": "AWS STS"},
    {"url_suffix": "list_awskeymanagementservice",           "prefix": "kms",            "display_name": "AWS KMS"},
    {"url_suffix": "list_awssecretsmanager",                 "prefix": "secretsmanager", "display_name": "AWS Secrets Manager"},
    {"url_suffix": "list_awsx-ray",                          "prefix": "xray",           "display_name": "AWS X-Ray"},
    # ... add more services here
]


class ActionCollector:
    """
    Scrapes the AWS Service Authorization Reference to collect IAM action
    documents for the retrieval corpus.

    Each action becomes a structured JSON document with a unified full_text
    field that concatenates the action name, service, description, access
    level, resource types, and condition keys into a single embedding string.
    """

    BASE_URL = "https://docs.aws.amazon.com/service-authorization/latest/reference/"

    def collect_service(self, service_cfg: dict) -> list[dict]:
        """
        Scrape all actions for a single AWS service.

        Returns a list of action dicts with fields:
          doc_id, action, action_name, service, service_prefix,
          description, access_level, resource_types, condition_keys,
          dependent_actions, full_text
        """
        url = self.BASE_URL + service_cfg["url_suffix"] + ".html"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  ✗ Failed to fetch {service_cfg['display_name']}: {e}")
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        return self._parse_actions(soup, service_cfg)

    def _parse_actions(self, soup: BeautifulSoup, service_cfg: dict) -> list[dict]:
        """Parse the actions table from the service authorization HTML page."""
        actions = []
        prefix       = service_cfg["prefix"]
        service_name = service_cfg["display_name"]

        for table in soup.find_all("table"):
            for row in table.find_all("tr")[1:]:  # skip header row
                cols = row.find_all("td")
                if len(cols) < 6:
                    continue

                action_name      = cols[0].get_text(strip=True)
                description      = cols[1].get_text(strip=True)
                access_level     = cols[2].get_text(strip=True)
                resource_types   = [r.strip() for r in cols[3].get_text().split("\n") if r.strip()]
                condition_keys   = [c.strip() for c in cols[4].get_text().split("\n") if c.strip()]
                dependent_actions= [d.strip() for d in cols[5].get_text().split("\n") if d.strip()]

                if not action_name or not description:
                    continue

                action_full  = f"{prefix}:{action_name}"
                resource_str = ", ".join(resource_types) if resource_types else "None"
                condition_str= ", ".join(condition_keys) if condition_keys else "None"

                full_text = (
                    f"IAM Action: {action_full}. "
                    f"Service: {service_name}. "
                    f"Description: {description} "
                    f"Access level: {access_level}. "
                    f"Applicable resource types: {resource_str}. "
                    f"Supported condition keys: {condition_str}."
                )

                actions.append({
                    "doc_id":            f"{prefix}_{action_name.lower()}",
                    "action":            action_full,
                    "action_name":       action_name,
                    "service":           service_name,
                    "service_prefix":    prefix,
                    "description":       description,
                    "access_level":      access_level,
                    "resource_types":    resource_types,
                    "condition_keys":    condition_keys,
                    "dependent_actions": dependent_actions,
                    "full_text":         full_text,
                })

        return actions

    def collect_all(self, services_config: list[dict]) -> list[dict]:
        """Collect actions for all services in the config list."""
        all_actions = []
        for svc in tqdm(services_config, desc="Collecting IAM actions"):
            actions = self.collect_service(svc)
            all_actions.extend(actions)
            print(f"  ✓ {svc['display_name']}: {len(actions)} actions")
            time.sleep(0.5)  # be polite to AWS docs servers
        return all_actions

    def save(self, actions: list[dict], output_path: str) -> None:
        """Save collected actions to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(actions, f, indent=2)
        print(f"Saved {len(actions)} actions to {output_path}")


class PolicyCollector:
    """
    Scrapes AWS Managed Policies from the Managed Policy Reference for
    the retrieval corpus.

    Applies relevance-based sampling so the 500 collected policies are
    balanced across the 56 core services rather than alphabetically skewed.

    Note: this class collects the full 500-policy corpus. Selecting which
    30 of these to hold out for evaluation (and removing them from the
    retrieval index) is the responsibility of GroundTruthBuilder in
    src/ground_truth_builder.py.
    """

    BASE_URL = "https://docs.aws.amazon.com/aws-managed-policy/latest/reference/"

    def collect(self, n: int = 500, target_services: set[str] | None = None) -> list[dict]:
        """
        Collect up to n managed policies, sampling for balanced service coverage.

        Returns list of policy dicts with fields:
          policy_name, url, policy_document, actions_used, relevance_score
        """
        if target_services is None:
            target_services = {
                "s3", "lambda", "dynamodb", "ec2", "iam", "logs",
                "cloudwatch", "ecs", "ecr", "ssm", "kms", "rds",
                "sqs", "sns", "secretsmanager", "xray", "cloudformation",
            }

        policy_list = self._get_policy_list()
        scored = self._score_policies(policy_list, target_services)
        scored.sort(key=lambda x: x["relevance_score"], reverse=True)

        collected = []
        for entry in tqdm(scored[:n], desc="Collecting managed policies"):
            policy = self._fetch_policy(entry)
            if policy:
                collected.append(policy)
            time.sleep(0.3)

        return collected

    def _get_policy_list(self) -> list[dict]:
        """Scrape the policy list index page."""
        resp = requests.get(self.BASE_URL + "policy-list.html", timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        return [
            {"name": link.get_text(strip=True), "url": self.BASE_URL + link["href"]}
            for link in soup.find_all("a", href=True)
            if link["href"].endswith(".html") and "policy-list" not in link["href"]
        ]

    def _score_policies(self, policies: list[dict], target_services: set[str]) -> list[dict]:
        """
        Score each policy by how many target-service names appear in its name.
        Avoids the alphabetically skewed default ordering from AWS docs.
        """
        for p in policies:
            name_lower = p["name"].lower()
            p["relevance_score"] = sum(svc in name_lower for svc in target_services)
        return policies

    def _fetch_policy(self, entry: dict) -> dict | None:
        """Fetch and parse a single managed policy page."""
        try:
            resp = requests.get(entry["url"], timeout=15)
            resp.raise_for_status()
        except requests.RequestException:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        code_block = soup.find("code")
        if not code_block:
            return None

        try:
            policy_doc = json.loads(code_block.get_text())
        except json.JSONDecodeError:
            return None

        actions_used = []
        for stmt in policy_doc.get("Statement", []):
            raw = stmt.get("Action", [])
            if isinstance(raw, str):
                raw = [raw]
            actions_used.extend(raw)

        return {
            "policy_name":     entry["name"],
            "url":             entry["url"],
            "policy_document": policy_doc,
            "actions_used":    list(set(actions_used)),
            "relevance_score": entry.get("relevance_score", 0),
        }

    def save(self, policies: list[dict], output_path: str) -> None:
        """Save collected policies to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(policies, f, indent=2)
        print(f"Saved {len(policies)} policies to {output_path}")
