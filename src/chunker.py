"""
chunker.py
==========
Converts raw IAM action documents and managed policies into
retrieval-ready chunks. Each chunk carries:

  text      → rich natural-language string for embedding
  metadata  → structured original data for generation + evaluation

Two chunking strategies, one per source:

  ActionChunker   → one chunk per IAM action (entity-level)
                    preserves action name, description, access level,
                    resource types, and condition keys

  PolicyChunker   → one chunk per managed policy
                    rebuilds a full-text representation that includes
                    effect, actions, resources, and conditions from
                    every statement (not just action names)

Usage
-----
    from src.chunker import ActionChunker, PolicyChunker

    action_chunks = ActionChunker().chunk(action_documents)
    policy_chunks = PolicyChunker().chunk(managed_policies)
"""

import json
from pathlib import Path


class ActionChunker:
    """
    Converts IAM action documents into retrieval chunks.
    One chunk per action — uses the full_text field produced by the
    collector as the embedding string, and preserves all structured
    fields in metadata for the generator.
    """

    def chunk(self, action_data: list[dict]) -> list[dict]:
        """
        Convert a list of action documents into chunks.

        Each output chunk has the shape:
          {
            "text":     "<embedding string>",
            "metadata": {
              "type":              "iam_action",
              "doc_id":            "s3_getobject",
              "action":            "s3:GetObject",
              "action_name":       "GetObject",
              "service":           "Amazon S3",
              "service_prefix":    "s3",
              "description":       "Grants permission to ...",
              "access_level":      "Read",
              "resource_types":    ["object"],
              "condition_keys":    ["s3:prefix", ...],
              "dependent_actions": [],
            }
          }
        """
        chunks = []
        skipped = 0

        for item in action_data:
            if not item.get("full_text") or not item.get("description"):
                skipped += 1
                continue

            chunks.append({
                "text": item["full_text"],
                "metadata": {
                    "type":              "iam_action",
                    "doc_id":            item["doc_id"],
                    "action":            item["action"],
                    "action_name":       item["action_name"],
                    "service":           item["service"],
                    "service_prefix":    item["service_prefix"],
                    "description":       item["description"],
                    "access_level":      item["access_level"],
                    "resource_types":    item["resource_types"],
                    "condition_keys":    item["condition_keys"],
                    "dependent_actions": item["dependent_actions"],
                },
            })

        print(f"ActionChunker: {len(chunks)} chunks, {skipped} skipped")
        return chunks


class PolicyChunker:
    """
    Converts managed policy documents into retrieval chunks.
    One chunk per policy — rebuilds a rich full_text from the policy
    structure (effect, actions, resources, conditions) rather than
    storing only action names. This improves retrieval by grounding
    the embedding in how actions are actually used together.
    """

    def chunk(self, policy_data: list[dict]) -> list[dict]:
        """
        Convert a list of managed policies into chunks.

        Each output chunk has the shape:
          {
            "text":     "<embedding string — full policy semantics>",
            "metadata": {
              "type":            "iam_policy",
              "policy_name":     "AmazonS3ReadOnlyAccess",
              "url":             "https://...",
              "actions_used":    ["s3:GetObject", "s3:ListBucket", ...],
              "policy_document": { <original JSON> },
            }
          }
        """
        chunks = []
        skipped = 0

        for policy in policy_data:
            if not policy.get("policy_document"):
                skipped += 1
                continue

            chunks.append({
                "text": self._build_full_text(policy),
                "metadata": {
                    "type":            "iam_policy",
                    "policy_name":     policy["policy_name"],
                    "url":             policy.get("url", ""),
                    # De-duplicate actions for cleaner metadata
                    "actions_used":    list(set(policy.get("actions_used", []))),
                    # Retain full JSON so generator can read correct policy structure
                    "policy_document": policy["policy_document"],
                },
            })

        print(f"PolicyChunker: {len(chunks)} chunks, {skipped} skipped")
        return chunks

    @staticmethod
    def _build_full_text(policy: dict) -> str:
        """
        Build a rich embedding string from a managed policy document.

        Includes all statement details (effect, actions, resources, conditions)
        rather than just action names, so the embedding captures how this
        policy combines permissions for specific tasks and resources.
        """
        lines = [f"AWS Managed Policy: {policy['policy_name']}."]
        statements = policy.get("policy_document", {}).get("Statement", [])

        for i, stmt in enumerate(statements):
            lines.append(f"Statement {i + 1}:")

            if "Sid" in stmt:
                lines.append(f"  Purpose: {stmt['Sid']}.")

            effect = stmt.get("Effect", "Allow")
            lines.append(f"  Effect: {effect}.")

            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            lines.append(f"  Actions: {', '.join(actions)}.")

            resources = stmt.get("Resource", [])
            if isinstance(resources, str):
                resources = [resources]
            lines.append(f"  Resources: {', '.join(resources)}.")

            conditions = stmt.get("Condition", {})
            if conditions:
                for operator, condition_block in conditions.items():
                    for key, value in condition_block.items():
                        value_str = (
                            ", ".join(value) if isinstance(value, list)
                            else str(value)
                        )
                        lines.append(
                            f"  Condition: {operator} {key} = {value_str}."
                        )

        return " ".join(lines)


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Save chunks to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(path: str) -> list[dict]:
    """Load chunks from a JSON file."""
    with open(path) as f:
        return json.load(f)
