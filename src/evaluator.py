"""
evaluator.py
============
Four-layer evaluation pipeline for IAM policy synthesis.

Since no existing benchmark addresses IAM policy generation from natural
language, we designed a domain-specific evaluation framework:

  Layer 1 – Syntactic Validity
    Checks that output is valid JSON with required IAM schema fields
    (Version, Statement, Effect, Action, Resource).

  Layer 2 – Hallucination Detection
    Compares generated actions against a ground-truth actions database
    (all 4,505 valid IAM actions collected from AWS docs). Any generated
    action not found in this database is a hallucination.
    Supports wildcard expansion via fnmatch pattern matching.

  Layer 3 – Semantic Action Matching
    Computes precision, recall, and F1 between the generated action set
    and the ground-truth policy action set. Also checks resource namespace
    compatibility (e.g., s3:GetObject should reference arn:aws:s3::).

  Layer 4 – LLM-as-Judge (Claude Sonnet)
    Scores functional correctness, security correctness, and resource
    scoping on a 0–10 scale. Also gives a binary acceptability decision.
    Explicitly evaluates condition block adequacy when ground truth
    contains conditions.

Usage
-----
    from src.evaluator import PolicyEvaluator

    evaluator = PolicyEvaluator(
        gt_actions_path="data/processed/ground_truth_actions.json",
        anthropic_api_key="...",
    )

    result = evaluator.evaluate(
        query="Allow Lambda to read from S3",
        generated_policy_str='{"Version": "2012-10-17", ...}',
        gt_policy_doc={"Version": "2012-10-17", "Statement": [...]},
    )
    print(result["layer3_recall"], result["layer2_hallucination_rate"])
"""

import fnmatch
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Resource namespace mapping
# Used in Layer 3 to check that generated resource ARNs reference the
# correct AWS service namespace for each action prefix.
# ---------------------------------------------------------------------------

RESOURCE_NAMESPACE_MAP = {
    "s3":             "arn:aws:s3",
    "lambda":         "arn:aws:lambda",
    "dynamodb":       "arn:aws:dynamodb",
    "ec2":            "arn:aws:ec2",
    "iam":            "arn:aws:iam",
    "logs":           "arn:aws:logs",
    "cloudwatch":     "arn:aws:cloudwatch",
    "events":         "arn:aws:events",
    "ecs":            "arn:aws:ecs",
    "ecr":            "arn:aws:ecr",
    "ssm":            "arn:aws:ssm",
    "kms":            "arn:aws:kms",
    "secretsmanager": "arn:aws:secretsmanager",
    "sqs":            "arn:aws:sqs",
    "sns":            "arn:aws:sns",
    "kinesis":        "arn:aws:kinesis",
    "rds":            "arn:aws:rds",
    "xray":           "arn:aws:xray",
    "cloudformation": "arn:aws:cloudformation",
    "codebuild":      "arn:aws:codebuild",
    "codepipeline":   "arn:aws:codepipeline",
    "states":         "arn:aws:states",
}


class PolicyEvaluator:
    """Four-layer IAM policy evaluation pipeline."""

    def __init__(
        self,
        gt_actions_path: str,
        anthropic_api_key: str | None = None,
    ):
        """
        Args:
            gt_actions_path:   path to ground_truth_actions.json
                               (list of all valid IAM action names)
            anthropic_api_key: Anthropic API key for Layer 4 LLM-as-Judge
                               (optional — skip Layer 4 if not provided)
        """
        with open(gt_actions_path) as f:
            raw = json.load(f)
        # Accept either a flat list of action strings or list of action dicts
        if raw and isinstance(raw[0], dict):
            self.valid_actions = {a["action"].lower() for a in raw if a.get("action")}
        else:
            self.valid_actions = {a.lower() for a in raw}

        self.anthropic_client = None
        if anthropic_api_key:
            import anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        generated_policy_str: str,
        gt_policy_doc: dict,
        run_llm_judge: bool = True,
    ) -> dict[str, Any]:
        """
        Run all four evaluation layers on a generated policy.

        Args:
            query:                natural language query used to generate policy
            generated_policy_str: raw string output from the generator
            gt_policy_doc:        ground-truth policy document (parsed dict)
            run_llm_judge:        whether to run Layer 4 (requires Anthropic key)

        Returns:
            dict of all evaluation metrics across all layers
        """
        results: dict[str, Any] = {"query": query}

        # Layer 1 — syntactic validity
        is_valid, message, policy_doc = self._layer1_syntactic(generated_policy_str)
        results["layer1_valid"]   = is_valid
        results["layer1_message"] = message

        if not is_valid:
            results["parse_failed"] = True
            for key in ["layer2_hallucination_rate", "layer3_precision",
                        "layer3_recall", "layer3_f1"]:
                results[key] = None
            return results

        results["parse_failed"] = False

        # Layer 2 — hallucination detection
        l2 = self._layer2_hallucination(policy_doc)
        results.update({
            "layer2_n_generated":        l2["n_generated"],
            "layer2_n_hallucinated":     l2["n_hallucinated"],
            "layer2_hallucination_rate": l2["hallucination_rate"],
            "layer2_hallucinated":       l2["hallucinated"],
        })

        # Layer 3 — semantic action matching
        l3 = self._layer3_semantic(policy_doc, gt_policy_doc)
        results.update({k: v for k, v in l3.items()})

        # Layer 4 — LLM judge
        if run_llm_judge and self.anthropic_client:
            l4 = self._layer4_llm_judge(query, policy_doc, gt_policy_doc, l3)
            if l4:
                results.update({f"layer4_{k}": v for k, v in l4.items()})

        return results

    # -------------------------------------------------------------------------
    # Layer 1 — Syntactic validity
    # -------------------------------------------------------------------------

    @staticmethod
    def _layer1_syntactic(
        policy_str: str,
    ) -> tuple[bool, str, dict | None]:
        """
        Check that the output is valid JSON with required IAM schema fields.

        Returns:
            (is_valid, message, parsed_dict_or_None)
        """
        try:
            doc = json.loads(policy_str)
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {e}", None

        if not isinstance(doc, dict):
            return False, "Output is not a JSON object", None

        required_keys = {"Version", "Statement"}
        missing = required_keys - doc.keys()
        if missing:
            return False, f"Missing keys: {missing}", None

        if not isinstance(doc["Statement"], list) or not doc["Statement"]:
            return False, "Statement must be a non-empty list", None

        for stmt in doc["Statement"]:
            for field in ("Effect", "Action", "Resource"):
                if field not in stmt:
                    return False, f"Statement missing '{field}'", None

        return True, "valid", doc

    # -------------------------------------------------------------------------
    # Layer 2 — Hallucination detection
    # -------------------------------------------------------------------------

    def _layer2_hallucination(self, policy_doc: dict) -> dict:
        """
        Compare generated actions against the ground-truth actions database.
        Supports wildcard expansion: s3:Get* expands against all s3:Get* actions.
        """
        generated_actions = self._extract_actions(policy_doc)
        hallucinated = []

        for action in generated_actions:
            if not self._is_valid_action(action):
                hallucinated.append(action)

        n = len(generated_actions)
        return {
            "n_generated":        n,
            "n_hallucinated":     len(hallucinated),
            "hallucination_rate": len(hallucinated) / n if n else 0.0,
            "hallucinated":       hallucinated,
        }

    def _is_valid_action(self, action: str) -> bool:
        """Check if an action exists in the ground-truth actions database."""
        action_lower = action.lower()

        # Exact match
        if action_lower in self.valid_actions:
            return True

        # Wildcard expansion: s3:Get* matches any s3:Get... action
        if "*" in action_lower:
            return any(fnmatch.fnmatch(a, action_lower) for a in self.valid_actions)

        return False

    # -------------------------------------------------------------------------
    # Layer 3 — Semantic action matching
    # -------------------------------------------------------------------------

    def _layer3_semantic(
        self, policy_doc: dict, gt_policy_doc: dict
    ) -> dict:
        """
        Precision, recall, F1 between generated and ground-truth action sets,
        plus resource scoping accuracy.
        """
        generated_actions = set(self._extract_actions(policy_doc))
        gt_actions        = set(self._extract_actions(gt_policy_doc))

        # Expand wildcards in generated actions against gt
        expanded_generated = self._expand_wildcards(generated_actions, gt_actions)

        covered = expanded_generated & gt_actions
        missing = gt_actions - expanded_generated
        extra   = expanded_generated - gt_actions

        precision = len(covered) / len(expanded_generated) if expanded_generated else 0.0
        recall    = len(covered) / len(gt_actions)          if gt_actions else 1.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        resource_score, resource_mismatches = self._check_resource_scoping(
            policy_doc, gt_policy_doc
        )

        return {
            "layer3_precision":       precision,
            "layer3_recall":          recall,
            "layer3_f1":              f1,
            "layer3_exact_match":     len(missing) == 0 and len(extra) == 0,
            "layer3_covered":         list(covered),
            "layer3_missing":         list(missing),
            "layer3_extra":           list(extra),
            "layer3_n_generated":     len(expanded_generated),
            "layer3_n_expected":      len(gt_actions),
            "layer3_resource_score":  resource_score,
            "layer3_resource_mismatches": resource_mismatches,
            "layer3_gt_has_conditions":   self._has_conditions(gt_policy_doc),
            "layer3_gen_has_conditions":  self._has_conditions(policy_doc),
            "layer3_condition_present_when_expected": (
                self._has_conditions(policy_doc)
                if self._has_conditions(gt_policy_doc)
                else None
            ),
            "layer3_has_deny": any(
                s.get("Effect") == "Deny"
                for s in policy_doc.get("Statement", [])
            ),
        }

    @staticmethod
    def _expand_wildcards(
        generated: set[str], gt: set[str]
    ) -> set[str]:
        """Expand wildcard actions (s3:Get*) against the ground truth set."""
        expanded = set()
        for action in generated:
            if "*" in action:
                matched = {a for a in gt if fnmatch.fnmatch(a.lower(), action.lower())}
                expanded |= (matched or {action})
            else:
                expanded.add(action)
        return expanded

    def _check_resource_scoping(
        self, policy_doc: dict, gt_policy_doc: dict
    ) -> tuple[float, list[str]]:
        """
        Check that generated resource ARNs use the correct AWS service namespace.
        Returns (score 0-1, list of mismatch descriptions).
        """
        gt_services = {
            a.split(":")[0].lower()
            for a in self._extract_actions(gt_policy_doc)
            if ":" in a
        }

        mismatches = []
        resources = self._extract_resources(policy_doc)
        non_wildcard = [r for r in resources if r != "*"]

        if not non_wildcard:
            return 1.0, []

        for resource in non_wildcard:
            resource_lower = resource.lower()
            if resource_lower == "*":
                continue
            matched = False
            for svc in gt_services:
                expected_ns = RESOURCE_NAMESPACE_MAP.get(svc, f"arn:aws:{svc}")
                if resource_lower.startswith(expected_ns):
                    matched = True
                    break
            if not matched:
                mismatches.append(resource)

        score = 1.0 - (len(mismatches) / len(non_wildcard))
        return max(0.0, score), mismatches

    # -------------------------------------------------------------------------
    # Layer 4 — LLM-as-Judge (Claude Sonnet)
    # -------------------------------------------------------------------------

    def _layer4_llm_judge(
        self,
        query: str,
        policy_doc: dict,
        gt_policy_doc: dict,
        layer3_results: dict,
    ) -> dict | None:
        """
        Use Claude Sonnet to evaluate the generated policy on three dimensions:
          - Functional correctness (0-10)
          - Security correctness / least-privilege (0-10)
          - Resource scoping appropriateness (0-10)
          - Overall acceptability (true/false)
          - Key issues (list of strings)
        """
        if self.anthropic_client is None:
            return None

        gt_conds  = layer3_results.get("layer3_gt_has_conditions", False)
        missing   = layer3_results.get("layer3_missing", [])
        extra     = layer3_results.get("layer3_extra", [])

        condition_section = ""
        if gt_conds:
            gt_conditions  = {
                i: s.get("Condition")
                for i, s in enumerate(gt_policy_doc.get("Statement", []))
                if s.get("Condition")
            }
            gen_conditions = {
                i: s.get("Condition")
                for i, s in enumerate(policy_doc.get("Statement", []))
                if s.get("Condition")
            }
            condition_section = f"""
The ground truth policy includes condition blocks:
{json.dumps(gt_conditions, indent=2)}

Generated policy conditions (if any):
{json.dumps(gen_conditions, indent=2) if gen_conditions else "None — generator omitted conditions entirely"}

Additionally evaluate:
- Whether appropriate conditions are included for this task
- Whether omitting conditions (if the generator did) creates a security risk
"""

        prompt = f"""You are an AWS IAM security expert evaluating a generated IAM policy.

Task description: {query}

Ground truth policy:
{json.dumps(gt_policy_doc, indent=2)}

Generated policy:
{json.dumps(policy_doc, indent=2)}

Objective analysis:
- Missing actions: {missing}
- Extra actions beyond expected: {extra}
- Resource compatibility score: {layer3_results.get('layer3_resource_score', 1.0):.3f}
{condition_section}

Evaluate on these dimensions:

1. FUNCTIONAL CORRECTNESS (0-10): Does the policy grant sufficient permissions?
   Penalize for missing actions that would prevent the task.

2. SECURITY CORRECTNESS (0-10): Does the policy follow least-privilege?
   Penalize for unnecessary broad permissions or wildcard actions.

3. RESOURCE SCOPING (0-10): Are resources appropriately scoped?
   Using * is acceptable when the task does not specify resources.
   Penalize only if resources reference clearly wrong service namespaces.

4. OVERALL ACCEPTABILITY: Would an AWS security engineer approve this policy?
   Answer true or false.

5. KEY ISSUES: List the 1-3 most important problems, if any.

Respond ONLY in this exact JSON format:
{{
  "functional_score":  <0-10>,
  "security_score":    <0-10>,
  "resource_score":    <0-10>,
  "is_acceptable":     <true/false>,
  "key_issues":        ["issue1", "issue2"],
  "reasoning":         "brief explanation"
}}"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as e:
            print(f"  Layer 4 LLM judge failed: {e}")
            return None

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
    def _extract_resources(policy_doc: dict) -> list[str]:
        """Flatten all Resource values from a policy document."""
        resources = []
        for stmt in policy_doc.get("Statement", []):
            if not isinstance(stmt, dict):
                continue
            raw = stmt.get("Resource", [])
            if isinstance(raw, str):
                raw = [raw]
            resources.extend(raw)
        return resources

    @staticmethod
    def _has_conditions(policy_doc: dict) -> bool:
        """Return True if any statement contains a Condition block."""
        return any(
            bool(stmt.get("Condition"))
            for stmt in policy_doc.get("Statement", [])
            if isinstance(stmt, dict)
        )
