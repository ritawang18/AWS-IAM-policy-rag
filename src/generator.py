"""
generator.py
============
IAM policy generation using retrieved context (RAG) or without context
(No-RAG baseline). Supports any OpenAI-compatible API endpoint, allowing
both GPT-3.5-turbo and Llama-3.1-8B (via Groq) to be used as generators.

Design decisions
----------------
- System prompt emphasizes least-privilege and exact action names to
  counteract the LLM's tendency to over-generate wildcard actions.
- Retrieved action chunks are included as a strict vocabulary list to
  anchor generation to real IAM actions.
- Retrieved policy chunks are included as structural examples to teach
  the LLM correct resource scoping and condition patterns.

Usage
-----
    from src.generator import PolicyGenerator

    # RAG generation
    gen = PolicyGenerator(api_key="...", model="gpt-3.5-turbo")
    policy_json = gen.generate_with_rag(query, retrieved_chunks)

    # No-RAG baseline
    policy_json = gen.generate_no_rag(query)
"""

import json
from openai import OpenAI


SYSTEM_PROMPT = """\
You are an AWS IAM policy expert. Generate a valid AWS IAM policy JSON document
based on the user's task description.

Rules:
1. Output ONLY valid JSON — no explanations, no markdown, no code fences.
2. Use exact IAM action names from the provided context. Do NOT invent actions.
3. Follow least-privilege: grant only the permissions required for the described task.
4. Avoid wildcard actions (*) unless the task explicitly requires all actions.
5. Structure: {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": [...], "Resource": "..."}]}
"""

RAG_USER_TEMPLATE = """\
Task: {query}

Available IAM actions (use ONLY these for the relevant services):
{action_context}

Example policy structures for reference:
{policy_context}

Generate a minimal IAM policy JSON for this task.
"""

NO_RAG_USER_TEMPLATE = """\
Task: {query}

Generate a minimal AWS IAM policy JSON that grants exactly the permissions
needed for this task. Follow least-privilege. Output only valid JSON.
"""


class PolicyGenerator:
    """
    Generates IAM policies from natural language task descriptions.

    Supports any OpenAI-compatible endpoint — use base_url to point to
    Groq for Llama-3.1-8B inference.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        base_url: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0,
    ):
        """
        Args:
            api_key:     OpenAI or Groq API key
            model:       model name (e.g. "gpt-3.5-turbo", "llama-3.1-8b-instant")
            base_url:    override API endpoint (e.g. "https://api.groq.com/openai/v1")
            max_tokens:  max tokens in generated policy
            temperature: generation temperature (0 for deterministic)
        """
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_with_rag(
        self, query: str, retrieved_chunks: list[dict]
    ) -> str:
        """
        Generate a policy using retrieved context chunks.

        Formats action chunks as an exact vocabulary list and policy chunks
        as structural examples, then prompts the generator to stay within
        those bounds.

        Returns:
            raw string output from the model (should be JSON)
        """
        action_chunks = [
            c for c in retrieved_chunks
            if c["metadata"]["type"] == "iam_action"
        ]
        policy_chunks = [
            c for c in retrieved_chunks
            if c["metadata"]["type"] == "iam_policy"
        ]

        # Build action vocabulary context
        action_lines = []
        for c in action_chunks:
            meta = c["metadata"]
            action_lines.append(
                f"- {meta['action']}: {meta['description'][:80]}"
            )
        action_context = "\n".join(action_lines) if action_lines else "No action context retrieved."

        # Build policy structure context
        policy_lines = []
        for c in policy_chunks:
            meta = c["metadata"]
            doc  = meta.get("policy_document", {})
            policy_lines.append(
                f"Policy: {meta['policy_name']}\n"
                + json.dumps(doc, indent=2)[:500]
                + ("\n..." if len(json.dumps(doc)) > 500 else "")
            )
        policy_context = "\n\n".join(policy_lines) if policy_lines else "No policy examples retrieved."

        user_message = RAG_USER_TEMPLATE.format(
            query=query,
            action_context=action_context,
            policy_context=policy_context,
        )

        return self._call(user_message)

    def generate_no_rag(self, query: str) -> str:
        """
        Generate a policy without any retrieval context.
        Used as the No-RAG baseline in evaluation.

        Returns:
            raw string output from the model (should be JSON)
        """
        user_message = NO_RAG_USER_TEMPLATE.format(query=query)
        return self._call(user_message)

    def _call(self, user_message: str) -> str:
        """Call the LLM API and return the raw text response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()
