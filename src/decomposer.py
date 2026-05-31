"""
decomposer.py
=============
GPT-4o-driven query decomposition for IAM-aware retrieval.

Problem
-------
Natural language queries operate at the task level ("allow Lambda to read
from S3"), while IAM policies must be specified at the individual API action
level. Standard single-query retrieval misses cross-service dependencies
that are never explicitly mentioned in the query.

Solution
--------
QueryDecomposer uses GPT-4o to break a user query into N targeted sub-queries,
each focused on one AWS service using action-bundle phrasing. These sub-queries
are consumed by DecomposedRetriever in src/retriever.py, which runs dense
retrieval independently per sub-query and merges the results.

Example
-------
Query: "launch an EC2 instance and write execution logs"
Sub-queries produced:
  - "ec2 run instances describe subnets describe security groups create tags"
  - "logs create log group create log stream put log events"
  - "iam pass role execution"   ← implicit dependency surfaced by decomposer

This module contains only the LLM decomposition logic. All retrieval
strategies — including the final DecomposedRetriever that uses this class —
live in src/retriever.py.

Usage
-----
    from src.decomposer import QueryDecomposer

    decomposer = QueryDecomposer(openai_api_key="...")
    sub_queries = decomposer.decompose("Allow Lambda to read from S3 and write logs")
    # → ["lambda read object s3 bucket get", "logs create group stream put events", ...]
"""

import json
from openai import OpenAI


DECOMPOSE_PROMPT = """\
You are an AWS IAM expert. Given a natural language task description, decompose
it into a list of specific retrieval sub-queries. Each sub-query should target
one AWS service and use IAM action-bundle phrasing (action name keywords, not
natural language sentences).

IMPORTANT:
- Include sub-queries for IMPLICIT dependencies (e.g., CloudWatch Logs for any
  Lambda function, iam:PassRole for ECS/Lambda/CodePipeline tasks).
- Do NOT use service or action names in the output — use descriptive keywords.
- Respond with a JSON array of strings only. No explanation.

Example input: "Allow Lambda to read from S3 and write execution logs"
Example output: ["lambda read object s3 bucket get", "logs create group stream put events", "iam pass role execution"]

Task: {query}
"""


class QueryDecomposer:
    """
    Decomposes a natural language task description into targeted IAM
    retrieval sub-queries using GPT-4o.

    Each sub-query targets one AWS service with action-bundle phrasing,
    so dense retrieval can independently find the right action chunks for
    each service rather than trying to satisfy all services with one query.
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model  = model

    def decompose(self, query: str) -> list[str]:
        """
        Decompose a natural language query into retrieval sub-queries.
        Falls back to the original query as a single-element list if the
        GPT-4o response is malformed.

        Args:
            query: natural language task description

        Returns:
            list of sub-query strings, each targeting one AWS service
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}],
            temperature=0,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()

        try:
            sub_queries = json.loads(text)
            if isinstance(sub_queries, list) and all(isinstance(s, str) for s in sub_queries):
                return sub_queries
        except json.JSONDecodeError:
            pass

        print("  Warning: decomposer returned malformed output, using original query")
        return [query]
