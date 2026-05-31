# Data Directory

This directory contains **sample files only**. Full corpus files are not committed to git (too large; see `.gitignore`).

---

## Raw Data

### `raw/sample_actions.json`
A small sample of IAM action documents scraped from the AWS Service Authorization Reference.

**Schema (per action):**
```json
{
  "doc_id":            "s3_getobject",
  "action":            "s3:GetObject",
  "action_name":       "GetObject",
  "service":           "Amazon S3",
  "service_prefix":    "s3",
  "description":       "Grants permission to retrieve objects from Amazon S3",
  "access_level":      "Read",
  "resource_types":    ["object"],
  "condition_keys":    ["s3:prefix", "s3:ExistingObjectTag/..."],
  "dependent_actions": [],
  "full_text":         "IAM Action: s3:GetObject. Service: Amazon S3. ..."
}
```

**Full corpus:** 4,505 actions across 56 AWS services. Regenerate with:
```bash
python -c "from src.collector import ActionCollector, SERVICES_CONFIG; c=ActionCollector(); c.save(c.collect_all(SERVICES_CONFIG), 'data/processed/action_documents.json')"
```

---

### `raw/sample_policies.json`
A small sample of AWS managed policies scraped from the Managed Policy Reference.

**Schema (per policy):**
```json
{
  "policy_name":     "AmazonS3ReadOnlyAccess",
  "url":             "https://docs.aws.amazon.com/...",
  "policy_document": { "Version": "2012-10-17", "Statement": [...] },
  "actions_used":    ["s3:GetObject", "s3:ListBucket"],
  "relevance_score": 2
}
```

**Full corpus:** 470 managed policies (500 scraped, 30 held out for evaluation). Regenerate with:
```bash
python -c "from src.collector import PolicyCollector; c=PolicyCollector(); c.save(c.collect(n=500), 'data/processed/managed_policies.json')"
```

---

## Evaluation Data

### `evaluation/evaluation_ground_truth.json`
75 real-world IAM policies paired with GPT-4o-generated natural language queries.

**Schema (per entry):**
```json
{
  "policy_name":    "LambdaS3ReadRole",
  "source":         "github",
  "query":          "Allow a Lambda function to read objects from S3 and write execution logs",
  "policy_document": { "Version": "2012-10-17", "Statement": [...] }
}
```

**Coverage:**
- 45 policies from GitHub CloudFormation templates (aws-samples, awslabs)
- 30 policies from AWS Managed Policy Reference (held out from corpus)
- 23 AWS services, 2–25 actions per policy, 50 multi-service policies

See `notebooks/04_ground_truth_construction.ipynb` for the full construction methodology.
