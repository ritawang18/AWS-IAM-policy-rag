"""
embedder.py
===========
Builds three retrieval indexes from the chunked corpus:

  DenseEmbedder   → Voyage Code-3 embeddings stored in a FAISS IndexFlatIP
                    (cosine similarity via inner product after L2 normalization)

  SparseEmbedder  → SPLADE-v3 sparse vectors stored as {token_id: weight} dicts
                    (dot-product similarity, vocabulary-expanded matching)

  BM25Indexer     → BM25 Okapi keyword index with IAM-aware tokenization
                    (preserves service:action colons and hyphenated identifiers)

All three indexes share the same action + policy chunk corpus.

Usage
-----
    from src.embedder import DenseEmbedder, SparseEmbedder, BM25Indexer

    # Build dense FAISS index
    de = DenseEmbedder(voyage_api_key="...")
    de.embed_and_save(action_chunks, "indexes/dense/action_chunks_embedded.json")
    de.embed_and_save(policy_chunks, "indexes/dense/policy_chunks_embedded.json")
    de.build_faiss_index(
        action_embedded_path="indexes/dense/action_chunks_embedded.json",
        policy_embedded_path="indexes/dense/policy_chunks_embedded.json",
        output_dir="indexes/dense/",
    )

    # Build SPLADE sparse index
    se = SparseEmbedder(hf_token="...")
    se.embed_and_save(action_chunks, "indexes/sparse/action_chunks_splade.json")
    se.embed_and_save(policy_chunks, "indexes/sparse/policy_chunks_splade.json")

    # Build BM25 index
    bi = BM25Indexer()
    bi.build_and_save(action_chunks + policy_chunks, "indexes/bm25/")
"""

import json
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


class DenseEmbedder:
    """
    Generates dense embeddings using Voyage Code-3 and builds a FAISS index.

    Voyage Code-3 is used because it is fine-tuned on code and technical
    schemas, making it well-suited for the rigid lexical register of IAM
    action names and JSON policy structures.

    Asymmetric embedding: documents use input_type="document", queries use
    input_type="query" — Voyage trains separate spaces for each, which
    improves retrieval over using "document" for both.
    """

    MODEL = "voyage-code-3"

    def __init__(self, voyage_api_key: str):
        import voyageai
        self.client = voyageai.Client(api_key=voyage_api_key)

    def embed_and_save(
        self,
        chunks: list[dict],
        output_path: str,
        batch_size: int = 128,
    ) -> list[dict]:
        """
        Generate embeddings for a list of chunks and save to JSON.

        Returns list of chunks with added "embedding" field.
        """
        print(f"Embedding {len(chunks)} chunks with {self.MODEL}...")
        embeddings = []

        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            response = self.client.embed(
                texts=texts,
                model=self.MODEL,
                input_type="document",  # asymmetric: document side
            )
            embeddings.extend(response.embeddings)

        result = [
            {"text": c["text"], "metadata": c["metadata"], "embedding": emb}
            for c, emb in zip(chunks, embeddings)
        ]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, ensure_ascii=False)

        print(f"✓ Saved {len(result)} embeddings to {output_path}")
        return result

    def build_faiss_index(
        self,
        action_embedded_path: str,
        policy_embedded_path: str,
        output_dir: str,
    ) -> tuple:
        """
        Combine action and policy embedded chunks into one unified FAISS index.

        Action chunks come first in the combined list — this ordering is
        critical because FAISS returns positional indices that must map
        1:1 to the saved combined_chunks.json.

        Returns:
            (faiss_index, combined_chunks)
        """
        with open(action_embedded_path) as f:
            action_chunks = json.load(f)
        with open(policy_embedded_path) as f:
            policy_chunks = json.load(f)

        combined = action_chunks + policy_chunks
        print(f"Building FAISS index: {len(action_chunks)} actions + {len(policy_chunks)} policies")

        # Save combined chunks — position in list = FAISS index position
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        combined_path = os.path.join(output_dir, "combined_chunks.json")
        with open(combined_path, "w") as f:
            json.dump(combined, f)

        # Build index
        vectors = np.array(
            [c["embedding"] for c in combined], dtype="float32"
        )
        faiss.normalize_L2(vectors)           # normalize for cosine similarity

        dim = vectors.shape[1]                # 1024 for voyage-code-3
        index = faiss.IndexFlatIP(dim)        # inner product = cosine after normalization
        index.add(vectors)

        index_path = os.path.join(output_dir, "combined_faiss.index")
        faiss.write_index(index, index_path)

        print(f"✓ FAISS index saved: {index_path} ({index.ntotal} vectors, dim={dim})")
        print(f"✓ Combined chunks saved: {combined_path}")
        return index, combined


class SparseEmbedder:
    """
    Generates SPLADE-v3 sparse embeddings.

    SPLADE encodes text into sparse {token_id: weight} dicts by learning to
    expand documents with semantically related vocabulary terms. Unlike BM25,
    it handles vocabulary mismatch between natural-language queries and IAM
    technical identifiers.

    Only non-zero weights are stored to keep file sizes manageable.
    """

    MODEL_ID = "naver/splade-v3"

    def __init__(self, hf_token: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, token=hf_token
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.MODEL_ID, token=hf_token
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"SPLADE model loaded on {self.device}")

    def embed_and_save(
        self,
        chunks: list[dict],
        output_path: str,
        batch_size: int = 16,
    ) -> list[dict]:
        """
        Generate SPLADE sparse vectors and save to JSON.

        Returns list of chunks with added "sparse_embedding" field.
        """
        print(f"Generating SPLADE embeddings for {len(chunks)} chunks...")
        sparse_embeddings = []

        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                # SPLADE core: log1p(ReLU(max-pooled logits))
                logits_relu = torch.log1p(torch.relu(logits))
                token_max, _ = torch.max(
                    logits_relu * inputs.attention_mask.unsqueeze(-1), dim=1
                )

            for j in range(len(texts)):
                nonzero_idx = torch.nonzero(token_max[j]).squeeze(-1).cpu().tolist()
                nonzero_w   = token_max[j][nonzero_idx].cpu().tolist()
                sparse_dict = {
                    str(idx): round(w, 4)
                    for idx, w in zip(nonzero_idx, nonzero_w)
                }
                sparse_embeddings.append(sparse_dict)

        result = [
            {"text": c["text"], "metadata": c["metadata"], "sparse_embedding": s}
            for c, s in zip(chunks, sparse_embeddings)
        ]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, ensure_ascii=False)

        print(f"✓ Saved {len(result)} SPLADE embeddings to {output_path}")
        return result


class BM25Indexer:
    """
    Builds a BM25 Okapi index with IAM-aware tokenization.

    Standard tokenizers split on colons and hyphens, destroying IAM
    identifiers like s3:GetObject and access-analyzer. Our custom tokenizer
    preserves these so BM25 can match exact action names from user queries.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        IAM-aware tokenizer.
        Keeps colon (s3:GetObject) and hyphen (access-analyzer) intact.
        Lowercases and splits on all other non-word characters.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s:\-]", " ", text)
        return text.split()

    def build_and_save(self, chunks: list[dict], output_dir: str) -> None:
        """
        Build a BM25 index over the combined chunk corpus and save to disk.

        Saves:
          combined_bm25.pkl          → serialized BM25Okapi object
          combined_chunks_mapped.json → chunks list (index position = BM25 position)
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"Building BM25 index over {len(chunks)} chunks...")
        tokenized = [self.tokenize(c["text"]) for c in tqdm(chunks)]
        bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        index_path  = os.path.join(output_dir, "combined_bm25.pkl")
        chunks_path = os.path.join(output_dir, "combined_chunks_mapped.json")

        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
        with open(chunks_path, "w") as f:
            json.dump(chunks, f)

        print(f"✓ BM25 index saved: {index_path}")
        print(f"✓ Mapped chunks saved: {chunks_path}")
