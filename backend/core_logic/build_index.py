# backend/core_logic/build_index.py
"""
Input JSONL format (one object per line):
{"id": "doc1-0001", "text": "chunk text here", "meta": {"source": "PI", "section": "Warnings"}}

Examples:
  # Single JSONL
  PINECONE_API_KEY=... PINECONE_ENVIRONMENT=us-east-1 PINECONE_INDEX=pharma-assistant \
  python build_index.py --jsonl /path/to/chunks.jsonl --namespace lilly

  # Folder of JSONL files
  python build_index.py --dir /path/to/jsonls --namespace lilly
"""
import argparse, os, json, glob, sys
from typing import Dict, Any, Iterable, List

from embeddings import EmbeddingModel
from config import settings

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception as e:
    print("ERROR: pinecone-client not installed. Run `pip install pinecone-client`.", file=sys.stderr)
    raise

BATCH = 128

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def iter_records(jsonl: str | None, directory: str | None) -> Iterable[Dict[str, Any]]:
    if jsonl:
        yield from read_jsonl(jsonl)
        return
    assert directory, "Provide --jsonl or --dir"
    for file in glob.glob(os.path.join(directory, "*.jsonl")):
        yield from read_jsonl(file)

def batched(it, size: int):
    batch: List[Dict[str, Any]] = []
    for item in it:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def ensure_index(pc: "Pinecone", name: str, dim: int, region: str):
    existing = [i["name"] for i in pc.list_indexes()]
    if name in existing:
        return
    pc.create_index(
        name=name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region or "us-east-1"),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="Path to JSONL")
    ap.add_argument("--dir", help="Folder with JSONL files")
    ap.add_argument("--namespace", default=None, help="Pinecone namespace (optional)")
    args = ap.parse_args()

    # Init embedder to determine dimension
    embedder = EmbeddingModel()
    dim = int(embedder.encode_one("ping").shape[-1])

    # Init Pinecone and ensure index exists
    api_key = os.getenv("PINECONE_API_KEY", settings.pinecone.API_KEY)
    region = os.getenv("PINECONE_ENVIRONMENT", settings.pinecone.ENVIRONMENT)
    index_name = os.getenv("PINECONE_INDEX", settings.pinecone.INDEX)

    pc = Pinecone(api_key=api_key)
    ensure_index(pc, index_name, dim, region)
    index = pc.Index(index_name)

    # Upsert vectors
    for batch in batched(iter_records(args.jsonl, args.dir), BATCH):
        ids = [rec["id"] for rec in batch]
        texts = [rec["text"] for rec in batch]
        metas = [rec.get("meta", {}) | {"text": rec["text"]} for rec in batch]
        vecs = embedder.encode(texts).tolist()
        to_upsert = list(zip(ids, vecs, metas))
        index.upsert(vectors=to_upsert, namespace=args.namespace)

    print(f"Done. Upserted into index '{index_name}' (namespace={args.namespace!r}).")

if __name__ == "__main__":
    main()
