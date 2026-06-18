"""
ETL script — runs on EC2.

Fetch, clean, split, and upload a dataset to S3.

Parallelism strategy:
  - Abstract download  : ThreadPoolExecutor  (I/O bound, network wait)
  - PDF parse + clean  : Ray                 (CPU bound, coming later)
  - Embedding          : Ray                 (GPU bound, coming later)

Usage:
    python scripts/etl.py --source arxiv --n 500
"""

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_materials.data.arxiv import ArxivSource

SOURCES = {
    "arxiv": ArxivSource,
    # "materials_project": MaterialsProjectSource,  # add later
}

VAL_RATIO = 0.1
MAX_WORKERS = 8


# ------------------------------------------------------------------
# S3
# ------------------------------------------------------------------

def upload_jsonl(records, bucket, s3_key):
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        tmp_path = f.name
    s3.upload_file(tmp_path, bucket, s3_key)
    os.unlink(tmp_path)
    print(f"Uploaded {len(records)} records to s3://{bucket}/{s3_key}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(SOURCES.keys()))
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    bucket = os.environ["S3_BUCKET"]
    source = SOURCES[args.source]()

    # 1. fetch IDs (paginate arXiv API serially to respect rate limit)
    print(f"Fetching {args.n} records from {args.source}...")
    raw = source.fetch(args.n)
    print(f"Fetched {len(raw)} raw records")

    # 2. clean in parallel with threads (text cleaning is fast but I/O safe with threads)
    print(f"Cleaning with ThreadPoolExecutor (workers={MAX_WORKERS})...")
    cleaned = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(source.clean, r): r for r in raw}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                cleaned.append(result)
    print(f"Kept {len(cleaned)} records after cleaning")

    # 3. train / val split
    split = int(len(cleaned) * (1 - VAL_RATIO))
    train, val = cleaned[:split], cleaned[split:]
    print(f"Split: {len(train)} train / {len(val)} val")

    # 4. upload to S3
    upload_jsonl(train, bucket, f"materialLLM/data/{args.source}/train.jsonl")
    upload_jsonl(val,   bucket, f"materialLLM/data/{args.source}/val.jsonl")

    print("ETL complete.")


if __name__ == "__main__":
    main()
