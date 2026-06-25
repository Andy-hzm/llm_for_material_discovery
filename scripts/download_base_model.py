"""
One-time script: download Qwen2.5-0.5B from HuggingFace and upload to S3.

Usage:
    python scripts/download_base_model.py

Run locally once. After this, train.py loads from S3 instead of HuggingFace.
S3 path: s3://<S3_BUCKET>/materialLLM/models/Qwen2.5-0.5B/
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "Qwen/Qwen2.5-0.5B"
S3_MODEL_PREFIX = "materialLLM/models/Qwen2.5-0.5B"

# Downloaded here so you can inspect the files. Gitignored.
LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "Qwen2.5-0.5B"
)


def main():
    import argparse
    import boto3
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true", help="Delete local model files after uploading to S3")
    args = parser.parse_args()

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2"))

    # Download model into project folder
    print(f"Downloading {MODEL_ID} from HuggingFace...")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=LOCAL_MODEL_DIR)
    print(f"Downloaded to: {LOCAL_MODEL_DIR}")

    # Upload all files to S3
    print(f"\nUploading to s3://{bucket}/{S3_MODEL_PREFIX}/")
    for dirpath, _, filenames in os.walk(LOCAL_MODEL_DIR):
        for fname in filenames:
            local_path = os.path.join(dirpath, fname)
            relative = os.path.relpath(local_path, LOCAL_MODEL_DIR)
            s3_key = f"{S3_MODEL_PREFIX}/{relative}"
            print(f"  {relative}")
            s3.upload_file(local_path, bucket, s3_key)

    print(f"\nDone. Model stored at s3://{bucket}/{S3_MODEL_PREFIX}/")
    print(f"Local copy at: {LOCAL_MODEL_DIR}")

    if args.cleanup:
        import shutil
        shutil.rmtree(LOCAL_MODEL_DIR)
        print("Local copy deleted.")
    else:
        print("Local copy kept — inspect the files, then run with --cleanup to delete.")


if __name__ == "__main__":
    main()
