"""
Launch training on the GPU EC2 instance (g4dn.xlarge).

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --no-stop   # keep instance running after job
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.ec2 import EC2, REMOTE_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_r16.json", help="Path to experiment config JSON")
    parser.add_argument("--no-stop", action="store_true", help="Keep instance running after job")
    args = parser.parse_args()

    ec2 = EC2("EC2_INSTANCE_ID")

    ec2.start()
    try:
        ec2.push_code()
        ec2.pull_code()
        ec2.run(
            f"cd {REMOTE_DIR} && "
            f"S3_BUCKET={os.environ['S3_BUCKET']} "
            f"AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']} "
            f"AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']} "
            f"AWS_REGION={os.environ.get('AWS_REGION', 'us-east-2')} "
            f".venv/bin/python scripts/train.py --config {args.config}"
        )
    finally:
        if not args.no_stop:
            ec2.stop()


if __name__ == "__main__":
    main()
