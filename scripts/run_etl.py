"""
Launch ETL on the ETL EC2 instance (t3.medium).

Usage:
    python scripts/run_etl.py --source arxiv --n 500
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infra.ec2 import EC2, REMOTE_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--no-stop", action="store_true", help="Keep instance running after job")
    args = parser.parse_args()

    ec2 = EC2("EC2_ETL_INSTANCE_ID")

    ec2.start()
    ec2.push_code()
    ec2.pull_code()
    ec2.run(
        f"cd {REMOTE_DIR} && "
        f"S3_BUCKET={os.environ['S3_BUCKET']} "
        f"AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']} "
        f"AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']} "
        f"AWS_REGION={os.environ.get('AWS_REGION', 'us-east-2')} "
        f".venv/bin/python scripts/etl.py --source {args.source} --n {args.n}"
    )

    if not args.no_stop:
        ec2.stop()


if __name__ == "__main__":
    main()
