"""
Baby-step smoke test for infra/ec2.py.
Run after the quota is approved and the instance is launched.

    EC2_INSTANCE_ID=i-xxxx AWS_REGION=us-east-2 python scripts/test_ec2.py

Steps:
    1. start  — instance transitions to running, SSM agent reachable
    2. run    — echo hello from EC2, confirm output comes back
    3. stop   — instance shuts down
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra.ec2 import EC2

ec2 = EC2("EC2_INSTANCE_ID")

print("=== Step 1: start ===")
ec2.start()

print("\n=== Step 2: run dummy command ===")
ec2.run("echo 'hello from EC2' && nvidia-smi")

print("\n=== Step 3: stop ===")
ec2.stop()

print("\nAll steps passed.")
