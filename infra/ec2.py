import os
import time
import boto3


class EC2:
    def __init__(self):
        region = os.environ.get("AWS_REGION", "us-east-2")
        self.instance_id = os.environ["EC2_INSTANCE_ID"]
        self._ec2 = boto3.client("ec2", region_name=region)
        self._ssm = boto3.client("ssm", region_name=region)

    # ------------------------------------------------------------------
    # Step 1: lifecycle
    # ------------------------------------------------------------------

    def start(self):
        print(f"Starting {self.instance_id}...")
        self._ec2.start_instances(InstanceIds=[self.instance_id])
        self._ec2.get_waiter("instance_running").wait(InstanceIds=[self.instance_id])
        print("Instance running — waiting for SSM agent...")
        self._wait_for_ssm()
        print("Ready")

    def stop(self):
        self._ec2.stop_instances(InstanceIds=[self.instance_id])
        print(f"Stop signal sent to {self.instance_id}")

    # ------------------------------------------------------------------
    # Step 2: run a command via SSM
    # ------------------------------------------------------------------

    def run(self, command, timeout=7200):
        print(f"Running: {command[:80]}...")
        resp = self._ssm.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": [command], "executionTimeout": [str(timeout)]},
        )
        cmd_id = resp["Command"]["CommandId"]
        self._wait_for_output(cmd_id, timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _wait_for_ssm(self, retries=20, pause=10):
        for _ in range(retries):
            info = self._ssm.describe_instance_information(
                Filters=[{"Key": "InstanceIds", "Values": [self.instance_id]}]
            )
            if info["InstanceInformationList"]:
                return
            time.sleep(pause)
        raise TimeoutError("SSM agent not reachable — check IAM role has AmazonSSMManagedInstanceCore")

    def _wait_for_output(self, cmd_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(5)
            result = self._ssm.get_command_invocation(
                CommandId=cmd_id,
                InstanceId=self.instance_id,
            )
            for line in result.get("StandardOutputContent", "").splitlines():
                print(f"[ec2] {line}")
            for line in result.get("StandardErrorContent", "").splitlines():
                print(f"[ec2:err] {line}")
            if result["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
                if result["Status"] != "Success":
                    raise RuntimeError(f"Command {result['Status']}")
                return
        raise TimeoutError("Command timed out")
