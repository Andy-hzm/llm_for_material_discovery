import io
import os
import time
import zipfile
import tempfile
import boto3

# project root = one level above this file (infra/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_DIR = "/home/ec2-user/llm_for_material_discovery"
S3_CODE_KEY = "materialLLM/code/code.zip"


class EC2:
    def __init__(self, instance_env_var="EC2_INSTANCE_ID"):
        region = os.environ.get("AWS_REGION", "us-east-2")
        self.instance_id = os.environ[instance_env_var]
        self.bucket = os.environ["S3_BUCKET"]
        self._ec2 = boto3.client("ec2", region_name=region)
        self._ssm = boto3.client("ssm", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)

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
    # S3 code sync
    # ------------------------------------------------------------------

    def push_code(self):
        """Zip local project and upload to S3."""
        dirs_to_zip = ["infra", "scripts", "llm_materials"]
        files_to_zip = ["requirements.txt"]

        print("Zipping and uploading code to S3...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for d in dirs_to_zip:
                    full = os.path.join(ROOT, d)
                    if not os.path.exists(full):
                        continue
                    for dirpath, _, filenames in os.walk(full):
                        for fname in filenames:
                            if fname.endswith(".pyc") or "__pycache__" in dirpath:
                                continue
                            fpath = os.path.join(dirpath, fname)
                            zf.write(fpath, os.path.relpath(fpath, ROOT))
                for f in files_to_zip:
                    fpath = os.path.join(ROOT, f)
                    if os.path.exists(fpath):
                        zf.write(fpath, f)
            tmp_path = tmp.name

        self._s3.upload_file(tmp_path, self.bucket, S3_CODE_KEY)
        os.unlink(tmp_path)
        print(f"Code uploaded to s3://{self.bucket}/{S3_CODE_KEY}")

    def pull_code(self):
        """On the EC2 instance: download code from S3, unzip, install deps."""
        self.run(
            f"aws s3 cp s3://{self.bucket}/{S3_CODE_KEY} /tmp/code.zip && "
            f"mkdir -p {REMOTE_DIR} && "
            f"unzip -o /tmp/code.zip -d {REMOTE_DIR} && "
            f"cd {REMOTE_DIR} && "
            f"([ -d .venv ] || python3 -m venv .venv) && "
            f".venv/bin/pip install -q --upgrade pip && "
            f".venv/bin/pip install -q -r requirements.txt"
        )

    # ------------------------------------------------------------------
    # Run a command via SSM
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
