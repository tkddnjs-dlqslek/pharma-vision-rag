"""One EC2 box for the compose stack (services/deploy.md, "AWS (한 달 실험)"): provision, deploy, inspect, tear down.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/31_aws_deploy.py up [--instance-type t4g.xlarge] [--spot] [--eip]
    ... deploy [--branch claude/sharp-tesla-po09dt]      clone or pull the repo, upload the remote .env, compose up, wait for health
    ... status | stop | start | logs | down --yes

AWS credentials come from .env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION; default region ap-northeast-2).
Everything is looked up by the Name=pharma-rag tag, so re-running `up` reuses the instance. The public host is
<public-ip>.sslip.io (a relaunch with a new IP is a new hostname and a new Let's Encrypt certificate).
Secrets are never printed: the remote .env is built from a key whitelist and shown as key names only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
NAME = "pharma-rag"
PEM = Path.home() / ".ssh" / f"{NAME}.pem"
REPO_URL = "https://github.com/tkddnjs-dlqslek/pharma-vision-rag.git"
REMOTE_DIR = "pharma-vision-rag"           # under /home/ubuntu
AMI_OWNER = "099720109477"                  # Canonical
AMI_NAME = "ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"
# The only local .env keys that go to the box (docker-compose.yml reads them); PUBLIC_HOST, MCP_PUBLIC_URL, ENCODER_URL are computed.
REMOTE_ENV_KEYS = ("ENCODER_TOKEN", "MCP_TOKEN", "MCP_USER", "MCP_PASSWORD", "QDRANT_CLOUD_URL", "QDRANT_CLOUD_API_KEY",
                   "DATA_REPO", "HF_TOKEN")
USER_DATA = """#!/bin/bash
set -eux
apt-get update -y && apt-get install -y git curl
curl -fsSL https://get.docker.com | sh
usermod -aG docker ubuntu
systemctl enable --now docker
"""
HEALTH_TIMEOUT_S = 30 * 60


# ─── pure helpers (tests/test_aws_deploy.py) ─────────────────────────────────

def pick_ami(images: list[dict]) -> dict:
    """The newest image of a describe_images result (Canonical publishes several per month)."""
    if not images:
        sys.exit(f"no AMI matches {AMI_NAME} (owner {AMI_OWNER}) in this region")
    return max(images, key=lambda i: i["CreationDate"])


def find_instance(described: dict) -> dict | None:
    """The live pharma-rag instance from describe_instances, running preferred; None when only terminated ones exist."""
    live = [i for r in described.get("Reservations", []) for i in r.get("Instances", [])
            if i["State"]["Name"] not in ("terminated", "shutting-down")]
    live.sort(key=lambda i: i["State"]["Name"] != "running")
    return live[0] if live else None


def sslip(ip: str) -> str:
    return f"{ip}.sslip.io"


def remote_env(local: dict, public_host: str) -> str:
    """The .env for the box: whitelisted keys that have a real value, plus the derived public URL keys."""
    lines = []
    for k in REMOTE_ENV_KEYS:
        v = (local.get(k) or "").strip()
        if not v or (k == "HF_TOKEN" and re.fullmatch(r"hf_x+", v)):   # .env.example placeholder breaks the download (deploy.md)
            continue
        lines.append(f"{k}={v}")
    lines += [f"PUBLIC_HOST={public_host}", f"MCP_PUBLIC_URL=https://{public_host}", "ENCODER_URL=http://encoders:7860"]
    return "\n".join(lines) + "\n"


def update_env_text(text: str, updates: dict[str, str]) -> str:
    """Rewrite only the given keys in a .env text (replace in place, append the missing ones)."""
    out = text
    for k, v in updates.items():
        line = f"{k}={v}"
        out, n = re.subn(rf"(?m)^{re.escape(k)}=.*$", line.replace("\\", r"\\"), out, count=1)
        if n == 0:
            out = out.rstrip("\n") + ("\n" if out.strip() else "") + line + "\n"
    return out


def write_env(updates: dict[str, str]) -> None:
    text = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    ENV_PATH.write_text(update_env_text(text, updates), encoding="utf-8")
    print(f".env updated: {', '.join(updates)}")


# ─── AWS ─────────────────────────────────────────────────────────────────────

def session():
    load_dotenv(ENV_PATH)
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        sys.exit("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY missing in .env")
    import boto3
    return boto3.session.Session(region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))


def my_ip() -> str:
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()


def tag_filter():
    return [{"Name": "tag:Name", "Values": [NAME]}]


def tags(resource: str):
    return [{"ResourceType": resource, "Tags": [{"Key": "Name", "Value": NAME}]}]


def current_instance(ec2) -> dict | None:
    return find_instance(ec2.describe_instances(Filters=tag_filter()))


def require_instance(ec2) -> dict:
    inst = current_instance(ec2)
    if inst is None:
        sys.exit(f"no instance tagged Name={NAME}; run `up` first")
    return inst


def ensure_key_pair(ec2) -> None:
    from botocore.exceptions import ClientError
    try:
        ec2.describe_key_pairs(KeyNames=[NAME])
        exists = True
    except ClientError as e:
        if e.response["Error"]["Code"] != "InvalidKeyPair.NotFound":
            raise
        exists = False
    if exists and PEM.exists():
        print(f"key pair {NAME}: reused ({PEM})")
        return
    if exists:
        sys.exit(f"key pair {NAME} exists in AWS but {PEM} is missing: delete the key pair (console or `down --yes`) and rerun")
    kp = ec2.create_key_pair(KeyName=NAME, KeyType="ed25519", KeyFormat="pem", TagSpecifications=tags("key-pair"))
    PEM.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(PEM, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="\n") as f:
        f.write(kp["KeyMaterial"])
    try:
        os.chmod(PEM, 0o600)
    except OSError:
        pass
    print(f"key pair {NAME}: created, private key saved to {PEM}")


def ensure_security_group(ec2) -> str:
    from botocore.exceptions import ClientError
    groups = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [NAME]}])["SecurityGroups"]
    if groups:
        sg_id = groups[0]["GroupId"]
        print(f"security group {NAME}: reused ({sg_id})")
    else:
        sg_id = ec2.create_security_group(GroupName=NAME, Description="pharma-rag: ssh from operator, http/https public",
                                          TagSpecifications=tags("security-group"))["GroupId"]
        print(f"security group {NAME}: created ({sg_id})")
    ip = my_ip()
    rules = [(22, f"{ip}/32", "operator ssh"), (80, "0.0.0.0/0", "acme + redirect"), (443, "0.0.0.0/0", "https")]
    for port, cidr, desc in rules:
        try:
            ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[{
                "IpProtocol": "tcp", "FromPort": port, "ToPort": port, "IpRanges": [{"CidrIp": cidr, "Description": desc}]}])
        except ClientError as e:
            if e.response["Error"]["Code"] != "InvalidPermission.Duplicate":
                raise
    print(f"ingress: 22 from {ip}/32, 80 and 443 from anywhere")
    return sg_id


def find_eip(ec2) -> dict | None:
    addrs = ec2.describe_addresses(Filters=tag_filter())["Addresses"]
    return addrs[0] if addrs else None


def public_ip(ec2, instance_id: str) -> str:
    inst = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
    return inst.get("PublicIpAddress", "")


def record_host(ec2, instance_id: str) -> str:
    ip = public_ip(ec2, instance_id)
    host = sslip(ip)
    write_env({"AWS_INSTANCE_ID": instance_id, "AWS_PUBLIC_IP": ip, "PUBLIC_HOST": host,
               "MCP_PUBLIC_URL": f"https://{host}", "MCP_URL": f"https://{host}/mcp"})
    print(f"public IP {ip}\nPUBLIC_HOST={host}")
    return host


def cmd_up(args) -> None:
    ec2 = session().client("ec2")
    ensure_key_pair(ec2)
    sg_id = ensure_security_group(ec2)
    inst = current_instance(ec2)
    if inst is not None:
        iid, state = inst["InstanceId"], inst["State"]["Name"]
        print(f"instance {iid} already exists ({state}): reusing")
        if state == "stopped":
            ec2.start_instances(InstanceIds=[iid])
        if state != "running":
            ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    else:
        ami = pick_ami(ec2.describe_images(Owners=[AMI_OWNER], Filters=[
            {"Name": "name", "Values": [AMI_NAME]}, {"Name": "state", "Values": ["available"]}])["Images"])
        print(f"AMI {ami['ImageId']} ({ami['Name']})")
        params = {
            "ImageId": ami["ImageId"], "InstanceType": args.instance_type, "KeyName": NAME, "SecurityGroupIds": [sg_id],
            "MinCount": 1, "MaxCount": 1, "UserData": USER_DATA, "TagSpecifications": tags("instance") + tags("volume"),
            "BlockDeviceMappings": [{"DeviceName": ami.get("RootDeviceName", "/dev/sda1"),
                                     "Ebs": {"VolumeSize": args.volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}]}
        if args.spot:
            params["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {
                "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
        iid = ec2.run_instances(**params)["Instances"][0]["InstanceId"]
        print(f"instance {iid} launched ({args.instance_type}{', spot' if args.spot else ''}, {args.volume_gb} GB gp3); waiting for running")
        ec2.get_waiter("instance_running").wait(InstanceIds=[iid])
    if args.eip:
        eip = find_eip(ec2)
        if eip is None:
            eip = ec2.allocate_address(Domain="vpc", TagSpecifications=tags("elastic-ip"))
            print(f"elastic IP allocated: {eip['PublicIp']} (billed while the instance is stopped)")
        if eip.get("InstanceId") != iid:
            ec2.associate_address(AllocationId=eip["AllocationId"], InstanceId=iid)
            print(f"elastic IP {eip['PublicIp']} associated")
    print("waiting for status checks (a few minutes)")
    ec2.get_waiter("instance_status_ok").wait(InstanceIds=[iid])
    record_host(ec2, iid)
    print("next: scripts/31_aws_deploy.py deploy")


# ─── SSH ─────────────────────────────────────────────────────────────────────

def ssh_connect(ip: str, tries: int = 30):
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())   # fresh box each launch; IP pinning would only annoy
    for i in range(tries):
        try:
            client.connect(ip, username="ubuntu", key_filename=str(PEM), timeout=15, banner_timeout=30)
            return client
        except Exception as e:  # sshd is not up yet right after launch
            if i == tries - 1:
                raise
            print(f"ssh not ready ({type(e).__name__}); retrying")
            time.sleep(10)


def ssh_run(client, cmd: str, stream: bool = True) -> tuple[int, str]:
    """Run cmd on the box; stream its combined output when asked; return (exit status, output)."""
    _, stdout, _ = client.exec_command(cmd, get_pty=True)
    buf = io.StringIO()
    for line in iter(stdout.readline, ""):
        buf.write(line)
        if stream:
            print("  | " + line.rstrip())
    return stdout.channel.recv_exit_status(), buf.getvalue()


def health(host: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"https://{host}/health", timeout=15) as r:
            return r.status, r.read().decode()[:300]
    except Exception as e:  # noqa: BLE001 - TLS not issued yet, connection refused, 502 while mcp loads
        return 0, f"{type(e).__name__}: {e}"


def cmd_deploy(args) -> None:
    ec2 = session().client("ec2")
    inst = require_instance(ec2)
    if inst["State"]["Name"] != "running":
        sys.exit(f"instance is {inst['State']['Name']}; run `start` or `up` first")
    ip = inst["PublicIpAddress"]
    host = sslip(ip)
    local = dotenv_values(ENV_PATH)
    env_text = remote_env(local, host)
    keys = [line.split("=", 1)[0] for line in env_text.splitlines()]
    print(f"remote .env keys: {', '.join(keys)}")
    missing = [k for k in ("ENCODER_TOKEN", "MCP_TOKEN") if k not in keys]
    if missing:
        sys.exit(f"local .env lacks {missing} (compose refuses to start without them)")

    client = ssh_connect(ip)
    print("waiting for cloud-init (docker install)")
    ssh_run(client, "cloud-init status --wait >/dev/null; docker --version")
    rc, _ = ssh_run(client, f"if [ -d {REMOTE_DIR}/.git ]; then cd {REMOTE_DIR} && git fetch -q origin {args.branch} "
                            f"&& git checkout -q {args.branch} && git reset -q --hard origin/{args.branch}; "
                            f"else git clone -q -b {args.branch} {REPO_URL} {REMOTE_DIR}; fi && cd {REMOTE_DIR} && git log --oneline -1")
    if rc != 0:
        sys.exit("git clone/pull failed")
    sftp = client.open_sftp()
    with sftp.file(f"{REMOTE_DIR}/.env", "w") as f:
        f.write(env_text)
    sftp.chmod(f"{REMOTE_DIR}/.env", 0o600)
    sftp.close()
    print("remote .env written")
    print("docker compose --profile serve up -d --build (first build: several minutes for torch on arm64)")
    rc, _ = ssh_run(client, f"cd {REMOTE_DIR} && docker compose --profile serve up -d --build 2>&1")
    if rc != 0:
        sys.exit("docker compose up failed")

    print(f"polling https://{host}/health (up to {HEALTH_TIMEOUT_S // 60} min; encoders log every 30 s)")
    t0 = time.time()
    while True:
        code, body = health(host)
        if code == 200:
            print(f"health 200: {body}")
            break
        if time.time() - t0 > HEALTH_TIMEOUT_S:
            sys.exit(f"health not ok after {HEALTH_TIMEOUT_S // 60} min: {body}")
        print(f"[{int(time.time() - t0)}s] health: {code or body[:80]}")
        ssh_run(client, f"cd {REMOTE_DIR} && docker compose logs --tail 3 --no-log-prefix encoders 2>/dev/null")
        time.sleep(30)
    client.close()
    print(f"""
Register:
  claude mcp add --transport http pharma-corpus-remote https://{host}/mcp --header "Authorization: Bearer $MCP_TOKEN"
  claude.ai / Claude Desktop: Settings > Connectors > Add custom connector, URL https://{host}/mcp, then log in with MCP_USER / MCP_PASSWORD
Note: search_pages answers 503 until the encoders health is ok (docker compose logs -f encoders on the box).""")


def cmd_status(args) -> None:
    s = session()
    ec2 = s.client("ec2")
    inst = current_instance(ec2)
    if inst is None:
        print(f"no instance tagged Name={NAME}")
    else:
        state = inst["State"]["Name"]
        up = dt.datetime.now(dt.UTC) - inst["LaunchTime"]
        print(f"instance {inst['InstanceId']} {inst['InstanceType']} {state}, public IP {inst.get('PublicIpAddress', '-')}, "
              f"since last start {up.total_seconds() / 3600:.1f} h" + (" (spot)" if inst.get("InstanceLifecycle") == "spot" else ""))
        eip = find_eip(ec2)
        if eip:
            print(f"elastic IP {eip['PublicIp']}")
        if state == "running":
            try:
                client = ssh_connect(inst["PublicIpAddress"], tries=1)
                ssh_run(client, f"cd {REMOTE_DIR} && docker compose --profile serve ps")
                client.close()
            except Exception as e:  # noqa: BLE001
                print(f"ssh failed: {type(e).__name__}: {e}")
            print(f"health: {health(sslip(inst['PublicIpAddress']))}")
    from botocore.exceptions import ClientError
    today = dt.datetime.now(dt.UTC).date()
    try:
        ce = s.client("ce", region_name="us-east-1")
        r = ce.get_cost_and_usage(TimePeriod={"Start": today.replace(day=1).isoformat(), "End": (today + dt.timedelta(days=1)).isoformat()},
                                  Granularity="MONTHLY", Metrics=["UnblendedCost"],
                                  Filter={"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Elastic Compute Cloud - Compute"]}})
        amt = r["ResultsByTime"][0]["Total"]["UnblendedCost"]
        print(f"EC2 cost month-to-date: {float(amt['Amount']):.2f} {amt['Unit']} (Cost Explorer lags about a day)")
    except ClientError as e:
        print(f"Cost Explorer not available ({e.response['Error']['Code']}): check Billing in the console, or allow ce:GetCostAndUsage")


def cmd_stop(args) -> None:
    ec2 = session().client("ec2")
    inst = require_instance(ec2)
    ec2.stop_instances(InstanceIds=[inst["InstanceId"]])
    print(f"stopping {inst['InstanceId']} (compute charge stops; EBS 40 GB and any elastic IP still bill)")
    ec2.get_waiter("instance_stopped").wait(InstanceIds=[inst["InstanceId"]])
    print("stopped. `start` gives a new public IP unless an elastic IP is attached (up --eip)")


def cmd_start(args) -> None:
    ec2 = session().client("ec2")
    inst = require_instance(ec2)
    old_ip = inst.get("PublicIpAddress")
    ec2.start_instances(InstanceIds=[inst["InstanceId"]])
    ec2.get_waiter("instance_running").wait(InstanceIds=[inst["InstanceId"]])
    host = record_host(ec2, inst["InstanceId"])
    if not host.startswith(str(old_ip or "")):
        print("public IP changed: run `deploy` again so the box gets the new PUBLIC_HOST (new certificate) and re-register the URL")
    else:
        print("same IP: the stack restarts by itself (restart: unless-stopped); allow a few minutes for the models")


def cmd_logs(args) -> None:
    """Copy the MCP call log out of the corpus_data volume to eval/results/mcp_calls.jsonl (scripts/32 reads it)."""
    ec2 = session().client("ec2")
    inst = require_instance(ec2)
    client = ssh_connect(inst["PublicIpAddress"], tries=1)
    rc, _ = ssh_run(client, "docker cp pharma-rag-mcp:/data/logs/mcp_calls.jsonl /tmp/mcp_calls.jsonl", stream=False)
    if rc != 0:
        sys.exit("no log yet (no tool call has been made) or the mcp container is not running")
    out = ROOT / "eval" / "results" / "mcp_calls.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    sftp.get("/tmp/mcp_calls.jsonl", str(out))
    sftp.close()
    client.close()
    print(f"{out} ({sum(1 for _ in out.open(encoding='utf-8'))} calls); next: scripts/32_mcp_usage_report.py")


def cmd_down(args) -> None:
    if not args.yes:
        sys.exit("this terminates the instance and deletes the security group, key pair and elastic IP: add --yes")
    from botocore.exceptions import ClientError
    ec2 = session().client("ec2")
    inst = current_instance(ec2)
    if inst is not None:
        ec2.terminate_instances(InstanceIds=[inst["InstanceId"]])
        print(f"terminating {inst['InstanceId']}")
        ec2.get_waiter("instance_terminated").wait(InstanceIds=[inst["InstanceId"]])
        print("instance terminated (root volume deleted with it)")
    eip = find_eip(ec2)
    if eip:
        ec2.release_address(AllocationId=eip["AllocationId"])
        print(f"elastic IP {eip['PublicIp']} released")
    groups = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [NAME]}])["SecurityGroups"]
    for g in groups:
        for _ in range(12):   # the ENI of a just-terminated instance can hold the group for a minute
            try:
                ec2.delete_security_group(GroupId=g["GroupId"])
                print(f"security group {g['GroupId']} deleted")
                break
            except ClientError as e:
                if e.response["Error"]["Code"] != "DependencyViolation":
                    raise
                time.sleep(10)
        else:
            print(f"security group {g['GroupId']} still in use; delete it from the console later")
    try:
        ec2.delete_key_pair(KeyName=NAME)
        print(f"key pair {NAME} deleted")
    except ClientError as e:
        print(f"key pair: {e.response['Error']['Code']}")
    if PEM.exists():
        PEM.unlink()
        print(f"{PEM} removed")
    print("done. Left in .env: AWS_INSTANCE_ID, AWS_PUBLIC_IP, PUBLIC_HOST, MCP_PUBLIC_URL, MCP_URL (stale; harmless)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="create or reuse key pair, security group and instance; write the host into .env")
    up.add_argument("--instance-type", default="t4g.xlarge")
    up.add_argument("--volume-gb", type=int, default=40)
    up.add_argument("--spot", action="store_true", help="one-time spot request (cheaper, can be reclaimed)")
    up.add_argument("--eip", action="store_true", help="allocate an elastic IP so stop/start keeps the hostname (billed while stopped)")
    up.set_defaults(fn=cmd_up)
    dp = sub.add_parser("deploy", help="git clone/pull, upload the remote .env, compose up --build, wait for https health")
    dp.add_argument("--branch", default="claude/sharp-tesla-po09dt")
    dp.set_defaults(fn=cmd_deploy)
    sub.add_parser("status", help="instance state, compose ps, month-to-date EC2 cost").set_defaults(fn=cmd_status)
    sub.add_parser("stop", help="stop the instance (no compute charge)").set_defaults(fn=cmd_stop)
    sub.add_parser("start", help="start a stopped instance and record the (new) IP").set_defaults(fn=cmd_start)
    sub.add_parser("logs", help="fetch the MCP call log to eval/results/mcp_calls.jsonl").set_defaults(fn=cmd_logs)
    dn = sub.add_parser("down", help="terminate and delete everything this script created")
    dn.add_argument("--yes", action="store_true")
    dn.set_defaults(fn=cmd_down)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
