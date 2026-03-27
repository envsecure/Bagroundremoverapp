"""
Pull the trained model from DagsHub using DVC.
Run this during Docker build or manually before serving.

Required env vars:
  DAGSHUB_REPO_OWNER   — e.g. "asish"
  DAGSHUB_REPO_NAME    — e.g. "bg-remover"
  DAGSHUB_TOKEN        — your DagsHub access token
"""

import os
import subprocess
import sys


def pull_model():
    owner = os.environ.get("DAGSHUB_REPO_OWNER")
    repo = os.environ.get("DAGSHUB_REPO_NAME")
    token = os.environ.get("DAGSHUB_TOKEN")

    if not all([owner, repo, token]):
        print("❌ Missing env vars. Set DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN")
        sys.exit(1)

    dvc_remote_url = f"https://dagshub.com/{owner}/{repo}.dvc"

    print(f"📦 Configuring DVC remote → {dvc_remote_url}")

    commands = [
        ["dvc", "remote", "add", "-f", "dagshub", dvc_remote_url],
        ["dvc", "remote", "modify", "dagshub", "auth", "basic"],
        ["dvc", "remote", "modify", "dagshub", "user", owner],
        ["dvc", "remote", "modify", "dagshub", "password", token],
        ["dvc", "pull", "-r", "dagshub"],
    ]

    for cmd in commands:
        print(f"  → {' '.join(cmd[:3])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ❌ Failed: {result.stderr}")
            sys.exit(1)

    print("✅ Model pulled successfully from DagsHub!")


if __name__ == "__main__":
    pull_model()
