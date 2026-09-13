#!/usr/bin/env python3
"""
Upload model artifacts to Hugging Face Hub.

Run with: python upload_to_hf.py

Requires:
- HF_ACCESS_TOKEN in .env file or environment variable
- HF_USERNAME in .env file or environment variable
"""

import os
import json
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

# Load environment variables from .env file
def load_env():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Handle both "export KEY=VALUE" and "KEY=VALUE" formats
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        os.environ[key] = value

load_env()

HF_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")

if not HF_TOKEN or not HF_USERNAME:
    raise ValueError("HF_ACCESS_TOKEN and HF_USERNAME must be set in .env file or environment variables")

REPO_ID = f"{HF_USERNAME}/stock-prediction-lstm"

# Files to upload
ARTIFACTS = [
    "model_AAPL.pt",
    "scalers_AAPL.pkl",
    "config_AAPL.json",
]

def main():
    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=REPO_ID, token=HF_TOKEN, private=False, exist_ok=True)
        print(f"Repository {REPO_ID} created/verified")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    # Upload each artifact
    for artifact in ARTIFACTS:
        local_path = Path(artifact)
        if not local_path.exists():
            print(f"Warning: {artifact} not found locally, skipping")
            continue
        
        try:
            print(f"Uploading {artifact}...")
            upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=artifact,
                repo_id=REPO_ID,
                token=HF_TOKEN,
            )
            print(f"✅ Uploaded {artifact}")
        except Exception as e:
            print(f"❌ Error uploading {artifact}: {e}")
    
    # Upload model card (README)
    model_card_path = Path("hf_model_readme.md")
    if model_card_path.exists():
        try:
            print("Uploading model card (README.md)...")
            upload_file(
                path_or_fileobj=str(model_card_path),
                path_in_repo="README.md",
                repo_id=REPO_ID,
                token=HF_TOKEN,
            )
            print("✅ Uploaded model card")
        except Exception as e:
            print(f"❌ Error uploading model card: {e}")
    
    print(f"\n✅ All artifacts uploaded to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
