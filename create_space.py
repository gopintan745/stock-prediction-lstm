#!/usr/bin/env python3
"""
Create and deploy Hugging Face Space for the Stock Prediction LSTM app.

Run with: python create_space.py

Requires:
- HF_ACCESS_TOKEN in .env file or environment variable
- HF_USERNAME in .env file or environment variable
"""

import os
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

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

SPACE_ID = f"{HF_USERNAME}/stock-prediction-lstm-space"

# Files/folders to include in the space
SPACE_FILES = [
    "app/",
    "data/",
    "models/",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Files/folders to ignore
IGNORE_PATTERNS = [
    "*.pt", "*.pkl", "*.db", "*.json",  # Don't upload model artifacts to space
    "__pycache__", "*.pyc", ".pyc",
    ".venv", "venv", "env",
    ".git", ".gitignore",
    "upload_to_hf.py", "create_space.py", "hf_model_readme.md",
    "debug_*.py", "test_*.py",
    "training/", "utils/", "config_*.json", "scalers_*.pkl", "model_*.pt",
    "optuna_study*",
    ".env",
]

def main():
    api = HfApi(token=HF_TOKEN)
    
    # Create space repo if it doesn't exist
    try:
        create_repo(
            repo_id=SPACE_ID, 
            token=HF_TOKEN, 
            private=False, 
            exist_ok=True,
            repo_type="space",
            space_sdk="static",
        )
        print(f"Space {SPACE_ID} created/verified")
    except Exception as e:
        print(f"Error creating space: {e}")
        return
    
    # Upload space files
    print("Uploading space files...")
    try:
        upload_folder(
            folder_path=".",
            repo_id=SPACE_ID,
            token=HF_TOKEN,
            repo_type="space",
            ignore_patterns=IGNORE_PATTERNS,
        )
        print(f"✅ Space files uploaded to https://huggingface.co/spaces/{SPACE_ID}")
    except Exception as e:
        print(f"❌ Error uploading space files: {e}")
        return
    
    print(f"\n🚀 Space deployed at: https://huggingface.co/spaces/{SPACE_ID}")
    print(f"   The space will build automatically. Check the build logs on the space page.")

if __name__ == "__main__":
    main()
