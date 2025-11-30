#!/usr/bin/env python3
"""Download PyAnnote model files directly with progress tracking"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import snapshot_download, hf_hub_download
import time

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("❌ HUGGINGFACE_TOKEN not found!")
    sys.exit(1)

print("=" * 60)
print("Downloading PyAnnote Model Files")
print("=" * 60)
print()

# Download main model with all files
print("📥 Step 1: Downloading pyannote/speech-separation-ami-1.0...")
print("   (This includes all model files)")
sys.stdout.flush()

try:
    model_path = snapshot_download(
        repo_id="pyannote/speech-separation-ami-1.0",
        token=hf_token,
        local_files_only=False,
        resume_download=True  # Resume if interrupted
    )
    print(f"✅ Main model downloaded to: {model_path}")
    
    # Check size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except:
                pass
    size_mb = total_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    print()
    
except Exception as e:
    print(f"❌ Error downloading main model: {e}")
    sys.exit(1)

# Dependency model is already downloaded (1.2 GB)
dep_path = os.path.expanduser("~/.cache/huggingface/hub/models--pyannote--separation-ami-1.0")
if os.path.exists(dep_path):
    print("✅ Dependency model already in cache (1.2 GB)")
    print()

print("=" * 60)
print("✅ All model files downloaded!")
print("=" * 60)
print()
print("Now testing if model can be loaded...")
print()

# Test loading
import pyannote_patch  # noqa: F401
from pyannote.audio import Pipeline
import torch

try:
    print("🔄 Loading pipeline...")
    sys.stdout.flush()
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token=hf_token
    )
    
    if pipeline is None:
        print("❌ Pipeline is None!")
        sys.exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    print("✅ Model loaded successfully!")
    print("🎉 Everything is ready!")
    
except Exception as e:
    import traceback
    print(f"❌ Error loading pipeline: {e}")
    print()
    print("Full traceback:")
    print(traceback.format_exc())
    sys.exit(1)

