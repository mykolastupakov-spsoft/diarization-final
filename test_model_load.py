#!/usr/bin/env python3
"""Test script to load PyAnnote model with detailed error reporting"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

import pyannote_patch  # noqa: F401

from pyannote.audio import Pipeline
import torch

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("❌ HUGGINGFACE_TOKEN not found!")
    sys.exit(1)

print("🔄 Attempting to load model...")
print(f"Token starts with: {hf_token[:10]}...")
print()

try:
    print("Step 1: Calling Pipeline.from_pretrained...")
    sys.stdout.flush()
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token=hf_token
    )
    
    print("Step 2: Checking if pipeline is None...")
    if pipeline is None:
        print("❌ Pipeline is None!")
        sys.exit(1)
    
    print("Step 3: Moving to device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    print("✅ Model loaded successfully!")
    print(f"Pipeline type: {type(pipeline)}")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    print()
    print("Full traceback:")
    print(traceback.format_exc())
    sys.exit(1)

