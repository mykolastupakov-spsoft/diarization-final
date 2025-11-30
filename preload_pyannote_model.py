#!/usr/bin/env python3
"""
Preload PyAnnote speech-separation-ami-1.0 model into cache
This script downloads the model once so it's ready for use
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import patch BEFORE pyannote to fix torchaudio compatibility
import pyannote_patch  # noqa: F401

try:
    from pyannote.audio import Pipeline
    import torch
except ImportError as e:
    print(f"❌ Missing required package: {e.name}")
    print("Please install: pip install pyannote.audio torch torchaudio")
    sys.exit(1)

def preload_model():
    """Preload the PyAnnote model into cache"""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not found in environment variables!")
        print("Please set it in .env file")
        sys.exit(1)
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache_path = os.path.join(cache_dir, "models--pyannote--speech-separation-ami-1.0")
    
    if os.path.exists(model_cache_path):
        print("✅ Model already in cache!")
        print(f"Cache location: {model_cache_path}")
        
        # Calculate cache size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_cache_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except:
                    pass
        size_mb = total_size / (1024 * 1024)
        print(f"Cache size: {size_mb:.1f} MB")
        return True
    
    print("📥 Model not in cache, starting download...")
    print("⏳ This may take 5-10 minutes (depending on internet speed)...")
    print("💡 You can monitor progress in the cache directory:")
    print(f"   {cache_dir}")
    print()
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 Using device: {device}")
        print("🔄 Loading model (this will download it)...")
        print()
        
        # First, try to authenticate and check access
        try:
            from huggingface_hub import whoami, login
            print("🔑 Checking Hugging Face authentication...")
            user_info = whoami(token=hf_token)
            print(f"✅ Authenticated as: {user_info.get('name', 'unknown')}")
            print()
        except Exception as auth_error:
            print(f"⚠️  Authentication check failed: {auth_error}")
            print("   Continuing anyway...")
            print()
        
        print("📥 Downloading model from Hugging Face...")
        print("   (This may take 5-10 minutes)")
        print()
        
        # First, try to download model files using huggingface_hub
        try:
            from huggingface_hub import snapshot_download
            print("📦 Step 1: Downloading model files...")
            sys.stdout.flush()
            
            # Download main model
            print("   Downloading pyannote/speech-separation-ami-1.0...")
            sys.stdout.flush()
            model_path = snapshot_download(
                repo_id="pyannote/speech-separation-ami-1.0",
                token=hf_token,
                local_files_only=False
            )
            print(f"✅ Main model files downloaded to: {model_path}")
            sys.stdout.flush()
            
            # Also download dependency model
            print("   Downloading dependency: pyannote/separation-ami-1.0...")
            sys.stdout.flush()
            try:
                dep_model_path = snapshot_download(
                    repo_id="pyannote/separation-ami-1.0",
                    token=hf_token,
                    local_files_only=False
                )
                print(f"✅ Dependency model files downloaded to: {dep_model_path}")
                sys.stdout.flush()
            except Exception as dep_error:
                dep_error_str = str(dep_error)
                if "gated" in dep_error_str.lower() or "accept" in dep_error_str.lower():
                    print("⚠️  Dependency model is gated!")
                    print("   Please visit https://hf.co/pyannote/separation-ami-1.0")
                    print("   and accept the user conditions there too.")
                    return False
                else:
                    print(f"⚠️  Dependency download warning: {dep_error}")
                    print("   Continuing anyway...")
                    sys.stdout.flush()
                    
        except Exception as download_error:
            error_str = str(download_error)
            if "gated" in error_str.lower() or "accept" in error_str.lower():
                print("❌ Model is still gated or access not propagated yet.")
                print("   Please wait 2-3 minutes after accepting and try again.")
                return False
            else:
                print(f"⚠️  Direct download failed: {download_error}")
                print("   Trying Pipeline.from_pretrained instead...")
                sys.stdout.flush()
        
        print("📦 Step 2: Loading pipeline...")
        sys.stdout.flush()
        
        # Try loading from local path if available
        try:
            if 'model_path' in locals() and os.path.exists(model_path):
                print(f"   Using local model path: {model_path}")
                sys.stdout.flush()
                pipeline = Pipeline.from_pretrained(model_path, use_auth_token=hf_token)
            else:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speech-separation-ami-1.0",
                    use_auth_token=hf_token
                )
        except Exception as pipeline_error:
            # If loading from path failed, try from repo again
            print(f"   Loading from path failed, trying from repo...")
            sys.stdout.flush()
            pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=hf_token
            )
        
        if pipeline is None:
            print("❌ Failed to load pipeline!")
            print()
            print("Please check:")
            print("1. Visit https://hf.co/pyannote/speech-separation-ami-1.0")
            print("   and click 'Agree and access repository'")
            print("2. Make sure your HUGGINGFACE_TOKEN is valid")
            print("3. Wait a few minutes after accepting, then try again")
            return False
        
        pipeline.to(device)
        
        # Verify cache
        if os.path.exists(model_cache_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_cache_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except:
                        pass
            size_mb = total_size / (1024 * 1024)
            
            print()
            print("✅ Model successfully downloaded and cached!")
            print(f"📦 Cache location: {model_cache_path}")
            print(f"💾 Cache size: {size_mb:.1f} MB")
            print()
            print("🎉 Model is ready to use! You can now use Mode 2 without waiting.")
            return True
        else:
            print("⚠️  Model loaded but cache path not found. This is unusual.")
            return False
            
    except Exception as e:
        error_msg = str(e)
        import traceback
        full_error = traceback.format_exc()
        
        print(f"❌ Error: {error_msg}")
        print()
        
        if "gated" in error_msg.lower() or "accept" in error_msg.lower() or "NoneType" in error_msg:
            print("🔒 Model access issue detected!")
            print()
            print("📋 Steps to fix:")
            print("1. Visit: https://hf.co/pyannote/speech-separation-ami-1.0")
            print("2. Click 'Agree and access repository' button")
            print("3. Wait 1-2 minutes for access to propagate")
            print("4. Make sure your HUGGINGFACE_TOKEN is set in .env file")
            print("5. Run this script again")
            print()
            print("💡 If you already accepted, wait a bit and try again.")
        elif "authentication" in error_msg.lower() or "token" in error_msg.lower():
            print("🔑 Authentication failed!")
            print()
            print("Please check:")
            print("1. Your HUGGINGFACE_TOKEN in .env file")
            print("2. Token is valid (visit https://hf.co/settings/tokens)")
            print("3. Token has 'read' permissions")
        else:
            print("❓ Unexpected error occurred.")
            print()
            print("Full error details:")
            print(full_error[-500:])  # Last 500 chars
        
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PyAnnote Model Preloader")
    print("=" * 60)
    print()
    
    success = preload_model()
    
    print()
    print("=" * 60)
    if success:
        print("✅ Preload completed successfully!")
        sys.exit(0)
    else:
        print("❌ Preload failed. Please check the errors above.")
        sys.exit(1)

