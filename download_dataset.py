#!/usr/bin/env python3
"""
Download speaker diarization datasets from Hugging Face
and save audio files to 'audio examples/Downloaded' directory
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Error: 'datasets' library not installed!")
    print("   Please run: pip install datasets")
    sys.exit(1)

# Available datasets for speaker diarization
AVAILABLE_DATASETS = {
    "callhome": "talkbank/callhome",
    "simsamu": "diarizers-community/simsamu",
    "ami": "diarizers-community/ami",
    "sakura": "talkbank/sakura",
    "callfriend": "talkbank/callfriend",
}

def download_dataset(dataset_name: str, output_dir: str = None, config: str = None):
    """
    Download a dataset from Hugging Face and extract audio files.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ami', 'callhome', etc.)
        output_dir: Directory to save audio files (default: 'audio examples/Downloaded')
        config: Optional configuration name (e.g., 'ihm', 'sdm' for AMI)
    """
    # Get dataset ID
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"   Available datasets: {', '.join(AVAILABLE_DATASETS.keys())}")
        sys.exit(1)
    
    dataset_id = AVAILABLE_DATASETS[dataset_name]
    
    # Set output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "audio examples" / "Downloaded"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Downloading Dataset: {dataset_id}")
    if config:
        print(f"Configuration: {config}")
    print("=" * 60)
    print()
    
    # Get Hugging Face token if available
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print("✅ Using HUGGINGFACE_TOKEN from environment")
    else:
        print("⚠️  No HUGGINGFACE_TOKEN found (may be needed for gated datasets)")
    
    print(f"📥 Loading dataset '{dataset_id}'...")
    print("   (This may take a while depending on dataset size)")
    print()
    
    try:
        # Try to get available configs first
        from datasets import get_dataset_config_names
        try:
            available_configs = get_dataset_config_names(dataset_id, token=hf_token)
            if available_configs:
                print(f"   Available configurations: {', '.join(available_configs)}")
                if not config:
                    # Use first config if none specified
                    config = available_configs[0]
                    print(f"   Using default configuration: {config}")
                elif config not in available_configs:
                    print(f"   ⚠️  Config '{config}' not found, using first available: {available_configs[0]}")
                    config = available_configs[0]
        except Exception as e:
            # If we can't get configs, try without config
            print(f"   Note: Could not check configurations: {e}")
        
        # Load dataset
        load_kwargs = {}
        if hf_token:
            load_kwargs['token'] = hf_token
        if config:
            load_kwargs['name'] = config
            
        dataset = load_dataset(dataset_id, **load_kwargs)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Splits available: {list(dataset.keys())}")
        print()
        
        # Process each split
        total_files = 0
        for split_name, split_data in dataset.items():
            print(f"📂 Processing split: {split_name}")
            print(f"   Number of examples: {len(split_data)}")
            
            # Determine audio column name
            audio_column = None
            for col in split_data.column_names:
                if 'audio' in col.lower() or 'path' in col.lower() or 'file' in col.lower():
                    audio_column = col
                    break
            
            if audio_column is None:
                print(f"   ⚠️  Could not find audio column in: {split_data.column_names}")
                print(f"   Columns available: {', '.join(split_data.column_names)}")
                continue
            
            print(f"   Using audio column: {audio_column}")
            
            # Extract audio files
            for idx, example in enumerate(split_data):
                try:
                    # Get audio data
                    audio_data = example[audio_column]
                    
                    # Handle different audio formats
                    if isinstance(audio_data, dict):
                        # Audio dict with 'array' and 'sampling_rate'
                        audio_array = audio_data.get('array')
                        sampling_rate = audio_data.get('sampling_rate', 16000)
                        
                        # Save as WAV file
                        import soundfile as sf
                        filename = f"{dataset_name}_{split_name}_{idx:04d}.wav"
                        filepath = output_dir / filename
                        
                        sf.write(str(filepath), audio_array, sampling_rate)
                        total_files += 1
                        
                        if (idx + 1) % 10 == 0:
                            print(f"   Processed {idx + 1}/{len(split_data)} files...")
                    
                    elif isinstance(audio_data, str):
                        # Path to audio file - copy it
                        import shutil
                        source_path = Path(audio_data)
                        if source_path.exists():
                            filename = f"{dataset_name}_{split_name}_{idx:04d}{source_path.suffix}"
                            filepath = output_dir / filename
                            shutil.copy2(source_path, filepath)
                            total_files += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing example {idx}: {e}")
                    continue
            
            print(f"   ✅ Processed {len(split_data)} examples from {split_name}")
            print()
        
        print("=" * 60)
        print(f"✅ Download complete!")
        print(f"   Total audio files saved: {total_files}")
        print(f"   Output directory: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        if "gated" in str(e).lower() or "accept" in str(e).lower():
            print()
            print("⚠️  This dataset is gated!")
            print(f"   Please visit https://huggingface.co/{dataset_id}")
            print("   and accept the user conditions.")
            print("   Then set HUGGINGFACE_TOKEN in your .env file.")
        sys.exit(1)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python download_dataset.py <dataset_name> [config] [output_dir]")
        print()
        print("Available datasets:")
        for name, dataset_id in AVAILABLE_DATASETS.items():
            print(f"  - {name}: {dataset_id}")
        print()
        print("Example:")
        print("  python download_dataset.py ami")
        print("  python download_dataset.py ami ihm")
        print("  python download_dataset.py callhome")
        sys.exit(1)
    
    dataset_name = sys.argv[1].lower()
    config = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('/') and not sys.argv[2].startswith('.') else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 and (sys.argv[2].startswith('/') or sys.argv[2].startswith('.')) else None)
    
    download_dataset(dataset_name, output_dir, config)


if __name__ == "__main__":
    main()

