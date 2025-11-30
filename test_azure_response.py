#!/usr/bin/env python3
"""
Test script to get raw Azure STT response and understand its structure.
This will help us create a proper parser for Azure responses without
trying to normalize them to Speechmatics format.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_azure_realtime():
    """Test Azure Realtime STT and show raw response structure."""
    try:
        from azure_realtime import AzureRealtimeDiarization
    except ImportError:
        logger.error("azure-cognitiveservices-speech is required. Install it: pip install azure-cognitiveservices-speech")
        return None
    
    api_key = os.getenv('AZURE_SPEECH_KEY')
    region = os.getenv('AZURE_SPEECH_REGION')
    
    if not api_key or not region:
        logger.error("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
        return None
    
    # Use example audio file - Azure Realtime requires WAV format
    audio_file_mp3 = Path(__file__).parent / "audio examples" / "OverlappingCallCenterWithoutBackground.MP3"
    if not audio_file_mp3.exists():
        logger.error(f"Audio file not found: {audio_file_mp3}")
        return None
    
    # Convert MP3 to WAV for Azure Realtime (requires ffmpeg)
    import subprocess
    import tempfile
    import shutil
    
    ffmpeg_bin = shutil.which('ffmpeg')
    if not ffmpeg_bin:
        logger.error("ffmpeg is required to convert MP3 to WAV for Azure Realtime")
        return None
    
    temp_dir = tempfile.mkdtemp()
    audio_file_wav = Path(temp_dir) / "test_audio.wav"
    
    logger.info(f"Converting {audio_file_mp3.name} to WAV format...")
    cmd = [
        ffmpeg_bin,
        '-hide_banner',
        '-loglevel', 'error',
        '-y',
        '-i', str(audio_file_mp3),
        '-ac', '1',
        '-ar', '16000',
        '-sample_fmt', 's16',
        str(audio_file_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not audio_file_wav.exists():
        logger.error(f"ffmpeg conversion failed: {result.stderr}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    
    logger.info(f"Testing Azure Realtime STT with: {audio_file_mp3.name} (converted to WAV)")
    
    try:
        client = AzureRealtimeDiarization(subscription_key=api_key, region=region)
        segments, raw_payload = client.transcribe_from_file(str(audio_file_wav), language="en-US")
    finally:
        # Cleanup temp file
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info(f"✅ Transcription completed: {len(segments)} segments")
    
    # Show raw response structure
    print("\n" + "="*80)
    print("RAW AZURE REALTIME RESPONSE STRUCTURE")
    print("="*80)
    print(json.dumps(raw_payload, indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("NORMALIZED SEGMENTS (current format)")
    print("="*80)
    print(json.dumps(segments[:5] if len(segments) > 5 else segments, indent=2, ensure_ascii=False))
    if len(segments) > 5:
        print(f"\n... and {len(segments) - 5} more segments")
    
    return {
        'raw_payload': raw_payload,
        'segments': segments,
        'total_segments': len(segments)
    }

def test_azure_batch():
    """Test Azure Batch STT and show raw response structure."""
    try:
        from azure_stt import AzureSpeechClient, upload_file_to_azure_blob
    except ImportError:
        logger.error("azure-storage-blob is required. Install it: pip install azure-storage-blob")
        return None
    
    api_key = os.getenv('AZURE_SPEECH_KEY')
    region = os.getenv('AZURE_SPEECH_REGION')
    endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')
    
    if not api_key or not region:
        logger.error("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env")
        return None
    
    # Use example audio file
    audio_file = Path(__file__).parent / "audio examples" / "OverlappingCallCenterWithoutBackground.MP3"
    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        return None
    
    logger.info(f"Testing Azure Batch STT with: {audio_file.name}")
    
    # Upload to blob storage
    logger.info("Uploading file to Azure Blob Storage...")
    audio_url = upload_file_to_azure_blob(str(audio_file))
    logger.info(f"✅ File uploaded: {audio_url[:80]}...")
    
    client = AzureSpeechClient(subscription_key=api_key, region=region, endpoint=endpoint)
    
    # Submit batch transcription
    logger.info("Submitting batch transcription job...")
    segments, azure_payload = client.full_workflow(
        audio_urls=[audio_url],
        locale="en-US",
        diarization_enabled=True,
        min_speakers=2,
        max_speakers=4
    )
    
    logger.info(f"✅ Transcription completed: {len(segments)} segments")
    
    # Show raw response structure
    print("\n" + "="*80)
    print("RAW AZURE BATCH RESPONSE STRUCTURE")
    print("="*80)
    print(json.dumps(azure_payload, indent=2, ensure_ascii=False))
    
    print("\n" + "="*80)
    print("NORMALIZED SEGMENTS (current format)")
    print("="*80)
    print(json.dumps(segments[:5] if len(segments) > 5 else segments, indent=2, ensure_ascii=False))
    if len(segments) > 5:
        print(f"\n... and {len(segments) - 5} more segments")
    
    return {
        'raw_payload': azure_payload,
        'segments': segments,
        'total_segments': len(segments)
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Azure STT and show raw response structure')
    parser.add_argument('--mode', choices=['realtime', 'batch', 'both'], default='realtime',
                       help='Which Azure STT mode to test')
    parser.add_argument('--output', help='Save response to JSON file')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.mode in ('realtime', 'both'):
        print("\n" + "="*80)
        print("TESTING AZURE REALTIME STT")
        print("="*80)
        result = test_azure_realtime()
        if result:
            results['realtime'] = result
    
    if args.mode in ('batch', 'both'):
        print("\n" + "="*80)
        print("TESTING AZURE BATCH STT")
        print("="*80)
        result = test_azure_batch()
        if result:
            results['batch'] = result
    
    if args.output and results:
        output_file = Path(args.output)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {output_file}")

