#!/usr/bin/env python3
"""
Quick validation helpers for the Azure Speech-to-Text client.

Usage:
    python test_azure_stt.py               # Parse built-in sample JSON
    python test_azure_stt.py --live-url https://.../audio.wav --language en-US
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

from azure_stt import AzureSpeechClient


def run_sample() -> None:
    """Parse the documentation sample payload and print normalized segments."""
    sample_payload: Dict[str, Any] = {
        "source": "https://your-bucket.blob.core.windows.net/audio/meeting.wav",
        "recognizedPhrases": [
            {
                "recognitionStatus": "Success",
                "channel": 0,
                "offset": "PT0.76S",
                "duration": "PT1.32S",
                "speaker": 1,
                "nBest": [
                    {
                        "confidence": 0.8643338,
                        "lexical": "hello this is speaker one",
                        "display": "Hello, this is speaker one.",
                        "words": [
                            {"word": "hello", "offset": "PT0.76S", "duration": "PT0.52S"},
                            {"word": "this", "offset": "PT1.28S", "duration": "PT0.39S"},
                            {"word": "is", "offset": "PT1.67S", "duration": "PT0.26S"},
                            {"word": "speaker", "offset": "PT1.93S", "duration": "PT0.45S"},
                            {"word": "one", "offset": "PT2.38S", "duration": "PT0.40S"},
                        ],
                    }
                ],
            },
            {
                "recognitionStatus": "Success",
                "channel": 0,
                "offset": "PT2.10S",
                "duration": "PT0.68S",
                "speaker": 2,
                "nBest": [
                    {
                        "confidence": 0.9012547,
                        "lexical": "and i am speaker two",
                        "display": "And I am speaker two.",
                        "words": [
                            {"word": "and", "offset": "PT2.10S", "duration": "PT0.30S"},
                            {"word": "i", "offset": "PT2.40S", "duration": "PT0.21S"},
                            {"word": "am", "offset": "PT2.61S", "duration": "PT0.24S"},
                            {"word": "speaker", "offset": "PT2.85S", "duration": "PT0.42S"},
                            {"word": "two", "offset": "PT3.27S", "duration": "PT0.35S"},
                        ],
                    }
                ],
            },
        ],
    }

    client = AzureSpeechClient(subscription_key="test-key", region="eastus")
    segments = client.parse_response_to_segments(sample_payload)
    print(f"Parsed {len(segments)} segments from sample payload:")
    print(json.dumps(segments, indent=2))


def run_live(audio_url: str, language: str) -> None:
    """Submit a real Azure batch transcription job (requires env vars)."""
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")

    if not key or not region:
        print("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set for live tests.", file=sys.stderr)
        sys.exit(1)

    client = AzureSpeechClient(subscription_key=key, region=region, endpoint=endpoint)
    segments, _raw = client.full_workflow(
        audio_urls=[audio_url],
        locale=language,
        diarization_enabled=True,
        min_speakers=1,
        max_speakers=4,
    )
    print(f"Live transcription returned {len(segments)} segments.")
    print(json.dumps(segments[:5], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Azure STT client quick validation")
    parser.add_argument("--live-url", help="Audio URL to submit to Azure STT (requires Azure credentials)")
    parser.add_argument("--language", default="en-US", help="Locale for live submissions (default: en-US)")
    args = parser.parse_args()

    if args.live_url:
        run_live(args.live_url, args.language)
    else:
        run_sample()


if __name__ == "__main__":
    main()

