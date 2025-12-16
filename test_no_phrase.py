#!/usr/bin/env python3
"""
Тест призначення фрази "No, it should be 200."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import enhance_main_speaker_audio

def test_no_phrase():
    audio_path = "audio examples/detecting main speakers/speaker_0.wav"
    
    output_path, main_speaker, segments_info = enhance_main_speaker_audio(
        audio_path,
        suppression_factor=0.0
    )
    
    transcription_segments = segments_info.get('transcription_segments', [])
    
    print(f"📊 Main speaker: {main_speaker}")
    print(f"\n📝 Segments around 55-65 seconds:")
    print("=" * 80)
    
    for seg in transcription_segments:
        start = seg['start']
        end = seg['end']
        speaker = seg['speaker']
        text = seg['text'].strip()
        
        if 50 <= start <= 70:
            is_main = speaker == main_speaker
            status = "[MAIN - KEPT]" if is_main else "[OTHER - SUPPRESSED]"
            print(f"   [{start:.2f}s - {end:.2f}s] Speaker {speaker} {status}: {text}")
            
            if "No, it should be 200" in text or "should be 200" in text or "No," in text:
                print(f"      ⚠️  FOUND PHRASE!")
                if speaker == main_speaker:
                    print(f"      ✅ OK: Assigned to MAIN speaker (correct)")
                else:
                    print(f"      ❌ PROBLEM: Assigned to OTHER speaker, but should be MAIN!")

if __name__ == "__main__":
    test_no_phrase()

