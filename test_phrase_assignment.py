#!/usr/bin/env python3
"""
Тест призначення фрази "I can't do this."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import enhance_main_speaker_audio

def test_phrase():
    audio_path = "audio examples/detecting main speakers/speaker_0.wav"
    
    output_path, main_speaker, segments_info = enhance_main_speaker_audio(
        audio_path,
        suppression_factor=0.0
    )
    
    transcription_segments = segments_info.get('transcription_segments', [])
    
    print(f"📊 Main speaker: {main_speaker}")
    print(f"📊 Total segments: {len(transcription_segments)}")
    print(f"\n📝 All segments around 23-42 seconds:")
    print("=" * 80)
    
    for seg in transcription_segments:
        start = seg['start']
        end = seg['end']
        speaker = seg['speaker']
        text = seg['text'].strip()
        
        # Показуємо сегменти в діапазоні 23-42 секунди
        if 20 <= start <= 45:
            is_main = speaker == main_speaker
            status = "[MAIN - KEPT]" if is_main else "[OTHER - SUPPRESSED]"
            print(f"   [{start:.2f}s - {end:.2f}s] Speaker {speaker} {status}: {text}")
            
            # Перевіряємо чи це наша фраза
            if "can't do this" in text.lower():
                print(f"      ⚠️  FOUND PHRASE!")
                if speaker == main_speaker:
                    print(f"      ❌ PROBLEM: Assigned to MAIN speaker, should be OTHER!")
                else:
                    print(f"      ✅ OK: Assigned to OTHER speaker (will be suppressed)")

if __name__ == "__main__":
    test_phrase()

