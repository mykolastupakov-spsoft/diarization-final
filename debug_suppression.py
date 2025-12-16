#!/usr/bin/env python3
"""
Діагностика проблеми з відсіканням реплік основного спікера
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import enhance_main_speaker_audio

def debug_suppression():
    """Діагностика відсікання"""
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    
    if not os.path.exists(test_file):
        print(f"❌ File not found: {test_file}")
        return
    
    print("=" * 80)
    print("DEBUGGING SUPPRESSION ISSUE")
    print("=" * 80)
    
    output_path, main_speaker, segments_info = enhance_main_speaker_audio(
        test_file,
        suppression_factor=0.0,
        num_speakers=2
    )
    
    transcription_segments = segments_info.get('transcription_segments', [])
    
    print(f"\n📊 Main speaker determined: {main_speaker}")
    print(f"\n📝 First 10 transcription segments with suppression info:")
    
    for i, seg in enumerate(transcription_segments[:10]):
        speaker = seg.get('speaker')
        text = seg.get('text', '')
        is_main = speaker == main_speaker
        will_be_suppressed = not is_main
        marker = " [MAIN - KEPT]" if is_main else f" [OTHER - {'SUPPRESSED' if will_be_suppressed else 'KEPT'}]"
        
        print(f"   {i+1}. [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s] Speaker {speaker}{marker}")
        print(f"      Text: {text[:80]}")
        
        # Перевіряємо проблемні слова
        if "Hey," in text:
            print(f"      ⚠️  'Hey,' is in this segment")
            if will_be_suppressed:
                print(f"      ❌ PROBLEM: 'Hey,' will be SUPPRESSED, but it's part of main speaker's line!")
            else:
                print(f"      ✅ OK: 'Hey,' will be KEPT (main speaker)")
        
        if "dropping" in text.lower():
            print(f"      ⚠️  'dropping' is in this segment")
            if will_be_suppressed:
                print(f"      ✅ OK: 'dropping' will be SUPPRESSED (other speaker)")
            else:
                print(f"      ❌ PROBLEM: 'dropping' will be KEPT, but it's from other speaker!")
    
    # Перевіряємо, які спікери є в транскрипції
    speakers_in_transcription = set(seg.get('speaker') for seg in transcription_segments)
    print(f"\n📊 Speakers in transcription: {sorted(speakers_in_transcription)}")
    print(f"   Main speaker: {main_speaker}")
    
    # Підраховуємо слова по спікерах
    speaker_word_counts = {}
    for seg in transcription_segments:
        speaker = seg.get('speaker')
        word_count = len(seg.get('text', '').split())
        if speaker not in speaker_word_counts:
            speaker_word_counts[speaker] = 0
        speaker_word_counts[speaker] += word_count
    
    print(f"\n📊 Word distribution by speaker:")
    for speaker, count in sorted(speaker_word_counts.items()):
        is_main = speaker == main_speaker
        marker = " 👑 [MAIN - KEPT]" if is_main else f" [OTHER - {'SUPPRESSED' if not is_main else 'KEPT'}]"
        print(f"   Speaker {speaker}: {count} words{marker}")

if __name__ == "__main__":
    debug_suppression()

