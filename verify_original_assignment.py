#!/usr/bin/env python3
"""
Перевірка правильності призначення слів на оригінальному файлі
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import enhance_main_speaker_audio

def verify_assignment():
    """Перевіряє правильність призначення слів"""
    test_file = "audio examples/Screen Recording 2025-12-05 at 07.29.15.m4a"
    
    if not os.path.exists(test_file):
        print(f"❌ File not found: {test_file}")
        return
    
    print("=" * 80)
    print("VERIFYING WORD ASSIGNMENT ON ORIGINAL FILE")
    print("=" * 80)
    
    output_path, main_speaker, segments_info = enhance_main_speaker_audio(
        test_file,
        suppression_factor=0.0,
        num_speakers=2
    )
    
    transcription_segments = segments_info.get('transcription_segments', [])
    
    # З оригінального діалогу знаємо:
    # Speaker 1 (клієнт) починає: "I have a problem with my internet connection..."
    # Speaker 0 (оператор) відповідає: "Hey, did you try to reset your modem?"
    
    print(f"\n📊 Main speaker: {main_speaker}")
    print(f"   (Expected: 1, because client starts the conversation)")
    
    print(f"\n📝 First 15 transcription segments:")
    for i, seg in enumerate(transcription_segments[:15]):
        speaker = seg.get('speaker')
        text = seg.get('text', '')
        is_main = speaker == main_speaker
        marker = " [MAIN]" if is_main else " [OTHER]"
        print(f"   {i+1}. [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s] Speaker {speaker}{marker}: {text[:60]}")
        
        # Перевіряємо перші репліки
        if i == 0:
            if "I have a problem" in text or "problem with my internet" in text:
                if speaker == 1:
                    print(f"      ✅ CORRECT: First line (client) assigned to speaker 1")
                else:
                    print(f"      ❌ ERROR: First line (client) assigned to speaker {speaker}, should be 1!")
        
        if "Hey, did you try" in text or "reset your modem" in text:
            if speaker == 0:
                print(f"      ✅ CORRECT: Operator's question assigned to speaker 0")
            else:
                print(f"      ❌ ERROR: Operator's question assigned to speaker {speaker}, should be 0!")
    
    # Підраховуємо слова
    speaker_word_counts = {}
    for seg in transcription_segments:
        speaker = seg.get('speaker')
        word_count = len(seg.get('text', '').split())
        if speaker not in speaker_word_counts:
            speaker_word_counts[speaker] = 0
        speaker_word_counts[speaker] += word_count
    
    print(f"\n📊 Word distribution:")
    for speaker, count in sorted(speaker_word_counts.items()):
        marker = " 👑" if speaker == main_speaker else ""
        print(f"   Speaker {speaker}: {count} words{marker}")
    
    # Перевірка
    if 0 in speaker_word_counts and 1 in speaker_word_counts:
        print(f"\n✅ SUCCESS: Both speakers (0 and 1) are present in transcription!")
        print(f"   Speaker 0: {speaker_word_counts[0]} words")
        print(f"   Speaker 1: {speaker_word_counts[1]} words")
        
        if main_speaker == 1:
            print(f"   ✅ Main speaker is 1 (client), so speaker 0 (operator) should be suppressed")
        else:
            print(f"   ⚠️  Main speaker is {main_speaker}, but expected 1 (client)")
    else:
        missing = []
        if 0 not in speaker_word_counts:
            missing.append(0)
        if 1 not in speaker_word_counts:
            missing.append(1)
        print(f"\n❌ PROBLEM: Missing speakers in transcription: {missing}")

if __name__ == "__main__":
    verify_assignment()

