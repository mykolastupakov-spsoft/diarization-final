#!/usr/bin/env python3
"""
Тест на оригінальному файлі з обома спікерами
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import enhance_main_speaker_audio

def test_original():
    """Тестує на оригінальному файлі"""
    # Спочатку тестуємо на speaker_0.wav (одноголосий)
    print("=" * 80)
    print("TEST 1: speaker_0.wav (single speaker file)")
    print("=" * 80)
    
    test_file1 = "audio examples/detecting main speakers/speaker_0.wav"
    if os.path.exists(test_file1):
        try:
            output_path, main_speaker, segments_info = enhance_main_speaker_audio(
                test_file1,
                suppression_factor=0.0,
                num_speakers=2
            )
            
            transcription_segments = segments_info.get('transcription_segments', [])
            speakers_in_transcription = set(seg.get('speaker') for seg in transcription_segments)
            
            print(f"\n📊 Results for speaker_0.wav:")
            print(f"   Main speaker: {main_speaker}")
            print(f"   Speakers in transcription: {sorted(speakers_in_transcription)}")
            
            speaker_word_counts = {}
            for seg in transcription_segments:
                speaker = seg.get('speaker')
                word_count = len(seg.get('text', '').split())
                if speaker not in speaker_word_counts:
                    speaker_word_counts[speaker] = 0
                speaker_word_counts[speaker] += word_count
            
            print(f"   Word distribution: {speaker_word_counts}")
            
            if len(speakers_in_transcription) > 1:
                print(f"   ⚠️  WARNING: Found {len(speakers_in_transcription)} speakers in single-speaker file!")
            else:
                print(f"   ✅ OK: Only one speaker found (as expected for single-speaker file)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Тестуємо на оригінальному файлі (якщо доступний)
    print("\n" + "=" * 80)
    print("TEST 2: Original file with both speakers")
    print("=" * 80)
    
    test_file2 = "audio examples/Screen Recording 2025-12-05 at 07.29.15.m4a"
    if os.path.exists(test_file2):
        try:
            print(f"   Testing on: {test_file2}")
            output_path, main_speaker, segments_info = enhance_main_speaker_audio(
                test_file2,
                suppression_factor=0.0,
                num_speakers=2
            )
            
            transcription_segments = segments_info.get('transcription_segments', [])
            speakers_in_transcription = set(seg.get('speaker') for seg in transcription_segments)
            
            print(f"\n📊 Results for original file:")
            print(f"   Main speaker: {main_speaker}")
            print(f"   Speakers in transcription: {sorted(speakers_in_transcription)}")
            
            speaker_word_counts = {}
            for seg in transcription_segments:
                speaker = seg.get('speaker')
                word_count = len(seg.get('text', '').split())
                if speaker not in speaker_word_counts:
                    speaker_word_counts[speaker] = 0
                speaker_word_counts[speaker] += word_count
            
            print(f"   Word distribution: {speaker_word_counts}")
            
            if 1 in speakers_in_transcription:
                print(f"   ✅ SUCCESS: Speaker 1 is present in transcription!")
                speaker1_words = speaker_word_counts.get(1, 0)
                speaker0_words = speaker_word_counts.get(0, 0)
                print(f"      Speaker 0: {speaker0_words} words")
                print(f"      Speaker 1: {speaker1_words} words")
                
                if main_speaker == 0:
                    print(f"   ✅ Main speaker is 0, so speaker 1 should be suppressed")
                else:
                    print(f"   ⚠️  Main speaker is {main_speaker}, not 0")
            else:
                print(f"   ❌ PROBLEM: Speaker 1 is NOT present in transcription!")
                print(f"      This means speaker 1's words are being assigned to speaker 0!")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ⚠️  Original file not found: {test_file2}")
        print(f"   Testing only on speaker_0.wav")

if __name__ == "__main__":
    test_original()

