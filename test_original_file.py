#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки enhance_main_speaker_audio на оригінальному файлі
"""
import os
import sys

# Додаємо поточну директорію до шляху
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Завантажуємо змінні середовища
from dotenv import load_dotenv
load_dotenv()

# Імпортуємо функцію
from app_ios_shortcuts import enhance_main_speaker_audio

def test_original_file(audio_path):
    """Тестує enhance_main_speaker_audio на оригінальному файлі"""
    print(f"🔍 Testing enhance_main_speaker_audio on: {audio_path}")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    try:
        # Викликаємо функцію
        output_path, main_speaker, segments_info = enhance_main_speaker_audio(
            audio_path,
            suppression_factor=0.0,  # Повне видалення неосновного спікера
            num_speakers=2
        )
        
        print(f"\n✅ Results:")
        print(f"   Output file: {output_path}")
        print(f"   Main speaker: {main_speaker}")
        print(f"   Total segments: {segments_info['total_segments']}")
        print(f"   Main speaker duration: {segments_info['main_speaker_duration']:.2f}s")
        print(f"   Main speaker percentage: {segments_info['main_speaker_percentage']:.1f}%")
        
        print(f"\n📊 All speakers:")
        for speaker, dur in sorted(segments_info['all_speakers'].items()):
            marker = " 👑" if speaker == main_speaker else ""
            print(f"   Speaker {speaker}: {dur:.2f}s{marker}")
        
        print(f"\n📝 Transcription segments (first 15):")
        for seg in segments_info['transcription_segments'][:15]:
            is_main = seg['speaker'] == main_speaker
            marker = " [MAIN]" if is_main else " [OTHER]"
            print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}{marker}: {seg['text'][:60]}")
        
        # Перевіряємо, чи є обидва спікери в транскрипції
        speakers_in_transcription = set(seg['speaker'] for seg in segments_info['transcription_segments'])
        print(f"\n📊 Speakers in transcription: {sorted(speakers_in_transcription)}")
        
        # Підраховуємо слова по спікерах
        speaker_word_counts = {}
        for seg in segments_info['transcription_segments']:
            speaker = seg['speaker']
            word_count = len(seg['text'].split())
            if speaker not in speaker_word_counts:
                speaker_word_counts[speaker] = 0
            speaker_word_counts[speaker] += word_count
        
        print(f"\n📊 Word distribution by speaker:")
        for speaker, count in sorted(speaker_word_counts.items()):
            marker = " 👑" if speaker == main_speaker else ""
            print(f"   Speaker {speaker}: {count} words{marker}")
        
        # Перевіряємо, чи спікер 1 присутній
        if 1 in speakers_in_transcription:
            print(f"\n✅ Speaker 1 is present in transcription")
            speaker_1_words = speaker_word_counts.get(1, 0)
            speaker_0_words = speaker_word_counts.get(0, 0)
            print(f"   Speaker 0: {speaker_0_words} words")
            print(f"   Speaker 1: {speaker_1_words} words")
            
            # Перевіряємо, чи спікер 1 правильно визначений
            if main_speaker == 0 and speaker_1_words > 0:
                print(f"\n⚠️  PROBLEM: Main speaker is 0, but speaker 1 has {speaker_1_words} words!")
                print(f"   Expected: Speaker 1 should be main (client starts conversation)")
        else:
            print(f"\n❌ PROBLEM: Speaker 1 is NOT present in transcription (only speaker 0 found)")
            print(f"   This means speaker 1's words are being assigned to speaker 0!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Тестуємо на оригінальному файлі
    test_file = "audio examples/Screen Recording 2025-12-05 at 07.29.15.m4a"
    if os.path.exists(test_file):
        test_original_file(test_file)
    else:
        print(f"❌ Original file not found: {test_file}")
        print("Testing on speaker_0.wav instead...")
        test_file = "audio examples/detecting main speakers/speaker_0.wav"
        test_original_file(test_file)

