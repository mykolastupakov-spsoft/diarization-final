#!/usr/bin/env python3
"""
Знаходить проблемні призначення спікерів - слова, які можуть бути неправильно призначені
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import enhance_main_speaker_audio

def find_problematic_assignments(audio_path):
    """Знаходить проблемні призначення спікерів"""
    print(f"🔍 Analyzing: {audio_path}")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    try:
        output_path, main_speaker, segments_info = enhance_main_speaker_audio(
            audio_path,
            suppression_factor=0.0
        )
        
        print(f"\n📊 Main speaker: {main_speaker}")
        
        # Отримуємо сегменти транскрипції
        transcription_segments = segments_info.get('transcription_segments', [])
        print(f"📊 Total segments: {len(transcription_segments)}")
        
        # Знаходимо проблемні сегменти
        print(f"\n🔍 Looking for problematic assignments...")
        print("=" * 80)
        
        problematic_segments = []
        
        for i, seg in enumerate(transcription_segments):
            speaker = seg['speaker']
            text = seg['text'].strip()
            start = seg['start']
            end = seg['end']
            duration = end - start
            
            is_main = speaker == main_speaker
            status = "[MAIN - KEPT]" if is_main else "[OTHER - SUPPRESSED]"
            
            # Критерії проблемних сегментів:
            # 1. Короткі сегменти (<1s) від іншого спікера, які можуть бути частиною репліки основного
            # 2. Сегменти, які починаються дуже близько до сегментів основного спікера (<0.5s)
            # 3. Сегменти з невеликим текстом, які можуть бути частиною більшої репліки
            
            is_problematic = False
            reasons = []
            
            if not is_main and duration < 1.0:
                # Короткий сегмент від іншого спікера
                # Перевіряємо, чи є сегменти основного спікера дуже близько
                for other_seg in transcription_segments:
                    if other_seg['speaker'] == main_speaker:
                        other_start = other_seg['start']
                        other_end = other_seg['end']
                        
                        # Перевіряємо відстань до попереднього або наступного сегмента
                        gap_before = start - other_end if start > other_end else float('inf')
                        gap_after = other_start - end if other_start > end else float('inf')
                        
                        if gap_before < 0.5 or gap_after < 0.5:
                            is_problematic = True
                            reasons.append(f"Short segment ({duration:.2f}s) close to main speaker segment (gap: {min(gap_before, gap_after):.2f}s)")
                            break
            
            if not is_main and len(text.split()) < 3:
                # Сегмент з малою кількістю слів
                # Перевіряємо, чи наступний сегмент від основного спікера
                if i < len(transcription_segments) - 1:
                    next_seg = transcription_segments[i + 1]
                    if next_seg['speaker'] == main_speaker:
                        gap = next_seg['start'] - end
                        if gap < 0.5:
                            is_problematic = True
                            reasons.append(f"Short text ({len(text.split())} words) followed by main speaker segment (gap: {gap:.2f}s)")
            
            if is_problematic:
                problematic_segments.append({
                    'index': i,
                    'segment': seg,
                    'reasons': reasons
                })
        
        if problematic_segments:
            print(f"\n⚠️  Found {len(problematic_segments)} potentially problematic segments:")
            print("=" * 80)
            
            for prob in problematic_segments:
                seg = prob['segment']
                i = prob['index']
                speaker = seg['speaker']
                text = seg['text'].strip()
                start = seg['start']
                end = seg['end']
                
                print(f"\n   {i+1}. [{start:.2f}s - {end:.2f}s] Speaker {speaker} [OTHER - SUPPRESSED]")
                print(f"      Text: {text}")
                print(f"      Reasons:")
                for reason in prob['reasons']:
                    print(f"        - {reason}")
                
                # Показуємо контекст (попередній і наступний сегменти)
                if i > 0:
                    prev_seg = transcription_segments[i - 1]
                    print(f"      Previous: [{prev_seg['start']:.2f}s - {prev_seg['end']:.2f}s] "
                          f"Speaker {prev_seg['speaker']}: {prev_seg['text'].strip()[:50]}")
                if i < len(transcription_segments) - 1:
                    next_seg = transcription_segments[i + 1]
                    print(f"      Next: [{next_seg['start']:.2f}s - {next_seg['end']:.2f}s] "
                          f"Speaker {next_seg['speaker']}: {next_seg['text'].strip()[:50]}")
        else:
            print(f"\n✅ No obviously problematic segments found!")
        
        # Показуємо всі сегменти для аналізу
        print(f"\n📝 All segments (first 20):")
        print("=" * 80)
        for i, seg in enumerate(transcription_segments[:20]):
            speaker = seg['speaker']
            text = seg['text'].strip()
            start = seg['start']
            end = seg['end']
            is_main = speaker == main_speaker
            status = "[MAIN - KEPT]" if is_main else "[OTHER - SUPPRESSED]"
            print(f"   {i+1}. [{start:.2f}s - {end:.2f}s] Speaker {speaker} {status}: {text[:60]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Тестуємо на файлі користувача
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    find_problematic_assignments(test_file)

