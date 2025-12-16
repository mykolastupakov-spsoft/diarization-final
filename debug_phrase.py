#!/usr/bin/env python3
"""
Діагностика конкретної фрази "I can't do this."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import (
    extract_speaker_embeddings,
    diarize_audio,
    transcribe_audio,
    combine_diarization_and_transcription
)

def debug_phrase(audio_path, phrase="I can't do this"):
    """Діагностика конкретної фрази"""
    print(f"🔍 Analyzing phrase: '{phrase}' in {audio_path}")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    # Діаризація
    embeddings, timestamps = extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5)
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
    # Транскрипція
    transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
    
    # Знаходимо слова фрази
    phrase_words = []
    for word in words:
        if any(phrase_word.lower() in word['word'].lower() for phrase_word in phrase.split()):
            phrase_words.append(word)
    
    if not phrase_words:
        print(f"❌ Phrase '{phrase}' not found in transcription")
        return
    
    print(f"\n📊 Found {len(phrase_words)} words matching phrase:")
    for word in phrase_words:
        print(f"   '{word['word']}' at [{word['start']:.2f}s - {word['end']:.2f}s]")
    
    # Знаходимо сегменти діаризації, що перетинаються з фразою
    phrase_start = min(w['start'] for w in phrase_words)
    phrase_end = max(w['end'] for w in phrase_words)
    phrase_center = (phrase_start + phrase_end) / 2.0
    
    print(f"\n📊 Phrase timing: [{phrase_start:.2f}s - {phrase_end:.2f}s], center: {phrase_center:.2f}s")
    
    print(f"\n📊 Diarization segments overlapping with phrase:")
    overlapping_diar = []
    for diar_seg in sorted_diar_segments:
        diar_start = diar_seg['start']
        diar_end = diar_seg['end']
        
        if phrase_start < diar_end and phrase_end > diar_start:
            overlap_start = max(phrase_start, diar_start)
            overlap_end = min(phrase_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            overlap_ratio = overlap / (phrase_end - phrase_start) if (phrase_end - phrase_start) > 0 else 0
            
            diar_center = (diar_start + diar_end) / 2.0
            center_distance = abs(phrase_center - diar_center)
            
            overlapping_diar.append({
                'segment': diar_seg,
                'speaker': diar_seg['speaker'],
                'overlap': overlap,
                'overlap_ratio': overlap_ratio,
                'center_distance': center_distance,
                'segment_duration': diar_end - diar_start
            })
            
            print(f"   Speaker {diar_seg['speaker']} [{diar_start:.2f}s - {diar_end:.2f}s]: "
                  f"overlap={overlap:.2f}s ({overlap_ratio*100:.1f}%), "
                  f"center_dist={center_distance:.2f}s, "
                  f"duration={diar_end - diar_start:.2f}s")
    
    # Тестуємо об'єднання
    print(f"\n🔗 Testing combine_diarization_and_transcription...")
    combined = combine_diarization_and_transcription(diarization_segments, words)
    
    # Знаходимо сегмент з фразою в об'єднаному результаті
    phrase_segment = None
    for seg in combined:
        if phrase.lower() in seg.get('text', '').lower():
            phrase_segment = seg
            break
    
    if phrase_segment:
        print(f"\n📝 Phrase in combined result:")
        print(f"   [{phrase_segment['start']:.2f}s - {phrase_segment['end']:.2f}s] "
              f"Speaker {phrase_segment['speaker']}: {phrase_segment['text']}")
        
        # Визначаємо основний спікер
        speaker_word_counts = {}
        for seg in combined:
            speaker = seg['speaker']
            word_count = len(seg.get('text', '').split())
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
        
        main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0]
        
        print(f"\n📊 Main speaker: {main_speaker} ({speaker_word_counts[main_speaker]} words)")
        print(f"📊 Phrase speaker: {phrase_segment['speaker']}")
        
        if phrase_segment['speaker'] == main_speaker:
            print(f"   ⚠️  PROBLEM: Phrase is assigned to MAIN speaker, but should be OTHER!")
        else:
            print(f"   ✅ OK: Phrase is assigned to OTHER speaker (will be suppressed)")
    else:
        print(f"\n❌ Phrase not found in combined result")

if __name__ == "__main__":
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    debug_phrase(test_file, "I can't do this")

