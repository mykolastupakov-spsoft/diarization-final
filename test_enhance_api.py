#!/usr/bin/env python3
"""
Тест через API для перевірки роботи enhance-main-speaker
"""
import requests
import os
import json

def test_enhance_api(audio_path):
    """Тестує API enhance-main-speaker"""
    print(f"🔍 Testing API: /api/enhance-main-speaker")
    print(f"   File: {audio_path}")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    url = "http://localhost:5005/api/enhance-main-speaker"
    
    with open(audio_path, 'rb') as f:
        files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
        data = {
            'suppression_factor': '0.0',
            'num_speakers': '2',
            'return_json': 'true'
        }
        
        print("📤 Sending request...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                print(f"✅ Success!")
                print(f"   Main speaker: {result.get('main_speaker')}")
                print(f"   Total segments: {result.get('segments_info', {}).get('total_segments', 0)}")
                
                transcription_segments = result.get('segments_info', {}).get('transcription_segments', [])
                speakers_in_transcription = set(seg.get('speaker') for seg in transcription_segments)
                
                print(f"\n📊 Speakers in transcription: {sorted(speakers_in_transcription)}")
                
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
                    marker = " 👑" if speaker == result.get('main_speaker') else ""
                    print(f"   Speaker {speaker}: {count} words{marker}")
                
                print(f"\n📝 First 10 transcription segments:")
                for i, seg in enumerate(transcription_segments[:10]):
                    is_main = seg.get('speaker') == result.get('main_speaker')
                    marker = " [MAIN]" if is_main else " [OTHER]"
                    print(f"   {i+1}. [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s] Speaker {seg.get('speaker')}{marker}: {seg.get('text', '')[:60]}")
                
                # Перевірка
                if 1 in speakers_in_transcription:
                    print(f"\n✅ SUCCESS: Speaker 1 is present in transcription!")
                else:
                    print(f"\n❌ PROBLEM: Speaker 1 is NOT present in transcription!")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:500]}")

if __name__ == "__main__":
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    test_enhance_api(test_file)

