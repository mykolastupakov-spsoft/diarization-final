#!/usr/bin/env python3
"""
Транскрибує одноголосі файли через Speechmatics API з діаризацією
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

# Завантажуємо змінні з .env файлу, якщо він існує
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv не встановлений, спробуємо без нього

def format_dialogue(segments, speaker_label_prefix="Speaker"):
    """Форматує сегменти у читабельний діалог"""
    lines = []
    for seg in segments:
        start_time = seg.get('start', 0)
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        speaker = seg.get('speaker', 'SPEAKER_00')
        # Нормалізуємо формат спікера (SPEAKER_00 -> 0, SPEAKER_01 -> 1)
        if isinstance(speaker, str) and speaker.startswith('SPEAKER_'):
            try:
                speaker_num = int(speaker.replace('SPEAKER_', ''))
            except:
                speaker_num = 0
        else:
            speaker_num = int(speaker) if isinstance(speaker, (int, str)) else 0
        
        text = seg.get('text', '').strip()
        
        if text:
            lines.append(f"{time_str} {speaker_label_prefix} {speaker_num}: {text}")
    
    return "\n".join(lines)

def upload_to_speechmatics(api_key, file_path, language='en', is_separated_track=True):
    """Завантажує файл до Speechmatics та повертає job_id"""
    url = 'https://asr.api.speechmatics.com/v2/jobs'
    
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    upload_timeout = max(300, int(60 + (file_size_mb / 10) * 60))
    
    print(f"📤 Uploading {os.path.basename(file_path)} ({file_size_mb:.2f}MB) to Speechmatics...")
    
    # Використовуємо 'speaker' mode для виявлення залишків інших спікерів
    diarization_mode = 'speaker'
    
    config = {
        'type': 'transcription',
        'transcription_config': {
            'language': language,
            'diarization': diarization_mode,
            'operating_point': 'enhanced',
            'speaker_diarization_config': {
                'get_speakers': True
            }
        }
    }
    
    data = {'config': json.dumps(config)}
    headers = {'Authorization': f'Bearer {api_key}'}
    
    # Визначаємо MIME type
    file_ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.mp4': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.webm': 'audio/webm',
        '.aac': 'audio/aac',
    }
    mime_type = mime_types.get(file_ext, 'audio/wav')
    
    with open(file_path, 'rb') as f:
        files = {'data_file': (os.path.basename(file_path), f, mime_type)}
        
        response = requests.post(
            url,
            files=files,
            data=data,
            headers=headers,
            timeout=upload_timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        if not result.get('id'):
            raise ValueError(f"No job ID in response: {result}")
        
        print(f"✅ Job created: {result['id']}")
        return result['id']

def poll_speechmatics_job(api_key, job_id, max_attempts=120, poll_interval=3):
    """Очікує завершення завдання та повертає результат"""
    url = f'https://asr.api.speechmatics.com/v2/jobs/{job_id}'
    headers = {'Authorization': f'Bearer {api_key}'}
    
    print(f"⏳ Waiting for job {job_id} to complete...")
    
    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        job_status = result.get('job', {}).get('status', 'unknown')
        
        if job_status == 'done':
            print(f"✅ Job completed!")
            # Отримуємо транскрипт
            transcript_url = f'https://asr.api.speechmatics.com/v2/jobs/{job_id}/transcript'
            transcript_response = requests.get(
                transcript_url,
                headers={**headers, 'Accept': 'application/json'},
                timeout=30
            )
            transcript_response.raise_for_status()
            transcript_data = transcript_response.json()
            
            return transcript_data
        elif job_status == 'rejected':
            failure_reason = result.get('job', {}).get('failure_reason', 'Unknown error')
            raise Exception(f"Job rejected: {failure_reason}")
        
        if attempt % 10 == 0:  # Логуємо кожні 10 спроб
            print(f"   Status: {job_status} (attempt {attempt + 1}/{max_attempts})")
        
        time.sleep(poll_interval)
    
    raise TimeoutError(f"Job {job_id} did not complete within {max_attempts * poll_interval} seconds")

def process_file(api_key, file_path, output_dir, language='en'):
    """Обробляє один файл через Speechmatics"""
    file_name = os.path.basename(file_path)
    file_base = os.path.splitext(file_name)[0]
    
    print(f"\n{'='*60}")
    print(f"📁 Processing: {file_name}")
    print(f"{'='*60}")
    
    # 1. Завантажуємо файл
    job_id = upload_to_speechmatics(api_key, file_path, language, is_separated_track=True)
    
    # 2. Очікуємо завершення
    transcript_data = poll_speechmatics_job(api_key, job_id)
    
    # 3. Витягуємо segments з формату Speechmatics v2
    words = []
    if transcript_data.get('results') and isinstance(transcript_data['results'], list):
        for result in transcript_data['results']:
            if result.get('type') == 'punctuation':
                continue
            
            if result.get('type') == 'word' and result.get('alternatives'):
                alt = result['alternatives'][0]
                speaker_label = alt.get('speaker', 'S1')
                
                # Convert "S1" -> 0, "S2" -> 1, etc.
                speaker_num = 0
                if speaker_label.startswith('S'):
                    num_str = speaker_label[1:]
                    speaker_num = int(num_str) - 1 if num_str.isdigit() else 0
                else:
                    speaker_num = int(speaker_label) if str(speaker_label).isdigit() else 0
                
                words.append({
                    'word': alt.get('content', ''),
                    'start': result.get('start_time', 0),
                    'end': result.get('end_time', result.get('start_time', 0)),
                    'speaker': speaker_num
                })
    
    if not words:
        print("⚠️  No words found in transcript")
        combined_segments = []
    else:
        # Групуємо слова по спікерах та сегментах
        combined_segments = []
        current_speaker = None
        current_start = None
        current_words = []
        
        for word_info in words:
            speaker = word_info.get('speaker', 0)
            word_start = word_info.get('start', 0)
            word_text = word_info.get('word', '').strip()
            
            if not word_text:
                continue
            
            if speaker != current_speaker:
                if current_speaker is not None and current_words:
                    combined_segments.append({
                        'speaker': current_speaker,
                        'start': round(current_start, 2),
                        'end': round(word_start, 2),
                        'text': ' '.join(current_words).strip()
                    })
                current_speaker = speaker
                current_start = word_start
                current_words = [word_text]
            else:
                current_words.append(word_text)
        
        if current_words:
            combined_segments.append({
                'speaker': current_speaker if current_speaker is not None else 0,
                'start': round(current_start, 2) if current_start is not None else 0,
                'end': round(words[-1].get('end', 0), 2) if words else 0,
                'text': ' '.join(current_words).strip()
            })
    
    # 4. Форматуємо та зберігаємо
    dialogue = format_dialogue(combined_segments)
    
    output_path = os.path.join(output_dir, f"{file_base}_speechmatics.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(dialogue)
    print(f"✅ Saved: {output_path}")
    
    # 5. Зберігаємо повний JSON
    json_path = os.path.join(output_dir, f"{file_base}_speechmatics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {json_path}")
    
    # 6. Статистика
    unique_speakers = sorted(set(seg.get('speaker', 'SPEAKER_00') for seg in combined_segments))
    print(f"📊 Detected {len(unique_speakers)} speaker(s): {unique_speakers}")
    print(f"📊 Total segments: {len(combined_segments)}")
    
    return {
        'file': file_name,
        'num_speakers': len(unique_speakers),
        'speakers': unique_speakers,
        'num_segments': len(combined_segments)
    }

def main():
    # Перевіряємо API key
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    if not api_key:
        print("❌ Error: SPEECHMATICS_API_KEY environment variable is not set")
        print("   Please set it in your .env file or export it:")
        print("   export SPEECHMATICS_API_KEY=your_key_here")
        return
    
    # Шляхи
    input_dir = "Audio Examples/detecting main speakers"
    output_dir = "Audio Examples/detecting main speakers"
    
    speaker_files = [
        os.path.join(input_dir, "speaker_0.wav"),
        os.path.join(input_dir, "speaker_1.wav")
    ]
    
    print("=" * 60)
    print("🎵 Transcribing single-speaker files with Speechmatics")
    print("=" * 60)
    
    results = []
    
    for speaker_file in speaker_files:
        if not os.path.exists(speaker_file):
            print(f"⚠️  File not found: {speaker_file}")
            continue
        
        try:
            result = process_file(api_key, speaker_file, output_dir, language='en')
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {speaker_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Підсумок
    print("\n" + "=" * 60)
    print("✅ Processing completed!")
    print("=" * 60)
    print("\n📊 Summary:")
    for result in results:
        print(f"  {result['file']}:")
        print(f"    - Speakers detected: {result['num_speakers']} ({result['speakers']})")
        print(f"    - Segments: {result['num_segments']}")

if __name__ == "__main__":
    main()

