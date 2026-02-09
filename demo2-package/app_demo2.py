#!/usr/bin/env python3
"""
Flask сервер для demo2: SpeechBrain діаризація + Whisper транскрипція
"""

import os
import json
import numpy as np
import torch
import librosa
import soundfile as sf
import requests
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import time
from werkzeug.utils import secure_filename
from cascading_diarization import CascadingDiarizationController, DiarizationSegment
import threading
import uuid
from datetime import datetime

# Патч для torchaudio сумісності з speechbrain (завантажуємо ДО імпорту speechbrain)
exec(open('patch_torchaudio.py').read())

from speechbrain.pretrained import SpeakerRecognition
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
import whisper
import warnings
from pathlib import Path
import tempfile

warnings.filterwarnings("ignore")

app = Flask(__name__)
# Налаштування CORS для iOS Shortcuts (підтримка preflight OPTIONS запитів)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size

# Константи для iOS Shortcuts API
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac'}
PROCESSING_TIMEOUT = 300  # 5 хвилин

# Дозволи для завантажень
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Асинхронна обробка: словник для зберігання статусів завдань
jobs = {}  # {job_id: {'status': 'pending'|'processing'|'completed'|'failed', 'result': {...}, 'error': '...', 'created_at': datetime}}
jobs_lock = threading.Lock()

# Глобальні змінні для моделей (завантажуються один раз)
speaker_model = None
whisper_model = None
separation_model = None

def diagnose_model_structure():
    """Діагностика структури моделі та версій бібліотек"""
    global speaker_model
    if speaker_model is None:
        print("⚠️  Cannot diagnose: speaker_model is None")
        return
    print("\n" + "="*60)
    print("🔍 DIAGNOSTICS: Model Structure and Versions")
    print("="*60)
    # Версії бібліотек
    try:
        import torch
        import torchaudio
        import speechbrain
        print(f"📦 torch version: {torch.__version__}")
        print(f"📦 torchaudio version: {torchaudio.__version__}")
        print(f"📦 speechbrain version: {speechbrain.__version__}")
    except Exception as e:
        print(f"⚠️  Error getting versions: {e}")
    # Тип та структура моделі
    print(f"\n📋 Model type: {type(speaker_model)}")
    print(f"📋 Model class: {speaker_model.__class__.__name__}")
    # Атрибути моделі
    print(f"\n📋 Model attributes (first 20): {[attr for attr in dir(speaker_model) if not attr.startswith('_')][:20]}")
    # Перевірка mods
    if hasattr(speaker_model, 'mods'):
        print(f"✅ Model has 'mods' attribute")
        print(f"📋 Mods type: {type(speaker_model.mods)}")
        print(f"📋 Mods attributes: {[attr for attr in dir(speaker_model.mods) if not attr.startswith('_')][:20]}")
        if hasattr(speaker_model.mods, 'encoder'):
            print(f"✅ Model has 'mods.encoder'")
            print(f"📋 Encoder type: {type(speaker_model.mods.encoder)}")
        else:
            print(f"❌ Model does NOT have 'mods.encoder'")
        if hasattr(speaker_model.mods, 'embedding_model'):
            print(f"✅ Model has 'mods.embedding_model'")
            print(f"📋 Embedding model type: {type(speaker_model.mods.embedding_model)}")
        else:
            print(f"❌ Model does NOT have 'mods.embedding_model'")
    else:
        print(f"❌ Model does NOT have 'mods' attribute")
    # Перевірка методів
    if hasattr(speaker_model, 'encode_batch'):
        print(f"✅ Model has 'encode_batch' method")
        try:
            import inspect
            sig = inspect.signature(speaker_model.encode_batch)
            print(f"📋 encode_batch signature: {sig}")
        except Exception as e:
            print(f"⚠️  Could not get signature: {e}")
    else:
        print(f"❌ Model does NOT have 'encode_batch' method")
    if hasattr(speaker_model, 'encode_file'):
        print(f"✅ Model has 'encode_file' method")
        try:
            import inspect
            sig = inspect.signature(speaker_model.encode_file)
            print(f"📋 encode_file signature: {sig}")
        except Exception as e:
            print(f"⚠️  Could not get signature: {e}")
    else:
        print(f"❌ Model does NOT have 'encode_file' method")
    # Device моделі
    try:
        if hasattr(speaker_model, 'parameters'):
            device = next(speaker_model.parameters()).device
            print(f"📱 Model device: {device}")
        else:
            print(f"⚠️  Cannot determine model device")
    except Exception as e:
        print(f"⚠️  Error getting device: {e}")
    print("="*60 + "\n")


def load_models():
    """Завантажує моделі SpeechBrain та Whisper один раз при старті"""
    global speaker_model, whisper_model, separation_model
    if speaker_model is None:
        print("🔄 Loading SpeechBrain speaker recognition model...")
        try:
            speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            print("✅ SpeechBrain model loaded successfully!")
            # Діагностика після завантаження
            diagnose_model_structure()
        except Exception as e:
            print(f"❌ Error loading SpeechBrain model: {e}")
            raise
    if whisper_model is None:
        print("🔄 Loading Whisper model...")
        try:
            # Використовуємо base модель для балансу між швидкістю та якістю
            whisper_model = whisper.load_model("base")
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            raise
    # Separation model завантажується на вимогу (lazy loading)

# Завантажуємо моделі при старті (з обробкою помилок)
try:
    load_models()
except Exception as e:
    print(f"⚠️  Warning: Could not load models at startup: {e}")
    print("   Models will be loaded on first request")
    import traceback
    traceback.print_exc()


def extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5):
    """
    Витягує ембеддинги спікера для сегментів аудіо.
    Args:
        audio_path: шлях до аудіофайлу
        segment_duration: довжина сегмента в секундах
        overlap: перекриття між сегментами (0-1)
    Returns:
        embeddings: матриця ембедингів (N, 192)
        timestamps: список (start, end) для кожного сегмента
    """
    global speaker_model
    if speaker_model is None:
        load_models()
    try:
        # Завантажуємо аудіо
        print(f"📂 Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"⏱️  Audio duration: {duration:.2f} seconds, sample rate: {sr} Hz, samples: {len(audio)}")
        # Перевірка мінімальної довжини
        min_duration = 0.5  # Мінімум 0.5 секунди
        if duration < min_duration:
            print(f"⚠️  Audio too short ({duration:.2f}s < {min_duration}s), using entire audio as single segment")
            # Використовуємо все аудіо як один сегмент
            embedding = None
            try:
                # Використовуємо формат [1, samples] (працює з поточною версією SpeechBrain)
                segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                embedding = speaker_model.encode_batch(segment_tensor, normalize=False).squeeze().cpu().detach().numpy()
            except Exception as e1:
                try:
                    # Fallback до encode_batch без normalize
                    segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                    embedding = speaker_model.encode_batch(segment_tensor).squeeze().cpu().detach().numpy()
                except Exception as e2:
                    try:
                        # Fallback до encode_file через тимчасовий файл
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            sf.write(tmp_file.name, audio, sr)
                            tmp_path = tmp_file.name
                        if hasattr(speaker_model, 'encode_file'):
                            embedding = speaker_model.encode_file(tmp_path).squeeze().cpu().detach().numpy()
                        else:
                            embedding = None
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    except Exception as e3:
                        print(f"❌ Error processing short audio: Method1={e1}, Method2={e2}, Method3={e3}")
                        return None, []
            if embedding is not None and len(embedding) > 0:
                return np.array([embedding]), [(0.0, duration)]
            else:
                return None, []
        embeddings = []
        timestamps = []
        # Ковзні вікна
        segment_samples = int(segment_duration * sr)
        stride_samples = int(segment_duration * (1 - overlap) * sr)
        print(f"🔍 Processing with segment_duration={segment_duration}s, overlap={overlap}, segment_samples={segment_samples}, stride_samples={stride_samples}")
        # Якщо аудіо коротше за один сегмент, використовуємо все аудіо
        if len(audio) < segment_samples:
            print(f"⚠️  Audio shorter than segment ({len(audio)} < {segment_samples}), using entire audio")
            embedding = None
            try:
                # Використовуємо формат [1, samples] (працює з поточною версією SpeechBrain)
                segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                embedding = speaker_model.encode_batch(segment_tensor, normalize=False).squeeze().cpu().detach().numpy()
            except Exception as e1:
                try:
                    # Fallback до encode_batch без normalize
                    segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                    embedding = speaker_model.encode_batch(segment_tensor).squeeze().cpu().detach().numpy()
                except Exception as e2:
                    try:
                        # Fallback до encode_file через тимчасовий файл
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            sf.write(tmp_file.name, audio, sr)
                            tmp_path = tmp_file.name
                        if hasattr(speaker_model, 'encode_file'):
                            embedding = speaker_model.encode_file(tmp_path).squeeze().cpu().detach().numpy()
                        else:
                            embedding = None
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    except Exception as e3:
                        print(f"❌ Error processing short audio segment: Method1={e1}, Method2={e2}, Method3={e3}")
                        return None, []
            if embedding is not None and len(embedding) > 0:
                return np.array([embedding]), [(0.0, duration)]
            else:
                return None, []
        # Обробка сегментів
        max_start = len(audio) - segment_samples
        if max_start < 0:
            max_start = 0
        segments_processed = 0
        for start_sample in range(0, max_start + 1, stride_samples):
            end_sample = min(start_sample + segment_samples, len(audio))
            segment = audio[start_sample:end_sample]
            # Перевірка, що сегмент не порожній
            if len(segment) == 0:
                continue
            # Витягуємо ембеддинг через SpeechBrain
            # Використовуємо кілька fallback методів для сумісності з різними версіями
            embedding = None
            # Отримуємо device моделі
            try:
                model_device = next(speaker_model.parameters()).device
            except:
                model_device = torch.device('cpu')
            try:
                # Метод 1: encode_batch з форматом [1, samples] (працює з поточною версією SpeechBrain)
                segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(model_device)  # [1, samples]
                if start_sample == 0 or segments_processed == 0:
                    print(f"🔍 Method 1: tensor shape={segment_tensor.shape}, dtype={segment_tensor.dtype}, device={segment_tensor.device}")
                embedding = speaker_model.encode_batch(segment_tensor, normalize=False)
                embedding = embedding.squeeze().cpu().detach().numpy()
                if embedding is not None and len(embedding) > 0:
                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                        print(f"⚠️  Method 1: NaN or Inf found in embedding, trying next method...")
                        embedding = None
                    else:
                        if start_sample == 0 or segments_processed == 0:
                            print(f"✅ Method 1 succeeded: embedding shape={embedding.shape}, dtype={embedding.dtype}")
            except Exception as e1:
                if start_sample == 0 or segments_processed == 0:
                    print(f"⚠️  Method 1 (encode_batch [1,samples] normalize=False) failed: {e1}")
                try:
                    # Метод 2: encode_batch без normalize
                    segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(model_device)  # [1, samples]
                    if start_sample == 0 or segments_processed == 0:
                        print(f"🔍 Method 2: tensor shape={segment_tensor.shape}, dtype={segment_tensor.dtype}, device={segment_tensor.device}")
                    embedding = speaker_model.encode_batch(segment_tensor)
                    embedding = embedding.squeeze().cpu().detach().numpy()
                    if embedding is not None and len(embedding) > 0:
                        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                            print(f"⚠️  Method 2: NaN or Inf found in embedding, trying next method...")
                            embedding = None
                        else:
                            if start_sample == 0 or segments_processed == 0:
                                print(f"✅ Method 2 succeeded: embedding shape={embedding.shape}, dtype={embedding.dtype}")
                except Exception as e2:
                    if start_sample == 0 or segments_processed == 0:
                        print(f"⚠️  Method 2 (encode_batch [1,samples] default) failed: {e2}")
                    # Метод 3: encode_file через тимчасовий файл (якщо доступний)
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            sf.write(tmp_file.name, segment, sr)
                            tmp_path = tmp_file.name
                        if hasattr(speaker_model, 'encode_file'):
                            if start_sample == 0 or segments_processed == 0:
                                print(f"🔍 Method 3: Using encode_file with temporary file")
                            embedding = speaker_model.encode_file(tmp_path)
                            embedding = embedding.squeeze().cpu().detach().numpy()
                            if embedding is not None and len(embedding) > 0:
                                if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                    print(f"⚠️  Method 3: NaN or Inf found in embedding")
                                    embedding = None
                                else:
                                    if start_sample == 0 or segments_processed == 0:
                                        print(f"✅ Method 3 succeeded: embedding shape={embedding.shape}, dtype={embedding.dtype}")
                            # Видаляємо тимчасовий файл
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                        else:
                            embedding = None
                    except Exception as e3:
                        if start_sample == 0 or segments_processed == 0:
                            print(f"⚠️  Method 3 (encode_file) failed: {e3}")
                        try:
                            if 'tmp_path' in locals():
                                os.unlink(tmp_path)
                        except:
                            pass
                        print(f"❌ All methods failed for segment at {start_sample}: Method1={type(e1).__name__}:{str(e1)[:100]}, Method2={type(e2).__name__}:{str(e2)[:100]}, Method3={type(e3).__name__}:{str(e3)[:100]}")
                        import traceback
                        traceback.print_exc()
                        continue
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                start_time = start_sample / sr
                end_time = end_sample / sr
                timestamps.append((start_time, min(end_time, duration)))
                segments_processed += 1
            else:
                print(f"⚠️  No embedding extracted for segment at {start_sample}")
                continue
        print(f"✅ Processed {segments_processed} segments, extracted {len(embeddings)} embeddings")
        if len(embeddings) == 0:
            print("❌ No embeddings extracted!")
            return None, []
        return np.array(embeddings), timestamps
    except Exception as e:
        print(f"❌ Error in extract_speaker_embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def diarize_audio(embeddings, timestamps, num_speakers=None):
    """
    Виконує діаризацію через spectral clustering на ембедингах.
    Args:
        embeddings: матриця ембедингів (N, 192)
        timestamps: список (start, end) для кожного сегмента
        num_speakers: кількість спікерів (якщо None, автоматично визначається)
    Returns:
        segments: список {'speaker': int, 'start': float, 'end': float}
    """
    if embeddings is None:
        print("❌ No embeddings provided for diarization")
        return []
    if len(embeddings) < 2:
        print(f"⚠️  Only {len(embeddings)} embedding(s) available, need at least 2 for clustering")
        # Якщо тільки один сегмент, повертаємо його як одного спікера
        if len(embeddings) == 1 and timestamps:
            return [{
                'speaker': 0,
                'start': round(timestamps[0][0], 2),
                'end': round(timestamps[0][1], 2)
            }]
        return []
    try:
        # Нормалізуємо ембеддинги (L2 нормалізація)
        from sklearn.preprocessing import normalize
        embeddings_normalized = normalize(embeddings, norm='l2')
        # Обчислюємо косинусну відстань між ембедингами
        distances = pdist(embeddings_normalized, metric='cosine')
        distance_matrix = squareform(distances)
        # Діагностика: перевіряємо розподіл відстаней
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        print(f"📊 Distance stats: mean={mean_dist:.4f}, std={std_dist:.4f}, min={np.min(distances):.4f}, max={np.max(distances):.4f}")
        # Створюємо similarity matrix для кластеризації
        if std_dist < 1e-6:
            print(f"⚠️  All distances are nearly identical, using uniform similarity")
            similarity_matrix = np.ones_like(distance_matrix) * 0.5
        else:
            # Використовуємо адаптивне масштабування
            scale = mean_dist if mean_dist > 0.01 else 0.1
            similarity_matrix = np.exp(-distance_matrix / scale)
        # Визначаємо кількість спікерів автоматично, якщо не задано
        if num_speakers is None:
            # Покращений алгоритм визначення кількості спікерів
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            best_k = 2
            best_score = -1
            # Перевіряємо k від 2 до min(5, кількість_сегментів/3)
            max_k = min(5, max(2, len(embeddings) // 3))
            scores = []
            for k in range(2, max_k + 1):
                try:
                    test_clustering = SpectralClustering(
                        n_clusters=k,
                        affinity='precomputed',
                        random_state=42,
                        assign_labels='kmeans',
                        n_init=10  # Більше спроб для стабільності
                    )
                    test_labels = test_clustering.fit_predict(similarity_matrix)
                    # Обчислюємо silhouette score (потребує принаймні 2 кластери)
                    if len(np.unique(test_labels)) > 1:
                        sil_score = silhouette_score(embeddings_normalized, test_labels, metric='cosine')
                        db_score = davies_bouldin_score(embeddings_normalized, test_labels)
                        # Комбінований score (silhouette вищий = краще, DB нижчий = краще)
                        combined_score = sil_score - (db_score / 10)  # Нормалізуємо DB score
                        print(f"   k={k}: silhouette={sil_score:.4f}, DB={db_score:.4f}, combined={combined_score:.4f}")
                        scores.append((k, combined_score, sil_score, db_score))
                        if combined_score > best_score:
                            best_score = combined_score
                            best_k = k
                except Exception as e:
                    print(f"   k={k}: error - {e}")
                    continue
            # Додаткова перевірка: якщо різниця між найкращим та другим невелика, вибираємо менше кластерів
            if len(scores) > 1:
                scores.sort(key=lambda x: x[1], reverse=True)
                best_score_val = scores[0][1]
                second_score_val = scores[1][1] if len(scores) > 1 else -1
                # Якщо різниця менше 0.1, вибираємо менше кластерів (простіше = краще)
                if best_score_val - second_score_val < 0.1:
                    best_k = min(scores[0][0], scores[1][0] if len(scores) > 1 else scores[0][0])
                    print(f"   ⚖️  Scores too close, choosing fewer clusters: k={best_k}")
            num_speakers = best_k
            print(f"🔍 Auto-detected {num_speakers} speakers (best combined_score={best_score:.4f})")
            # Якщо всі дуже схожі, примусово встановлюємо мінімум 2
            if mean_dist < 0.05:
                num_speakers = 2
                print(f"⚠️  Very low distance ({mean_dist:.4f}), forcing 2 speakers")
        # Перевіряємо, чи достатньо сегментів для кластеризації
        if len(embeddings) < num_speakers:
            print(f"⚠️  Not enough segments ({len(embeddings)}) for {num_speakers} speakers, using {len(embeddings)}")
            num_speakers = len(embeddings)
        # Спробуємо різні алгоритми кластеризації з покращеними параметрами
        labels = None
        # Метод 1: Spectral clustering з покращеними параметрами
        try:
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans',
                n_init=20,  # Більше спроб для кращої стабільності
                n_jobs=-1  # Використовуємо всі ядра
            )
            labels = clustering.fit_predict(similarity_matrix)
            # Перевіряємо якість кластеризації
            unique_labels = np.unique(labels)
            if len(unique_labels) < num_speakers:
                print(f"⚠️  Spectral clustering produced only {len(unique_labels)} clusters, expected {num_speakers}")
                # Якщо отримали менше кластерів, пробуємо інший метод
                raise ValueError(f"Only {len(unique_labels)} clusters found")
            # Перевіряємо баланс кластерів
            label_counts = {label: np.sum(labels == label) for label in unique_labels}
            min_count = min(label_counts.values())
            max_count = max(label_counts.values())
            # Якщо один кластер занадто малий (< 5% сегментів), це підозріло
            if min_count < len(embeddings) * 0.05:
                print(f"⚠️  Unbalanced clusters detected (min={min_count}, max={max_count}), trying alternative method")
                # Не викидаємо помилку, але логуємо попередження
            print(f"✅ Used SpectralClustering: {len(unique_labels)} clusters, balance: {min_count}-{max_count} segments per cluster")
        except Exception as e:
            print(f"⚠️  Spectral clustering failed: {e}, trying AgglomerativeClustering")
            # Fallback до AgglomerativeClustering з покращеними параметрами
            try:
                from sklearn.cluster import AgglomerativeClustering
                # Пробуємо різні типи зв'язку
                for linkage_type in ['ward', 'average', 'complete']:
                    try:
                        clustering = AgglomerativeClustering(
                            n_clusters=num_speakers,
                            linkage=linkage_type,
                            affinity='precomputed' if linkage_type != 'ward' else 'euclidean'
                        )
                        if linkage_type == 'ward':
                            # Для ward потрібна евклідова відстань, використовуємо оригінальні ембеддинги
                            labels = clustering.fit_predict(embeddings_normalized)
                        else:
                            labels = clustering.fit_predict(similarity_matrix)
                        unique_labels = np.unique(labels)
                        if len(unique_labels) == num_speakers:
                            print(f"✅ Used AgglomerativeClustering (linkage={linkage_type})")
                            break
                    except Exception as e2:
                        print(f"   linkage={linkage_type} failed: {e2}")
                        continue
                if labels is None:
                    raise Exception("All AgglomerativeClustering methods failed")
            except Exception as e2:
                print(f"❌ AgglomerativeClustering also failed: {e2}")
                # Останній fallback: простий k-means на нормалізованих ембеддингах
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=20)
                labels = kmeans.fit_predict(embeddings_normalized)
                print(f"✅ Used KMeans as fallback")
        if labels is None:
            print("❌ Clustering failed completely")
            return []
        # Діагностика: перевіряємо розподіл лейблів
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"📊 Clustering result: {len(unique_labels)} unique speakers found")
        for label, count in zip(unique_labels, counts):
            print(f"   Speaker {label}: {count} segments ({count/len(labels)*100:.1f}%)")
        # Якщо всі сегменти одного спікера, спробуємо інший підхід
        if len(unique_labels) == 1 and num_speakers > 1:
            print(f"⚠️  All segments assigned to one speaker, trying alternative clustering...")
            # Спробуємо використати відстані безпосередньо
            from sklearn.cluster import AgglomerativeClustering
            clustering_alt = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward',
                metric='euclidean'
            )
            labels_alt = clustering_alt.fit_predict(embeddings_normalized)
            unique_alt, counts_alt = np.unique(labels_alt, return_counts=True)
            if len(unique_alt) > 1:
                print(f"✅ Alternative clustering found {len(unique_alt)} speakers")
                labels = labels_alt
                unique_labels, counts = unique_alt, counts_alt
        # Покращена нормалізація лейблів: завжди починаємо з 0
        # Переіменовуємо лейбли так, щоб найбільш ранній сегмент мав лейбл 0
        if len(unique_labels) > 0:
            # Знаходимо, який лейбл має найранніший сегмент
            first_segment_label = labels[0]
            # Якщо перший сегмент не має лейбл 0, міняємо місцями
            if first_segment_label != 0:
                print(f"⚠️  First segment has label {first_segment_label}, normalizing to start with label 0...")
                # Створюємо мапу: first_segment_label → 0, 0 → first_segment_label
                label_map = {}
                for old_label in unique_labels:
                    if old_label == first_segment_label:
                        label_map[old_label] = 0
                    elif old_label == 0:
                        label_map[old_label] = first_segment_label
                    else:
                        label_map[old_label] = old_label
                # Застосовуємо мапу
                labels = np.array([label_map[label] for label in labels])
                unique_labels = np.array([label_map[label] for label in unique_labels])
                print(f"✅ Normalized labels: first segment now has label 0")
        # Зливаємо сусідні сегменти одного спікера
        segments = []
        current_speaker = None
        current_start = None
        for i, (label, (start, end)) in enumerate(zip(labels, timestamps)):
            if label != current_speaker:
                if current_speaker is not None:
                    segments.append({
                        'speaker': int(current_speaker),
                        'start': round(current_start, 2),
                        'end': round(timestamps[i-1][1], 2)
                    })
                current_speaker = label
                current_start = start
        # Додаємо останній сегмент
        if current_speaker is not None:
            segments.append({
                'speaker': int(current_speaker),
                'start': round(current_start, 2),
                'end': round(timestamps[-1][1], 2)
            })
        print(f"✅ Created {len(segments)} diarization segments")
        return segments
    except Exception as e:
        print(f"❌ Error in diarize_audio: {e}")
        import traceback
        traceback.print_exc()
        return []


def transcribe_audio_groq(audio_path, language=None, model="whisper-large-v3-turbo"):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    print(f"🧠 [Groq] Transcriber model: {model}")
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    data = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
        "timestamp_granularities": '["segment","word"]'
    }
    if language:
        data["language"] = language
    with open(audio_path, "rb") as audio_file:
        files = {
            "file": (os.path.basename(audio_path), audio_file, "application/octet-stream")
        }
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files=files,
            timeout=180
        )
    if response.status_code >= 400:
        raise ValueError(f"Groq API error {response.status_code}: {response.text}")
    payload = response.json()
    transcription = payload.get("text", "")
    segments = []
    words = []
    payload_segments = payload.get("segments") or []
    for seg in payload_segments:
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text", "")
        if start is None or end is None:
            continue
        segments.append({
            "start": round(float(start), 2),
            "end": round(float(end), 2),
            "text": text.strip()
        })
    payload_words = payload.get("words")
    if isinstance(payload_words, list):
        for word_info in payload_words:
            word_text = (word_info.get("word") or word_info.get("text") or "").strip()
            if not word_text:
                continue
            words.append({
                "word": word_text,
                "start": round(float(word_info.get("start", 0) or 0), 2),
                "end": round(float(word_info.get("end", 0) or 0), 2)
            })
    elif payload_segments:
        for seg in payload_segments:
            seg_words = seg.get("words")
            if not isinstance(seg_words, list):
                continue
            for word_info in seg_words:
                word_text = (word_info.get("word") or word_info.get("text") or "").strip()
                if not word_text:
                    continue
                words.append({
                    "word": word_text,
                    "start": round(float(word_info.get("start", 0) or 0), 2),
                    "end": round(float(word_info.get("end", 0) or 0), 2)
                })
    if not words and segments:
        for seg in segments:
            seg_text = seg.get("text", "")
            seg_words = [word for word in seg_text.split() if word]
            if not seg_words:
                continue
            seg_start = float(seg.get("start", 0) or 0)
            seg_end = float(seg.get("end", seg_start) or seg_start)
            duration = max(seg_end - seg_start, 0)
            step = duration / max(len(seg_words), 1)
            for idx, word_text in enumerate(seg_words):
                start = seg_start + (idx * step)
                end = seg_start + ((idx + 1) * step) if step else seg_end
                words.append({
                    "word": word_text,
                    "start": round(start, 2),
                    "end": round(end, 2)
                })
    return transcription, segments, words


def transcribe_audio_speechmatics(audio_path, language=None):
    """
    Transcribe + diarize with Speechmatics (word-level speakers).
    Returns:
        transcription: full text
        segments: list of {speaker, start, end, text}
        words: list of {word, start, end, speaker}
    """
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    if not api_key:
        raise ValueError("SPEECHMATICS_API_KEY is not set")

    base_url = 'https://asr.api.speechmatics.com/v2'
    headers = {'Authorization': f'Bearer {api_key}'}
    file_size = os.path.getsize(audio_path)
    file_size_mb = file_size / (1024 * 1024)
    upload_timeout = max(300, int(60 + (file_size_mb / 10) * 60))

    file_ext = os.path.splitext(audio_path)[1].lower()
    mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.mp4': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.webm': 'audio/webm',
        '.aac': 'audio/aac'
    }
    mime_type = mime_types.get(file_ext, 'audio/wav')

    config = {
        'type': 'transcription',
        'transcription_config': {
            'language': language or 'en',
            'diarization': 'speaker',
            'operating_point': 'enhanced',
            'speaker_diarization_config': {
                'get_speakers': True
            }
        }
    }

    with open(audio_path, 'rb') as audio_file:
        files = {'data_file': (os.path.basename(audio_path), audio_file, mime_type)}
        data = {'config': json.dumps(config)}
        response = requests.post(
            f'{base_url}/jobs',
            files=files,
            data=data,
            headers=headers,
            timeout=upload_timeout
        )
    if response.status_code >= 400:
        raise ValueError(f"Speechmatics API error {response.status_code}: {response.text}")
    job_id = response.json().get('id') or response.json().get('job_id')
    if not job_id:
        raise ValueError("Speechmatics job_id missing")

    status = 'running'
    for _ in range(120):
        status_response = requests.get(f'{base_url}/jobs/{job_id}', headers=headers, timeout=30)
        status_response.raise_for_status()
        job = status_response.json().get('job', {})
        status = job.get('status', 'unknown')
        if status == 'done':
            break
        if status == 'rejected':
            raise ValueError(f"Speechmatics job rejected: {job.get('failure_reason', 'unknown')}")
        time.sleep(3)
    if status != 'done':
        raise ValueError("Speechmatics job did not complete in time")

    transcript_response = requests.get(
        f'{base_url}/jobs/{job_id}/transcript',
        headers={**headers, 'Accept': 'application/json'},
        timeout=30
    )
    transcript_response.raise_for_status()
    transcript_data = transcript_response.json()

    words = []
    results = transcript_data.get('results') or []
    last_word_index = None
    for item in results:
        item_type = item.get('type')
        if item_type == 'punctuation':
            punct = (item.get('alternatives') or [{}])[0].get('content') or ''
            if punct and last_word_index is not None:
                words[last_word_index]['word'] = f"{words[last_word_index]['word']}{punct}"
            continue
        if item_type != 'word':
            continue
        alternatives = item.get('alternatives') or []
        if not alternatives:
            continue
        alt = alternatives[0]
        speaker_label = alt.get('speaker', 'S1')
        if isinstance(speaker_label, str) and speaker_label.startswith('S'):
            number = speaker_label[1:]
            speaker_num = int(number) - 1 if number.isdigit() else 0
        else:
            try:
                speaker_num = int(speaker_label)
            except (TypeError, ValueError):
                speaker_num = 0
        word_text = (alt.get('content') or '').strip()
        if not word_text:
            continue
        words.append({
            'word': word_text,
            'start': round(float(item.get('start_time', 0) or 0), 2),
            'end': round(float(item.get('end_time', item.get('start_time', 0)) or 0), 2),
            'speaker': speaker_num
        })
        last_word_index = len(words) - 1

    if not words:
        return '', [], []

    segments = []
    current_speaker = None
    current_start = None
    current_words = []
    last_end = None
    for word in words:
        speaker = word['speaker']
        if current_speaker is None:
            current_speaker = speaker
            current_start = word['start']
            current_words = [word['word']]
            last_end = word['end']
            continue
        if speaker != current_speaker:
            segments.append({
                'speaker': current_speaker,
                'start': round(current_start, 2),
                'end': round(last_end, 2),
                'text': ' '.join(current_words).strip()
            })
            current_speaker = speaker
            current_start = word['start']
            current_words = [word['word']]
        else:
            current_words.append(word['word'])
        last_end = word['end']
    if current_words:
        segments.append({
            'speaker': current_speaker if current_speaker is not None else 0,
            'start': round(current_start, 2) if current_start is not None else 0,
            'end': round(last_end, 2) if last_end is not None else 0,
            'text': ' '.join(current_words).strip()
        })

    transcription = ' '.join([w['word'] for w in words]).strip()
    return transcription, segments, words


def transcribe_audio(audio_path, language=None, transcriber=None):
    """
    Транскрибує аудіо за допомогою Whisper з word timestamps.
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk', 'en', 'ar') або None для авто-визначення
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    try:
        selected_transcriber = (transcriber or "local").strip().lower()
        if selected_transcriber in ("speechmatics",):
            print(f"🎤 Transcribing audio with Speechmatics: {audio_path}")
            return transcribe_audio_speechmatics(audio_path, language=language)
        if selected_transcriber in ("groq", "groq_whisper", "groq-whisper", "whisper-large-v3-turbo"):
            print(f"🎤 Transcribing audio with Groq: {audio_path}")
            return transcribe_audio_groq(audio_path, language=language)
        global whisper_model
        if whisper_model is None:
            load_models()
        try:
            model_name = whisper_model.model_name if hasattr(whisper_model, "model_name") else "local"
        except Exception:
            model_name = "local"
        print(f"🧠 [Whisper] Transcriber model: {model_name}")
        print(f"🎤 Transcribing audio (local Whisper): {audio_path}")
        # Налаштування для транскрипції
        transcribe_options = {
            'word_timestamps': True,
            'verbose': False,
            'task': 'transcribe'  # Завжди транскрибуємо, не перекладаємо
        }
        if language:
            transcribe_options['language'] = language
            print(f"🌐 Using specified language: {language}")
        else:
            # Автоматичне визначення мови - Whisper зробить це автоматично
            print(f"🌐 Auto-detecting language (Whisper will detect automatically)")
        # Транскрибуємо з детальними сегментами та word timestamps
        result = whisper_model.transcribe(
            audio_path,
            **transcribe_options
        )
        detected_lang = result.get('language', 'unknown')
        print(f"🌐 Detected language: {detected_lang}")
        transcription = result["text"]
        segments = []
        words = []
        # Формуємо сегменти з текстом та збираємо всі слова
        for seg in result["segments"]:
            segments.append({
                'start': round(seg['start'], 2),
                'end': round(seg['end'], 2),
                'text': seg['text'].strip()
            })
            # Збираємо слова з timestamps
            if 'words' in seg:
                for word_info in seg['words']:
                    words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': round(word_info.get('start', 0), 2),
                        'end': round(word_info.get('end', 0), 2)
                    })
        print(f"✅ Transcribed {len(segments)} segments, language: {detected_lang}")
        return transcription, segments, words
    except Exception as e:
        print(f"❌ Error in transcribe_audio: {e}")
        import traceback
        traceback.print_exc()
        return "", [], []


def separate_speakers(audio_path, output_dir=None):
    """
    Розділяє аудіо на окремі треки для кожного спікера за допомогою SpeechBrain.
    Args:
        audio_path: шлях до аудіофайлу
        output_dir: директорія для збереження розділених треків (якщо None, створюється тимчасова)
    Returns:
        dict з ключами:
            - success: bool
            - speakers: список словників з інформацією про кожен трек
            - output_dir: шлях до директорії з треками
            - num_speakers: кількість спікерів
    """
    global separation_model
    if separation_model is None:
        print("🔄 Loading SpeechBrain separation model...")
        try:
            from speechbrain.pretrained import SepformerSeparation as Separator
            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            separation_model = Separator.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix",
                run_opts={"device": device}
            )
            print(f"✅ SpeechBrain separation model loaded successfully on {device}!")
        except Exception as e:
            print(f"❌ Error loading SpeechBrain separation model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    try:
        # Створюємо output директорію, якщо не вказана
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="speechbrain_separation_")
        os.makedirs(output_dir, exist_ok=True)
        # Завантажуємо аудіо (librosa завжди повертає mono якщо не вказано інакше)
        waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
        # Конвертуємо в torch tensor [1, samples]
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        # Ресемплінг до 8kHz (вимога моделі)
        if sample_rate != 8000:
            print(f"🔄 Resampling from {sample_rate}Hz to 8000Hz...")
            import torchaudio
            resampler = torchaudio.transforms.Resample(sample_rate, 8000)
            waveform_tensor = resampler(waveform_tensor)
            sample_rate = 8000
        # Розділяємо спікерів
        print("🔀 Separating speakers...")
        device = next(separation_model.parameters()).device
        waveform_tensor = waveform_tensor.to(device)
        with torch.no_grad():
            est_sources = separation_model.separate_batch(waveform_tensor)
        # Обробка результату
        if est_sources.dim() == 3:
            est_sources = est_sources[0]  # [batch, num_speakers, time] -> [num_speakers, time]
        if est_sources.dim() == 2:
            if est_sources.shape[0] == separation_model.hparams.num_spks:
                # [num_speakers, time]
                sources_tensor = est_sources
            elif est_sources.shape[1] == separation_model.hparams.num_spks:
                # [time, num_speakers] -> транспонуємо
                sources_tensor = est_sources.transpose(0, 1)
            else:
                raise ValueError(f"Unexpected est_sources shape: {est_sources.shape}")
        else:
            raise ValueError(f"Unsupported est_sources dimension: {est_sources.dim()}")
        # Зберігаємо кожен трек
        speakers = []
        for idx, source in enumerate(sources_tensor):
            speaker_name = f"SPEAKER_{idx:02d}"
            output_path = os.path.join(output_dir, f"{speaker_name}.wav")
            # Конвертуємо назад до оригінальної sample rate, якщо потрібно
            source_np = source.cpu().squeeze().numpy()
            # Ресемплінг назад до 16kHz для подальшої обробки
            if sample_rate != 16000:
                import torchaudio
                source_tensor = torch.tensor(source_np, dtype=torch.float32).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                source_tensor = resampler(source_tensor)
                source_np = source_tensor.squeeze().numpy()
                final_sr = 16000
            else:
                final_sr = sample_rate
            sf.write(output_path, source_np, final_sr)
            speakers.append({
                "name": speaker_name,
                "path": output_path,
                "index": idx
            })
        print(f"✅ Separated into {len(speakers)} speaker tracks")
        return {
            "success": True,
            "speakers": speakers,
            "output_dir": output_dir,
            "num_speakers": len(speakers)
        }
    except Exception as e:
        print(f"❌ Error in separate_speakers: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


# Глобальна змінна для зберігання інформації про LLM ітерації
_llm_iterations_cache = []


def detect_and_fix_speaker_mismatch_after_complete_statement(segments, max_gap=2.0):
    """
    Виявляє та виправляє помилки призначення спікерів, коли завершена думка (statement)
    з одного спікера закінчується, а наступне питання також приписується тому ж спікеру.
    
    Шаблон виявлення:
    - Перша репліка закінчується на знак завершення речення (., !)
    - Друга репліка закінчується на знак питання (?)
    - Обидві репліки від одного спікера (помилка діаризації)
    - Мала пауза між ними (< max_gap)
    
    Такі випадки позначаються для перевірки спікера через LLM.
    
    Args:
        segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
        max_gap: максимальний проміжок між сегментами для перевірки (секунди)
    
    Returns:
        fixed_segments: список сегментів з виправленими або позначеними для перевірки сегментами
    """
    if not segments or len(segments) < 2:
        return segments
    
    print(f"🔍 Detecting speaker mismatch after complete statements in {len(segments)} segments...")
    fixed = []
    i = 0
    mismatch_count = 0
    
    while i < len(segments):
        if i == len(segments) - 1:
            # Останній сегмент - просто додаємо
            fixed.append(segments[i])
            i += 1
            continue
        
        current = segments[i]
        next_seg = segments[i + 1]
        
        # Перевіряємо умови для помилки призначення спікера
        current_text = current['text'].strip()
        next_text = next_seg['text'].strip()
        
        # Умова 1: Перша закінчується на знак завершення речення (statement)
        ends_with_statement = current_text.endswith(('.', '!'))
        
        # Умова 2: Друга закінчується на знак питання
        ends_with_question = next_text.endswith('?')
        
        # Умова 3: Один і той же спікер (помилка діаризації)
        same_speaker = current['speaker'] == next_seg['speaker']
        
        # Умова 4: Мала пауза між сегментами
        pause = next_seg['start'] - current['end']
        short_pause = pause >= 0 and pause < max_gap
        
        # Додаткова перевірка: питання виглядає як окреме питання (починається з великої літери або питального слова)
        looks_like_separate_question = (
            len(next_text) > 0 and (
                next_text[0].isupper() or
                next_text.lower().startswith(('did you', 'can you', 'will you', 'have you', 'are you', 
                                             'is it', 'do you', 'would you', 'could you', 'should you',
                                             'what', 'when', 'where', 'who', 'which', 'how', 'why'))
            )
        )
        
        if (ends_with_statement and 
            ends_with_question and 
            same_speaker and 
            short_pause and
            looks_like_separate_question):
            
            # Це помилка призначення спікера - питання має бути від іншого спікера
            mismatch_count += 1
            print(f"  ⚠️  Detected speaker mismatch after complete statement:")
            print(f"     Statement: Speaker {current['speaker']+1} [{current['start']:.2f}s-{current['end']:.2f}s]: \"{current_text}\"")
            print(f"     Question: Speaker {next_seg['speaker']+1} [{next_seg['start']:.2f}s-{next_seg['end']:.2f}s]: \"{next_text}\"")
            
            # Позначаємо наступний сегмент (питання) для перевірки спікера
            # Питання має бути від іншого спікера (1 - current['speaker'])
            other_speaker = 1 - current['speaker']  # Альтернативний спікер (0 ↔ 1)
            
            # Створюємо копію наступного сегмента з позначкою для перевірки
            next_seg_fixed = next_seg.copy()
            next_seg_fixed['speaker'] = other_speaker  # Тимчасово призначаємо іншого спікера
            next_seg_fixed['needs_role_verification'] = True  # Позначаємо для перевірки ролі
            next_seg_fixed['reassignment_reason'] = 'question_after_complete_statement'
            next_seg_fixed['original_speaker'] = next_seg['speaker']  # Зберігаємо оригінального спікера
            
            fixed.append(current)
            fixed.append(next_seg_fixed)
            print(f"     ✅ Marked question for verification: reassigned to Speaker {other_speaker+1} (was {next_seg['speaker']+1})")
            i += 2  # Пропускаємо обидва сегменти
            continue
        
        # Якщо не виявлено помилку - просто додаємо поточний сегмент
        fixed.append(current)
        i += 1
    
    if mismatch_count > 0:
        print(f"✅ Detected and marked {mismatch_count} speaker mismatch(es) after complete statements")
    else:
        print(f"✅ No speaker mismatches after complete statements detected")
    
    return fixed


def fix_answer_after_question_speaker_assignment_v2(segments, max_gap=3.0):
    """
    Покращена версія, яка обробляє випадок:
    "Hey, did you try to reset your modem? Yes," (Спікер 2)
    "I tried, but the problem is still existing..." (Спікер 1)
    
    Правило: "Yes," має належати Спікеру 1 (тому, хто відповідає).
    """
    if not segments or len(segments) < 2:
        return segments
    
    print(f"🔍 Fixing answer-after-question speaker assignments (v2) in {len(segments)} segments...")
    fixed = []
    i = 0
    fixed_count = 0
    
    while i < len(segments):
        if i == len(segments) - 1:
            fixed.append(segments[i])
            i += 1
            continue
        
        current = segments[i]
        next_seg = segments[i + 1]
        
        current_text = (current.get('text') or '').strip()
        next_text = (next_seg.get('text') or '').strip()
        
        if not current_text or not next_text:
            fixed.append(current)
            i += 1
            continue
        
        # Перевіряємо, чи поточна репліка містить питання + коротку відповідь
        # Наприклад: "Hey, did you try to reset your modem? Yes,"
        if '?' in current_text:
            # Знаходимо останній знак питання (на випадок, якщо їх кілька)
            last_question_mark = current_text.rfind('?')
            if last_question_mark >= 0:
                question_part = current_text[:last_question_mark + 1].strip()
                answer_part = current_text[last_question_mark + 1:].strip()
                
                # Перевіряємо, чи після питання є коротка відповідь
                short_answers = ['yes', 'no', 'sure', 'okay', 'ok', 'alright', 'yeah', 'yep', 'of course', 'certainly']
                answer_lower = answer_part.lower().rstrip(',.!?;:').strip()
                
                # Діагностика
                if answer_lower:
                    print(f"  🔍 Checking segment {i+1}: \"{current_text[:60]}...\" (Speaker {current.get('speaker')+1})")
                    print(f"     Question part: \"{question_part}\"")
                    print(f"     Answer part: \"{answer_part}\" (lower: \"{answer_lower}\")")
                
                is_short_answer = any(answer_lower.startswith(short) for short in short_answers)
                
                if is_short_answer:
                    print(f"     ✅ Detected short answer: \"{answer_lower}\"")
                    
                    # Перевіряємо, чи наступна репліка є продовженням відповіді
                    answer_continuations = [
                        'i tried', 'i did', 'i have', 'i can', 'i will', 
                        'i think', 'i believe', 'i guess', 'i know', 'i see',
                        'i understand', 'i need', 'i want', 'i\'m', 'i am',
                        'i was', 'i would', 'i could', 'i should', 'i might'
                    ]
                    next_lower = next_text.lower().strip()
                    next_is_continuation = any(next_lower.startswith(cont) for cont in answer_continuations)
                    
                    print(f"     Next segment: \"{next_text[:60]}...\" (Speaker {next_seg.get('speaker')+1})")
                    print(f"     Is continuation: {next_is_continuation}")
                    
                    # Перевіряємо паузу
                    pause = next_seg.get('start', 0) - current.get('end', 0)
                    short_pause = 0 <= pause <= max_gap
                    print(f"     Pause: {pause:.2f}s (max: {max_gap}s, short: {short_pause})")
                    
                    # Якщо наступна репліка є продовженням відповіді
                    if next_is_continuation and short_pause:
                        # Знаходимо іншого спікера (того, хто відповідає)
                        current_speaker = current.get('speaker')
                        answer_speaker = next_seg.get('speaker')
                        
                        fixed_count += 1
                        print(f"  🔧 Moving answer fragment to answer speaker:")
                        print(f"     Question: Speaker {current_speaker+1} [{current.get('start'):.2f}s-{current.get('end'):.2f}s]: \"{question_part}\"")
                        print(f"     Answer fragment: \"{answer_part}\" (currently with Speaker {current_speaker+1})")
                        print(f"     Answer continuation: Speaker {answer_speaker+1} [{next_seg.get('start'):.2f}s-{next_seg.get('end'):.2f}s]: \"{next_text}\"")
                        
                        # Розділяємо: питання залишається з поточним спікером
                        # Оцінюємо час для питання (приблизно 80-85% від загального часу)
                        question_duration = current.get('end') - current.get('start')
                        question_end_time = current.get('start') + question_duration * 0.85
                        
                        question_seg = {
                            'speaker': current_speaker,
                            'start': current.get('start'),
                            'end': question_end_time,
                            'text': question_part,
                            'question_answer_split': True
                        }
                        
                        # Відповідь (коротка частина + продовження) переходить до спікера, який відповідає
                        combined_answer = (answer_part + ' ' + next_text).strip()
                        answer_seg = {
                            'speaker': answer_speaker,  # Спікер, який відповідає
                            'start': question_end_time,
                            'end': next_seg.get('end'),
                            'text': combined_answer,
                            'question_answer_split': True,
                            'original_speakers': [current_speaker, answer_speaker]
                        }
                        
                        fixed.append(question_seg)
                        fixed.append(answer_seg)
                        print(f"     ✅ Fixed: Question → Speaker {current_speaker+1}, Answer → Speaker {answer_speaker+1}")
                        print(f"        Combined answer: \"{combined_answer}\"")
                        
                        i += 2
                        continue
                    else:
                        print(f"     ⚠️  Conditions not met: continuation={next_is_continuation}, pause={short_pause}")
        
        # Якщо не виправляємо - додаємо як є
        fixed.append(current)
        i += 1
    
    if fixed_count > 0:
        print(f"✅ Fixed {fixed_count} answer-after-question speaker assignment(s)")
    else:
        print(f"✅ No answer-after-question issues detected")
    
    return fixed


def enforce_speaker_continuity_rule(segments, max_gap=3.0):
    """
    Застосовує правило: якщо спікер починає фразу, він має її закінчити.
    Новий спікер не може початися, поки речення не доведено до крапки.
    
    Критерії для об'єднання:
    1. Поточна репліка НЕ закінчується на крапку/знак питання/вигук
    2. Наступна репліка граматично продовжує думку
    3. Між ними мала пауза (< max_gap)
    4. Різні спікери (помилка діаризації)
    
    Args:
        segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
        max_gap: максимальний проміжок між сегментами для об'єднання (секунди)
    
    Returns:
        merged_segments: список сегментів з об'єднаними незавершеними фразами
    """
    if not segments or len(segments) < 2:
        return segments
    
    print(f"🔍 Enforcing speaker continuity rule in {len(segments)} segments...")
    merged = []
    i = 0
    merged_count = 0
    
    while i < len(segments):
        if i == len(segments) - 1:
            # Останній сегмент - просто додаємо
            merged.append(segments[i])
            i += 1
            continue
        
        current = segments[i]
        next_seg = segments[i + 1]
        
        current_text = (current.get('text') or '').strip()
        next_text = (next_seg.get('text') or '').strip()
        
        if not current_text or not next_text:
            merged.append(current)
            i += 1
            continue
        
        # Критерій 1: Поточна репліка НЕ закінчується на завершальний знак
        sentence_endings = ['.', '!', '?']
        current_ends_properly = any(current_text.endswith(ending) for ending in sentence_endings)
        
        # Критерій 2: Різні спікери (потенційна помилка діаризації)
        different_speakers = current.get('speaker') != next_seg.get('speaker')
        
        # Критерій 3: Мала пауза між сегментами
        pause = next_seg.get('start', 0) - current.get('end', 0)
        short_pause = 0 <= pause <= max_gap
        
        # Критерій 4: Наступна репліка граматично продовжує думку
        is_grammatical_continuation = False
        
        if not current_ends_properly and different_speakers and short_pause:
            # Перевірка граматичної зв'язності
            current_lower = current_text.lower()
            next_lower = next_text.lower()
            
            # Останні слова поточної репліки (без пунктуації)
            current_words = current_lower.split()
            next_words = next_lower.split()
            
            if current_words and next_words:
                # Маркери незавершеної фрази
                incomplete_markers = [
                    'the', 'a', 'an', 'this', 'that', 'these', 'those',
                    'but', 'and', 'or', 'so', 'because', 'although',
                    'to', 'for', 'with', 'from', 'in', 'on', 'at',
                    'i', 'you', 'he', 'she', 'it', 'we', 'they',
                    'is', 'are', 'was', 'were', 'has', 'have', 'had',
                    'can', 'could', 'will', 'would', 'should', 'may', 'might'
                ]
                
                last_word = current_words[-1].rstrip('.,!?;:')
                first_word = next_words[0].rstrip('.,!?;:')
                
                # Перевірка 1: Останнє слово поточної репліки - маркер незавершеності
                if last_word in incomplete_markers:
                    is_grammatical_continuation = True
                
                # Перевірка 2: Перше слово наступної репліки не з великої літери
                # (якщо не починається з великої, це продовження)
                elif not next_text[0].isupper() and len(next_words) < 15:
                    is_grammatical_continuation = True
                
                # Перевірка 3: Разом утворюють граматично правильне речення
                combined_text = (current_text + ' ' + next_text).strip()
                combined_words = combined_text.split()
                
                # Якщо об'єднаний текст має сенс (не дуже довгий, немає подвійних пробілів)
                if (len(combined_words) < 30 and 
                    '  ' not in combined_text and
                    not combined_text.startswith(next_text.split()[0] if next_text.split() else '')):
                    
                    # Додаткова перевірка: чи закінчується об'єднана фраза на крапку
                    if any(combined_text.endswith(ending) for ending in sentence_endings):
                        is_grammatical_continuation = True
                
                # Перевірка 4: Конкретні випадки з прикладу
                # "I tried, but the" + "problem is still existing" = продовження
                if (last_word in ['the', 'a', 'an', 'this', 'that'] and 
                    first_word in ['problem', 'issue', 'connection', 'device', 'modem', 'router']):
                    is_grammatical_continuation = True
        
        # Якщо всі критерії виконані - об'єднуємо
        if (not current_ends_properly and 
            different_speakers and 
            short_pause and 
            is_grammatical_continuation):
            
            merged_count += 1
            print(f"  🔗 Merging incomplete phrase:")
            print(f"     Segment 1: Speaker {current.get('speaker')+1} [{current.get('start'):.2f}s-{current.get('end'):.2f}s]: \"{current_text}\"")
            print(f"     Segment 2: Speaker {next_seg.get('speaker')+1} [{next_seg.get('start'):.2f}s-{next_seg.get('end'):.2f}s]: \"{next_text}\"")
            
            # Об'єднуємо сегменти, залишаючи спікера, який почав фразу
            merged_seg = {
                'speaker': current.get('speaker'),  # Спікер, який почав фразу
                'start': current.get('start'),
                'end': next_seg.get('end'),
                'text': (current_text + ' ' + next_text).strip(),
                'speaker_continuity_fix': True,  # Позначаємо для логування
                'original_speakers': [current.get('speaker'), next_seg.get('speaker')]
            }
            merged.append(merged_seg)
            print(f"     ✅ Merged: \"{merged_seg['text']}\" → Speaker {merged_seg['speaker']+1} (started the phrase)")
            i += 2  # Пропускаємо обидва сегменти
            continue
        
        # Якщо не об'єднуємо - додаємо поточний сегмент
        merged.append(current)
        i += 1
    
    if merged_count > 0:
        print(f"✅ Speaker continuity rule applied: merged {merged_count} incomplete phrase(s)")
    else:
        print(f"✅ No incomplete phrases detected")
    
    return merged


def combine_diarization_and_transcription(diarization_segments, words):
    """
    Об'єднує результати діаризації та транскрипції на рівні слів для точності.
    Args:
        diarization_segments: сегменти діаризації [{'speaker': int, 'start': float, 'end': float}]
        words: список слів з timestamps [{'word': str, 'start': float, 'end': float}]
    Returns:
        combined: список об'єднаних сегментів з спікером та текстом
    """
    if not words:
        print("⚠️  No words provided for combination")
        return []
    if not diarization_segments:
        print("⚠️  No diarization segments provided, using default speaker 0")
        # Якщо немає діаризації, повертаємо транскрипцію з одним спікером
        combined = []
        current_start = words[0]['start']
        current_words = []
        for word in words:
            if not word['word'].strip():
                continue
            current_words.append(word['word'])
            # Створюємо сегменти по проміжках >1 сек
            if len(combined) == 0 or (word['start'] - combined[-1]['end'] > 1.0):
                if current_words:
                    combined.append({
                        'speaker': 0,
                        'start': round(current_start, 2),
                        'end': round(word['end'], 2),
                        'text': ' '.join(current_words).strip()
                    })
                    current_words = []
                    current_start = word['start']
        if current_words:
            combined.append({
                'speaker': 0,
                'start': round(current_start, 2),
                'end': round(words[-1]['end'], 2),
                'text': ' '.join(current_words).strip()
            })
        return combined
    print(f"🔗 Combining {len(words)} words with {len(diarization_segments)} diarization segments")
    # Сортуємо сегменти діаризації за часом для швидшого пошуку
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    # Для кожного слова знаходимо найкраще перекриття з сегментами діаризації
    word_speakers = []
    for word_idx, word in enumerate(words):
        word_start = word['start']
        word_end = word['end']
        word_center = (word_start + word_end) / 2.0
        word_text = word['word']
        if not word_text.strip():
            continue
        # Знаходимо спікера з найбільшим перекриттям
        best_speaker = None
        best_overlap = 0
        best_overlap_ratio = 0
        # Спочатку шукаємо сегменти, які повністю містять слово
        fully_contained_segments = []
        for diar_seg in sorted_diar_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            # Перевіряємо, чи слово повністю в межах сегмента
            if word_start >= diar_start and word_end <= diar_end:
                fully_contained_segments.append((diar_seg, diar_seg['speaker']))
        # Якщо є сегменти, що повністю містять слово, використовуємо їх
        if fully_contained_segments:
            # Якщо кілька сегментів містять слово (overlap), вибираємо той, де центр слова
            for diar_seg, speaker in fully_contained_segments:
                diar_start = diar_seg['start']
                diar_end = diar_seg['end']
                # Перевіряємо, чи центр слова в межах сегмента
                if word_center >= diar_start and word_center <= diar_end:
                    best_speaker = speaker
                    best_overlap = word_end - word_start  # Повне перекриття
                    best_overlap_ratio = 1.0
                    break
            # Якщо центр не в жодному сегменті, вибираємо найближчий
            if best_speaker is None:
                min_distance = float('inf')
                for diar_seg, speaker in fully_contained_segments:
                    diar_start = diar_seg['start']
                    diar_end = diar_seg['end']
                    # Відстань від центру слова до центру сегмента
                    seg_center = (diar_start + diar_end) / 2.0
                    distance = abs(word_center - seg_center)
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = speaker
                        best_overlap = word_end - word_start
                        best_overlap_ratio = 1.0
        # Якщо слово не повністю в жодному сегменті, шукаємо найкраще перекриття
        if best_speaker is None:
            for diar_seg in sorted_diar_segments:
                diar_start = diar_seg['start']
                diar_end = diar_seg['end']
                # Обчислюємо перекриття
                overlap_start = max(word_start, diar_start)
                overlap_end = min(word_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > 0:
                    # Обчислюємо відношення перекриття до довжини слова
                    word_duration = word_end - word_start
                    overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                    # Враховуємо також, чи центр слова в межах сегмента
                    center_in_segment = (word_center >= diar_start and word_center <= diar_end)
                    # Пріоритет: центр в сегменті > більше перекриття > більше відношення
                    if (center_in_segment and best_overlap_ratio < 0.5) or \
                       (overlap > best_overlap) or \
                       (overlap == best_overlap and overlap_ratio > best_overlap_ratio):
                        best_overlap = overlap
                        best_overlap_ratio = overlap_ratio
                        best_speaker = diar_seg['speaker']
        # Якщо все ще не знайдено, використовуємо найближчий сегмент за часом
        if best_speaker is None:
            min_distance = float('inf')
            for diar_seg in sorted_diar_segments:
                diar_start = diar_seg['start']
                diar_end = diar_seg['end']
                # Відстань від центру слова до найближчої точки сегмента
                if word_center < diar_start:
                    distance = diar_start - word_center
                elif word_center > diar_end:
                    distance = word_center - diar_end
                else:
                    distance = 0
                if distance < min_distance:
                    min_distance = distance
                    best_speaker = diar_seg['speaker']
        speaker_id = best_speaker if best_speaker is not None else 0
        word_speakers.append({
            'word': word_text,
            'start': word_start,
            'end': word_end,
            'speaker': speaker_id
        })
    # Додаткова перевірка: виправляємо прив'язку слів на межах сегментів
    # Якщо слово починається після закінчення сегмента попереднього спікера,
    # воно має належати наступному спікеру
    for i in range(len(word_speakers)):
        word_info = word_speakers[i]
        word_start = word_info['start']
        word_center = (word_info['start'] + word_info['end']) / 2.0
        # Знаходимо сегмент, що закінчується найближче перед початком цього слова
        segments_ending_before = [seg for seg in sorted_diar_segments 
                                   if seg['end'] <= word_start]
        # Знаходимо сегмент, що починається найближче після початку цього слова
        segments_starting_after = [seg for seg in sorted_diar_segments 
                                     if seg['start'] >= word_start]
        # Якщо є сегмент, що починається після закінчення попереднього
        if segments_ending_before and segments_starting_after:
            last_ending_seg = max(segments_ending_before, key=lambda x: x['end'])
            first_starting_seg = min(segments_starting_after, key=lambda x: x['start'])
            # Якщо є чіткий перехід між спікерами (попередній закінчився, наступний почався)
            if last_ending_seg['end'] <= first_starting_seg['start']:
                # Якщо центр слова знаходиться після закінчення попереднього сегмента
                # або слово починається після закінчення попереднього сегмента,
                # воно має належати наступному спікеру
                if word_center >= last_ending_seg['end'] or word_start >= last_ending_seg['end']:
                    # Зберігаємо поточного спікера для логування
                    current_speaker = word_info['speaker']
                    # Перевіряємо, чи поточний спікер не відповідає наступному сегменту
                    if current_speaker != first_starting_seg['speaker']:
                        # Якщо слово ближче до наступного сегмента, прив'язуємо до нього
                        distance_to_prev = word_start - last_ending_seg['end']
                        distance_to_next = first_starting_seg['start'] - word_start
                        # Якщо слово починається після закінчення попереднього сегмента
                        # і ближче до наступного, прив'язуємо до наступного
                        if word_start >= last_ending_seg['end'] and (distance_to_next < distance_to_prev or distance_to_next < 0.3):
                            word_info['speaker'] = first_starting_seg['speaker']
                            print(f"🔧 Fixed word '{word_info['word']}' at {word_start:.2f}s: assigned to speaker {first_starting_seg['speaker']} (was {current_speaker})")
    # Діагностика: перевіряємо розподіл спікерів
    speakers_found = set(w['speaker'] for w in word_speakers)
    print(f"📊 Word-level speakers: {len(speakers_found)} unique speakers found: {sorted(speakers_found)}")
    # Групуємо послідовні слова одного спікера в сегменти
    combined = []
    if not word_speakers:
        return combined
    current_speaker = word_speakers[0]['speaker']
    current_start = word_speakers[0]['start']
    current_words = [word_speakers[0]['word']]
    for i in range(1, len(word_speakers)):
        word_info = word_speakers[i]
        prev_word_info = word_speakers[i-1]
        # Перевіряємо, чи є сегмент діаризації, що починається між словами
        # Якщо так, і слово належить іншому спікеру, це перебивка
        gap_start = prev_word_info['end']
        gap_end = word_info['start']
        # Знаходимо сегменти, що починаються в проміжку між словами
        segments_in_gap = [seg for seg in sorted_diar_segments 
                           if seg['start'] >= gap_start and seg['start'] <= gap_end]
        # Якщо спікер змінився або великий проміжок (>1 сек), створюємо новий сегмент
        # Або якщо є сегмент іншого спікера в проміжку
        should_split = False
        if word_info['speaker'] != current_speaker:
            should_split = True
        elif gap_end - gap_start > 1.0:
            should_split = True
        elif segments_in_gap:
            # Якщо є сегмент іншого спікера в проміжку, розділяємо
            for seg in segments_in_gap:
                if seg['speaker'] != current_speaker:
                    should_split = True
                    break
        if should_split:
            # Зберігаємо поточний сегмент
            combined.append({
                'speaker': current_speaker,
                'start': round(current_start, 2),
                'end': round(prev_word_info['end'], 2),
                'text': ' '.join(current_words).strip()
            })
            # Починаємо новий сегмент
            current_speaker = word_info['speaker']
            current_start = word_info['start']
            current_words = [word_info['word']]
        else:
            # Додаємо слово до поточного сегмента
            current_words.append(word_info['word'])
    # Додаємо останній сегмент
    if current_words:
        combined.append({
            'speaker': current_speaker,
            'start': round(current_start, 2),
            'end': round(word_speakers[-1]['end'], 2),
            'text': ' '.join(current_words).strip()
        })
    # Діагностика: перевіряємо фінальний результат
    final_speakers = set(seg['speaker'] for seg in combined)
    print(f"✅ Combined result: {len(combined)} segments, {len(final_speakers)} unique speakers: {sorted(final_speakers)}")
    
    # КРИТИЧНА ПЕРЕВІРКА 0: Виправлення помилок "питання + коротка відповідь" в одному сегменті
    print(f"🔍 Fixing answer-after-question speaker assignments...")
    combined = fix_answer_after_question_speaker_assignment_v2(combined, max_gap=3.0)
    
    # КРИТИЧНА ПЕРЕВІРКА 1: Правило неперервності спікера (спікер, який почав фразу, має її закінчити)
    print(f"🔍 Applying speaker continuity rule (speaker who started phrase must finish it)...")
    combined = enforce_speaker_continuity_rule(combined, max_gap=3.0)
    
    # Об'єднуємо сусідні сегменти одного спікера для зменшення фрагментації
    # АЛЕ: зберігаємо всі сегменти, не об'єднуємо занадто агресивно
    print(f"📊 Before merging: {len(combined)} segments")
    combined = merge_consecutive_speaker_segments(combined, max_gap=1.5)  # Зменшуємо max_gap для менш агресивного об'єднання
    print(f"📊 After merging: {len(combined)} segments")
    
    # КРИТИЧНА ПЕРЕВІРКА 2: Виявлення помилок призначення спікерів (завершена думка → питання)
    print(f"🔍 Checking for speaker assignment errors (complete statement → question pattern)...")
    combined = detect_and_fix_speaker_mismatch_after_complete_statement(combined)
    
    # КРИТИЧНА ПЕРЕВІРКА 3: Виявлення розбитих фраз (перша починається з великої, друга закінчується на ?)
    print(f"🔍 Checking for fragmented phrases (split sentences)...")
    combined = detect_and_merge_fragmented_phrases(combined)
    
    # Нормалізуємо порядок спікерів ПЕРЕД LLM обробкою
    print(f"🔧 Normalizing speaker order before LLM processing...")
    combined = normalize_speaker_order(combined)
    
    # НОВА КАСКАДНА СИСТЕМА: Використовуємо двоетапну систему з ескалацією
    try:
        global _llm_iterations_cache
        print(f"🤖 Starting Cascading Diarization System...")
        print(f"📊 Input: {len(combined)} segments")
        
        # Формуємо текст для каскадної системи
        full_text = "\n".join([
            f"Speaker {seg['speaker']+1} [{seg['start']:.2f}s-{seg['end']:.2f}s]: {seg['text']}"
            for seg in combined
        ])
        
        # Створюємо адаптери для моделей
        def call_fast_model(prompt):
            """Adapter for 1B model (fast)"""
            return _llm_request(
                "http://127.0.0.1:3001/v1/chat/completions",
                "google/gemma-3-1b",
                "You are a fast diarization tool. Return only JSON.",
                prompt,
                max_tokens=500
            )
        
        def call_smart_model(prompt):
            """Adapter for 20B model (smart, high reasoning)"""
            return _llm_request(
                "http://127.0.0.1:3001/v1/chat/completions",
                "openai/gpt-oss-20b",
                "You are an expert dialogue analyst with advanced reasoning capabilities. Use heavy reasoning.",
                prompt,
                max_tokens=800
            )
        
        # Створюємо контролер каскадної системи
        controller = CascadingDiarizationController(
            fast_model_func=call_fast_model,
            smart_model_func=call_smart_model,
            agent_context="Customer service representative, professional, offers solutions",
            client_context="Customer seeking help, may be emotional, describes problems"
        )
        
        # Спочатку перевіряємо сегменти, які потребують перевірки спікера
        # 1. Fragmented phrases
        segments_needing_verification = [
            seg for seg in combined 
            if seg.get('needs_speaker_verification', False) and seg.get('fragmented_merge', False)
        ]
        
        # 2. Segments with reassignment after complete statement
        segments_needing_role_verification = [
            seg for seg in combined 
            if seg.get('needs_role_verification', False) and seg.get('reassignment_reason') == 'question_after_complete_statement'
        ]
        
        # Об'єднуємо обидва типи для перевірки
        all_segments_for_verification = segments_needing_verification + segments_needing_role_verification
        
        if all_segments_for_verification:
            print(f"🔍 Found {len(all_segments_for_verification)} segments needing speaker/role verification...")
            for seg in all_segments_for_verification:
                # Створюємо спеціальний промпт для перевірки спікера
                if seg.get('fragmented_merge', False):
                    # Fragmented phrase
                    verification_prompt = f"""You are an expert dialogue analyst.

CONTEXT:
- Agent Role: Customer service representative, professional, offers solutions, asks process questions.
- Client Role: Customer seeking help, may be emotional, describes problems, asks about product/service.

MERGED FRAGMENTED PHRASE (was split incorrectly by diarization):
"{seg['text']}"

This phrase was incorrectly split into two segments with different speakers:
- Original Speaker 1: {seg.get('original_speakers', [0, 1])[0] + 1}
- Original Speaker 2: {seg.get('original_speakers', [0, 1])[1] + 1}

CRITICAL: Determine the CORRECT speaker for this complete phrase.

Consider:
- Who typically asks questions like this? (Agent asks process questions, Client asks for help)
- The grammatical structure and tone
- Question patterns: "Did you try to..." is typically from Agent (checking what client tried)

Return ONLY a JSON object:
{{
  "speaker": "Agent" or "Client",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
                else:
                    # Question after complete statement
                    verification_prompt = f"""You are an expert dialogue analyst.

CONTEXT:
- Agent Role: Customer service representative, professional, offers solutions, asks process questions.
- Client Role: Customer seeking help, may be emotional, describes problems, asks about product/service.

SEGMENT TO VERIFY:
"{seg['text']}"

This segment was detected as a QUESTION that follows a complete statement from the same speaker (likely a diarization error).

CRITICAL: Determine the CORRECT speaker for this question.

Consider:
- Questions like "Hey, did you try to...", "Can you...", "Have you..." are typically from Agent (checking what client tried)
- The context: if previous segment was a problem description, this question is likely from Agent
- Question patterns: process questions are typically from Agent

Return ONLY a JSON object:
{{
  "speaker": "Agent" or "Client",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
                
                # Використовуємо smart model для перевірки
                smart_response = call_smart_model(verification_prompt)
                
                try:
                    import json
                    import re
                    json_match = re.search(r'\{.*?\}', smart_response, re.DOTALL)
                    if json_match:
                        verification_data = json.loads(json_match.group())
                        correct_speaker = verification_data.get('speaker', 'Client')
                        speaker_num = 0 if correct_speaker == 'Agent' else 1
                        seg['speaker'] = speaker_num
                        seg['verification_confidence'] = verification_data.get('confidence', 0.8)
                        seg['verification_reasoning'] = verification_data.get('reasoning', '')
                        print(f"  ✅ Verified fragmented phrase: Speaker {speaker_num+1} ({correct_speaker}) - {verification_data.get('reasoning', '')}")
                except Exception as e:
                    print(f"  ⚠️  Could not parse verification response: {e}")
                    # Залишаємо оригінального спікера першого сегмента
        
        # Обробляємо через каскадну систему (якщо потрібно)
        # Але спочатку перевіримо, чи є ще сегменти для обробки
        if len(combined) > 0:
            # Формуємо текст для каскадної системи (оновлений після об'єднання)
            full_text = "\n".join([
                f"Speaker {seg['speaker']+1} [{seg['start']:.2f}s-{seg['end']:.2f}s]: {seg['text']}"
                for seg in combined
            ])
            
            cascading_segments = controller.process_full_text(
                full_text=full_text,
                context_summary="Customer service conversation",
                max_chunk_size=1000
            )
        
        # КРИТИЧНО: Конвертуємо результати каскадної системи назад у формат combined
        # Зберігаємо ВСІ оригінальні сегменти та їх timestamps
        # Маппимо 'Agent' -> 0, 'Client' -> 1
        combined_cascading = []
        used_original_indices = set()  # Відстежуємо, які оригінальні сегменти вже використані
        
        for casc_seg in cascading_segments:
            speaker_num = 0 if casc_seg.speaker == 'Agent' else 1
            # Знаходимо найкращий відповідний сегмент у combined для timestamp
            matching_seg = None
            best_match_idx = -1
            best_similarity = 0.0
            
            for idx, orig_seg in enumerate(combined):
                if idx in used_original_indices:
                    continue
                
                orig_text = orig_seg['text'].strip().lower()
                casc_text = casc_seg.text.strip().lower()
                
                # Перевіряємо різні типи відповідності
                if orig_text in casc_text or casc_text in orig_text:
                    # Точна відповідність
                    similarity = min(len(orig_text), len(casc_text)) / max(len(orig_text), len(casc_text), 1)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        matching_seg = orig_seg
                        best_match_idx = idx
                elif len(orig_text) > 10 and len(casc_text) > 10:
                    # Перевіряємо перетин слів
                    orig_words = set(orig_text.split())
                    casc_words = set(casc_text.split())
                    if orig_words and casc_words:
                        word_overlap = len(orig_words & casc_words) / len(orig_words | casc_words)
                        if word_overlap > 0.5 and word_overlap > best_similarity:
                            best_similarity = word_overlap
                            matching_seg = orig_seg
                            best_match_idx = idx
            
            if matching_seg and best_match_idx >= 0:
                used_original_indices.add(best_match_idx)
                combined_cascading.append({
                    'speaker': speaker_num,
                    'start': matching_seg['start'],
                    'end': matching_seg['end'],
                    'text': casc_seg.text,  # Використовуємо текст з каскадної системи (може бути виправлений)
                    'confidence': casc_seg.confidence,
                    'needs_escalation': casc_seg.needs_escalation,
                    'final_decision_basis': casc_seg.final_decision_basis
                })
            else:
                # Якщо не знайшли відповідність, намагаємося знайти найближчий сегмент за часом
                # Але це небезпечно - краще не додавати сегменти без timestamps
                print(f"  ⚠️  Could not match cascading segment: \"{casc_seg.text[:50]}...\" - skipping to preserve timestamps")
                # НЕ додаємо сегменти без timestamps - вони втрачаються
        
        # Додаємо оригінальні сегменти, які не були оброблені каскадною системою
        for idx, orig_seg in enumerate(combined):
            if idx not in used_original_indices:
                print(f"  📌 Preserving original segment {idx+1}: Speaker {orig_seg['speaker']+1} [{orig_seg['start']:.2f}s-{orig_seg['end']:.2f}s]")
                combined_cascading.append(orig_seg)
        
        # Сортуємо за часом початку
        combined_cascading.sort(key=lambda x: x.get('start', 0))
        
        # КРИТИЧНО: Перевіряємо, чи каскадна система не втратила сегменти
        if combined_cascading and len(combined_cascading) > 0:
            # Перевіряємо, чи кількість сегментів не зменшилася критично
            original_count = len(combined)
            cascading_count = len(combined_cascading)
            
            if cascading_count < original_count * 0.5:  # Втрачено більше 50% сегментів
                print(f"⚠️  Cascading system lost too many segments ({original_count} → {cascading_count}), falling back to standard LLM...")
                combined, llm_iterations = fix_diarization_errors_with_llm(combined)
                _llm_iterations_cache = llm_iterations
            else:
                # Перевіряємо, чи всі сегменти мають правильні timestamps
                segments_with_zero_timestamp = [s for s in combined_cascading if s.get('start', 0) == 0.0 and s.get('end', 0) == 0.0]
                if len(segments_with_zero_timestamp) > len(combined_cascading) * 0.3:  # Більше 30% без timestamps
                    print(f"⚠️  Too many segments without timestamps ({len(segments_with_zero_timestamp)}/{len(combined_cascading)}), falling back to standard LLM...")
                    combined, llm_iterations = fix_diarization_errors_with_llm(combined)
                    _llm_iterations_cache = llm_iterations
                else:
                    print(f"✅ Cascading system completed: {len(combined_cascading)} segments (was {original_count})")
                    combined = combined_cascading
                    _llm_iterations_cache = []  # Каскадна система має свою логіку
        else:
            # Fallback до старої системи
            print(f"⚠️  Cascading system returned no results, falling back to standard LLM...")
            combined, llm_iterations = fix_diarization_errors_with_llm(combined)
            _llm_iterations_cache = llm_iterations
        
        # Після LLM обробки знову нормалізуємо
        combined = normalize_speaker_order(combined)
        
    except Exception as e:
        print(f"⚠️  Error in cascading diarization system: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"📋 Cascading Error traceback:\n{error_traceback}")
        traceback.print_exc()
        
        # Fallback до старої системи
        try:
            print(f"🔄 Falling back to standard LLM system...")
            combined, llm_iterations = fix_diarization_errors_with_llm(combined)
            _llm_iterations_cache = llm_iterations
        except Exception as e2:
            print(f"⚠️  Error in fallback LLM system: {e2}")
            _llm_iterations_cache = []
        
        # Якщо все не вдалося, нормалізуємо порядок
        combined = normalize_speaker_order(combined)
    
    return combined


def assign_speakers_to_transcription_segments(diarization_segments, transcription_segments):
    """
    Призначає спікера для кожного сегмента транскрипції на основі перекриття з діаризацією.
    Використовує сегменти ASR як основу для читабельного formatted_dialogue.
    """
    if not transcription_segments:
        return []
    diar_segments = sorted(diarization_segments or [], key=lambda x: x.get('start', 0))
    combined = []
    for seg in transcription_segments:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        try:
            start = float(seg.get('start', 0) or 0)
            end = float(seg.get('end', start) or start)
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        speaker = 0
        if diar_segments:
            best_speaker = None
            best_overlap = -1.0
            for diar_seg in diar_segments:
                d_start = float(diar_seg.get('start', 0) or 0)
                d_end = float(diar_seg.get('end', d_start) or d_start)
                overlap = max(0.0, min(end, d_end) - max(start, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg.get('speaker', 0)
            if best_speaker is None or best_overlap <= 0:
                center = (start + end) / 2.0
                best_distance = None
                for diar_seg in diar_segments:
                    d_start = float(diar_seg.get('start', 0) or 0)
                    d_end = float(diar_seg.get('end', d_start) or d_start)
                    if center < d_start:
                        distance = d_start - center
                    elif center > d_end:
                        distance = center - d_end
                    else:
                        distance = 0.0
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_speaker = diar_seg.get('speaker', 0)
            speaker = int(best_speaker) if best_speaker is not None else 0
        combined.append({
            'speaker': speaker,
            'start': round(start, 2),
            'end': round(end, 2),
            'text': text
        })
    return combined


def detect_and_merge_fragmented_phrases(segments, max_gap=2.0):
    """
    Виявляє та об'єднує фрагментовані фрази, які були неправильно розділені діаризатором.
    
    Шаблон виявлення:
    - Дві репліки стоять поруч (мала пауза < max_gap)
    - Перша починається з великої літери (початок речення)
    - Друга закінчується на знак питання (кінець речення)
    - Різні спікери (помилка діаризації)
    
    Такі сегменти об'єднуються і позначаються для ескалації до smart model.
    
    Args:
        segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
        max_gap: максимальний проміжок між сегментами для об'єднання (секунди)
    
    Returns:
        merged_segments: список сегментів з об'єднаними фрагментованими фразами
    """
    if not segments or len(segments) < 2:
        return segments
    
    print(f"🔍 Detecting fragmented phrases in {len(segments)} segments...")
    merged = []
    i = 0
    fragmented_count = 0
    
    while i < len(segments):
        if i == len(segments) - 1:
            # Останній сегмент - просто додаємо
            merged.append(segments[i])
            i += 1
            continue
        
        current = segments[i]
        next_seg = segments[i + 1]
        
        # Перевіряємо умови для фрагментованої фрази
        current_text = current['text'].strip()
        next_text = next_seg['text'].strip()
        
        # Умова 1: Мала пауза між сегментами
        pause = next_seg['start'] - current['end']
        short_pause = pause >= 0 and pause < max_gap
        
        # Умова 2: Перша починається з великої літери (початок речення)
        starts_with_capital = len(current_text) > 0 and current_text[0].isupper()
        
        # Умова 3: Друга закінчується на знак питання
        ends_with_question = next_text.endswith('?')
        
        # Умова 4: Різні спікери (помилка діаризації)
        different_speakers = current['speaker'] != next_seg['speaker']
        
        # Умова 5: Перша не закінчується на знак завершення речення
        current_ends_properly = current_text.endswith(('.', '!', '?'))
        
        # Додаткова перевірка: перша фраза виглядає як початок речення
        # (не закінчується на знак завершення, або закінчується на кому/тире)
        looks_like_start = not current_ends_properly or current_text.endswith((',', '-', '—', '–'))
        
        # Перевірка на граматичну зв'язність
        # Якщо разом вони утворюють граматично правильне речення
        combined_text = (current_text + ' ' + next_text).strip()
        is_grammatically_connected = (
            len(combined_text.split()) < 30 and  # Не дуже довге
            '  ' not in combined_text and  # Немає подвійних пробілів
            not combined_text.startswith(next_text.split()[0] if next_text.split() else '')  # Не дублікат
        )
        
        if (short_pause and 
            starts_with_capital and 
            ends_with_question and 
            different_speakers and 
            looks_like_start and
            is_grammatically_connected):
            
            # Це фрагментована фраза - об'єднуємо
            fragmented_count += 1
            print(f"  🔗 Detected fragmented phrase:")
            print(f"     Segment 1: Speaker {current['speaker']+1} [{current['start']:.2f}s-{current['end']:.2f}s]: \"{current_text}\"")
            print(f"     Segment 2: Speaker {next_seg['speaker']+1} [{next_seg['start']:.2f}s-{next_seg['end']:.2f}s]: \"{next_text}\"")
            
            # Об'єднуємо сегменти
            merged_seg = {
                'speaker': current['speaker'],  # Тимчасово залишаємо спікера першого сегмента
                'start': current['start'],
                'end': next_seg['end'],
                'text': combined_text,
                'needs_speaker_verification': True,  # Позначаємо для перевірки спікера
                'original_speakers': [current['speaker'], next_seg['speaker']],  # Зберігаємо оригінальних спікерів
                'fragmented_merge': True  # Позначаємо як об'єднання фрагментів
            }
            merged.append(merged_seg)
            print(f"     ✅ Merged: \"{combined_text}\" (needs speaker verification)")
            i += 2  # Пропускаємо обидва сегменти
            continue
        
        # Якщо не фрагментована фраза - просто додаємо поточний сегмент
        merged.append(current)
        i += 1
    
    if fragmented_count > 0:
        print(f"✅ Detected and merged {fragmented_count} fragmented phrase(s)")
    else:
        print(f"✅ No fragmented phrases detected")
    
    return merged


def merge_consecutive_speaker_segments(segments, max_gap=2.0):
    """
    Об'єднує сусідні сегменти одного спікера для зменшення фрагментації.
    Args:
        segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
        max_gap: максимальний проміжок між сегментами для об'єднання (секунди)
    Returns:
        merged_segments: об'єднаний список сегментів
    """
    if not segments or len(segments) < 2:
        return segments
    print(f"🔗 Merging consecutive segments from same speaker (max_gap={max_gap}s)...")
    merged = []
    current_seg = None
    for seg in segments:
        if current_seg is None:
            current_seg = seg.copy()
            continue
        # Перевіряємо, чи можна об'єднати
        gap = seg['start'] - current_seg['end']
        same_speaker = current_seg['speaker'] == seg['speaker']
        has_overlap = seg['start'] < current_seg['end']
        # Об'єднуємо, якщо:
        # 1. Один і той же спікер
        # 2. Перекриваються або мають невеликий проміжок (< max_gap)
        if same_speaker and (has_overlap or gap <= max_gap):
            # Об'єднуємо сегменти
            current_seg['end'] = max(current_seg['end'], seg['end'])
            # Об'єднуємо текст
            current_seg['text'] = (current_seg['text'] + ' ' + seg['text']).strip()
            print(f"  🔗 Merged: Speaker {current_seg['speaker']} [{current_seg['start']:.2f}s-{seg['end']:.2f}s]")
        else:
            # Зберігаємо поточний сегмент і починаємо новий
            merged.append(current_seg)
            current_seg = seg.copy()
    # Додаємо останній сегмент
    if current_seg:
        merged.append(current_seg)
    if len(merged) < len(segments):
        print(f"✅ Merged: {len(segments)} → {len(merged)} segments")
    else:
        print(f"✅ No merging needed: {len(segments)} segments")
    return merged


def _llm_request(lm_studio_url, model, system_prompt, user_prompt, max_tokens=500):
    """Допоміжна функція для відправки запиту до LLM"""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": max_tokens
        }
        response = requests.post(
            lm_studio_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # Короткий timeout для мікропромптів
        )
        if response.status_code != 200:
            return None
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    except Exception as e:
        print(f"⚠️  LLM request error: {e}")
        return None


def analyze_dialogue_zones(segments):
    """
    Аналізує діалог і виявляє різні типи зон:
    - overlaps: перетини реплік (короткі паузи між різними спікерами)
    - clean_speech: чиста мова одного спікера (довгі сегменти без переривань)
    - pauses: паузи та тиша (великі проміжки між сегментами)
    - short_segments: дуже короткі сегменти (можливі шуми або фрагменти)
    
    Returns:
        dict: {
            'overlaps': [{'start_idx': int, 'end_idx': int, 'segments': [...]}],
            'clean_speech': [...],
            'pauses': [...],
            'short_segments': [...]
        }
    """
    if not segments or len(segments) < 2:
        return {'overlaps': [], 'clean_speech': [], 'pauses': [], 'short_segments': []}
    
    overlaps = []
    clean_speech = []
    pauses = []
    short_segments = []
    
    # Виявляємо перетини реплік (короткі паузи між різними спікерами)
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        pause = next_seg['start'] - current['end']
        duration_current = current['end'] - current['start']
        duration_next = next_seg['end'] - next_seg['start']
        
        # Перетин: різні спікери + коротка пауза (< 1.5 сек) + короткі сегменти
        if current['speaker'] != next_seg['speaker'] and pause < 1.5 and pause >= 0:
            # Перевіряємо, чи це не початок нового перетину
            if not overlaps or overlaps[-1]['end_idx'] < i:
                overlaps.append({
                    'start_idx': i,
                    'end_idx': i + 1,
                    'segments': [current, next_seg],
                    'pause': pause
                })
            else:
                # Розширюємо поточний перетин
                overlaps[-1]['end_idx'] = i + 1
                overlaps[-1]['segments'].append(next_seg)
        
        # Пауза: великий проміжок (> 2 сек)
        if pause > 2.0:
            pauses.append({
                'start_idx': i,
                'end_idx': i + 1,
                'pause_duration': pause,
                'segments': [current, next_seg]
            })
        
        # Короткі сегменти (можливі шуми або фрагменти)
        if duration_current < 0.5 or duration_next < 0.5:
            if duration_current < 0.5:
                short_segments.append({
                    'idx': i,
                    'segment': current,
                    'duration': duration_current
                })
            if duration_next < 0.5 and i == len(segments) - 2:
                short_segments.append({
                    'idx': i + 1,
                    'segment': next_seg,
                    'duration': duration_next
                })
    
    # Виявляємо чисту мову (довгі сегменти одного спікера без переривань)
    i = 0
    while i < len(segments):
        current = segments[i]
        duration = current['end'] - current['start']
        
        # Чиста мова: довгий сегмент (> 3 сек) або послідовні сегменти одного спікера
        if duration > 3.0:
            clean_speech.append({
                'start_idx': i,
                'end_idx': i,
                'segments': [current],
                'speaker': current['speaker']
            })
        else:
            # Перевіряємо послідовні сегменти одного спікера
            clean_zone = [current]
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                pause = next_seg['start'] - segments[j-1]['end']
                if next_seg['speaker'] == current['speaker'] and pause < 1.0:
                    clean_zone.append(next_seg)
                    j += 1
                else:
                    break
            
            if len(clean_zone) > 1:
                total_duration = clean_zone[-1]['end'] - clean_zone[0]['start']
                if total_duration > 2.0:
                    clean_speech.append({
                        'start_idx': i,
                        'end_idx': j - 1,
                        'segments': clean_zone,
                        'speaker': current['speaker']
                    })
                    i = j - 1
        
        i += 1
    
    return {
        'overlaps': overlaps,
        'clean_speech': clean_speech,
        'pauses': pauses,
        'short_segments': short_segments
    }


def assess_llm_confidence(response, expected_format="json", min_items=1):
    """
    Оцінює впевненість LLM у своїй відповіді.
    
    Args:
        response: відповідь від LLM
        expected_format: очікуваний формат ("json")
        min_items: мінімальна кількість елементів для впевненої відповіді
    
    Returns:
        tuple: (confidence_score: float 0-1, is_confident: bool, reason: str)
    """
    if not response:
        return 0.0, False, "No response"
    
    confidence = 0.5  # Базовий рівень
    reasons = []
    
    # Перевірка 1: Чи правильно сформований JSON
    try:
        import json
        import re
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                confidence += 0.3
                reasons.append("Valid JSON array")
                
                # Перевірка 2: Кількість елементів
                if len(data) >= min_items:
                    confidence += 0.1
                    reasons.append(f"Has {len(data)} items")
                elif len(data) == 0:
                    confidence -= 0.2
                    reasons.append("Empty array (might be uncertain)")
        else:
            confidence -= 0.3
            reasons.append("No valid JSON found")
    except Exception as e:
        confidence -= 0.4
        reasons.append(f"JSON parse error: {str(e)[:30]}")
    
    # Перевірка 3: Наявність маркерів невпевненості
    uncertainty_markers = ["uncertain", "not sure", "maybe", "possibly", "might", "could be", "not confident"]
    response_lower = response.lower()
    for marker in uncertainty_markers:
        if marker in response_lower:
            confidence -= 0.2
            reasons.append(f"Uncertainty marker: {marker}")
            break
    
    # Перевірка 4: Наявність коментарів про складність
    complexity_markers = ["difficult", "complex", "hard to determine", "ambiguous"]
    for marker in complexity_markers:
        if marker in response_lower:
            confidence -= 0.15
            reasons.append(f"Complexity marker: {marker}")
            break
    
    confidence = max(0.0, min(1.0, confidence))  # Обмежуємо 0-1
    is_confident = confidence >= 0.6  # Поріг впевненості
    
    return confidence, is_confident, "; ".join(reasons) if reasons else "Basic assessment"


def fix_diarization_errors_with_llm(combined_segments, lm_studio_url="http://127.0.0.1:3001/v1/chat/completions", model="google/gemma-3-1b", escalation_model="openai/gpt-oss-20b"):
    """
    Виправляє помилки діаризації за допомогою LLM (LM Studio) через ланцюжок мікропромптів.
    НОВИЙ ПІДХІД: Даємо повний діалог як контекст, але з простими фокусованими інструкціями.
    Обробляємо той самий діалог послідовно з різними інструкціями.
    Args:
        combined_segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
        lm_studio_url: URL для LM Studio API
        model: назва моделі
    Returns:
        tuple: (fixed_segments, llm_iterations_info)
        - fixed_segments: виправлений список сегментів
        - llm_iterations_info: список інформації про ітерації [{'iteration': int, 'total': int, 'result': str}]
    """
    try:
        if not combined_segments or len(combined_segments) < 2:
            print("⚠️  LLM: Not enough segments to process")
            return combined_segments, []
        print(f"🤖 Fixing diarization errors with LLM ({model}) using full dialogue context with micro-instructions...")
        # Формуємо повний діалог для контексту
        full_dialogue = ""
        for idx, seg in enumerate(combined_segments):
            full_dialogue += f"{idx+1}. Speaker {seg['speaker']+1} [{seg['start']:.2f}s-{seg['end']:.2f}s]: \"{seg['text']}\"\n"
        llm_iterations = []  # Інформація про ітерації для дебаг консолі
        # ІТЕРАЦІЯ 1: Виявлення сегментів, які належать одному спікеру (об'єднання)
        print("📋 LLM Iteration 1: Identifying segments that belong to the same speaker...")
        current_iteration = 1
        total_iterations = 3  # 3 ітерації з різними інструкціями
        system_prompt_1 = "You are a helpful assistant. Analyze the dialogue and identify which consecutive segments belong to the same speaker."
        user_prompt_1 = f"""FULL DIALOGUE:
{full_dialogue}

INSTRUCTION: Look at consecutive segments where the speaker changes. For each pair of consecutive segments with DIFFERENT speakers, determine if they actually belong to the SAME speaker (one person's speech was incorrectly split).

Return ONLY a JSON array of pairs that should be merged. Format:
[
  {{"segment1": 1, "segment2": 2, "should_merge": true}},
  {{"segment1": 5, "segment2": 6, "should_merge": false}},
  ...
]

If no merges needed, return empty array: []"""
        # Спочатку пробуємо слабку модель
        response_1 = _llm_request(lm_studio_url, model, system_prompt_1, user_prompt_1, max_tokens=500)
        confidence_1, is_confident_1, reason_1 = assess_llm_confidence(response_1, min_items=0)
        
        result_text_1 = f"Iteration 1 (Merge detection): {response_1[:200] if response_1 else 'NO RESPONSE'}..."
        print(f"  [{current_iteration}/{total_iterations}] {result_text_1}")
        print(f"  📊 Confidence: {confidence_1:.2f} ({'✓ Confident' if is_confident_1 else '✗ Uncertain'}) - {reason_1}")
        
        # Ескалація для merge detection (якщо дуже низька впевненість)
        if not is_confident_1 and escalation_model and confidence_1 < 0.5:
            print(f"  ⬆️  Escalating to {escalation_model} for merge detection...")
            response_1_escalated = _llm_request(lm_studio_url, escalation_model, system_prompt_1, user_prompt_1, max_tokens=800)
            confidence_1_escalated, _, _ = assess_llm_confidence(response_1_escalated, min_items=0)
            
            if confidence_1_escalated > confidence_1:
                response_1 = response_1_escalated
                result_text_1 = f"Iteration 1 (Merge detection) [ESCALATED to {escalation_model}]: {response_1[:200] if response_1 else 'NO RESPONSE'}..."
                print(f"  ✅ Escalation improved confidence: {confidence_1:.2f} → {confidence_1_escalated:.2f}")
        
        llm_iterations.append({
            'iteration': current_iteration,
            'total': total_iterations,
            'result': f"{result_text_1} [Confidence: {confidence_1:.2f}]"
        })
        # Парсимо відповідь про об'єднання
        merge_decisions = []
        try:
            import json
            import re
            # Спробуємо витягти JSON з відповіді
            json_match = re.search(r'\[.*?\]', response_1, re.DOTALL)
            if json_match:
                merge_data = json.loads(json_match.group())
                for item in merge_data:
                    if item.get('should_merge', False):
                        seg1_idx = item.get('segment1', 0) - 1  # Конвертуємо з 1-based в 0-based
                        seg2_idx = item.get('segment2', 0) - 1
                        if 0 <= seg1_idx < len(combined_segments) and 0 <= seg2_idx < len(combined_segments):
                            merge_decisions.append((seg1_idx, seg2_idx))
        except Exception as e:
            print(f"  ⚠️  Could not parse merge decisions: {e}")
        # ІТЕРАЦІЯ 2: Виправлення помилок у визначенні спікерів (питання-відповіді)
        print("📋 LLM Iteration 2: Fixing speaker assignment errors (question-answer patterns)...")
        current_iteration = 2
        system_prompt_2 = "You are a helpful assistant. Analyze the dialogue and fix speaker assignment errors based on question-answer patterns and role relevance. DO NOT blindly alternate speakers."
        user_prompt_2 = f"""FULL DIALOGUE:
{full_dialogue}

INSTRUCTION: Identify and fix speaker assignment errors ONLY when there is clear evidence of role mismatch:
- If a segment contains a QUESTION and the next segment contains an ANSWER, they should be from DIFFERENT speakers (ONLY if this is a clear question-answer pattern)
- If a segment contains an ANSWER and the previous segment contains a QUESTION, they should be from DIFFERENT speakers (ONLY if this is a clear question-answer pattern)
- DO NOT blindly alternate speakers - preserve the diarization result unless there is clear evidence of error
- Only correct when there is a clear role mismatch (e.g., Agent asking for help, Client offering solutions)

CRITICAL: Preserve the diarization result unless there is strong evidence of error. Do not force alternation.

Return ONLY a JSON array of corrections. Format:
[
  {{"segment": 2, "correct_speaker": 1, "reason": "clear question-answer pattern"}},
  {{"segment": 5, "correct_speaker": 2, "reason": "role mismatch detected"}},
  ...
]

If no corrections needed, return empty array: []"""
        response_2 = _llm_request(lm_studio_url, model, system_prompt_2, user_prompt_2, max_tokens=500)
        result_text_2 = f"Iteration 2 (Speaker correction): {response_2[:200] if response_2 else 'NO RESPONSE'}..."
        llm_iterations.append({
            'iteration': current_iteration,
            'total': total_iterations,
            'result': result_text_2
        })
        print(f"  [{current_iteration}/{total_iterations}] {result_text_2}")
        # Парсимо відповідь про виправлення спікерів
        speaker_corrections = {}
        try:
            json_match = re.search(r'\[.*?\]', response_2, re.DOTALL)
            if json_match:
                corrections_data = json.loads(json_match.group())
                for item in corrections_data:
                    seg_idx = item.get('segment', 0) - 1  # Конвертуємо з 1-based в 0-based
                    correct_speaker = item.get('correct_speaker', 0) - 1  # Конвертуємо з 1-based в 0-based
                    if 0 <= seg_idx < len(combined_segments):
                        speaker_corrections[seg_idx] = correct_speaker
        except Exception as e:
            print(f"  ⚠️  Could not parse speaker corrections: {e}")
        
        # Ініціалізуємо overlap_corrections як порожній словник (якщо не було обробки overlap zones)
        overlap_corrections = {}
        
        # ІТЕРАЦІЯ 4: Фінальна перевірка та нормалізація
        print("📋 LLM Iteration 4: Final validation and normalization...")
        current_iteration = 4
        system_prompt_3 = "You are a helpful assistant. Validate the dialogue structure. DO NOT blindly alternate speakers - only fix clear errors."
        user_prompt_3 = f"""FULL DIALOGUE:
{full_dialogue}

INSTRUCTION: Validate the dialogue structure and identify ONLY clear errors:
- Check for obvious role mismatches (e.g., Agent describing problems, Client offering solutions)
- Check for clear question-answer patterns where speakers are incorrectly assigned
- DO NOT force alternation - preserve diarization result unless there is clear evidence of error
- DO NOT change speakers just because they don't alternate - only change when role is clearly wrong

CRITICAL: Preserve the diarization result. Only correct when there is strong evidence of role mismatch.

Return ONLY a JSON array of final corrections. Format:
[
  {{"segment": 1, "correct_speaker": 1, "reason": "clear role mismatch"}},
  ...
]

If no corrections needed, return empty array: []"""
        # Спочатку пробуємо слабку модель
        response_3 = _llm_request(lm_studio_url, model, system_prompt_3, user_prompt_3, max_tokens=500)
        confidence_3, is_confident_3, reason_3 = assess_llm_confidence(response_3, min_items=0)
        
        result_text_3 = f"Iteration 4 (Final validation): {response_3[:200] if response_3 else 'NO RESPONSE'}..."
        print(f"  [{current_iteration}/{total_iterations}] {result_text_3}")
        print(f"  📊 Confidence: {confidence_3:.2f} ({'✓ Confident' if is_confident_3 else '✗ Uncertain'}) - {reason_3}")
        
        # Ескалація для фінальної валідації (якщо дуже низька впевненість)
        if not is_confident_3 and escalation_model and confidence_3 < 0.5:
            print(f"  ⬆️  Escalating to {escalation_model} for final validation...")
            response_3_escalated = _llm_request(lm_studio_url, escalation_model, system_prompt_3, user_prompt_3, max_tokens=800)
            confidence_3_escalated, _, _ = assess_llm_confidence(response_3_escalated, min_items=0)
            
            if confidence_3_escalated > confidence_3:
                response_3 = response_3_escalated
                result_text_3 = f"Iteration 4 (Final validation) [ESCALATED to {escalation_model}]: {response_3[:200] if response_3 else 'NO RESPONSE'}..."
                print(f"  ✅ Escalation improved confidence: {confidence_3:.2f} → {confidence_3_escalated:.2f}")
        
        llm_iterations.append({
            'iteration': current_iteration,
            'total': total_iterations,
            'result': f"{result_text_3} [Confidence: {confidence_3:.2f}]"
        })
        # Парсимо фінальні виправлення
        final_corrections = {}
        try:
            json_match = re.search(r'\[.*?\]', response_3, re.DOTALL)
            if json_match:
                final_data = json.loads(json_match.group())
                for item in final_data:
                    seg_idx = item.get('segment', 0) - 1
                    correct_speaker = item.get('correct_speaker', 0) - 1
                    if 0 <= seg_idx < len(combined_segments):
                        final_corrections[seg_idx] = correct_speaker
        except Exception as e:
            print(f"  ⚠️  Could not parse final corrections: {e}")
        # Застосовуємо всі виправлення
        fixed_segments = [seg.copy() for seg in combined_segments]
        # Крок 1: Об'єднуємо сегменти
        if merge_decisions:
            print(f"  🔗 Applying {len(merge_decisions)} merge decisions...")
            # Створюємо множину індексів, які вже об'єднані
            merged_indices = set()
            new_fixed_segments = []
            i = 0
            while i < len(fixed_segments):
                if i in merged_indices:
                    i += 1
                    continue
                # Перевіряємо, чи потрібно об'єднати з наступним
                should_merge = False
                merge_end = i
                for merge_i, merge_j in merge_decisions:
                    if merge_i == i:
                        should_merge = True
                        merge_end = merge_j
                        merged_indices.add(merge_j)
                        break
                if should_merge:
                    # Об'єднуємо сегменти
                    merged_seg = {
                        'speaker': fixed_segments[i]['speaker'],
                        'start': fixed_segments[i]['start'],
                        'end': fixed_segments[merge_end]['end'],
                        'text': (fixed_segments[i]['text'] + ' ' + fixed_segments[merge_end]['text']).strip()
                    }
                    new_fixed_segments.append(merged_seg)
                    i = merge_end + 1
                else:
                    new_fixed_segments.append(fixed_segments[i])
                    i += 1
            fixed_segments = new_fixed_segments
            print(f"  ✅ Merged: {len(combined_segments)} → {len(fixed_segments)} segments")
        # Крок 2: Застосовуємо виправлення спікерів (з перетинів + загальні + фінальні)
        all_corrections = {**overlap_corrections, **speaker_corrections, **final_corrections}
        if all_corrections:
            print(f"  🔧 Applying {len(all_corrections)} speaker corrections...")
            # Перераховуємо індекси після об'єднання
            correction_map = {}
            current_idx = 0
            for orig_idx in range(len(combined_segments)):
                if orig_idx in all_corrections:
                    correction_map[current_idx] = all_corrections[orig_idx]
                current_idx += 1
            for seg_idx, correct_speaker in correction_map.items():
                if 0 <= seg_idx < len(fixed_segments):
                    fixed_segments[seg_idx]['speaker'] = correct_speaker
            print(f"  ✅ Applied speaker corrections")
        if len(fixed_segments) < len(combined_segments) or all_corrections:
            print(f"✅ LLM fix: {len(combined_segments)} → {len(fixed_segments)} segments")
        else:
            print(f"✅ LLM fix: {len(fixed_segments)} segments (no changes)")
        return fixed_segments, llm_iterations
    except Exception as e:
        print(f"❌ Critical error in fix_diarization_errors_with_llm: {e}")
        import traceback
        traceback.print_exc()
        return combined_segments, []


def normalize_speaker_order(segments):
    """
    Нормалізує порядок спікерів БЕЗ сліпого чергування.
    Виправляє тільки очевидні помилки:
    1. Нормалізує до 2 спікерів (якщо більше)
    2. Виправляє тільки якщо перший сегмент має явно неправильного спікера (на основі контексту)
    3. НЕ робить сліпого чергування - залишає те, що визначив діаризатор
    """
    if not segments or len(segments) == 0:
        return segments
    print(f"🔧 Normalizing speaker order for {len(segments)} segments (NO blind alternation)...")
    # Створюємо копію відразу, щоб не модифікувати оригінал
    fixed_segments = [seg.copy() for seg in segments]
    
    # Крок 1: Нормалізуємо до 2 спікерів (якщо більше)
    unique_speakers = sorted(set(seg['speaker'] for seg in fixed_segments))
    if len(unique_speakers) > 2:
        print(f"⚠️  Found {len(unique_speakers)} speakers, normalizing to 2 speakers")
        # Групуємо спікерів: перші 50% → 0, останні 50% → 1
        speaker_group_map = {}
        mid_point = len(unique_speakers) // 2
        for idx, sp in enumerate(unique_speakers):
            speaker_group_map[sp] = 0 if idx < mid_point else 1
        for seg in fixed_segments:
            seg['speaker'] = speaker_group_map.get(seg['speaker'], seg['speaker'] % 2)
        unique_speakers = [0, 1]
    
    # Крок 2: Перевірка першого сегмента - тільки якщо явно неправильно
    # НЕ робимо сліпого чергування - залишаємо те, що визначив діаризатор
    first_speaker = fixed_segments[0]['speaker']
    
    # Перевіряємо тільки якщо перший сегмент має спікера 1, але контекст вказує на спікера 0
    # Але НЕ змінюємо, якщо діаризатор правильно визначив
    if first_speaker == 1 and len(fixed_segments) > 1:
        # Перевіряємо, чи перші кілька сегментів мають однакового спікера
        # Якщо так, і це спікер 1, можливо діаризатор правильно визначив
        first_few_speakers = [seg['speaker'] for seg in fixed_segments[:min(3, len(fixed_segments))]]
        if all(sp == 1 for sp in first_few_speakers):
            # Всі перші сегменти мають спікера 1 - можливо діаризатор правильно визначив
            # Перевіряємо контекст тексту для визначення ролі
            first_text = fixed_segments[0]['text'].lower()
            # Якщо текст виглядає як клієнт (описує проблему, просить допомогу), залишаємо як є
            client_indicators = ['i have', 'i need', 'i can\'t', 'help me', 'problem', 'issue', 'error']
            if any(indicator in first_text for indicator in client_indicators):
                print(f"✅ First segment appears to be Client (Speaker 2), keeping diarization result")
                # Залишаємо як є - діаризатор правильно визначив
            else:
                # Можливо помилка - але не змінюємо безпідставно
                print(f"⚠️  First segment is Speaker 2, but context unclear. Keeping diarization result.")
        else:
            # Різні спікери на початку - залишаємо як є
            print(f"✅ Mixed speakers at start, keeping diarization result")
    
    # Крок 3: Перевірка балансу спікерів (тільки для логування)
    speaker_counts = {}
    for seg in fixed_segments:
        sp = seg['speaker']
        speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
    print(f"📊 Final speaker distribution (preserving diarization):")
    for sp, count in sorted(speaker_counts.items()):
        print(f"   Speaker {sp+1}: {count} segments ({count/len(fixed_segments)*100:.1f}%)")
    print(f"✅ Speaker normalization complete. First segment: Speaker {fixed_segments[0]['speaker']+1} (preserved from diarization)")
    return fixed_segments


def fix_diarization_errors_advanced(combined_segments):
    """
    Покращена функція виправлення помилок діаризації з використанням семантики та контексту діалогу.
    Виправляє помилки, коли:
    - Репліки одного спікера розбиваються на кілька сегментів з різними спікерами
    - Дуже короткі сегменти (< 0.5 сек) є частиною більших реплік
    - Діаризація неправильно визначає спікера на межах реплік
    Args:
        combined_segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
    Returns:
        fixed_segments: виправлений список сегментів
    """
    if not combined_segments or len(combined_segments) < 2:
        return combined_segments
    print(f"🔧 Advanced fixing diarization errors in {len(combined_segments)} segments...")
    # Граматичні маркери неповних фраз (англійська)
    incomplete_endings = [
        'to', 'and', 'or', 'but',
        'did you', 'can you', 'will you', 'would you', 'could you', 'should you',
        'try to', 'want to', 'need to', 'have to', 'going to', 'supposed to',
        'if', 'when', 'where', 'what', 'who', 'which', 'how',
        'that', 'this', 'these', 'those',
        'because', 'although', 'however', 'therefore',
        'i', 'it', 'the', 'a', 'an'  # Дуже короткі слова на кінці
    ]
    # Маркери продовження (початок наступного сегмента)
    continuation_markers = [
        'reset', 'open', 'close', 'check', 'try', 'do', 'make', 'get', 'set',
        'configure', 'connect', 'disconnect', 'restart', 'reboot', 'update',
        'enter', 'access', 'see', 'show', 'find', 'look'
    ]
    fixed_segments = []
    i = 0
    while i < len(combined_segments):
        current_seg = combined_segments[i]
        current_text = current_seg['text'].strip()
        current_text_lower = current_text.lower()
        current_duration = current_seg['end'] - current_seg['start']
        # Крок 1: Обробка дуже коротких сегментів (< 0.5 сек)
        if current_duration < 0.5 and i > 0 and i < len(combined_segments) - 1:
            prev_seg = combined_segments[i - 1]
            next_seg = combined_segments[i + 1]
            # Визначаємо, до якого сегмента приєднати короткий
            gap_to_prev = current_seg['start'] - prev_seg['end']
            gap_to_next = next_seg['start'] - current_seg['end']
            # Приєднуємо до найближчого сегмента
            if gap_to_prev < gap_to_next and gap_to_prev < 1.0:
                # Приєднуємо до попереднього
                prev_seg['end'] = max(prev_seg['end'], current_seg['end'])
                prev_seg['text'] = (prev_seg['text'] + ' ' + current_text).strip()
                print(f"  🔗 Merged short segment '{current_text[:30]}...' ({current_duration:.2f}s) to previous")
                i += 1
                continue
            elif gap_to_next < 1.0:
                # Приєднуємо до наступного
                next_seg['start'] = min(next_seg['start'], current_seg['start'])
                next_seg['text'] = (current_text + ' ' + next_seg['text']).strip()
                print(f"  🔗 Merged short segment '{current_text[:30]}...' ({current_duration:.2f}s) to next")
                i += 1
                continue
        # Крок 2: Перевірка неповних фраз
        words = current_text_lower.split()
        is_incomplete = False
        if len(words) > 0:
            last_word = words[-1].rstrip('.,!?;:')
            last_two_words = ' '.join(words[-2:]).rstrip('.,!?;:') if len(words) >= 2 else ''
            last_three_words = ' '.join(words[-3:]).rstrip('.,!?;:') if len(words) >= 3 else ''
            # Перевіряємо граматичні маркери неповноти
            if last_word in incomplete_endings:
                is_incomplete = True
            elif last_two_words in incomplete_endings:
                is_incomplete = True
            elif last_three_words in incomplete_endings:
                is_incomplete = True
            # Додаткова перевірка: дуже короткі фрази (< 3 слова) часто неповні
            elif len(words) < 3 and current_duration < 2.0:
                is_incomplete = True
        # Крок 3: Перевірка наступного сегмента
        if is_incomplete and i + 1 < len(combined_segments):
            next_seg = combined_segments[i + 1]
            next_text = next_seg['text'].strip()
            next_text_lower = next_text.lower()
            # Перевіряємо умови для об'єднання:
            different_speakers = current_seg['speaker'] != next_seg['speaker']
            pause = next_seg['start'] - current_seg['end']
            short_pause = pause < 1.5 and pause >= 0
            # Перевіряємо, чи наступний сегмент є продовженням
            is_continuation = False
            next_words = next_text_lower.split()
            if next_words:
                first_word = next_words[0].rstrip('.,!?;:')
                # Перевіряємо маркери продовження
                if first_word in continuation_markers:
                    is_continuation = True
                # Перевіряємо, чи перше слово не з великої літери (продовження речення)
                elif not next_text[0].isupper() and len(next_words) < 10:
                    is_continuation = True
                # Перевіряємо, чи разом утворюють граматично правильну фразу
                combined_text = (current_text + ' ' + next_text).strip()
                if len(combined_text.split()) < 25 and '  ' not in combined_text:
                    is_continuation = True
            # Крок 4: Аналіз контексту діалогу (альтернація спікерів)
            # Перевіряємо попередні сегменти для визначення паттерну
            speaker_pattern = []
            for j in range(max(0, i - 3), i + 2):
                if j < len(combined_segments):
                    speaker_pattern.append(combined_segments[j]['speaker'])
            # Якщо спікери постійно змінюються (аномалія), це може бути помилка
            frequent_changes = sum(1 for k in range(len(speaker_pattern) - 1) 
                                 if speaker_pattern[k] != speaker_pattern[k + 1])
            is_anomaly = frequent_changes >= len(speaker_pattern) - 1
            # Крок 5: Визначення правильного спікера
            if is_continuation and different_speakers and short_pause:
                # Визначаємо правильного спікера
                current_word_count = len(words)
                next_word_count = len(next_words)
                current_duration = current_seg['end'] - current_seg['start']
                next_duration = next_seg['end'] - next_seg['start']
                # Критерій 1: Хто почав фразу (перший сегмент) - найважливіший
                # Критерій 2: Контекст діалогу (альтернація спікерів)
                # Критерій 3: Більша частина фрази
                # Якщо це аномалія (часто змінюються спікери), використовуємо того, хто почав
                if is_anomaly:
                    correct_speaker = current_seg['speaker']
                # Якщо поточний сегмент значно більший, він правильний
                elif current_word_count >= next_word_count * 1.2 or current_duration >= next_duration * 1.2:
                    correct_speaker = current_seg['speaker']
                # Якщо наступний сегмент значно більший, він правильний
                elif next_word_count > current_word_count * 1.5:
                    correct_speaker = next_seg['speaker']
                # За замовчуванням - той, хто почав фразу
                else:
                    correct_speaker = current_seg['speaker']
                # Об'єднуємо сегменти
                merged_seg = {
                    'speaker': correct_speaker,
                    'start': current_seg['start'],
                    'end': next_seg['end'],
                    'text': (current_text + ' ' + next_text).strip()
                }
                fixed_segments.append(merged_seg)
                print(f"  🔧 Merged: '{current_text[:40]}...' + '{next_text[:40]}...' → Speaker {correct_speaker} (was {current_seg['speaker']} + {next_seg['speaker']})")
                # Пропускаємо наступний сегмент
                i += 2
                continue
        # Якщо не об'єднували, додаємо поточний сегмент як є
        fixed_segments.append(current_seg)
        i += 1
    if len(fixed_segments) < len(combined_segments):
        print(f"✅ Advanced fix: {len(combined_segments)} → {len(fixed_segments)} segments")
    else:
        print(f"✅ No advanced fixes needed: {len(fixed_segments)} segments")
    return fixed_segments


def fix_diarization_errors(combined_segments):
    """
    Виправляє помилки діаризації, об'єднуючи сегменти, які насправді належать одному спікеру.
    Виявляє та об'єднує сегменти, де:
    - Попередній сегмент закінчується неповною фразою
    - Наступний сегмент є продовженням попередньої фрази
    - Між ними коротка пауза (< 1.5 сек)
    - Різні спікери (помилка діаризації)
    Args:
        combined_segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
    Returns:
        fixed_segments: виправлений список сегментів
    """
    if not combined_segments or len(combined_segments) < 2:
        return combined_segments
    print(f"🔧 Fixing diarization errors in {len(combined_segments)} segments...")
    # Граматичні маркери неповних фраз (англійська)
    incomplete_phrase_markers = [
        ' to ', ' to', ' and ', ' and', ' or ', ' or', ' but ', ' but',
        ' did you', ' can you', ' will you', ' would you', ' could you', ' should you',
        ' try to', ' want to', ' need to', ' have to', ' going to', ' supposed to',
        ' if ', ' when ', ' where ', ' what ', ' who ', ' which ', ' how ',
        ' that ', ' this ', ' these ', ' those ',
        ' because ', ' although ', ' however ', ' therefore '
    ]
    # Маркери продовження (початок наступного сегмента)
    continuation_markers = [
        'reset', 'open', 'close', 'check', 'try', 'do', 'make', 'get', 'set',
        'configure', 'connect', 'disconnect', 'restart', 'reboot', 'update'
    ]
    fixed_segments = []
    i = 0
    while i < len(combined_segments):
        current_seg = combined_segments[i]
        current_text = current_seg['text'].strip().lower()
        # Перевіряємо, чи поточний сегмент закінчується неповною фразою
        is_incomplete = False
        words = current_text.split()
        if len(words) > 0:
            # Перевіряємо останні слова фрази
            last_word = words[-1].rstrip('.,!?;:')
            last_two_words = ' '.join(words[-2:]).rstrip('.,!?;:') if len(words) >= 2 else ''
            last_three_words = ' '.join(words[-3:]).rstrip('.,!?;:') if len(words) >= 3 else ''
            # Перевіряємо граматичні маркери неповноти
            incomplete_endings = [
                'to', 'and', 'or', 'but',
                'did you', 'can you', 'will you', 'would you', 'could you', 'should you',
                'try to', 'want to', 'need to', 'have to', 'going to', 'supposed to',
                'if', 'when', 'where', 'what', 'who', 'which', 'how',
                'that', 'this', 'these', 'those',
                'because', 'although', 'however', 'therefore'
            ]
            # Перевіряємо останнє слово
            if last_word in incomplete_endings:
                is_incomplete = True
            # Перевіряємо останні два слова
            elif last_two_words in incomplete_endings:
                is_incomplete = True
            # Перевіряємо останні три слова
            elif last_three_words in incomplete_endings:
                is_incomplete = True
            # Перевіряємо, чи фраза закінчується на маркер з пробілом перед ним
            for marker in incomplete_phrase_markers:
                marker_clean = marker.strip()
                if current_text.endswith(marker_clean) or current_text.endswith(marker_clean + '.'):
                    is_incomplete = True
                    break
                # Перевіряємо, чи маркер в останніх словах
                if marker_clean in last_three_words or marker_clean in last_two_words:
                    is_incomplete = True
                    break
        # Перевіряємо наступний сегмент, якщо є
        if is_incomplete and i + 1 < len(combined_segments):
            next_seg = combined_segments[i + 1]
            next_text = next_seg['text'].strip().lower()
            # Перевіряємо умови для об'єднання:
            # 1. Різні спікери (помилка діаризації)
            different_speakers = current_seg['speaker'] != next_seg['speaker']
            # 2. Коротка пауза між сегментами (< 1.5 сек)
            pause = next_seg['start'] - current_seg['end']
            short_pause = pause < 1.5 and pause >= 0
            # 3. Наступний сегмент виглядає як продовження
            is_continuation = False
            next_words = next_text.split()
            if next_words:
                first_word = next_words[0].rstrip('.,!?;:').lower()
                # Перевіряємо, чи починається з дієслова (продовження)
                if first_word in continuation_markers:
                    is_continuation = True
                # Або якщо перше слово не з великої літери (продовження речення)
                elif not next_seg['text'][0].isupper() and len(next_words) < 10:
                    is_continuation = True
                # Або якщо разом утворюють граматично правильну фразу
                combined_text = (current_seg['text'] + ' ' + next_seg['text']).strip()
                # Перевіряємо, чи виглядає як одне речення (не дуже довга фраза, немає подвійних пробілів)
                if len(combined_text.split()) < 25 and '  ' not in combined_text:
                    # Додаткова перевірка: чи перше слово наступного сегмента логічно продовжує попередній
                    if first_word in continuation_markers or first_word in ['reset', 'open', 'close', 'check', 'try', 'do', 'make', 'get', 'set', 'configure', 'connect']:
                        is_continuation = True
            # 4. Перевіряємо, чи разом утворюють логічну фразу
            # (наприклад, "did you try to" + "reset" = "did you try to reset")
            if is_continuation and different_speakers and short_pause:
                # Визначаємо правильного спікера
                # Критерій 1: хто почав фразу (перший сегмент)
                # Критерій 2: більше слів
                # Критерій 3: більша тривалість
                current_word_count = len(current_text.split())
                next_word_count = len(next_text.split())
                current_duration = current_seg['end'] - current_seg['start']
                next_duration = next_seg['end'] - next_seg['start']
                # Визначаємо правильного спікера
                if current_word_count >= next_word_count and current_duration >= next_duration:
                    correct_speaker = current_seg['speaker']
                elif next_word_count > current_word_count * 1.5:  # Наступний значно більший
                    correct_speaker = next_seg['speaker']
                else:
                    # За замовчуванням - той, хто почав фразу
                    correct_speaker = current_seg['speaker']
                # Об'єднуємо сегменти
                merged_seg = {
                    'speaker': correct_speaker,
                    'start': current_seg['start'],
                    'end': next_seg['end'],
                    'text': (current_seg['text'] + ' ' + next_seg['text']).strip()
                }
                fixed_segments.append(merged_seg)
                print(f"🔧 Merged segments: '{current_seg['text'][:50]}...' + '{next_seg['text'][:50]}...' → Speaker {correct_speaker}")
                # Пропускаємо наступний сегмент, бо вже об'єднано
                i += 2
                continue
        # Якщо не об'єднували, додаємо поточний сегмент як є
        fixed_segments.append(current_seg)
        i += 1
    if len(fixed_segments) < len(combined_segments):
        print(f"✅ Fixed: {len(combined_segments)} → {len(fixed_segments)} segments")
    else:
        print(f"✅ No errors found, kept {len(fixed_segments)} segments")
    return fixed_segments


def allowed_file(filename):
    """Перевіряє, чи дозволений формат файлу"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health():
    """Перевірка стану сервера"""
    return jsonify({
        'status': 'ok',
        'speaker_model_loaded': speaker_model is not None,
        'whisper_model_loaded': whisper_model is not None
    })


def process_diarization_async(job_id, filepath, filename, num_speakers, language, segment_duration, overlap, include_transcription, use_separation):
    """Фонова функція для обробки діаризації"""
    try:
        with jobs_lock:
            jobs[job_id]['status'] = 'processing'
        
        print(f"📁 [Job {job_id}] Processing file: {filename}")
        print(f"🔀 [Job {job_id}] Use separation: {use_separation}")
        engine_env = (os.environ.get('TRANSCRIPTION_ENGINE') or '').strip().lower()
        if engine_env == 'speechmatics':
            print(f"🎙️ [Job {job_id}] Using Speechmatics for transcription + diarization")
            transcription, sm_segments, words = transcribe_audio_speechmatics(filepath, language)
            if not sm_segments:
                raise ValueError("Speechmatics returned no segments")

            diarization_segments = [
                {
                    'speaker': int(seg.get('speaker', 0) or 0),
                    'start': round(float(seg.get('start', 0) or 0), 2),
                    'end': round(float(seg.get('end', 0) or 0), 2)
                }
                for seg in sm_segments
            ]
            transcription_segments = [
                {
                    'start': round(float(seg.get('start', 0) or 0), 2),
                    'end': round(float(seg.get('end', 0) or 0), 2),
                    'text': (seg.get('text') or '').strip()
                }
                for seg in sm_segments
                if (seg.get('text') or '').strip()
            ]
            combined = [
                {
                    'speaker': int(seg.get('speaker', 0) or 0),
                    'start': round(float(seg.get('start', 0) or 0), 2),
                    'end': round(float(seg.get('end', 0) or 0), 2),
                    'text': (seg.get('text') or '').strip()
                }
                for seg in sm_segments
                if (seg.get('text') or '').strip()
            ]
            if combined:
                combined = normalize_speaker_order(combined)

            result = {
                'success': True,
                'diarization': {
                    'segments': diarization_segments,
                    'num_speakers': len(set(seg.get('speaker', 0) for seg in diarization_segments)) if diarization_segments else 0
                }
            }
            if include_transcription:
                result['transcription'] = {
                    'full_text': transcription or '',
                    'segments': transcription_segments or []
                }
            result['combined'] = {
                'segments': combined if combined else [],
                'num_speakers': len(set(seg.get('speaker', 0) for seg in combined)) if combined else 0,
                'num_segments': len(combined) if combined else 0,
                'llm_iterations': []
            }

            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

            with jobs_lock:
                jobs[job_id]['status'] = 'completed'
                jobs[job_id]['result'] = result

            print(f"✅ [Job {job_id}] Speechmatics processing complete!")
            return
        # Крок 1: Завжди виконуємо стандартну діаризацію спочатку
        print(f"🔍 [Job {job_id}] Step 1: Standard diarization...")
        embeddings, timestamps = extract_speaker_embeddings(
            filepath,
            segment_duration=segment_duration, 
            overlap=overlap
        )
        print(f"👥 [Job {job_id}] Performing standard diarization...")
        standard_diarization_segments = diarize_audio(embeddings, timestamps, num_speakers)
        print(f"✅ [Job {job_id}] Step 1 finished: Standard diarization completed")
        # Якщо використовуємо розділення спікерів
        if use_separation:
            print(f"🔀 [Job {job_id}] Step 1: Separating speakers...")
            separation_result = separate_speakers(filepath)
            if not separation_result.get('success'):
                with jobs_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = f"Separation failed: {separation_result.get('error', 'Unknown error')}"
                return
            
            # Діаризуємо кожен розділений трек окремо
            all_diarization_segments = []
            separation_output_dir = separation_result['output_dir']
            for speaker_info in separation_result['speakers']:
                speaker_path = speaker_info['path']
                speaker_name = speaker_info['name']
                speaker_index = speaker_info['index']
                print(f"🔍 [Job {job_id}] Processing {speaker_name}...")
                # Витягуємо ембеддинги для цього треку
                embeddings, timestamps = extract_speaker_embeddings(
                    speaker_path,
                    segment_duration=segment_duration,
                    overlap=overlap
                )
                if embeddings is not None and len(embeddings) > 0:
                    # Діаризуємо (для одного треку має бути один спікер, але перевіряємо)
                    track_segments = diarize_audio(embeddings, timestamps, num_speakers=1)
                    # Додаємо сегменти з правильним спікером
                    for seg in track_segments:
                        seg['speaker'] = speaker_index  # Використовуємо індекс з розділення
                        all_diarization_segments.append(seg)
                else:
                    print(f"⚠️  [Job {job_id}] No embeddings extracted for {speaker_name}")
            # Сортуємо всі сегменти за часом
            all_diarization_segments.sort(key=lambda x: x['start'])
            # Зливаємо сусідні сегменти одного спікера
            diarization_segments = []
            if all_diarization_segments:
                current_speaker = all_diarization_segments[0]['speaker']
                current_start = all_diarization_segments[0]['start']
                prev_end = all_diarization_segments[0]['end']
                for seg in all_diarization_segments[1:]:
                    if seg['speaker'] != current_speaker:
                        # Зберігаємо попередній сегмент
                        diarization_segments.append({
                            'speaker': current_speaker,
                            'start': round(current_start, 2),
                            'end': round(prev_end, 2)
                        })
                        # Починаємо новий сегмент
                        current_speaker = seg['speaker']
                        current_start = seg['start']
                    prev_end = seg['end']
                # Додаємо останній сегмент
                diarization_segments.append({
                    'speaker': current_speaker,
                    'start': round(current_start, 2),
                    'end': round(prev_end, 2)
                })
            print(f"✅ [Job {job_id}] Combined diarization from {len(separation_result['speakers'])} separated tracks: {len(diarization_segments)} segments")
            # Очищаємо тимчасові файли розділення
            try:
                import shutil
                if os.path.exists(separation_output_dir):
                    shutil.rmtree(separation_output_dir)
            except Exception as e:
                print(f"⚠️  [Job {job_id}] Could not clean up separation directory: {e}")
        else:
            # Використовуємо результати стандартної діаризації
            diarization_segments = standard_diarization_segments
        
        result = {
            'success': True,
            'diarization': {
                'segments': diarization_segments,
                'num_speakers': len(set(seg.get('speaker', 0) for seg in diarization_segments)) if diarization_segments else 0
            }
        }
        # Додаємо транскрипцію, якщо потрібно
        print(f"📝 [Job {job_id}] Include transcription: {include_transcription}")
        if include_transcription:
            print(f"📝 [Job {job_id}] Transcribing audio...")
            try:
                transcription, transcription_segments, words = transcribe_audio(filepath, language)
                print(f"📊 [Job {job_id}] Transcription result: transcription={bool(transcription)}, segments={len(transcription_segments) if transcription_segments else 0}, words={len(words) if words else 0}")
                if words and len(words) > 0:
                    print(f"✅ [Job {job_id}] Transcription completed: {len(words)} words extracted")
                else:
                    print(f"⚠️  [Job {job_id}] Warning: No words extracted from transcription")
            except Exception as e:
                print(f"❌ [Job {job_id}] Error in transcribe_audio: {e}")
                import traceback
                traceback.print_exc()
                transcription, transcription_segments, words = None, [], []
            
            result['transcription'] = {
                'full_text': transcription or '',
                'segments': transcription_segments or []
            }
            # Об'єднуємо діаризацію та транскрипцію
            if transcription_segments:
                print(f"🔗 [Job {job_id}] Assigning speakers to transcription segments...")
                print(f"📊 [Job {job_id}] Input: {len(diarization_segments)} diarization segments, {len(transcription_segments)} transcription segments")
                combined = assign_speakers_to_transcription_segments(
                    diarization_segments,
                    transcription_segments
                )
                if combined:
                    combined = normalize_speaker_order(combined)
                result['combined'] = {
                    'segments': combined if combined else [],
                    'num_speakers': len(set(seg.get('speaker', 0) for seg in combined)) if combined else 0,
                    'num_segments': len(combined) if combined else 0,
                    'llm_iterations': []
                }
                print(f"✅ [Job {job_id}] Combined result prepared (ASR segments): {len(combined) if combined else 0} segments")
            elif words and len(words) > 0:
                print(f"🔗 [Job {job_id}] Combining diarization and transcription (word-level fallback)...")
                print(f"📊 [Job {job_id}] Input: {len(diarization_segments)} diarization segments, {len(words)} words")
                try:
                    # Скидаємо кеш перед обробкою
                    global _llm_iterations_cache
                    _llm_iterations_cache = []
                    print(f"🔄 [Job {job_id}] Reset LLM iterations cache")
                    combined = combine_diarization_and_transcription(
                        diarization_segments,
                        words
                    )
                    print(f"📊 [Job {job_id}] After combine_diarization_and_transcription: {len(combined) if combined else 0} segments")
                    # Отримуємо llm_iterations з кешу (якщо він є)
                    llm_iterations = _llm_iterations_cache if '_llm_iterations_cache' in globals() else []
                    print(f"📊 [Job {job_id}] LLM iterations from cache: {len(llm_iterations)}")
                    result['combined'] = {
                        'segments': combined if combined else [],
                        'num_speakers': len(set(seg.get('speaker', 0) for seg in combined)) if combined else 0,
                        'num_segments': len(combined) if combined else 0,
                        'llm_iterations': llm_iterations
                    }
                    print(f"✅ [Job {job_id}] Combined result prepared: {len(combined) if combined else 0} segments, {len(llm_iterations)} LLM iterations")
                except Exception as e:
                    print(f"❌ [Job {job_id}] Error in combine_diarization_and_transcription: {e}")
                    import traceback
                    traceback.print_exc()
                    result['combined'] = {
                        'segments': [],
                        'num_speakers': 0,
                        'num_segments': 0,
                        'llm_iterations': []
                    }
            else:
                print(f"⚠️  [Job {job_id}] Warning: Cannot combine - no transcription segments or words available")
                result['combined'] = {
                    'segments': [],
                    'num_speakers': 0,
                    'num_segments': 0,
                    'llm_iterations': []
                }
        # Видаляємо тимчасовий файл
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        
        # Зберігаємо результат
        with jobs_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = result
        
        print(f"✅ [Job {job_id}] Processing complete!")
    except Exception as e:
        print(f"❌ [Job {job_id}] Error in process_diarization_async: {e}")
        import traceback
        traceback.print_exc()
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
        # Видаляємо тимчасовий файл у разі помилки
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


@app.route('/api/diarize', methods=['POST', 'OPTIONS'])
def api_diarize():
    """API ендпоінт для діаризації та транскрипції (асинхронний)"""
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    # Генеруємо job_id ДО try блоку
    job_id = str(uuid.uuid4())
    filepath = None
    
    try:
        filename = None
        
        # Перевіряємо Content-Type для визначення формату запиту
        content_type = request.content_type or ''
        
        # Обробка JSON запитів (base64 файл)
        if 'application/json' in content_type:
            data = request.get_json() or {}
            print(f"📥 JSON request received. Keys: {list(data.keys())}")
            
            if 'file' not in data:
                return jsonify({'success': False, 'error': 'No file uploaded in JSON'}), 400
            
            file_base64 = data.get('file', '')
            filename = data.get('filename', 'audio.wav')
            mode = data.get('mode', 'fast')
            
            if not file_base64:
                return jsonify({'success': False, 'error': 'File data is empty'}), 400
            
            # Декодуємо base64
            try:
                import base64
                # Видаляємо data URI префікс якщо є
                if ',' in file_base64:
                    file_base64 = file_base64.split(',', 1)[1]
                
                # Очищаємо base64
                file_base64 = file_base64.replace('\n', '').replace('\r', '').replace(' ', '')
                
                # Конвертуємо base64url в стандартний base64
                file_base64 = file_base64.replace('-', '+').replace('_', '/')
                
                # Додаємо padding якщо потрібно
                missing_padding = len(file_base64) % 4
                if missing_padding:
                    file_base64 += '=' * (4 - missing_padding)
                
                audio_data = base64.b64decode(file_base64)
                print(f"✅ Decoded base64: {len(audio_data)} bytes")
            except Exception as e:
                print(f"❌ Base64 decode error: {e}")
                return jsonify({'success': False, 'error': f'Invalid base64 data: {str(e)}'}), 400
            
            # Отримуємо параметри з JSON
            num_speakers = data.get('num_speakers', type=int) if 'num_speakers' in data else None
            language = (data.get('language') or '').strip().lower()
            if not language or language == 'auto':
                language = 'en'
            segment_duration = float(data.get('segment_duration', 1.5))
            overlap = float(data.get('overlap', 0.5))
            include_transcription = data.get('include_transcription', True)
            use_separation = data.get('use_separation', False)
            
            # Створюємо завдання ДО декодування файлу
            with jobs_lock:
                jobs[job_id] = {
                    'status': 'pending',
                    'result': None,
                    'error': None,
                    'created_at': datetime.now(),
                    'include_transcription': include_transcription
                }
            
            print(f"✅ [Job {job_id}] Job created, returning job_id IMMEDIATELY")
            
            # Повертаємо job_id ОДРАЗУ (ДО декодування base64!)
            response = jsonify({
                'success': True,
                'job_id': job_id,
                'status': 'pending',
                'message': 'Processing started. Use GET /api/diarize/{job_id}/status to check progress.'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            
            # Декодуємо base64 та зберігаємо файл в фоні ПОСЛЯ відправки відповіді
            def decode_and_process():
                try:
                    import base64
                    # Видаляємо data URI префікс якщо є
                    file_base64_clean = file_base64
                    if ',' in file_base64_clean:
                        file_base64_clean = file_base64_clean.split(',', 1)[1]
                    
                    # Очищаємо base64
                    file_base64_clean = file_base64_clean.replace('\n', '').replace('\r', '').replace(' ', '')
                    
                    # Конвертуємо base64url в стандартний base64
                    file_base64_clean = file_base64_clean.replace('-', '+').replace('_', '/')
                    
                    # Додаємо padding якщо потрібно
                    missing_padding = len(file_base64_clean) % 4
                    if missing_padding:
                        file_base64_clean += '=' * (4 - missing_padding)
                    
                    audio_data = base64.b64decode(file_base64_clean)
                    print(f"✅ [Job {job_id}] Decoded base64: {len(audio_data)} bytes")
                    
                    # Зберігаємо файл тимчасово
                    filename_clean = secure_filename(filename)
                    filepath_local = os.path.join(UPLOAD_FOLDER, filename_clean)
                    with open(filepath_local, 'wb') as f:
                        f.write(audio_data)
                    print(f"💾 [Job {job_id}] Saved file: {filepath_local} ({len(audio_data)} bytes)")
                    
                    # Запускаємо обробку
                    process_diarization_async(job_id, filepath_local, filename_clean, num_speakers, language, segment_duration, overlap, include_transcription, use_separation)
                except Exception as e:
                    print(f"❌ [Job {job_id}] Error in decode_and_process: {e}")
                    import traceback
                    traceback.print_exc()
                    with jobs_lock:
                        jobs[job_id]['status'] = 'failed'
                        jobs[job_id]['error'] = str(e)
            
            # Запускаємо в окремому потоці
            thread = threading.Thread(target=decode_and_process)
            thread.daemon = True
            thread.start()
            
            return response
            
        # Обробка multipart/form-data запитів (legacy, синхронна)
        elif 'multipart/form-data' in content_type:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Отримуємо параметри з form
            num_speakers = request.form.get('num_speakers', type=int)
            language = (request.form.get('language', type=str) or '').strip().lower()
            if not language or language == 'auto':
                language = 'en'
            segment_duration = float(request.form.get('segment_duration', 1.5))
            overlap = float(request.form.get('overlap', 0.5))
            include_transcription = request.form.get('include_transcription', 'true').lower() == 'true'
            use_separation = request.form.get('use_separation', 'false').lower() == 'true'
            
            # Створюємо завдання
            with jobs_lock:
                jobs[job_id] = {
                    'status': 'pending',
                    'result': None,
                    'error': None,
                    'created_at': datetime.now(),
                    'include_transcription': include_transcription
                }
            
            # Повертаємо job_id ОДРАЗУ
            response = jsonify({
                'success': True,
                'job_id': job_id,
                'status': 'pending',
                'message': 'Processing started. Use GET /api/diarize/{job_id}/status to check progress.'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            
            # Запускаємо обробку в фоні
            def process_multipart():
                try:
                    process_diarization_async(job_id, filepath, filename, num_speakers, language, segment_duration, overlap, include_transcription, use_separation)
                except Exception as e:
                    print(f"❌ [Job {job_id}] Error in process_multipart: {e}")
                    import traceback
                    traceback.print_exc()
                    with jobs_lock:
                        jobs[job_id]['status'] = 'failed'
                        jobs[job_id]['error'] = str(e)
            
            thread = threading.Thread(target=process_multipart)
            thread.daemon = True
            thread.start()
            
            return response
        else:
            return jsonify({'success': False, 'error': f'Unsupported Content-Type: {content_type}. Expected application/json or multipart/form-data'}), 400
    except Exception as e:
        print(f"❌ [Job {job_id}] Error in api_diarize: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"📋 Full traceback:\n{error_traceback}")
        traceback.print_exc()
        # Видаляємо тимчасовий файл у разі помилки
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as cleanup_error:
            print(f"⚠️  Could not clean up file: {cleanup_error}")
        # Оновлюємо статус завдання
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = str(e)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback if app.debug else None
        }), 500


@app.route('/api/diarize/<job_id>/status', methods=['GET', 'OPTIONS'])
def get_diarize_status(job_id):
    """Перевірка статусу завдання діаризації"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found',
                'code': 'JOB_NOT_FOUND'
            }), 404
        
        job = jobs[job_id]
        
        if job['status'] == 'completed':
            result = job['result']
            response = jsonify({
                'success': True,
                'status': 'completed',
                **result
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        elif job['status'] == 'failed':
            response = jsonify({
                'success': False,
                'status': 'failed',
                'error': job.get('error', 'Unknown error')
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        else:
            response = jsonify({
                'success': True,
                'status': job['status'],
                'message': 'Processing in progress...'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200


def remove_filler_words(text):
    """
    Видаляє filler words (Uh., Um.) з тексту як окремі слова, не частини інших слів.
    
    Args:
        text: текст для очищення
    
    Returns:
        очищений текст без filler words
    """
    import re
    # Видаляємо "Uh." та "Um." як окремі слова (з word boundaries)
    # Також обробляємо варіанти з пробілами та пунктуацією
    # \b - word boundary, щоб не видаляти частини інших слів
    text = re.sub(r'\bUh\.\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUm\.\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUh\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUm\s+', '', text, flags=re.IGNORECASE)
    # Видаляємо подвійні пробіли, які могли залишитися
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_dialogue_from_segments(segments):
    """
    Форматує сегменти діалогу у читабельний текст з таймстемпами та спікерами.
    Формат кожної репліки в одному рядку:
        MM:SS Speaker X: [text]
    
    Між репліками - перенос рядка (\n) для зручного розбиття в Shortcut.
    
    Args:
        segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
    
    Returns:
        formatted_text: відформатований діалог, де кожна репліка в одному рядку
    """
    if not segments:
        return "Error: No dialogue segments found"
    
    formatted_replicas = []
    
    for seg in segments:
        # Конвертуємо час з секунд у MM:SS
        start_time = seg.get('start', 0)
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Отримуємо спікера та текст
        speaker = seg.get('speaker', 0)
        text = seg.get('text', '').strip()
        
        if not text:
            continue
        
        # Видаляємо filler words (Uh., Um.) перед форматуванням
        text = remove_filler_words(text)
        
        if not text:  # Якщо після видалення filler words текст став порожнім
            continue
        
        # Форматуємо одну репліку в одному рядку: MM:SS Speaker X: [text]
        replica = f"{time_str} Speaker {speaker}: {text}"
        formatted_replicas.append(replica)
    
    # Об'єднуємо всі репліки переносом рядка (для зручного розбиття в Shortcut)
    return "\n".join(formatted_replicas)


@app.route('/api/diarize/<job_id>/formatted', methods=['GET', 'OPTIONS'])
def get_diarize_formatted(job_id):
    """Отримує відформатований діалог у читабельному форматі"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found',
                'code': 'JOB_NOT_FOUND'
            }), 404
        
        job = jobs[job_id]
        
        if job['status'] == 'completed':
            result = job.get('result', {})
            combined = result.get('combined', {})
            segments = combined.get('segments', [])
            
            if not segments:
                return jsonify({
                    'success': False,
                    'error': 'No dialogue segments found in result',
                    'code': 'NO_SEGMENTS'
                }), 200
            
            # Форматуємо діалог
            formatted_dialogue = format_dialogue_from_segments(segments)
            
            response = jsonify({
                'success': True,
                'status': 'completed',
                'formatted_dialogue': formatted_dialogue
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        elif job['status'] == 'failed':
            response = jsonify({
                'success': False,
                'status': 'failed',
                'error': job.get('error', 'Unknown error'),
                'code': job.get('code', 'PROCESSING_ERROR')
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        else:
            response = jsonify({
                'success': True,
                'status': job['status'],
                'message': 'Processing in progress...'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200


def separate_speakers_with_speechbrain(audio_path, output_dir):
    """
    Розділяє спікерів за допомогою SpeechBrain SepformerSeparation.
    Використовує той самий підхід, що і в speechbrain_separation.py для якісної нарізки.
    
    Args:
        audio_path: шлях до оригінального аудіо файлу
        output_dir: директорія для збереження розділених файлів
    
    Returns:
        dict: {
            'success': bool,
            'speaker_files': {speaker_id: {'path': str, 'speaker_label': str}},
            'error': str (якщо помилка)
        }
    """
    import sys
    
    try:
        # Імпортуємо необхідні бібліотеки
        try:
            import pyannote_patch  # noqa: F401
            from speechbrain.pretrained import SepformerSeparation as Separator
            import torch
            import torchaudio
        except ImportError as e:
            print(f"⚠️ SpeechBrain separation not available: {e}, falling back to simple extraction")
            sys.stdout.flush()
            return {'success': False, 'error': f'SpeechBrain separation not available: {e}'}
        
        # Визначаємо device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"🔀 [SpeechBrain] Using device: {device}")
        sys.stdout.flush()
        
        # Завантажуємо модель
        cache_dir = os.path.expanduser(
            os.getenv("SPEECHBRAIN_CACHE_DIR", "~/.cache/speechbrain/sepformer-wsj02mix")
        )
        
        print(f"📦 [SpeechBrain] Loading sepformer-wsj02mix model...")
        sys.stdout.flush()
        
        try:
            # Використовуємо той самий підхід, що і в separate_speakers
            model = Separator.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir="pretrained_models/sepformer-wsj02mix",
                run_opts={"device": device},
            )
            print(f"✅ [SpeechBrain] Model loaded successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ [SpeechBrain] Failed to load model: {e}, falling back to simple extraction")
            sys.stdout.flush()
            return {'success': False, 'error': f'Failed to load model: {e}'}
        
        # Завантажуємо аудіо через librosa (підтримує більше форматів, включаючи m4a)
        try:
            # Використовуємо librosa для завантаження (підтримує m4a, mp3, тощо)
            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            # Конвертуємо в torch tensor
            if len(audio_data.shape) == 1:
                # Mono audio - додаємо вимір каналу
                waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
            else:
                # Multi-channel audio - shape [channels, samples]
                waveform = torch.from_numpy(audio_data).float()
            
            print(f"✅ [SpeechBrain] Loaded via librosa: shape={waveform.shape}, sr={sample_rate}")
            sys.stdout.flush()
        except Exception as load_error:
            print(f"❌ [SpeechBrain] Audio loading failed with librosa: {load_error}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return {'success': False, 'error': f'Audio loading failed: {load_error}'}
        
        # Конвертуємо в mono якщо потрібно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample до 8kHz (SpeechBrain вимагає 8kHz)
        if sample_rate != 8000:
            print(f"🔄 [SpeechBrain] Resampling from {sample_rate}Hz to 8000Hz")
            sys.stdout.flush()
            resampler = torchaudio.transforms.Resample(sample_rate, 8000)
            waveform = resampler(waveform)
            sample_rate = 8000
        
        total_samples = waveform.shape[1]
        
        # Налаштування chunking (як в speechbrain_separation.py)
        max_chunk_seconds = float(os.getenv("SPEECHBRAIN_CHUNK_SECONDS", "30"))
        max_chunk_samples = int(max_chunk_seconds * sample_rate)
        max_chunk_samples = max(max_chunk_samples, sample_rate * 5)  # мінімум 5 секунд
        
        print(f"🔍 [SpeechBrain] Waveform shape: {waveform.shape}, chunk size: {max_chunk_samples} samples")
        sys.stdout.flush()
        
        # Функція для обробки chunk
        def separate_chunk(chunk_tensor: torch.Tensor):
            chunk_tensor = chunk_tensor.to(device)
            with torch.no_grad():
                result = model.separate_batch(chunk_tensor)
            return result.cpu()
        
        # Запускаємо separation з chunking для довгих файлів
        print(f"🔄 [SpeechBrain] Running speaker separation...")
        sys.stdout.flush()
        
        if total_samples > max_chunk_samples:
            print(f"📦 [SpeechBrain] Processing in chunks (total: {total_samples} samples)")
            sys.stdout.flush()
            chunk_outputs = []
            for start in range(0, total_samples, max_chunk_samples):
                end = min(start + max_chunk_samples, total_samples)
                print(f"   🔄 [SpeechBrain] Separating chunk {start}:{end} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
                sys.stdout.flush()
                chunk = waveform[:, start:end]
                chunk_outputs.append(separate_chunk(chunk))
            est_sources = torch.cat(chunk_outputs, dim=1)
        else:
            waveform = waveform.to(device)
            est_sources = separate_chunk(waveform)
        
        # Обробляємо результат (як в speechbrain_separation.py)
        if est_sources.dim() == 3:
            est_sources = est_sources[0]  # [time, num_speakers]
        
        if est_sources.dim() == 2:
            if est_sources.shape[0] == model.hparams.num_spks:
                # shape [num_speakers, time]
                sources_tensor = est_sources
            elif est_sources.shape[1] == model.hparams.num_spks:
                sources_tensor = est_sources.transpose(0, 1)
            else:
                raise ValueError(f"Unexpected est_sources shape: {est_sources.shape}")
        else:
            raise ValueError(f"Unsupported est_sources dimension: {est_sources.dim()}")
        
        sources_tensor = sources_tensor.cpu()
        
        num_speakers = sources_tensor.shape[0]
        print(f"✅ [SpeechBrain] Found {num_speakers} speakers")
        sys.stdout.flush()
        
        # Застосовуємо сильне приглушення слабких сигналів
        print(f"🔇 [SpeechBrain] Applying noise gate to suppress weak signals...")
        sys.stdout.flush()
        
        def apply_noise_gate(audio_tensor, threshold=0.05, ratio=10.0, attack=0.01, release=0.1):
            """Застосовує noise gate для приглушення слабких сигналів."""
            # Конвертуємо в numpy якщо потрібно
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.numpy()
            else:
                audio_np = audio_tensor
            
            # Обчислюємо енергію сигналу (RMS)
            if len(audio_np.shape) == 1:
                # Mono
                energy = np.abs(audio_np)
            else:
                # Multi-channel - беремо середнє
                energy = np.abs(audio_np).mean(axis=0)
            
            # Обчислюємо RMS в склянному вікні (для плавності)
            window_size = int(sample_rate * 0.05)  # 50ms вікно
            if window_size < 1:
                window_size = 1
            
            # Обчислюємо RMS
            rms = np.sqrt(np.convolve(energy ** 2, np.ones(window_size) / window_size, mode='same'))
            rms_normalized = rms / (np.max(rms) + 1e-8)  # Нормалізуємо до 0-1
            
            # Створюємо gate mask
            gate_mask = np.ones_like(rms_normalized)
            
            # Застосовуємо threshold
            below_threshold = rms_normalized < threshold
            gate_mask[below_threshold] = 1.0 / ratio  # Сильне приглушення слабких сигналів
            
            # Плавні переходи (attack/release)
            attack_samples = int(sample_rate * attack)
            release_samples = int(sample_rate * release)
            
            # Застосовуємо плавні переходи
            smoothed_mask = np.copy(gate_mask)
            for i in range(1, len(gate_mask)):
                if gate_mask[i] > gate_mask[i-1]:
                    # Attack - швидко відкриваємо
                    start = max(0, i - attack_samples)
                    smoothed_mask[start:i] = np.linspace(gate_mask[i-1], gate_mask[i], i - start)
                elif gate_mask[i] < gate_mask[i-1]:
                    # Release - повільно закриваємо
                    end = min(len(gate_mask), i + release_samples)
                    smoothed_mask[i:end] = np.linspace(gate_mask[i], gate_mask[i-1], end - i)
            
            gate_mask = smoothed_mask
            
            # Застосовуємо mask до аудіо
            if len(audio_np.shape) == 1:
                gated_audio = audio_np * gate_mask
            else:
                gated_audio = audio_np * gate_mask[np.newaxis, :]
            
            return gated_audio
        
        # Застосовуємо noise gate до кожного спікера
        gated_sources = []
        for idx in range(num_speakers):
            source_tensor = sources_tensor[idx]
            source_np = source_tensor.squeeze().numpy()
            
            # Застосовуємо noise gate з сильним приглушенням
            gated_audio = apply_noise_gate(
                source_np, 
                threshold=0.15,  # Поріг 15% від максимуму
                ratio=20.0,  # Сильне приглушення (20:1)
                attack=0.01,  # Швидка атака
                release=0.1  # Повільне відпускання
            )
            
            gated_sources.append(gated_audio)
        
        print(f"✅ [SpeechBrain] Noise gate applied (threshold=0.15, ratio=20:1)")
        sys.stdout.flush()
        
        # Створюємо output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Зберігаємо файли для кожного спікера
        speaker_files = {}
        for idx in range(num_speakers):
            speaker_id = idx
            speaker_name = f"SPEAKER_{idx:02d}"
            output_path = os.path.join(output_dir, f"speaker_{speaker_id}.wav")
            
            gated_audio = gated_sources[idx]
            sf.write(output_path, gated_audio, sample_rate)
            
            speaker_files[speaker_id] = {
                'path': output_path,
                'speaker_label': speaker_name
            }
            
            duration = len(gated_audio) / sample_rate
            print(f"✅ [SpeechBrain] Saved speaker {speaker_id} ({speaker_name}): {duration:.2f}s")
            sys.stdout.flush()
        
        return {
            'success': True,
            'speaker_files': speaker_files,
            'speaker_map': {f"SPEAKER_{i:02d}": i for i in range(num_speakers)}
        }
        
    except Exception as e:
        print(f"❌ [SpeechBrain] Error in separation: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return {'success': False, 'error': str(e)}


@app.route('/api/separate-audio', methods=['POST', 'OPTIONS'])
def api_separate_audio():
    """
    Новий ендпоїнт для розділення аудіо на окремі голоси.
    Приймає аудіо файл, розбиває його на два голоси за допомогою SpeechBrain separation,
    і повертає два аудіо треки в JSON з полями file1, file2.
    
    Returns:
        JSON з полями:
        - file1: base64 encoded аудіо або URL до файлу
        - file2: base64 encoded аудіо або URL до файлу
        - success: bool
    """
    import sys
    import base64
    import uuid
    import shutil
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Перевіряємо, чи є файл
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided',
                'code': 'NO_FILE'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        print(f"🎵 [Separate Audio] Received file: {audio_file.filename}")
        sys.stdout.flush()
        
        # Зберігаємо тимчасовий файл
        job_id = str(uuid.uuid4())
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_filename = f"separate_{job_id}{file_extension}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        audio_file.save(temp_path)
        
        print(f"💾 [Separate Audio] Saved to: {temp_path}")
        sys.stdout.flush()

        # Створюємо тимчасову директорію для розділених файлів
        output_dir = os.path.join(UPLOAD_FOLDER, f"separated_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Виконуємо розділення за допомогою SpeechBrain
        print(f"🔀 [Separate Audio] Starting SpeechBrain separation...")
        sys.stdout.flush()
        
        separation_result = separate_speakers_with_speechbrain(temp_path, output_dir)
        
        if not separation_result.get('success'):
            # Видаляємо тимчасовий файл
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return jsonify({
                'success': False,
                'error': separation_result.get('error', 'Separation failed'),
                'code': 'SEPARATION_FAILED'
            }), 500
        
        speaker_files = separation_result['speaker_files']
        
        # Перевіряємо, чи є принаймні два спікери
        if len(speaker_files) < 2:
            # Видаляємо тимчасові файли
            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return jsonify({
                'success': False,
                'error': f'Found only {len(speaker_files)} speaker(s), need at least 2',
                'code': 'INSUFFICIENT_SPEAKERS'
            }), 400
        
        # Беремо перші два спікери
        speaker_ids = sorted(speaker_files.keys())[:2]
        speaker_0_file = speaker_files[speaker_ids[0]]['path']
        speaker_1_file = speaker_files[speaker_ids[1]]['path']
        
        print(f"✅ [Separate Audio] Separation completed: speaker {speaker_ids[0]} and {speaker_ids[1]}")
        sys.stdout.flush()
        
        # Видаляємо тимчасовий оригінальний файл
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        # Створюємо URL-и для завантаження файлів
        base_url = request.host_url.rstrip('/')
        file1_url = f"{base_url}/api/separate-audio-file/{job_id}/0"
        file2_url = f"{base_url}/api/separate-audio-file/{job_id}/1"
        
        # Переміщуємо файли в постійну директорію для завантаження
        download_dir = os.path.join(UPLOAD_FOLDER, 'separated_audio')
        os.makedirs(download_dir, exist_ok=True)
        
        file1_download_path = os.path.join(download_dir, f"{job_id}_speaker_0.wav")
        file2_download_path = os.path.join(download_dir, f"{job_id}_speaker_1.wav")
        
        shutil.copy2(speaker_0_file, file1_download_path)
        shutil.copy2(speaker_1_file, file2_download_path)
        
        # Видаляємо тимчасову директорію з розділеними файлами
        try:
            shutil.rmtree(output_dir)
        except:
            pass
        
        # Повертаємо результат з URL-ами
        response_data = {
            'success': True,
            'file1': file1_url,
            'file2': file2_url
        }
        
        print(f"📤 [Separate Audio] Returning separated audio files")
        sys.stdout.flush()
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
        
    except Exception as e:
        print(f"❌ [Separate Audio] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/api/separate-audio-file/<job_id>/<int:speaker_id>', methods=['GET', 'OPTIONS'])
def get_separate_audio_file(job_id, speaker_id):
    """
    Ендпоїнт для завантаження розділеного аудіо файлу.
    Після завантаження файл автоматично видаляється.
    
    Args:
        job_id: ID завдання розділення
        speaker_id: ID спікера (0 або 1)
    """
    import sys
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    try:
        # Знаходимо файл
        download_dir = os.path.join(UPLOAD_FOLDER, 'separated_audio')
        download_filename = f"{job_id}_speaker_{speaker_id}.wav"
        download_path = os.path.join(download_dir, download_filename)
        
        # Перевіряємо, чи файл існує
        if not os.path.exists(download_path):
            print(f"❌ [Separate Audio Download] File not found: {download_path}")
            sys.stdout.flush()
            return jsonify({
                'success': False,
                'error': f'Audio file for speaker {speaker_id} not found',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        print(f"📥 [Separate Audio Download] Serving file: {download_path} for job {job_id}, speaker {speaker_id}")
        sys.stdout.flush()
        
        # Відправляємо файл
        response = send_file(
            download_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"speaker_{speaker_id}.wav"
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        # Видаляємо файл після відправки (в фоні, щоб не блокувати відповідь)
        def delete_file_after_delay():
            import time
            time.sleep(2)  # Затримка, щоб файл точно відправився
            try:
                if os.path.exists(download_path):
                    os.remove(download_path)
                    print(f"🗑️ [Separate Audio Download] Deleted file: {download_path}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ [Separate Audio Download] Failed to delete file {download_path}: {e}")
                sys.stdout.flush()
        
        thread = threading.Thread(target=delete_file_after_delay)
        thread.daemon = True
        thread.start()
        
        return response, 200
        
    except Exception as e:
        print(f"❌ [Separate Audio Download] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/process', methods=['POST', 'OPTIONS'])
def process_audio():
    """
    Основний ендпоінт для iOS Shortcuts.
    Повертає формат згідно специфікації.
    """
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS preflight request received from", request.remote_addr)
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    print(f"📥 POST /process request received from {request.remote_addr}")
    print(f"📋 Headers: {dict(request.headers)}")
    print(f"📦 Files in request: {list(request.files.keys())}")
    print(f"📝 Form data keys: {list(request.form.keys())}")
    filepath = None
    start_time = time.time()
    try:
        # Валідація файлу
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded',
                'code': 'NO_FILE'
            }), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'code': 'NO_FILE'
            }), 400
        # Перевірка формату файлу
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid audio format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'code': 'INVALID_FORMAT'
            }), 400
        # Зберігаємо файл тимчасово
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        # Перевірка розміру файлу
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB',
                'code': 'FILE_SIZE_EXCEEDED'
            }), 413
        # Отримуємо параметри
        num_speakers = request.form.get('num_speakers', type=int)
        language = (request.form.get('language', type=str) or '').strip().lower()
        if not language or language == 'auto':
            language = 'en'
        segment_duration = float(request.form.get('segment_duration', 1.5))
        overlap = float(request.form.get('overlap', 0.5))
        print(f"📁 Processing file: {filename} ({file_size / (1024*1024):.2f} MB)")
        # Обчислюємо тривалість аудіо
        try:
            audio_duration = librosa.get_duration(path=filepath)
            print(f"⏱️  Audio duration: {audio_duration:.2f} seconds")
        except Exception as e:
            print(f"⚠️  Could not determine audio duration: {e}")
            audio_duration = 0
        # Витягуємо ембеддинги та виконуємо діаризацію
        print("🔍 Extracting speaker embeddings...")
        embeddings, timestamps = extract_speaker_embeddings(
            filepath, 
            segment_duration=segment_duration, 
            overlap=overlap
        )
        if embeddings is None:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': 'Failed to extract speaker embeddings. Audio may be corrupted or unsupported format.',
                'code': 'PROCESSING_ERROR'
            }), 500
        if len(embeddings) == 0:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'Audio too short (duration: {audio_duration:.2f}s). Minimum recommended: 2 seconds.',
                'code': 'PROCESSING_ERROR'
            }), 500
        print("👥 Performing diarization...")
        diarization_segments = diarize_audio(embeddings, timestamps, num_speakers)
        if not diarization_segments:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': 'Diarization failed. Could not identify speakers.',
                'code': 'PROCESSING_ERROR'
            }), 500
        # Транскрибуємо аудіо
        print("📝 Transcribing audio...")
        transcription, transcription_segments, words = transcribe_audio(filepath, language)
        if not transcription or not words:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': 'Transcription failed. Could not transcribe audio.',
                'code': 'PROCESSING_ERROR'
            }), 500
        # Об'єднуємо діаризацію та транскрипцію
        print("🔗 Combining diarization and transcription...")
        combined_segments = combine_diarization_and_transcription(
            diarization_segments, 
            words
        )
        # Виправляємо помилки діаризації (об'єднуємо сегменти, які насправді належать одному спікеру)
        # ТИМЧАСОВО ВИМКНЕНО - потребує доопрацювання
        # if combined_segments and len(combined_segments) > 0:
        #     combined_segments = fix_diarization_errors(combined_segments)
        # Перевірка таймауту
        processing_time = time.time() - start_time
        if processing_time > PROCESSING_TIMEOUT:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': 'Processing timeout',
                'code': 'TIMEOUT'
            }), 408
        # Формуємо відповідь згідно специфікації
        result = {
            'success': True,
            'duration': round(audio_duration, 2),
            'full_text': transcription,
            'segments': [
                {
                    'speaker': seg['speaker'],
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                }
                for seg in combined_segments
            ]
        }
        # Видаляємо тимчасовий файл
        try:
            os.remove(filepath)
        except:
            pass
        processing_time = time.time() - start_time
        print(f"✅ Processing complete! Time: {processing_time:.2f}s")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error in process_audio: {e}")
        import traceback
        traceback.print_exc()
        # Видаляємо тимчасовий файл у разі помилки
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/api/diarize-and-transcribe', methods=['POST', 'OPTIONS'])
def api_diarize_and_transcribe():
    """
    Ендпоїнт для діаризації та транскрипції аудіо файлу.
    Приймає аудіо файл, виконує діаризацію та транскрипцію,
    повертає транскрипт з розділенням по спікерам.
    
    Returns:
        JSON з полями:
        - success: bool
        - transcript: список рядків у форматі "Timestamp - Speaker number - Utterance"
    """
    import sys
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Перевіряємо, чи є файл
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided',
                'code': 'NO_FILE'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        # Отримуємо параметри (опціонально)
        processing_mode = request.form.get('mode', 'fast')  # 'smart' або 'fast'
        transcriber_override = (request.form.get('transcriber') or '').strip().lower()
        if transcriber_override in ('speechmatics',):
            transcriber = 'speechmatics'
        elif transcriber_override in ('groq', 'whisper_cloud', 'cloud', 'whisper-large-v3-turbo'):
            transcriber = 'groq'
        elif transcriber_override in ('local', 'whisper_local'):
            transcriber = 'local'
        elif transcriber_override:
            return jsonify({
                'success': False,
                'error': f'Transcription engine "{transcriber_override}" is not supported by demo2 diarize-and-transcribe.',
                'code': 'UNSUPPORTED_TRANSCRIBER'
            }), 400
        else:
            engine_env = (os.environ.get('TRANSCRIPTION_ENGINE') or '').strip().lower()
            if engine_env in ('speechmatics',):
                transcriber = 'speechmatics'
            elif engine_env in ('whisper_cloud', 'groq', 'groq_whisper', 'whisper-large-v3-turbo'):
                transcriber = 'groq'
            elif engine_env in ('whisper_local', 'local', ''):
                transcriber_env = os.environ.get('DEMO2_TRANSCRIBER') or os.environ.get('DEMO2_TRANSCRIBERS') or 'local'
                transcriber = transcriber_env.split(',')[0].strip() if transcriber_env else 'local'
            elif engine_env in ('azure', 'azure_realtime', 'azure-realtime'):
                return jsonify({
                    'success': False,
                    'error': f'Transcription engine "{engine_env}" is not supported by demo2 diarize-and-transcribe.',
                    'code': 'UNSUPPORTED_TRANSCRIBER'
                }), 400
            else:
                transcriber_env = os.environ.get('DEMO2_TRANSCRIBER') or os.environ.get('DEMO2_TRANSCRIBERS') or 'local'
                transcriber = transcriber_env.split(',')[0].strip() if transcriber_env else 'local'
        language = 'en'
        num_speakers = request.form.get('num_speakers', None)
        if num_speakers:
            try:
                num_speakers = int(num_speakers)
            except:
                num_speakers = None
        
        print(f"🎵 [Diarize & Transcribe] Received file: {audio_file.filename}, mode: {processing_mode}, transcriber: {transcriber}")
        sys.stdout.flush()
        
        # Зберігаємо тимчасовий файл
        job_id = str(uuid.uuid4())
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_filename = f"diarize_{job_id}{file_extension}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        audio_file.save(temp_path)
        
        print(f"💾 [Diarize & Transcribe] Saved to: {temp_path}")
        sys.stdout.flush()

        try:
            # Крок 1: Виконуємо діаризацію
            print(f"🔍 [Diarize & Transcribe] Step 1: Performing speaker diarization...")
            sys.stdout.flush()
            
            # Використовуємо SpeechBrain для діаризації
            embeddings, timestamps = extract_speaker_embeddings(
                temp_path,
                segment_duration=1.5,
                overlap=0.5
            )
            
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Failed to extract speaker embeddings")
            
            # Виконуємо діаризацію
            diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=num_speakers)
            
            if not diarization_segments:
                raise ValueError("Diarization failed - no segments found")
            
            print(f"✅ [Diarize & Transcribe] Found {len(diarization_segments)} diarization segments")
            sys.stdout.flush()
            
            # Крок 2: Транскрибуємо аудіо
            print(f"📝 [Diarize & Transcribe] Step 2: Transcribing audio...")
            sys.stdout.flush()
            
            transcription_text, transcription_segments, words = transcribe_audio(
                temp_path,
                language=language,  # Авто-визначення, якщо не вказано
                transcriber=transcriber
            )
            
            if not words:
                raise ValueError("Transcription failed - no words found")
            
            print(f"✅ [Diarize & Transcribe] Transcribed {len(words)} words")
            sys.stdout.flush()
            
            # Крок 3: Об'єднуємо діаризацію з транскрипцією
            print(f"🔗 [Diarize & Transcribe] Step 3: Combining diarization with transcription...")
            sys.stdout.flush()
            
            # Використовуємо простий спосіб об'єднання
            used_word_indices = set()
            combined_segments = []
            
            # Сортуємо сегменти діаризації за часом початку
            sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
            
            for diar_seg in sorted_diar_segments:
                # Знаходимо слова, які потрапляють в цей сегмент
                segment_words = []
                for word_idx, word in enumerate(words):
                    if word_idx in used_word_indices:
                        continue
                    
                    word_start = word.get('start', 0)
                    word_end = word.get('end', 0)
                    word_center = (word_start + word_end) / 2.0
                    
                    # Перевіряємо, чи слово потрапляє в сегмент
                    if (word_center >= diar_seg['start'] and word_center <= diar_seg['end']) or \
                       (word_start < diar_seg['end'] and word_end > diar_seg['start']):
                        segment_words.append((word_idx, word.get('word', '')))
                
                # Якщо знайшли слова для цього сегменту, додаємо їх
                if segment_words:
                    text = ' '.join([w[1] for w in segment_words]).strip()
                    if text:
                        # Позначаємо слова як використані
                        for word_idx, _ in segment_words:
                            used_word_indices.add(word_idx)
                        
                        combined_segments.append({
                            'speaker': diar_seg['speaker'],
                            'start': diar_seg['start'],
                            'end': diar_seg['end'],
                            'text': text
                        })
            
            print(f"✅ [Diarize & Transcribe] Combined {len(combined_segments)} segments")
            sys.stdout.flush()
            
            # Крок 4: Форматуємо результат
            print(f"📋 [Diarize & Transcribe] Step 4: Formatting transcript...")
            sys.stdout.flush()
            
            transcript_lines = []
            for seg in combined_segments:
                start_time = seg['start']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                speaker_num = seg['speaker']
                text = seg['text']
                
                transcript_lines.append(f"{timestamp} - Speaker {speaker_num} - {text}")
            
            # Видаляємо тимчасовий файл
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
            # Повертаємо результат
            response_data = {
                'success': True,
                'transcript': transcript_lines
            }
            
            print(f"📤 [Diarize & Transcribe] Returning transcript with {len(transcript_lines)} lines")
            sys.stdout.flush()
            
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
            
        except Exception as processing_error:
            # Видаляємо тимчасовий файл при помилці
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise processing_error
        
    except Exception as e:
        print(f"❌ [Diarize & Transcribe] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('DEMO2_PORT', 5005))
    print(f"🚀 Starting Flask server for iOS Shortcuts on port {port}")
    print(f"📂 Upload folder: {UPLOAD_FOLDER}")
    print(f"🌐 Server will be accessible at: http://0.0.0.0:{port}")
    print(f"📱 Use your Mac's IP address for iOS Shortcuts")
    app.run(host='0.0.0.0', port=port, debug=False)
