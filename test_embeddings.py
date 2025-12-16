#!/usr/bin/env python3
"""
Тестовий скрипт для діагностики витягування ембеддингів SpeechBrain
"""

import os
import sys
import numpy as np
import torch
import traceback
import tempfile
import soundfile as sf

# Патч для torchaudio сумісності з speechbrain
exec(open('patch_torchaudio.py').read())

from speechbrain.pretrained import SpeakerRecognition

print("="*60)
print("🧪 TEST: SpeechBrain Embedding Extraction")
print("="*60)

# Завантаження моделі
print("\n🔄 Loading SpeechBrain model...")
try:
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Device моделі
try:
    device = next(speaker_model.parameters()).device
    print(f"📱 Model device: {device}")
except:
    device = torch.device('cpu')
    print(f"📱 Using default device: {device}")

# Створення тестового сегмента (2 секунди, 16kHz)
print("\n🎵 Creating test audio segment (2 seconds, 16kHz)...")
sr = 16000
duration = 2.0
test_audio = np.random.randn(int(sr * duration)).astype(np.float32)
print(f"✅ Test audio created: shape={test_audio.shape}, dtype={test_audio.dtype}, samples={len(test_audio)}")

# Тест 1: encode_batch з normalize=False, формат [1, 1, samples]
print("\n" + "-"*60)
print("TEST 1: encode_batch(tensor [1,1,samples], normalize=False)")
print("-"*60)
try:
    segment_tensor = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"📊 Tensor shape: {segment_tensor.shape}, dtype: {segment_tensor.dtype}, device: {segment_tensor.device}")
    
    embedding = speaker_model.encode_batch(segment_tensor, normalize=False)
    embedding = embedding.squeeze().cpu().detach().numpy()
    
    print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        print("⚠️  WARNING: NaN or Inf found in embedding!")
    else:
        print("✅ No NaN or Inf in embedding")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Тест 2: encode_batch без normalize, формат [1, 1, samples]
print("\n" + "-"*60)
print("TEST 2: encode_batch(tensor [1,1,samples]) - без normalize")
print("-"*60)
try:
    segment_tensor = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"📊 Tensor shape: {segment_tensor.shape}, dtype: {segment_tensor.dtype}, device: {segment_tensor.device}")
    
    embedding = speaker_model.encode_batch(segment_tensor)
    embedding = embedding.squeeze().cpu().detach().numpy()
    
    print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        print("⚠️  WARNING: NaN or Inf found in embedding!")
    else:
        print("✅ No NaN or Inf in embedding")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Тест 3: encode_batch з формат [1, samples] (без подвійного unsqueeze)
print("\n" + "-"*60)
print("TEST 3: encode_batch(tensor [1,samples]) - без подвійного unsqueeze")
print("-"*60)
try:
    segment_tensor = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0)
    print(f"📊 Tensor shape: {segment_tensor.shape}, dtype: {segment_tensor.dtype}, device: {segment_tensor.device}")
    
    embedding = speaker_model.encode_batch(segment_tensor)
    embedding = embedding.squeeze().cpu().detach().numpy()
    
    print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        print("⚠️  WARNING: NaN or Inf found in embedding!")
    else:
        print("✅ No NaN or Inf in embedding")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

# Тест 4: encode_file (якщо доступний)
print("\n" + "-"*60)
print("TEST 4: encode_file() - через тимчасовий файл")
print("-"*60)
if hasattr(speaker_model, 'encode_file'):
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, test_audio, sr)
            tmp_path = tmp_file.name
        
        print(f"📁 Temporary file created: {tmp_path}")
        
        embedding = speaker_model.encode_file(tmp_path)
        embedding = embedding.squeeze().cpu().detach().numpy()
        
        print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("⚠️  WARNING: NaN or Inf found in embedding!")
        else:
            print("✅ No NaN or Inf in embedding")
        
        # Видаляємо тимчасовий файл
        os.unlink(tmp_path)
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass
else:
    print("❌ Model does not have 'encode_file' method")

# Тест 5: Прямий доступ через encoder + embedding_model
print("\n" + "-"*60)
print("TEST 5: Direct access via mods.encoder() + mods.embedding_model()")
print("-"*60)
if hasattr(speaker_model, 'mods') and hasattr(speaker_model.mods, 'encoder'):
    try:
        segment_tensor = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0)
        wav_lens = torch.tensor([len(test_audio) / sr], dtype=torch.float32)
        
        print(f"📊 Tensor shape: {segment_tensor.shape}, dtype: {segment_tensor.dtype}, device: {segment_tensor.device}")
        print(f"📊 wav_lens: {wav_lens}")
        
        with torch.no_grad():
            features = speaker_model.mods.encoder(segment_tensor, wav_lens=wav_lens)
            print(f"📊 Features shape after encoder: {features.shape}")
            
            if hasattr(speaker_model.mods, 'embedding_model'):
                embedding = speaker_model.mods.embedding_model(features, wav_lens=wav_lens)
            else:
                embedding = features
            
            embedding = embedding.squeeze().cpu().detach().numpy()
        
        print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("⚠️  WARNING: NaN or Inf found in embedding!")
        else:
            print("✅ No NaN or Inf in embedding")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
else:
    print("❌ Model does not have 'mods.encoder'")

# Тест 6: Тільки encoder без embedding_model
print("\n" + "-"*60)
print("TEST 6: Direct access via mods.encoder() only (no embedding_model)")
print("-"*60)
if hasattr(speaker_model, 'mods') and hasattr(speaker_model.mods, 'encoder'):
    try:
        segment_tensor = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0)
        wav_lens = torch.tensor([len(test_audio) / sr], dtype=torch.float32)
        
        print(f"📊 Tensor shape: {segment_tensor.shape}, dtype: {segment_tensor.dtype}, device: {segment_tensor.device}")
        
        with torch.no_grad():
            embedding = speaker_model.mods.encoder(segment_tensor, wav_lens=wav_lens)
            embedding = embedding.squeeze().cpu().detach().numpy()
        
        print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
        print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("⚠️  WARNING: NaN or Inf found in embedding!")
        else:
            print("✅ No NaN or Inf in embedding")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        traceback.print_exc()
else:
    print("❌ Model does not have 'mods.encoder'")

# Тест 7: Перевірка device (конвертація на device моделі)
print("\n" + "-"*60)
print("TEST 7: encode_batch with tensor on model device")
print("-"*60)
try:
    segment_tensor = torch.tensor(test_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    print(f"📊 Tensor shape: {segment_tensor.shape}, dtype: {segment_tensor.dtype}, device: {segment_tensor.device}")
    
    embedding = speaker_model.encode_batch(segment_tensor, normalize=False)
    embedding = embedding.squeeze().cpu().detach().numpy()
    
    print(f"✅ SUCCESS! Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
    print(f"📊 Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        print("⚠️  WARNING: NaN or Inf found in embedding!")
    else:
        print("✅ No NaN or Inf in embedding")
except Exception as e:
    print(f"❌ FAILED: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("🏁 TESTING COMPLETE")
print("="*60)



