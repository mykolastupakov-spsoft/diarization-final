#!/usr/bin/env python3
"""
PyAnnote Speech Separation Script
Uses pyannote/speech-separation-ami-1.0 model for speaker separation (local with cache)
"""

import os
import sys
import json
import tempfile
import argparse
import threading
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import patch BEFORE pyannote to fix torchaudio compatibility
import pyannote_patch  # noqa: F401

try:
    import torch
    import torchaudio
    from pyannote.audio import Pipeline
    import scipy.io.wavfile
    import numpy as np
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"Missing required package: {e.name}. Please install: pip install pyannote.audio torch torchaudio scipy"
    }), file=sys.stderr)
    sys.exit(1)


def separate_speakers(audio_path, output_dir=None):
    """
    Separate speakers using PyAnnote speech-separation-ami-1.0 (local with cache)
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated audio files (optional)
    
    Returns:
        dict with separation results
    """
    try:
        # Get HuggingFace token
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            return {
                "success": False,
                "error": "HUGGINGFACE_TOKEN not found in environment variables. Please set it in .env file."
            }
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}", file=sys.stderr)
        sys.stderr.flush()
        
        # Check cache before loading
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_path = os.path.join(cache_dir, "models--pyannote--speech-separation-ami-1.0")
        
        was_cached = os.path.exists(model_cache_path)
        if was_cached:
            print("✅ Model found in cache, loading from cache...", file=sys.stderr)
            sys.stderr.flush()
        else:
            print("📥 Model not in cache, will download from Hugging Face...", file=sys.stderr)
            print("⏳ This may take 5-10 minutes on first run (depending on internet speed)...", file=sys.stderr)
            sys.stderr.flush()
        
        # Heartbeat for long operations
        loading_complete = threading.Event()
        heartbeat_count = [0]
        
        def heartbeat():
            """Print heartbeat messages while model is loading"""
            while not loading_complete.wait(30):  # Every 30 seconds
                heartbeat_count[0] += 1
                msg = f"⏳ [Progress {heartbeat_count[0] * 30}s] Still downloading/loading model...\n"
                sys.stderr.write(msg)
                sys.stderr.flush()
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        
        # Load pipeline
        print("🔄 Starting model loading process...", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            print("📦 Calling Pipeline.from_pretrained()...", file=sys.stderr)
            sys.stderr.flush()
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speech-separation-ami-1.0",
                use_auth_token=hf_token
            )
            
            loading_complete.set()  # Stop heartbeat
            
            # Check if model was downloaded and show cache size
            if not was_cached and os.path.exists(model_cache_path):
                # Calculate cache size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_cache_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        try:
                            total_size += os.path.getsize(fp)
                        except:
                            pass
                size_mb = total_size / (1024 * 1024)
                print(f"✅ Model successfully downloaded and cached! ({size_mb:.1f} MB)", file=sys.stderr)
                sys.stderr.flush()
            
            if pipeline is None:
                return {
                    "success": False,
                    "error": "Failed to load pipeline. The model might be gated. Please:\n1. Visit https://hf.co/pyannote/speech-separation-ami-1.0 to accept the user conditions\n2. Make sure your HUGGINGFACE_TOKEN is valid and has access to this model"
                }
            
            print("🔄 Moving pipeline to device...", file=sys.stderr)
            sys.stderr.flush()
            pipeline.to(device)
            print("✅ Model loaded successfully and ready to use!", file=sys.stderr)
            sys.stderr.flush()
            
        except Exception as e:
            loading_complete.set()  # Stop heartbeat on error
            error_msg = str(e)
            if "gated" in error_msg.lower() or "accept" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Model access denied. Please visit https://hf.co/pyannote/speech-separation-ami-1.0 to accept the user conditions. Original error: {error_msg}"
                }
            elif "authentication" in error_msg.lower() or "token" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Authentication failed. Please check your HUGGINGFACE_TOKEN. Original error: {error_msg}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to load pipeline: {error_msg}"
                }
        
        # Load audio using soundfile directly to avoid torchcodec dependency
        print(f"Loading audio from {audio_path}...", file=sys.stderr)
        sys.stderr.flush()
        
        try:
            import soundfile as sf
            # Load audio using soundfile directly
            data, sample_rate = sf.read(audio_path, dtype='float32')
            # Convert to torch tensor
            if len(data.shape) == 1:
                # Mono audio
                waveform = torch.from_numpy(data).unsqueeze(0).float()
            else:
                # Multi-channel audio - transpose to (channels, samples)
                waveform = torch.from_numpy(data).transpose(0, 1).float()
            
            print(f"Loaded audio with soundfile: shape={waveform.shape}, sample_rate={sample_rate}", file=sys.stderr)
            sys.stderr.flush()
        except ImportError:
            # Fallback to torchaudio if soundfile not available
            print("Warning: soundfile not available, trying torchaudio...", file=sys.stderr)
            sys.stderr.flush()
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as load_error:
            print(f"Error loading with soundfile: {load_error}, trying torchaudio...", file=sys.stderr)
            sys.stderr.flush()
            waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        print(f"Audio shape: {waveform.shape}, Sample rate: {sample_rate}", file=sys.stderr)
        sys.stderr.flush()
        
        # Run separation
        print("Running speaker separation...", file=sys.stderr)
        sys.stderr.flush()
        diarization, sources = pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
        })
        
        # Get speakers
        speakers = list(diarization.labels())
        print(f"Found {len(speakers)} speakers: {speakers}", file=sys.stderr)
        sys.stderr.flush()
        
        # Debug: Check sources structure
        print(f"Sources type: {type(sources)}", file=sys.stderr)
        print(f"Sources has 'data' attribute: {hasattr(sources, 'data')}", file=sys.stderr)
        if hasattr(sources, 'data'):
            print(f"Sources.data type: {type(sources.data)}", file=sys.stderr)
            if hasattr(sources.data, 'shape'):
                print(f"Sources.data shape: {sources.data.shape}", file=sys.stderr)
            elif isinstance(sources.data, (list, tuple)):
                print(f"Sources.data length: {len(sources.data)}", file=sys.stderr)
        
        # Check if sources is a SlidingWindowFeature and how to access speaker data
        if hasattr(sources, '__getitem__'):
            print(f"Sources supports indexing", file=sys.stderr)
            try:
                test_item = sources[0]
                print(f"Sources[0] type: {type(test_item)}, shape: {getattr(test_item, 'shape', 'N/A')}", file=sys.stderr)
            except:
                pass
        
        # Check if sources has channels or other attributes
        print(f"Sources attributes: {[attr for attr in dir(sources) if not attr.startswith('_')][:20]}", file=sys.stderr)
        sys.stderr.flush()
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = tempfile.mkdtemp(prefix="pyannote_separation_")
        
        # Save separated audio files and create timeline
        speakers_data = []
        timeline = []
        
        # Handle different API versions of itertracks
        try:
            # Try new API (returns segment, track, label)
            for segment, track, speaker in diarization.itertracks(yield_label=True):
                timeline.append({
                    "speaker": speaker,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "duration": round(segment.end - segment.start, 2)
                })
        except ValueError:
            # Fallback to old API (returns turn, speaker)
            try:
                for turn, speaker in diarization.itertracks(yield_label=True):
                    timeline.append({
                        "speaker": speaker,
                        "start": round(turn.start, 2),
                        "end": round(turn.end, 2),
                        "duration": round(turn.end - turn.start, 2)
                    })
            except ValueError:
                # If that also fails, try without yield_label
                for turn in diarization.itertracks():
                    speaker = turn.label if hasattr(turn, 'label') else diarization[turn]
                    timeline.append({
                        "speaker": speaker,
                        "start": round(turn.start, 2),
                        "end": round(turn.end, 2),
                        "duration": round(turn.end - turn.start, 2)
                    })
        
        # Extract audio for each speaker
        # sources is a SlidingWindowFeature with shape (samples, channels)
        # We need to extract segments for each speaker based on diarization
        sources_data = sources.data  # Shape: (samples, channels) - channels should be num_speakers
        
        print(f"Extracting audio for {len(speakers)} speakers from sources with shape {sources_data.shape}", file=sys.stderr)
        sys.stderr.flush()
        
        # Check if sources has multiple channels (one per speaker)
        if sources_data.shape[1] == len(speakers):
            # Each channel corresponds to a speaker
            print("Using channel-based separation (each channel = one speaker)", file=sys.stderr)
            sys.stderr.flush()
            for s, speaker in enumerate(speakers):
                try:
                    # Get audio for this speaker from the corresponding channel
                    speaker_audio = sources_data[:, s].astype(np.float32)
                    
                    # Save to file
                    output_path = os.path.join(output_dir, f"{speaker}.wav")
                    scipy.io.wavfile.write(output_path, 16000, speaker_audio)
                    
                    speakers_data.append({
                        "name": speaker,
                        "format": "wav",
                        "local_path": output_path,
                        "isBackground": False
                    })
                    print(f"Saved {speaker} audio: {len(speaker_audio)} samples", file=sys.stderr)
                    sys.stderr.flush()
                except Exception as speaker_error:
                    print(f"Error processing speaker {speaker}: {speaker_error}", file=sys.stderr)
                    sys.stderr.flush()
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    continue
        else:
            # Single channel - need to extract segments based on diarization
            print("Using diarization-based segmentation (single channel, extracting segments)", file=sys.stderr)
            sys.stderr.flush()
            
            # Get the single channel audio
            mixed_audio = sources_data[:, 0].astype(np.float32)
            total_samples = len(mixed_audio)
            duration = total_samples / sample_rate
            
            # Create separate audio tracks for each speaker by extracting their segments
            for s, speaker in enumerate(speakers):
                try:
                    # Initialize speaker audio with zeros
                    speaker_audio = np.zeros_like(mixed_audio)
                    
                    # Extract segments for this speaker from diarization
                    speaker_segments = []
                    try:
                        # Try new API (returns segment, track, label)
                        for segment, track, label in diarization.itertracks(yield_label=True):
                            if label == speaker:
                                speaker_segments.append(segment)
                    except ValueError:
                        # Fallback to old API (returns turn, speaker)
                        try:
                            for turn, label in diarization.itertracks(yield_label=True):
                                if label == speaker:
                                    speaker_segments.append(turn)
                        except ValueError:
                            # If that also fails, try without yield_label
                            for turn in diarization.itertracks():
                                label = turn.label if hasattr(turn, 'label') else diarization[turn]
                                if label == speaker:
                                    speaker_segments.append(turn)
                    
                    # Fill in the segments for this speaker
                    for seg in speaker_segments:
                        start_sample = int(seg.start * sample_rate)
                        end_sample = int(seg.end * sample_rate)
                        if start_sample < total_samples and end_sample <= total_samples:
                            speaker_audio[start_sample:end_sample] = mixed_audio[start_sample:end_sample]
                    
                    # Save to file
                    output_path = os.path.join(output_dir, f"{speaker}.wav")
                    scipy.io.wavfile.write(output_path, 16000, speaker_audio)
                    
                    speakers_data.append({
                        "name": speaker,
                        "format": "wav",
                        "local_path": output_path,
                        "isBackground": False
                    })
                    print(f"Saved {speaker} audio: {len(speaker_audio)} samples, {len(speaker_segments)} segments", file=sys.stderr)
                    sys.stderr.flush()
                except Exception as speaker_error:
                    print(f"Error processing speaker {speaker}: {speaker_error}", file=sys.stderr)
                    sys.stderr.flush()
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    continue
        
        return {
            "success": True,
            "speakers": speakers_data,
            "timeline": timeline,
            "num_speakers": len(speakers),
            "output_dir": output_dir
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="PyAnnote Speaker Separation")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--output-dir", help="Directory to save output files", default=None)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        result = {
            "success": False,
            "error": f"Audio file not found: {args.audio_path}"
        }
        print(json.dumps(result))
        sys.exit(1)
    
    result = separate_speakers(args.audio_path, args.output_dir)
    print(json.dumps(result))
    
    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
