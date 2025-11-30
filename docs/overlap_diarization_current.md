## Overlap Diarization – Current Implementation (Modes 1–3)

This document summarizes how overlap diarization currently works across all three modes (AudioShake, PyAnnote, SpeechBrain) so you can modify or extend the system with confidence.

---

### 1. High-Level Pipeline (shared across modes)

1. **Step 1 – Primary audio analysis**  
   `runPythonDiarization()` (Speechmatics-based) runs on the uploaded file/URL and produces the baseline diarization (`primaryDiarization`).

2. **Step 2 – Speaker separation**  
   - Mode 1: AudioShake API.
   - Mode 2: Local PyAnnote (`pyannote/speech-separation-ami-1.0`).
   - Mode 3: Local SpeechBrain SepFormer (`speechbrain/sepformer-wsj02mix`).
   Separation outputs per-speaker WAV stems and a timeline.

3. **Step 3 – Per-speaker transcription**  
   Each separated track is sent back through `runPythonDiarization()` (same engine as step 1) to get text + timestamps for that speaker only.

4. **Step 4 – Role analysis**  
   `analyzeVoiceRole()` (LLM via OpenRouter) labels each speaker (operator / client) and stores confidence + summary.

5. **Step 5 – Overlap correction**  
   `correctPrimaryDiarizationWithTracks()` feeds the primary diarization + voice-track transcripts into an LLM prompt (`prompts/diarization_overlap_correction_prompt.txt`).  
   The LLM outputs the final overlap-corrected JSON that replaces the recording’s `results`.

6. **Frontend rendering**  
   `app.js` renders progress, voice tracks, transcripts, and downloads inside the “Overlap Diarization” tab.

Steps 2–4 are the only pieces that depend on the separation backend. Everything else is separation-agnostic.

---

### 2. Backend Entry Point (`server.js`)

**Route:** `POST /api/diarize-overlap`

Key excerpt showing how step 2 is chosen:

```javascript
const { pipelineMode } = req.body;
const overlapPipelineMode = pipelineMode || 'mode1';

if (overlapPipelineMode === 'mode2') {
  separation = await separateSpeakersWithPyAnnote(sourcePath);
} else if (overlapPipelineMode === 'mode3') {
  separation = await separateSpeakersWithSpeechBrain(sourcePath);
} else {
  separation = await separateSpeakersWithAudioShake(sourcePathOrUrl);
}
```

After `separation` is resolved, the code loops through `separation.speakers` and runs transcription + role analysis per track:

```javascript
for (let i = 0; i < separation.speakers.length; i++) {
  const speaker = separation.speakers[i];

  // Step 3: transcription (Speechmatics pipeline)
  const transcription = await runPythonDiarization({
    url: speaker.url,
    language
  });

  // Step 4: role classification (LLM)
  const roleAnalysis = await analyzeVoiceRole(
    transcriptText,
    language,
    mode
  );

  voiceTracks.push({
    speaker: speaker.name,
    audioUrl: speaker.url,
    transcription,
    roleAnalysis
  });
}
```

`voiceTracks`, `primaryDiarization`, `separation`, and `steps` are finally passed to `correctPrimaryDiarizationWithTracks()` (step 5) and then returned to the client.

---

### 3. Mode 2 – PyAnnote Separation (`pyannote_separation.py`)

This script is spawned from `separateSpeakersWithPyAnnote()` via Node’s `child_process.spawn`. It:

1. Loads environment (`.env` + `HUGGINGFACE_TOKEN`).
2. Applies compatibility patches (`pyannote_patch.py`) before importing `pyannote.audio`.
3. Calls:
   ```python
   pipeline = Pipeline.from_pretrained(
       "pyannote/speech-separation-ami-1.0",
       use_auth_token=hf_token
   )
   diarization, sources = pipeline({"waveform": waveform, "sample_rate": sample_rate})
   ```
4. Iterates over `diarization.itertracks()` to build a timeline and extracts per-speaker audio from `sources`.
5. Writes each speaker to `<tmp_dir>/SPEAKER_xx.wav` and prints JSON to stdout:
   ```json
   {
     "success": true,
     "speakers": [
       { "name": "SPEAKER_00", "format": "wav", "local_path": ".../SPEAKER_00.wav" },
       { "name": "SPEAKER_01", "format": "wav", "local_path": ".../SPEAKER_01.wav" }
     ],
     "timeline": [...],
     "output_dir": ".../pyannote_separation_1764..."
   }
   ```

Node copies those WAVs into `/uploads/<timestamp>_<idx>_<speaker>.wav`, exposes them via `/api/pyannote-stems/:filename`, and stores a public or relative URL in `separation.speakers[n].url`.

Any alternative engine should mimic the same JSON contract so no other code needs to change.

---

### 4. Mode 3 – SpeechBrain SepFormer (`speechbrain_separation.py`)

The new Mode 3 uses SpeechBrain’s SepFormer (`speechbrain/sepformer-wsj02mix`) for local separation. The script behaves like the PyAnnote version:

1. **Device selection** – prefers `SPEECHBRAIN_DEVICE` env value, otherwise picks `mps` → `cuda` → `cpu`.
2. **Model loading** – `Separator.from_hparams(...)` with cache in `~/.cache/speechbrain/sepformer-wsj02mix`.
3. **Input prep** – converts audio to mono and resamples to 8 kHz (SepFormer requirement), writes a temp WAV (`temp_input.wav`).
4. **Separation** – `model.separate_file()` returns tensors per speaker.
5. **Output** – emits the same JSON payload (`success`, `speakers`, `timeline`, `output_dir`). Each speaker’s duration is the whole track because SepFormer doesn’t emit diarization metadata.

`server.js` consumes the JSON via `separateSpeakersWithSpeechBrain()`:

```javascript
const separation = await separateSpeakersWithSpeechBrain(sourcePath);

// The helper copies SPEAKER_xx.wav into /uploads and assigns:
{
  name: 'SPEAKER_00',
  url: getPublicFileUrl(filename),
  downloadUrl: `/uploads/${filename}`
}
```

Timeout guards (5 min) and stderr logging keep runaway processes under control. Because SepFormer is local, there are no HTTPS / API-key requirements, and the error handling treats modes 2 and 3 the same way (never surface HTTPS errors to the user).

---

### 5. Frontend Hooks (`app.js` / `index.html`)

- User selects Mode 1 / 2 / 3 via `<select id="overlapPipelineMode">` (each option shows a contextual info box).
- `app.startOverlapDiarization()` posts the form data (including `pipelineMode`) and renders progress for all five steps.
- Once results arrive, `renderOverlapDiarizationResults()`:
  - Shows separation stats and per-speaker audio players.
  - Stores the overlap metadata inside the active recording so other tabs can display it.

Frontend expects the API response shape:

```json
{
  "success": true,
  "primaryDiarization": {...},
  "separation": {
    "speakers": [
      {
        "name": "SPEAKER_00",
        "url": "https://.../uploads/...",
        "downloadUrl": "/uploads/..."
      }
    ]
  },
  "voiceTracks": [...],
  "steps": {
    "step1": { "name": "...", "duration": "14.3s", ... },
    "step2": { "speakersCount": 2, ... },
    "step3": { "processedTracks": 2, ... },
    "step5": {...}
  }
}
```

Any new mode must keep this interface intact.

---

### 6. Extending to Additional Separation Modes

All three modes share the same contract. To add another engine in the future:

1. Add a frontend option so the user can select the new mode.
2. Branch inside `/api/diarize-overlap` and call a helper similar to `separateSpeakersWithPyAnnote` / `separateSpeakersWithSpeechBrain`.
3. Ensure the helper emits the **same JSON + WAV artifacts** (`speakers[n].url`, `downloadUrl`, `timeline`, `taskId`).
4. Keep the `/uploads` copy step so downstream transcription can fetch local files via HTTP.

As long as the JSON contract stays unchanged, steps 3–5 (transcription, role analysis, correction) and the UI will continue to work automatically.

---

### 7. Key Environment & Dependencies

- `server.js` (Node 18+)
- Python 3.10+ with:
  - `pyannote.audio>=3.1,<3.5`
  - `torch`, `torchaudio`, `numpy`, `soundfile`, `scipy`
  - `python-dotenv`
- `.env`:
  ```env
  HUGGINGFACE_TOKEN=...
  AUDIOSHAKE_API_KEY=...   # for mode 1
  OPENROUTER_API_KEY=...   # for LLM roles/correction
  ```
- `pyannote_patch.py` handles torchaudio + hyperpyyaml quirks (Python 3.14 support).
- `speechbrain>=0.5.14` for Mode 3 (auto-installs the SepFormer checkpoints on first run).

---

### 8. Files to Review

| File | Purpose |
| --- | --- |
| `server.js` | Main Express server, `/api/diarize-overlap`, mode dispatch, transcription, role analysis, overlap correction. |
| `pyannote_separation.py` | Current mode 2 implementation (model loading, waveform prep, separation, JSON output). Use as blueprint for SpeechBrain script. |
| `speechbrain_separation.py` | Mode 3 implementation using `speechbrain/sepformer-wsj02mix`; selects device, resamples to 8 kHz, writes WAVs, emits JSON identical to PyAnnote helper. |
| `pyannote_patch.py` | Compatibility patches executed before importing PyAnnote. Might not be needed for SpeechBrain but shows how we solve dependency issues. |
| `app.js` / `index.html` | Frontend controls, progress UI, rendering separated tracks. Needs an extra option + status label for mode 3. |
| `prompts/diarization_overlap_correction_prompt.txt` | Defines how the LLM consumes voice-track transcripts to fix overlap segments. No changes required. |

---

### 9. Summary

- Modes 1–3 differ only in how they produce separated WAVs; the transcription/LLM/overlap logic is shared.
- PyAnnote (mode 2) and SpeechBrain (mode 3) are fully local and bypass HTTPS requirements; AudioShake (mode 1) still needs a tunnel / HTTPS URL.
- Any future separation engine just needs to follow the established JSON contract so the rest of the pipeline remains untouched.

This context should be sufficient for an AI assistant (or developer) to maintain or extend the overlap diarization workflow.

