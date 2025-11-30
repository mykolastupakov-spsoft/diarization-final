# Project Context – Arabic/English Speaker Diarization

This document is intended to be copy-pasted into another LLM to provide full context about the diarization platform that lives in this repository. It captures the product mission, runtime architecture, key code paths, external dependencies, and development conventions so another assistant can answer deep follow-up questions.

---

## 1. Product Overview

- **Goal:** Provide a desktop-friendly web application that ingests Arabic/English call-center conversations and produces structured diarization (speaker segmentation + roles) plus quality analysis.
- **Primary workflows:**
  1. **LLM-only** diarization: convert transcripts into Speechmatics-compatible JSON using OpenRouter models.
  2. **Audio diarization**: run Speechmatics (via Python helper) on raw audio and bring results into the UI.
  3. **Combined pipeline**: merge audio + LLM outputs, perform comparison + correction passes, optionally using Gemini 2.5 Pro multimodal mode.
  4. **Overlap pipeline**: perform audio separation (AudioShake cloud or local PyAnnote / SpeechBrain), re-run transcription + role analysis per stem, then LLM-correct overlaps.
- **Secondary tooling:** Prompt manager, AI-powered audio generator, debug dashboards, diarization metrics calculator, and prompt templates for verification/correction.

---

## 2. Tech Stack & Key Dependencies

| Layer | Technologies |
| --- | --- |
| Frontend | `index.html`, `app.js`, vanilla JS modules (`metrics.js`, `visualization.js`), localStorage state |
| Backend | Node.js 18+, Express 4, Axios, Multer, Localtunnel, Cloudinary SDK, custom libs under `lib/` |
| Python services | `process_audio_temp.py`, `pyannote_separation.py`, `speechbrain_separation.py`, `processor.py`, `audioshake_client.py` |
| AI/ML services | OpenRouter (multiple models), Google Gemini 2.5 Pro, Speechmatics ASR, AudioShake Tasks API, PyAnnote, SpeechBrain SepFormer |
| Storage | Local `uploads/`, `temp_uploads/`, `Debug/` dashboards, `noise-samples/` |
| Observability | `server_debug.log`, `/api/log` endpoint, HTML dashboards under `Debug/` |

Package manifests: `package.json` (Node) and `requirements.txt` (Python).

---

## 3. High-Level Architecture

### 3.1 Frontend (`index.html`, `app.js`)

- Single-page application with multiple “tabs”: **LLM**, **Audio**, **Combined**, **Overlap**, **Upload**, **Prompt Manager**, **Audio Generator**.
- State container (`app` object) tracks recordings, API keys, active tab, translations, role filters, speechbrain samples, AudioShake tunnel status, etc.
- Features:
  - Drag-and-drop upload, progress steppers, role-based filtering, transcript textarea auto-cleanup.
  - Local persistence of API keys + service toggles in `localStorage`.
  - Event bridge with `Features/Realistic-audio-generator.html` via `postMessage` for synthesized test audio.
  - Prompt edit UI (`prompt-manager.html`) loads templates via `/api/prompts`.
  - Metrics visualization uses `metrics.js` (DER/JER) and `visualization.js`.

### 3.2 Backend (`server.js`)

- Express server bootstraps `.env`, ensures directories (`uploads/`, `temp_uploads/`, `Debug/`), configures body limits, static serving, and logging.
- Key helpers:
  - **Localtunnel bootstrap** to expose `/uploads` for AudioShake remote fetches (with optional `LOCALTUNNEL_SUBDOMAIN` or `PUBLIC_URL` override).
  - **`writeLog`** stores structured logs in `server_debug.log` and mirrors to stdout; `/api/log` lets the frontend push console logs.
  - **File lifecycle**: `multer` writes to `temp_uploads/`, `persistUploadedFile` moves into `uploads/` and returns public URLs.
  - **Python orchestration**: `runPythonDiarization`, `separateSpeakersWithPyAnnote`, `separateSpeakersWithSpeechBrain`, `callGeminiMultimodal`, `analyzeVoiceRole`, `correctPrimaryDiarizationWithTracks`, etc.
  - **Prompt loading/caching** for all LLM flows; `/api/prompts` endpoints expose them to the UI.

### 3.3 Python Processing Tier

- `process_audio_temp.py`: downloads/ingests audio, uploads to Speechmatics, polls job completion, parses segments, and formats results.
- `pyannote_separation.py`: runs `pyannote/speech-separation-ami-1.0`, writes per-speaker WAVs, returns JSON contract consumed by Node.
- `speechbrain_separation.py`: uses `speechbrain/sepformer-wsj02mix` with MPS/CUDA/CPU fallback.
- `audioshake_client.py` & `lib/audioshake-client.js`: typed wrappers around the AudioShake Tasks API (HTTPS URL required).
- `processor.py`: orchestrates AudioShake job submission, transcript stubs, role classification via OpenRouter.

### 3.4 Storage, Debugging & Assets

- `uploads/`: persistent audio accessible to AudioShake/tunnel and downloadable from the UI.
- `temp_uploads/`: transient files cleaned after processing.
- `Debug/`: test dashboards (`speechbrain_dashboard.html`, `pyannote_dashboard.html`), sample outputs, case data.
- `audio examples/` & `noise-samples/`: curated fixtures for demos/tests.

---

## 4. End-to-End Pipelines

### 4.1 LLM-only Diarization (`/api/diarize`, `/api/diarize-llm`)

1. Frontend collects transcript text, mode selection, and optional verification flags.
2. Backend cleans service markers, loads prompt templates (system + user), chooses OpenRouter model (`FAST_MODEL_ID`, `SMART_MODEL_ID`, `SMART_2_MODEL_ID`).
3. Response is parsed via `parseToStructuredJSON`, normalized to `recordings[0].results['text-service']`.
4. Optional verification/fix uses dedicated prompt templates (`prompts/diarization_verification_prompt.txt`, etc.).
5. `/api/diarize-llm` adds optional Gemini 2.5 Pro multimodal processing if audio is provided and `mode` requests it; falls back to standard OpenRouter path.

### 4.2 Audio Diarization (`/api/diarize-audio`)

- Accepts file upload or remote URL.
- Invokes `runPythonDiarization` → `process_audio_temp.py`, which uploads to Speechmatics with optional speaker count and language hints.
- Returns Speechmatics JSON (segments with start/end/speaker) and cleans temp files.

### 4.3 Combined Pipeline (`/api/diarize-combined`)

Steps (tracked in `steps` object for UI progress):
1. **Audio transcription** via Speechmatics (unless multimodal mode).
2. **LLM diarization** via `handleDiarizationRequest` (OpenRouter) or `callGeminiMultimodal`.
3. **Comparison agent** uses `prompts/diarization_comparison_prompt.txt` to analyze disagreements and role quality.
4. **Correction agent** (`prompts/diarization_correction_prompt.txt`) merges the strongest pieces into a final JSON.
5. Multimodal branch (if `pipelineMode=multimodal`) streams audio into Gemini 2.5 Pro (Google API first, fallback to OpenRouter version).
6. Optional manual transcripts can bypass audio step; results include intermediate artifacts for UI inspection/download.

### 4.4 Overlap Pipeline (`/api/diarize-overlap`)

Pipeline summary (matches `docs/overlap_diarization_current.md`):
1. **Primary diarization**: baseline Speechmatics result.
2. **Speaker separation**: mode-select between AudioShake (mode1), PyAnnote (mode2), SpeechBrain (mode3).
3. **Per-speaker transcription**: run `runPythonDiarization` on each stem.
4. **Role analysis**: `analyzeVoiceRole` hits OpenRouter to label Agent/Client, record confidence/summary.
5. **Overlap correction**: `correctPrimaryDiarizationWithTracks` feeds primary + voice tracks into LLM prompt (`prompts/diarization_overlap_correction_prompt.txt`) to rebuild segments with overlaps resolved.
6. UI shows separation stats, stem download links, and corrected diarization.

### 4.5 Audio Separation Engines

| Mode | Engine | Notes |
| --- | --- | --- |
| Mode 1 | AudioShake Tasks API | Requires HTTPS-accessible audio via localtunnel or `PUBLIC_URL`. Upload optionally proxied through Cloudinary via `lib/cloudinary-upload.js`. |
| Mode 2 | PyAnnote local script | Needs `HUGGINGFACE_TOKEN`, runs `pyannote/audio` pipeline, emits timeline data, stored in `/uploads` for playback. |
| Mode 3 | SpeechBrain SepFormer | Purely local, uses `speechbrain/sepformer-wsj02mix`, auto-selects device, resamples audio. |

Each mode must emit the same JSON contract with `speakers[n].url`, `downloadUrl`, `timeline`, `taskId` so downstream steps stay untouched.

### 4.6 Multimodal Gemini 2.5 Pro

- Implemented in `callGeminiMultimodal`:
  - Read prompts, adapt wording to requested language.
  - Construct `contents` array with optional inline audio data (base64) + text.
  - Try Google API first (`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent`), fallback to OpenRouter model `google/gemini-2.5-pro`.
  - Parse response, run through `parseToStructuredJSON`, annotate metadata (`source`, `provider`, `multimodal`).

---

## 5. API Surface (Server)

| Endpoint | Description |
| --- | --- |
| `POST /api/log` | Receive frontend console logs for `server_debug.log`. |
| `GET /api/tunnel-status` / `POST /api/tunnel-start` | Manage localtunnel lifecycle and expose current public URL. |
| `POST /api/diarize` | Core LLM diarization (OpenRouter). |
| `POST /api/diarize-llm` | Adds Gemini 2.5 Pro multimodal support + file upload option. |
| `POST /api/diarize-audio` | Audio → Speechmatics pipeline (Python). |
| `POST /api/diarize-combined` | Hybrid pipeline (audio + LLM + comparison + correction). |
| `POST /api/diarize-overlap` | Overlap correction pipeline with separation modes. |
| `POST /api/audioshake-tasks` | AudioShake Tasks fast path (Cloudinary/local uploads). |
| `POST /api/audioshake-pipeline` | Legacy AudioShake flow (polling + download). |
| `GET /api/pyannote-stems/:filename`, `/api/speechbrain-stems/:filename`, `/api/audioshake-stems/:jobId/:filename` | Serve separated stems via HTTP. |
| `GET /speechbrain_dashboard.html`, `/dashboards/*` | Serve debug dashboards. |
| `GET /api/prompts`, `GET /api/prompts/:filename`, `POST /api/prompts/:filename/reload` | Prompt manager API. |
| `POST /api/translate` | Translation service (LLM-backed) for transcripts UI. |
| `GET /api/generate-dialogue/test`, `POST /api/generate-dialogue` | Synthetic dialogue/audio generator hooks. |
| `POST /api/process-audio-temp` | Direct access to Python temp processing helper. |
| `POST /api/overlap-audio`, `/api/overlap-llm`, `/api/overlap-llm-patterns` | Experimental overlap testing endpoints. |
| `GET /api/speechmatics-key` | Exposes Speechmatics key existence (no value) for UI gating. |

Static routes also serve `/prompt-manager`, `/audio-generator`, `/uploads/*`, `/speechbrain_samples/*`, etc.

---

## 6. Prompting & Templates (`/prompts`)

- `arabic_diarization_complete_prompt.txt`: Primary diarization instruction (LLM + Gemini).
- `diarization_verification_prompt.txt`: Verification/fix pass template.
- `diarization_comparison_prompt.txt`: Combined pipeline comparison agent.
- `diarization_correction_prompt.txt`: Correction/merging agent prompt.
- `diarization_overlap_correction_prompt.txt`: Overlap-specific correction.
- `system_diarization.txt`, `system_translation.txt`: Shared system messages.
- `translation_prompt_template.txt`, `ai_analysis_prompt_template.txt`, `arabic_diarization_complete_prompt.txt`: Ad-hoc exports.
- Prompts are editable at runtime via `/prompt-manager` UI and reloaded through the prompt API.

---

## 7. Configuration & Secrets (`env.example`)

Environment variables of note:

- `OPENROUTER_API_KEY`, optional overrides per tier (`FAST_MODEL_ID`, `SMART_MODEL_ID`, `SMART_2_MODEL_ID`).
- `SPEECHMATICS_API_KEY` for audio transcription.
- `AUDIOSHAKE_API_KEY` plus `PUBLIC_URL`, `LOCALTUNNEL_SUBDOMAIN`, `DISABLE_LOCALTUNNEL`.
- `APP_URL`, `PORT`, `TUNNEL_PORT`, `REQUEST_BODY_LIMIT`.
- `OPENAI_API_KEY` (legacy text services), `GOOGLE_GEMINI_API_KEY`, `OPENAI_*` fallback.
- `HUGGINGFACE_TOKEN` for PyAnnote.
- `CLOUDINARY_*` for optional Cloudinary uploads.
- `PYTHON_BIN` to point at virtualenv interpreter.
- Additional flags referenced in code: `NODE_ENV`, `SPEECHBRAIN_DEVICE`.

Create `.env` from `env.example`, ensure `.venv` (if used) exposes Python dependencies.

---

## 8. External Integrations

| Service | Purpose | Integration Notes |
| --- | --- | --- |
| **OpenRouter** | LLM diarization, role analysis, comparison, correction, translation | All LLM HTTP calls include `HTTP-Referer` (`APP_URL`) and `X-Title`. Models: `gpt-oss-120b`, `gpt-5.1`, `google/gemini-3-pro-preview`, `google/gemini-2.5-pro`. |
| **Speechmatics** | Baseline audio diarization | Python script uploads file, polls job, extracts segments. Requires bearer token. |
| **AudioShake Tasks API** | Cloud speaker separation | Needs HTTPS audio URLs; local server uses localtunnel + optional Cloudinary upload. Polling + download handled by both Node (`lib/audioshake-client.js`) and Python (`audioshake_client.py`). |
| **PyAnnote** | Local separation fallback | Needs Hugging Face auth token, patched via `pyannote_patch.py` and `fix_hyperpyyaml.py`. |
| **SpeechBrain (SepFormer)** | Local separation fallback | Heavy models cached in `~/.cache`; script handles device selection/resampling. |
| **Google Gemini 2.5 Pro** | Multimodal diarization | Direct Google API call (if `GOOGLE_GEMINI_API_KEY`), fallback to OpenRouter variant. |
| **Cloudinary** | Optional hosting for AudioShake uploads | `lib/cloudinary-upload.js` handles upload; ensures audio is reachable over HTTPS. |
| **Localtunnel** | Expose `uploads/` to the public when running locally | Auto-started unless `DISABLE_LOCALTUNNEL=true`. `/api/tunnel-status` gives health info for UI gating. |

---

## 9. Frontend State & UX Highlights

- `SERVICES` & `OPENROUTER_MODELS` define UI cards for configuration.
- `app.recordings[]` holds normalized diarization outputs with results keyed by service; Combined/Overlap writes additional metadata (voice tracks, comparison analysis, corrections).
- Buttons automatically disable until transcripts/audio are present; role filters (`{ agent, client, none }`) drive transcript rendering.
- Progress steppers (LLM Combined, Overlap AudioShake) show intermediate states and durations.
- Speechbrain sample browser under `Debug/speechbrain_samples` is exposed inside the app when assets exist.
- `setupAudioGeneratorListener` listens for `message` events that drop generated audio files into the upload queue.

---

## 10. Metrics, Debugging & QA

- `metrics.js`: calculates DER/JER (with simplified Hungarian alignment) for comparing hypotheses vs. references.
- Debug assets (`Debug/`) include:
  - `speechbrain_dashboard.html` + `speechbrain_test_results.json`
  - `pyannote_dashboard.html`, `pyannote_test_results.json`
  - `case_results/CASE06.json`, diarization experiments.
- `TEST_SCENARIOS.md`: exhaustive manual QA checklist covering each feature area (LLM, audio, combined, overlap, Gemini, UI/UX, exports).
- Server logging: `server_debug.log` records JSON lines (timestamp, level, message, data). Frontend can POST console logs for correlation.

---

## 11. Developer Tooling & Scripts

- **Node scripts** (`package.json`):
  - `npm start`/`npm run dev`: launch Express server (includes UI + APIs).
  - `npm run check`: `check-setup.js` ensures dependencies, env, key files.
  - `npm run test-api`: verify OpenRouter connectivity.
  - `npm run test:mode3`: exercises overlap mode 3 (SpeechBrain).
- **Python utilities:**
  - `download_model_files.py`, `preload_pyannote_model.py`: warm caches.
  - `fix_hyperpyyaml.py`, `pyannote_patch.py`: compatibility patches.
  - `test_model_load.py`, `process_audio_temp.py`, `speechbrain_separation.py`, `pyannote_separation.py`: direct entrypoints.
  - `processor.py`, `audioshake_client.py`: pipeline orchestrators used in CLI/debug contexts.
- **Bash helpers:** `update-env.sh` for env templating.

---

## 12. Sample Data & Fixtures

- `audio examples/`: curated WAV/MP3 call center recordings for regression.
- `Features/`: `Realistic-audio-generator.html` (ElevenLabs + prompts) and older `prompt-manager.html`.
- `noise-samples/`: background noise segments for augmentation experiments.
- `temp_uploads/` + `uploads/`: runtime directories for in-flight and persisted audio (clean up manually when needed).

---

## 13. Known Constraints & Future Extensions

- **Localtunnel dependency:** AudioShake mode fails until the tunnel is up; consider production-ready HTTPS hosting (set `PUBLIC_URL`).
- **Model loading time:** PyAnnote/SpeechBrain scripts are heavy; prime caches with `download_model_files.py` before first run.
- **Transcript parsing:** `parseToStructuredJSON` expects well-formed JSON from LLMs; malformed outputs require manual inspection via logs.
- **Role consistency:** Correction/verification prompts enforce Agent/Client consistency, but downstream UI still shows original `speaker` labels if roles missing.
- **Multimodal fallback:** Gemini 2.5 Pro requires actual audio bytes; remote URLs must be downloaded first (see TODO comment in `callGeminiMultimodal`).
- **Storage hygiene:** `/uploads` retains files for AudioShake; set up cron or manual cleanup if disk usage matters.

Potential enhancements:
- Swap localtunnel with production reverse proxy.
- Add caching layer for prompt templates with hot reload watchers.
- Expand metrics dashboard to visualize DER/JER across experiments.
- Bundle CLI harnesses for overlap pipeline to aid automation.

---

## 14. Quick Reference – File Map

```
├── index.html / app.js            # Frontend UI + logic
├── server.js                      # Express server & API surface
├── lib/
│   ├── audioshake-client.js       # Node AudioShake Tasks integration
│   └── cloudinary-upload.js       # Cloudinary helper for HTTPS uploads
├── Python core
│   ├── process_audio_temp.py      # Speechmatics pipeline
│   ├── pyannote_separation.py     # Mode 2 separation
│   ├── speechbrain_separation.py  # Mode 3 separation
│   ├── audioshake_client.py       # Python AudioShake wrapper
│   └── processor.py               # Separation + role classification pipeline
├── prompts/                       # All prompt templates (LLM flows)
├── docs/
│   └── overlap_diarization_current.md  # Detailed overlap pipeline notes
├── Debug/                         # Dashboards, case data, sample stems
├── Features/                      # Prompt manager & audio generator standalone UIs
├── metrics.js / visualization.js  # Metrics + UI helpers
├── TEST_SCENARIOS.md              # Manual QA checklist
├── requirements.txt               # Python deps (pyannote, torch, speechbrain, etc.)
└── README.md                      # Quickstart + feature overview
```

Use this file as the canonical context package when onboarding contributors, prompting external LLMs, or planning feature changes.


