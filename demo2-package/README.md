# Demo2 package (user UI + Flask backend)

This package contains only the Demo2 user page and the Flask backend used by it.

## Stack
- UI: static HTML + JS (`public/demo2_user.html`)
- Backend: Python Flask (diarization + transcription)
- UI server: Node.js + Express (serves the page + local LLM proxy)
- Models: SpeechBrain (speaker embeddings) + Whisper (transcription)

## What's inside
- `app_demo2.py` - Flask backend for diarization + transcription.
- `server.js` - Node UI server that serves the Demo2 user page and proxies local LLM requests.
- `public/demo2_user.html` - Demo2 user page.
- `samples/` - Audio samples for manual testing.

## Requirements
- Python 3.11+ (recommended 3.11)
- Node.js 18+

## Install

### Python (Flask backend)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Node (UI server)
```bash
npm install
```

## Run

### 1) Start Flask backend
```bash
python app_demo2.py
```
Default port: `5005` (can be overridden with `DEMO2_PORT`).

### 2) Start UI server
```bash
npm start
```
Open `http://localhost:3000/`.

If Flask is running on a different host/port, open the UI with a query param:
```
http://localhost:3000/?api_base=http://HOST:5005
```

## How to upload a file
1. Open `http://localhost:3000/`.
2. Click "Upload file" and select an audio file (wav, mp3, m4a).
3. Wait for progress and the final markdown table.

## Webhook
Final post-processing webhook:
```
https://spsoft.app.n8n.cloud/webhook/diarization-send
```
This URL is called from `public/demo2_user.html`.

## Transcription engine switching (env)
Default engine is local Whisper. You can switch engines with env vars:

```
# Speechmatics
TRANSCRIPTION_ENGINE=speechmatics

# Groq Whisper (cloud)
TRANSCRIPTION_ENGINE=whisper_cloud
```

Notes:
- `TRANSCRIPTION_ENGINE=speechmatics` affects `/api/diarize` and `/api/diarize-and-transcribe`.
- `TRANSCRIPTION_ENGINE=whisper_cloud` (Groq) affects `/api/diarize-and-transcribe` only.
- You can also override per request with `transcriber=local|speechmatics|groq` in form-data.

## Speechmatics (optional)
If you want to use Speechmatics transcription instead of local Whisper, set the API key:
```
SPEECHMATICS_API_KEY=your_key_here
```
The backend reads this from `.env` (copy `env.example`).

## Groq Whisper (optional, cloud)
If you want to use cloud Whisper via Groq (for `/api/diarize-and-transcribe`), set:
```
GROQ_API_KEY=your_groq_key_here
```
Then either send `transcriber=groq` in the form-data request or set `TRANSCRIPTION_ENGINE=whisper_cloud`.

## Local LLM (optional, for in-page post-processing)
The UI calls `/api/llm/chat-completions-local`, which proxies to a local OpenAI-compatible server.
Set in `.env` (copy from `env.example`):
```
LOCAL_LLM_BASE_URL=http://127.0.0.1:3001
LOCAL_LLM_MODEL=openai/gpt-oss-20b
```

## Samples
- `samples/testdemopage.wav`
- `samples/audio-examples/`
- `samples/Call centre example.MP3`
