# Demo2 package (user UI + Flask backend)

This package contains only the Demo2 user page and the Flask backend used by it.

## What's inside
- `app_demo2.py` - Flask backend for diarization + transcription.
- `server.js` - Node UI server that serves the Demo2 user page and proxies local LLM requests.
- `public/demo2_user.html` - Demo2 user page.
- `samples/` - Audio samples for manual testing.

## Requirements
- Python 3.10+ (recommended 3.11)
- Node.js 18+

## Setup

### Python (Flask backend)
```bash
python -m venv .venv
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
