### Automated Testing

| Command | Description |
| --- | --- |
| `npm run test:overlap-fixes` | Runs the regression test for the “✅ Apply Overlap Fixes” flow using `audio examples/OverlappingCallCenterWithoutBackground.MP3`. Requires the local server to be running with valid Azure/Speechmatics credentials. |

# Arabic/English Speaker Diarization

Web application for speaker diarization and linguistic analysis of Arabic/English conversations.

## ✨ Features

- **Direct OpenRouter Integration** - No n8n dependency
- **Two AI Models**:
  - **Fast Mode** (gpt-oss-120b) - ~2-3 sec, Good quality
  - **Smart Mode** (gpt-5.1) - ~3-5 sec, Excellent quality
- **Structured JSON Output** - Compatible with Speechmatics format
- **Text Preservation** - Original transcript text is never modified
- **Verification Step** - Optional additional testing and correction
- **Fix Mistakes** - Manual trigger for verification on existing results
- **AudioShake Tasks Integration** - Optional Cloudinary-powered speaker separation before Speechmatics/Gemini processing

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure API Key

Create `.env` file:

```bash
cp env.example .env
```

Edit `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-v1-your_key_here
APP_URL=http://localhost:3000
PORT=3000
```

**Get API Key:**
1. Go to https://openrouter.ai/
2. Sign up / Log in
3. Navigate to "Keys" section
4. Create new API key
5. Copy to `.env` file

### 3. Check Setup

```bash
npm run check
```

This will verify:
- ✅ Dependencies installed
- ✅ Required packages present
- ✅ Environment configuration
- ✅ Required files exist

### 4. Start Server

```bash
npm start
```

Open browser: **http://localhost:3000**

## 📖 Usage

1. Navigate to **"Transcript Context Diarization"** tab
2. Select model (Fast or Smart)
3. Paste transcript text or upload file
4. Click **"Analyze Speakers"**
5. Review results

## 🔊 AudioShake Tasks via localtunnel

AudioShake requires a publicly reachable HTTPS URL for each uploaded file. The server now spins up a **localtunnel** automatically so your local `uploads/` folder is visible to AudioShake without any extra services.

### Quick start

1. Install dependencies: `npm install`
2. Copy `.env.example` to `.env` and set `AUDIOSHAKE_API_KEY`
3. Start the server with `npm start`
4. Watch the console for:

   ```
   🌐 Localtunnel URL: https://funny-cat-42.loca.lt
   ✓ Files accessible at: https://funny-cat-42.loca.lt/uploads/
   ```

5. Upload an audio file via the frontend → choose **Combined → AudioShake Agent Classifier**

### Fixed subdomain (optional)

Add this to `.env` to request a persistent URL (if available):

```
LOCALTUNNEL_SUBDOMAIN=my-unique-name
```

Now the server will always expose `https://my-unique-name.loca.lt/uploads/filename`.

### Checking tunnel status

- The frontend pings `/api/tunnel-status` before each AudioShake run
- You can also visit that endpoint in the browser to verify readiness

### Troubleshooting

| Issue | Fix |
| --- | --- |
| "Tunnel not ready" alert | Wait a few seconds after server start, then retry |
| AudioShake can't fetch file | Open the tunnel URL in your browser once, then click "Continue" to trust the cert |
| Need a stable hostname | Set `LOCALTUNNEL_SUBDOMAIN` in `.env`, relaunch |
| Use your own domain | Set `PUBLIC_URL=https://your-domain.com` (skips localtunnel entirely) |

> ⚠️ Files stay in `uploads/` so that AudioShake can download them. Clean up manually if needed.

## 🎙️ Azure Speech to Text (optional)

You can replace Speechmatics with Azure Speech-to-Text while keeping the rest of the overlap pipeline unchanged.

1. Create an Azure Speech resource and grab the subscription key + region.
2. Provision (or reuse) an Azure Blob Storage container that the app can upload audio files to.
3. Update `.env` with:

```
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=eastus
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=...
AZURE_STORAGE_CONTAINER=diarization-audio
# Optional if AccountKey missing from connection string
# AZURE_STORAGE_ACCOUNT_KEY=abc123
```

When the frontend Audio Settings → “Transcription Model” is set to one of the Azure options, the backend behaves as follows:

- **Model 2 – Azure STT (Batch)**  
  Uploads audio to your blob container, submits a batch diarization job via `azure_stt.py`, then normalizes the response into the Speechmatics schema. Highest accuracy but requires a stable REST endpoint/region.
- **Model 3 – Azure STT Realtime**  
  Streams the audio through the Azure Speech SDK (`azure_realtime.py`), so there’s no blob upload and no polling. Works well on UAEnorth, supports up to ~30 minutes of audio, and emits live speaker labels.

Dependencies:

```
pip install azure-storage-blob azure-cognitiveservices-speech
```

If the Azure variables are missing (or you stick with Model 1), the system automatically falls back to Speechmatics.

## 🔊 AudioShake Tasks Setup (Optional but recommended)

1. **Create Cloudinary account** (or reuse an existing one) and note the cloud name, API key, and API secret.
2. **Request AudioShake Tasks API access** and generate an API key from https://developer.audioshake.ai/tasks.
3. **Update `.env`** with the following values:

```env
AUDIOSHAKE_API_KEY=ashke_xxx
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_key
CLOUDINARY_API_SECRET=your_cloudinary_secret
```

4. Restart the server. The Combined → “AudioShake Agent Classifier” mode will now upload audio to Cloudinary, call AudioShake Tasks, and run the existing Speechmatics/Gemini pipeline on each separated stem. If these variables are missing, the app automatically falls back to the legacy Python-based AudioShake pipeline.

## 📁 Project Structure

```
.
├── server.js              # Backend server with OpenRouter integration
├── app.js                 # Frontend JavaScript
├── index.html             # Main UI
├── prompts/               # Prompt templates folder (all prompts editable here)
│   ├── arabic_diarization_complete_prompt.txt  # Main diarization prompt template
│   ├── diarization_verification_prompt.txt      # Verification/correction prompt
│   ├── system_diarization.txt                  # System message for diarization
│   ├── system_translation.txt                  # System message for translation
│   ├── translation_prompt_template.txt         # Translation prompt template
│   └── ai_analysis_prompt_template.txt         # AI analysis prompt template
├── package.json           # Dependencies
├── env.example            # Environment variables template
├── check-setup.js         # Setup verification script
├── QUICKSTART.md          # Quick start guide
├── OPENROUTER_SETUP.md    # Detailed setup documentation
└── README.md              # This file
```

## 🔧 Scripts

- `npm start` - Start the server
- `npm run dev` - Start in development mode (same as start)
- `npm run check` - Verify setup and configuration
- `npm run test-api` - Test OpenRouter API connection (requires API key)

## 📚 Documentation

- **QUICKSTART.md** - Quick start guide (3 steps)
- **OPENROUTER_SETUP.md** - Detailed setup and architecture documentation
- **diagnostic_prompt_for_perplexity.md** - Diagnostic information for research

## 🔒 Security

- API key is stored on backend only (`.env` file)
- Never exposed in frontend JavaScript
- `.env` file is in `.gitignore`

## 💰 Cost Estimation

**FAST MODE:**
- ~$0.0008-0.0016 per request
- ~$0.80-1.60 per 1000 requests/month

**SMART MODE:**
- ~$0.0016-0.0032 per request
- ~$1.60-3.20 per 1000 requests/month

## 🐛 Troubleshooting

**"OPENROUTER_API_KEY not set"**
- Check that `.env` file exists
- Verify it contains `OPENROUTER_API_KEY=sk-or-v1-...`
- Restart server after editing `.env`

**"Cannot find module 'axios'"**
```bash
npm install
```

**"ECONNREFUSED"**
- Check internet connection
- Verify OpenRouter API is accessible

**Run setup check:**
```bash
npm run check
```

## 📝 License

ISC

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

---

**Ready to use!** 🎉

For detailed information, see `OPENROUTER_SETUP.md`.

