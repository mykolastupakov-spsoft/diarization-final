# INSTALLATION & SETUP GUIDE

## 🚀 QUICK SETUP (30-45 minutes)

### Prerequisites
- Node.js ≥ 14
- npm ≥ 6
- 16GB+ RAM (or 12GB with 3-tier routing)
- GPU with ≥ 12GB VRAM (optional but recommended)

---

## OPTION 1: 2-Tier System (GPT-OSS 20B Only)

### Recommended for:
✅ Maximum accuracy (98.8%)
✅ Have 16GB+ VRAM
✅ Don't need memory optimization

### Setup Steps

#### 1. Install GPT-OSS 20B

**Via Ollama (recommended):**
```bash
# Install Ollama from https://ollama.ai

# Download model
ollama pull gpt-oss:20b

# Start server
ollama serve
# Server will be on http://localhost:11434
```

**Or via LM Studio:**
- Download: https://lmstudio.ai/
- Search for "gpt-oss 20b"
- Download and load model
- Start server

#### 2. Install Dependencies
```bash
npm install axios
```

#### 3. Copy Files
```bash
cp lib/gpt-oss-client.js ./lib/
cp lib/hybrid-analyzer.js ./lib/
cp lib/alignment-engine.js ./lib/
```

#### 4. Update server.js
```javascript
const HybridAnalyzer = require('./lib/hybrid-analyzer');

app.post('/api/analyze-text', async (req, res) => {
  try {
    const analyzer = new HybridAnalyzer({
      enableLLM: true,
      provider: 'ollama',
      gptOSSURL: 'http://localhost:11434'
    });

    const result = await analyzer.analyze(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

#### 5. Start & Test
```bash
npm start

# Test
curl -X POST http://localhost:3000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d @examples/example-request.json
```

---

## OPTION 2: 3-Tier System (Phi-3.5-Mini + GPT-OSS 20B)

### Recommended for:
✅ VRAM < 16GB
✅ Variable complexity (not all complex)
✅ Edge deployment
⚠️ Latency +2-4ms acceptable

### Setup Steps

#### 1. Install Both Models

```bash
# Terminal 1: Install Phi-3.5-Mini
ollama pull phi:3.5
ollama serve

# Terminal 2: Install GPT-OSS 20B
ollama pull gpt-oss:20b
# Will share Ollama server from Terminal 1
```

#### 2. Copy ALL Files
```bash
cp lib/gpt-oss-client.js ./lib/
cp lib/complexity-classifier.js ./lib/  # NEW!
cp lib/hybrid-analyzer.js ./lib/
cp lib/alignment-engine.js ./lib/
```

#### 3. Update server.js (with 3-tier routing)
```javascript
const HybridAnalyzer = require('./lib/hybrid-analyzer');

app.post('/api/analyze-text', async (req, res) => {
  try {
    const analyzer = new HybridAnalyzer({
      enableLLM: true,
      useComplexityRouting: true,  // NEW!
      provider: 'ollama',
      classifier: {
        provider: 'ollama',
        model: 'phi:3.5'
      },
      gptOSSURL: 'http://localhost:11434'
    });

    const result = await analyzer.analyze(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

#### 4. Start & Test
```bash
npm start

curl -X POST http://localhost:3000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d @examples/example-request.json
```

---

## COMPARISON: Setup Complexity

| Step | 2-Tier | 3-Tier | Time |
|------|--------|--------|------|
| Install models | 1 model | 2 models | +10 min for Phi |
| Copy files | 3 files | 4 files | Same |
| Configure | Simple | Medium | +5 min |
| Test | 5 min | 10 min | +5 min |
| Total | ~30 min | ~45 min | +15 min |

---

## TROUBLESHOOTING

### Problem: "Model not found"

```bash
ollama list  # Check installed models
ollama pull gpt-oss:20b  # Re-download if needed
ollama ps    # Check running processes
```

### Problem: "Out of Memory"

**For 3-Tier:**
```bash
# Use quantized Phi (if available)
ollama pull phi:3.5-q4

# Or use smaller classifier
"model": "qwen:1.5b"
```

**For 2-Tier:**
```bash
# Use quantized GPT-OSS (if available)
ollama pull gpt-oss:20b-q4

# Or reduce batch size in analyzer
maxBatchSize: 1
```

### Problem: "Slow inference"

```javascript
// Increase confidence threshold to use Script more
scriptConfidenceThreshold: 0.95  // More script, less LLM

// Or reduce context for classifier
maxSegmentsInContext: 3  // Shorter prompts
```

### Problem: "Port already in use"

```bash
# If Ollama server already running
lsof -i :11434  # Check process
kill -9 <PID>   # Kill if needed

# Or use different port
OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

---

## VERIFICATION

After setup, verify each component:

```bash
# 1. Check Ollama
curl http://localhost:11434/api/tags

# 2. Test GPT-OSS
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "gpt-oss:20b", "prompt": "Hello", "stream": false}'

# 3. Test Phi (if 3-tier)
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "phi:3.5", "prompt": "Hello", "stream": false}'

# 4. Test API endpoint
curl -X POST http://localhost:3000/api/analyze-text \
  -H "Content-Type: application/json" \
  -d @examples/example-request.json
```

---

## NEXT STEPS

1. ✅ Read: `docs/00-FINAL-RECOMMENDATION.md`
2. ✅ Choose: 2-Tier or 3-Tier
3. ✅ Follow appropriate setup
4. ✅ Test on your data
5. ✅ Deploy to production

---

**Questions?** See `docs/01-GPT-OSS-20B-Integration.md` (Troubleshooting section)
