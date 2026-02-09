const path = require('path');
const express = require('express');
const axios = require('axios');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const LOCAL_LLM_BASE_URL = (process.env.LOCAL_LLM_BASE_URL || 'http://127.0.0.1:3001').replace(/\/$/, '');
const LOCAL_LLM_MODEL = process.env.LOCAL_LLM_MODEL || 'openai/gpt-oss-20b';
const LOCAL_LLM_API_KEY = process.env.LOCAL_LLM_API_KEY || '';

app.use(express.json({ limit: '2mb' }));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'demo2_user.html'));
});

app.post('/api/llm/chat-completions-local', async (req, res) => {
  try {
    if (!LOCAL_LLM_BASE_URL) {
      return res.status(500).json({
        success: false,
        error: 'LOCAL_LLM_BASE_URL is not configured'
      });
    }

    const apiUrl = `${LOCAL_LLM_BASE_URL}/v1/chat/completions`;
    const headers = { 'Content-Type': 'application/json' };

    if (LOCAL_LLM_API_KEY) {
      headers.Authorization = `Bearer ${LOCAL_LLM_API_KEY}`;
    }

    const payload = {
      ...req.body,
      model: req.body.model || LOCAL_LLM_MODEL
    };

    const response = await axios.post(apiUrl, payload, {
      headers,
      timeout: 300000
    });

    return res.json(response.data);
  } catch (err) {
    console.error('[LLM] Proxy error:', err.message);
    return res.status(500).json({
      success: false,
      error: err.message
    });
  }
});

app.use(express.static(path.join(__dirname, 'public')));

app.listen(PORT, () => {
  console.log(`Demo2 UI server listening on http://localhost:${PORT}`);
});
