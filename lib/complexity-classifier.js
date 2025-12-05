/**
 * Complexity Classifier using Phi-3.5-Mini
 * 
 * Classifies text segments as SIMPLE, MEDIUM, or COMPLEX
 * for intelligent routing to appropriate processing tiers
 */

const axios = require('axios');

class ComplexityClassifier {
  constructor(config = {}) {
    this.provider = config.provider || 'ollama';

    if (this.provider === 'ollama') {
      this.baseURL = config.baseURL || 'http://localhost:11434';
      this.model = config.model || 'phi:3.5';
    } else if (this.provider === 'lmstudio') {
      this.baseURL = config.baseURL || 'http://localhost:1234/v1';
      this.model = config.model || 'microsoft/phi-3.5-mini-instruct';
    }

    this.timeout = config.timeout || 5000;
    this.maxRetries = config.maxRetries || 1;

    console.log(`[Classifier] Initialized with provider: ${this.provider}, model: ${this.model}`);
  }

  /**
   * Classify segment complexity
   */
  async classify(segment, sources) {
    const prompt = this._buildPrompt(segment, sources);

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        let result;

        if (this.provider === 'ollama') {
          result = await this._ollamaRequest(prompt);
        } else {
          result = await this._lmstudioRequest(prompt);
        }

        const complexity = this._parseComplexity(result);
        return complexity;

      } catch (error) {
        console.warn(`[Classifier] Attempt ${attempt} failed:`, error.message);
        if (attempt === this.maxRetries) {
          return { complexity: 'UNKNOWN', score: 0.5, error: true };
        }
      }
    }
  }

  _buildPrompt(segment, sources) {
    const generalCount = sources.general?.segments?.length || 0;
    const speaker1Count = sources.speaker1?.segments?.length || 0;
    const speaker2Count = sources.speaker2?.segments?.length || 0;

    return `You are a task complexity classifier for audio diarization segments.

TASK: Classify the complexity of analyzing this transcript segment.

TARGET SEGMENT:
Text: "${segment.text}"
Length: ${segment.text.split(' ').length} words
Start: ${segment.start}s, End: ${segment.end}s

AVAILABLE SOURCES:
- General (primary diarization): ${generalCount} segments
- Speaker1 (isolated track): ${speaker1Count} segments  
- Speaker2 (isolated track): ${speaker2Count} segments

COMPLEXITY LEVELS:
- SIMPLE: Segment is clearly in all sources or all sources
  → Script can handle easily (confidence needed: >0.9)

- MEDIUM: Segment has mixed elements but mostly clear
  → Script with verification needed (confidence: 0.6-0.9)

- COMPLEX: Segment is mixed-replica, ambiguous
  → Needs full LLM analysis (confidence: <0.6)

RESPOND WITH JSON ONLY:
{
  "complexity": "SIMPLE|MEDIUM|COMPLEX",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

CLASSIFY NOW:`;
  }

  async _ollamaRequest(prompt) {
    const response = await axios.post(
      `${this.baseURL}/api/generate`,
      {
        model: this.model,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0, // Zero temperature for deterministic results
          num_predict: 150,
          top_p: 0.9
        }
      },
      { timeout: this.timeout }
    );

    return response.data.response;
  }

  async _lmstudioRequest(prompt) {
    const response = await axios.post(
      `${this.baseURL}/chat/completions`,
      {
        model: this.model,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0,
        max_tokens: 150
      },
      { timeout: this.timeout }
    );

    return response.data.choices[0].message.content;
  }

  _parseComplexity(text) {
    text = text.trim();

    try {
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          complexity: parsed.complexity || 'UNKNOWN',
          score: parsed.confidence || 0.5,
          reasoning: parsed.reasoning || ''
        };
      }
    } catch (error) {
      // Fallback: heuristic parsing
    }

    // Simple heuristic fallback
    const lower = text.toLowerCase();

    if (lower.includes('simple')) {
      return { complexity: 'SIMPLE', score: 0.7, reasoning: 'Heuristic: keyword match' };
    } else if (lower.includes('complex')) {
      return { complexity: 'COMPLEX', score: 0.7, reasoning: 'Heuristic: keyword match' };
    } else {
      return { complexity: 'MEDIUM', score: 0.5, reasoning: 'Heuristic: default' };
    }
  }

  async healthCheck() {
    try {
      if (this.provider === 'ollama') {
        const response = await axios.get(
          `${this.baseURL}/api/tags`,
          { timeout: 5000 }
        );

        const hasModel = response.data.models?.some(m => m.name.includes('phi'));
        if (!hasModel) {
          console.warn('[Classifier] Model not found in Ollama. Run: ollama pull phi:3.5');
          return false;
        }
      }

      console.log('[Classifier] Health check passed ✓');
      return true;
    } catch (error) {
      console.error('[Classifier] Health check failed:', error.message);
      return false;
    }
  }
}

module.exports = ComplexityClassifier;
