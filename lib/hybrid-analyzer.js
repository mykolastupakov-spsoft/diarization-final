/**
 * Hybrid Analyzer with 3-Tier Classification System
 * 
 * Uses ComplexityClassifier (Phi-3.5-Mini) to route segments to appropriate processing:
 * - SIMPLE (conf > 0.9) → Script only
 * - MEDIUM (conf 0.6-0.9) → Script + quick LLM verification
 * - COMPLEX (conf < 0.6) → Full LLM analysis
 */

const ComplexityClassifier = require('./complexity-classifier');
const { analyzeText, analyzeTextWithLLM } = require('./textAnalysis');

class HybridAnalyzer {
  constructor(config = {}) {
    this.enableLLM = config.enableLLM !== false;
    this.useComplexityRouting = config.useComplexityRouting || false;
    this.provider = config.provider || 'ollama';
    
    // LLM configuration
    this.llmModel = config.llmModel;
    this.apiUrl = config.apiUrl;
    this.apiKey = config.apiKey;
    this.useLocalLLM = config.useLocalLLM || false;
    this.mode = config.mode || 'smart';
    
    // Complexity routing configuration
    if (this.useComplexityRouting) {
      this.classifier = new ComplexityClassifier({
        provider: config.classifier?.provider || this.provider,
        baseURL: config.classifier?.baseURL || config.baseURL || 'http://localhost:11434',
        model: config.classifier?.model || 'phi:3.5',
        timeout: config.classifier?.timeout || 5000
      });
      
      // Thresholds for routing
      this.simpleThreshold = config.simpleThreshold || 0.9;
      this.mediumThreshold = config.mediumThreshold || 0.6;
      
      console.log('[HybridAnalyzer] Initialized with 3-tier routing:', {
        provider: this.provider,
        classifierModel: this.classifier.model,
        simpleThreshold: this.simpleThreshold,
        mediumThreshold: this.mediumThreshold
      });
    } else {
      console.log('[HybridAnalyzer] Initialized with 2-tier (no complexity routing)');
    }
  }

  /**
   * Main analysis method
   */
  async analyze(payload) {
    if (!this.useComplexityRouting) {
      // 2-tier mode: use existing logic
      return this._analyze2Tier(payload);
    }

    // 3-tier mode: classify complexity first
    return this._analyze3Tier(payload);
  }

  /**
   * 2-Tier analysis (existing behavior)
   */
  async _analyze2Tier(payload) {
    if (!this.enableLLM) {
      // Script only
      return analyzeText(payload);
    }

    // Full LLM analysis
    try {
      return await analyzeTextWithLLM(
        payload,
        this.llmModel,
        this.apiUrl,
        this.apiKey,
        this.useLocalLLM,
        this.mode
      );
    } catch (error) {
      console.warn('[HybridAnalyzer] LLM analysis failed, falling back to script:', error.message);
      return analyzeText(payload);
    }
  }

  /**
   * 3-Tier analysis with complexity routing
   */
  async _analyze3Tier(payload) {
    // Parse markdown to get segments
    const segments = this._parseMarkdownSegments(payload.markdown);
    
    if (!segments || segments.length === 0) {
      console.warn('[HybridAnalyzer] No segments found in markdown, using script mode');
      return analyzeText(payload);
    }

    // Classify each segment
    const classifiedSegments = await this._classifySegments(segments, payload);
    
    // Group segments by complexity
    const simpleSegments = classifiedSegments.filter(s => s.complexity === 'SIMPLE');
    const mediumSegments = classifiedSegments.filter(s => s.complexity === 'MEDIUM');
    const complexSegments = classifiedSegments.filter(s => s.complexity === 'COMPLEX');
    
    console.log('[HybridAnalyzer] Classification results:', {
      total: segments.length,
      simple: simpleSegments.length,
      medium: mediumSegments.length,
      complex: complexSegments.length
    });

    // Process each group
    const results = {
      Blue: [],
      Green: [],
      Red: []
    };

    // SIMPLE: Script only
    if (simpleSegments.length > 0) {
      const simplePayload = this._createPayloadForSegments(simpleSegments, payload);
      const simpleResult = analyzeText(simplePayload);
      this._mergeResults(results, simpleResult);
    }

    // MEDIUM: Script + quick LLM verification (optional, can use script for now)
    if (mediumSegments.length > 0) {
      const mediumPayload = this._createPayloadForSegments(mediumSegments, payload);
      // For now, use script. Can add quick LLM verification later
      const mediumResult = analyzeText(mediumPayload);
      this._mergeResults(results, mediumResult);
    }

    // COMPLEX: Full LLM analysis
    if (complexSegments.length > 0 && this.enableLLM) {
      try {
        const complexPayload = this._createPayloadForSegments(complexSegments, payload);
        const complexResult = await analyzeTextWithLLM(
          complexPayload,
          this.llmModel,
          this.apiUrl,
          this.apiKey,
          this.useLocalLLM,
          this.mode
        );
        this._mergeResults(results, complexResult);
      } catch (error) {
        console.warn('[HybridAnalyzer] LLM analysis for complex segments failed, using script:', error.message);
        const complexPayload = this._createPayloadForSegments(complexSegments, payload);
        const complexResult = analyzeText(complexPayload);
        this._mergeResults(results, complexResult);
      }
    } else if (complexSegments.length > 0) {
      // LLM disabled, use script
      const complexPayload = this._createPayloadForSegments(complexSegments, payload);
      const complexResult = analyzeText(complexPayload);
      this._mergeResults(results, complexResult);
    }

    return results;
  }

  /**
   * Classify segments by complexity
   */
  async _classifySegments(segments, payload) {
    const sources = {
      general: payload.general,
      speaker1: payload.speaker1,
      speaker2: payload.speaker2
    };

    const classified = [];
    
    for (const segment of segments) {
      try {
        const classification = await this.classifier.classify(segment, sources);
        
        // Determine complexity based on confidence
        let complexity = 'MEDIUM';
        if (classification.score >= this.simpleThreshold) {
          complexity = 'SIMPLE';
        } else if (classification.score < this.mediumThreshold) {
          complexity = 'COMPLEX';
        }

        classified.push({
          ...segment,
          complexity: complexity,
          confidence: classification.score,
          reasoning: classification.reasoning
        });
      } catch (error) {
        console.warn(`[HybridAnalyzer] Classification failed for segment, defaulting to MEDIUM:`, error.message);
        classified.push({
          ...segment,
          complexity: 'MEDIUM',
          confidence: 0.5,
          reasoning: 'Classification failed, defaulting to MEDIUM'
        });
      }
    }

    return classified;
  }

  /**
   * Parse markdown table to extract segments
   */
  _parseMarkdownSegments(markdown) {
    if (!markdown || typeof markdown !== 'string') {
      return [];
    }

    const segments = [];
    const lines = markdown.split('\n');
    
    // Look for markdown table rows
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Skip header and separator lines
      if (line.startsWith('|') && !line.includes('---') && !line.toLowerCase().includes('segment id')) {
        const cells = line.split('|').map(c => c.trim()).filter(c => c);
        
        if (cells.length >= 5) {
          // Expected format: Segment ID | Speaker | Text | Start Time | End Time
          const segmentId = cells[0];
          const speaker = cells[1];
          const text = cells[2];
          const startTime = parseFloat(cells[3]) || 0;
          const endTime = parseFloat(cells[4]) || 0;

          if (text && text.length > 0) {
            segments.push({
              id: segmentId,
              speaker: speaker,
              text: text,
              start: startTime,
              end: endTime
            });
          }
        }
      }
    }

    return segments;
  }

  /**
   * Create payload for specific segments
   */
  _createPayloadForSegments(segments, originalPayload) {
    // Create a filtered markdown table with only these segments
    const filteredMarkdown = this._createMarkdownFromSegments(segments);
    
    return {
      ...originalPayload,
      markdown: filteredMarkdown
    };
  }

  /**
   * Create markdown table from segments
   */
  _createMarkdownFromSegments(segments) {
    if (segments.length === 0) {
      return '';
    }

    let markdown = '| Segment ID | Speaker | Text | Start Time | End Time |\n';
    markdown += '|------------|---------|------|------------|----------|\n';
    
    for (const segment of segments) {
      markdown += `| ${segment.id || ''} | ${segment.speaker || ''} | ${segment.text || ''} | ${segment.start || 0} | ${segment.end || 0} |\n`;
    }

    return markdown;
  }

  /**
   * Merge results from different processing tiers
   */
  _mergeResults(target, source) {
    if (source.Blue) {
      target.Blue = target.Blue.concat(source.Blue);
    }
    if (source.Green) {
      target.Green = target.Green.concat(source.Green);
    }
    if (source.Red) {
      target.Red = target.Red.concat(source.Red);
    }
  }

  /**
   * Health check for classifier
   */
  async healthCheck() {
    if (this.useComplexityRouting && this.classifier) {
      return await this.classifier.healthCheck();
    }
    return true;
  }
}

module.exports = HybridAnalyzer;

