// Override console methods to send logs to server
// NOTE: On demo page (/demo), logging is disabled
(function() {
  'use strict';
  
  // Only override in browser environment
  if (typeof window === 'undefined' || typeof fetch === 'undefined') {
    return;
  }
  
  // Check if we're on demo page
  const isDemoPage = () => {
    try {
      return window.location.pathname === '/demo' || window.location.pathname.includes('/demo');
    } catch (e) {
      return false;
    }
  };
  
  // Store original console methods
  const originalConsole = {
    log: console.log.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
    info: console.info.bind(console),
    debug: console.debug.bind(console)
  };
  
  // Send log to server
  const sendLogToServer = (level, message, args = []) => {
    // Don't log on demo page
    if (isDemoPage()) {
      return;
    }
    
    // Convert args to serializable format
    const data = args.map(arg => {
      if (typeof arg === 'object' && arg !== null) {
        try {
          return JSON.stringify(arg);
        } catch (e) {
          return String(arg);
        }
      }
      return arg;
    });
    
    // Send log to server (fire and forget)
    try {
      fetch('/api/client-log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          level,
          message: String(message),
          data: { args: data },
          timestamp: new Date().toISOString(),
          url: window.location.href,
          userAgent: navigator.userAgent || 'unknown'
        })
      }).catch(() => {
        // Silently fail if server is not available
      });
    } catch (e) {
      // Silently fail if fetch is not available or other error
    }
  };
  
  // Override console methods
  console.log = function(...args) {
    originalConsole.log(...args);
    if (args.length > 0) {
      sendLogToServer('info', args[0], args.slice(1));
    }
  };
  
  console.warn = function(...args) {
    originalConsole.warn(...args);
    if (args.length > 0) {
      sendLogToServer('warn', args[0], args.slice(1));
    }
  };
  
  console.error = function(...args) {
    originalConsole.error(...args);
    if (args.length > 0) {
      sendLogToServer('error', args[0], args.slice(1));
    }
  };
  
  console.info = function(...args) {
    originalConsole.info(...args);
    if (args.length > 0) {
      sendLogToServer('info', args[0], args.slice(1));
    }
  };
  
  console.debug = function(...args) {
    originalConsole.debug(...args);
    if (args.length > 0) {
      sendLogToServer('debug', args[0], args.slice(1));
    }
  };
})();

// Data
const SERVICES = [
  {
    id: 'speechmatics',
    name: 'Model3',
    displayLabel: '',
    freeTier: '480 minutes/month',
    pricingPerHour: 0.30,
    features: '32 languages, real-time translation',
    signupUrl: 'https://portal.speechmatics.com/signup',
    docsUrl: 'https://docs.speechmatics.com'
  }
];

const OPENROUTER_MODELS = [
  {
    id: 'meta-llama/llama-3.2-3b-instruct:free',
    name: 'Model4',
    description: 'Fast and efficient model'
  },
  {
    id: 'google/gemini-flash-1.5:free',
    name: 'Model4',
    description: 'Google model, good understanding'
  },
  {
    id: 'mistralai/mistral-7b-instruct:free',
    name: 'Model4',
    description: 'Balanced performance'
  },
  {
    id: 'qwen/qwen-2-7b-instruct:free',
    name: 'Model4',
    description: 'Strong multilingual support'
  }
];

const PLAYBACK_SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

const SAMPLE_PHRASES = [
  'Yes, I understand what you mean',
  'Let us discuss this topic in more detail',
  'I agree with your point of view',
  'This is a very interesting idea',
  'Perhaps we should consider alternative options',
  'I think this is the right approach',
  'Thank you for the explanation',
  'This is an important point to consider',
  'Hello, I understand what you mean',
  'Let\'s explore this question step by step',
  'I agree with your approach',
  'That sounds like a solid plan',
  'Maybe we should evaluate a few alternatives',
  'I think this solution will work',
  'Appreciate the clarification',
  'This aspect is critical for success'
];

const overlapMergeUtils = typeof window !== 'undefined' ? window.OverlapMergeUtils : null;

const TEXT_SERVICE_KEY = 'text-service';
const TEXT_SERVICE_NAME = 'Text Mode 📝';
const LEGACY_TEXT_SERVICE_KEYS = ['openai-text'];

// Cached regex patterns for better performance
const SERVICE_MARKER_REGEX = /^--\s+[^-]+\s+--\s*$/i;
const SEPARATOR_REGEX = /^---+\s*$/;
const RECORDING_HEADER_PATTERN = /^={3,}\s*(?:запис|recording|transcript|conversation|разговор|диалог|record|file|audio|call|session|تسجيل|محادثة|مكالمة|جلسة|ملف|صوتي)\s+\d+\s*:/i;
const HEADER_SEPARATOR_PATTERN = /^={3,}\s*$/;
const METADATA_PATTERN = /^(AI SUMMARY|GENERATED|SUMMARY)/i;
const RECORDING_NAME_EXTRACT_PATTERN = /^={3,}\s*(.+?)\s*={3,}$/;
const RECORDING_NAME_PREFIX_PATTERN = /^(?:запис|recording|transcript|conversation|разговор|диалог|record|file|audio|call|session|تسجيل|محادثة|مكالمة|جلسة|ملف|صوتي)\s+\d+\s*:\s*(.+)$/i;

const STORAGE_KEYS = {
  apiKeys: 'diarization_api_keys',
  enabledServices: 'diarization_enabled_services'
};

function getServiceDisplayName(service) {
  if (!service) return '';
  return service.displayLabel !== undefined ? service.displayLabel : service.name;
}

// Application State
const app = {
  config: {
    services: {},
    selectedServices: []
  },
  uploadedFile: null,
  fileDuration: 0,
  testResults: {},
  audioshakeResults: null,
  llmMultimodalAudioFile: null,
  testingInProgress: false,
  testingCancelled: false,
  groundTruth: null,
  comprehensiveMetrics: {},
  currentTab: 'summary',
  errorLogs: {}, // Store detailed error logs for each service
  translationState: {
    currentLanguage: 'original',
    isTranslating: false,
    lastError: null
  },
  replicaAudioObjectUrl: null,
  translationConfig: {
    endpoint: '/api/translate',
    model: 'openai/gpt-oss-20b'
  },
  pendingFlowIntent: null,
  requestedView: null,
  recordings: [],
  activeRecordingId: null,
  recordingQueue: [],
  currentRecordingLanguage: null,
  currentRecordingSpeakerCount: null,
  currentDiarizationMethod: 'llm', // 'llm', 'audio', or 'combined'
  roleFilter: { agent: true, client: true, none: true }, // Role filtering state
  currentTunnelUrl: null,
  tunnelStatusCache: null,
  lastTunnelStatusCheck: 0,
  tunnelStatusInterval: null,
  speechbrainSamples: {
    data: [],
    lastLoaded: null,
    loading: false,
    error: null
  },

  async init() {
    await this.loadSavedConfig();
    this.renderServiceConfig();
    this.setupDragAndDrop();
    this.loadSavedApiKeys();
    this.updateTranslationStatusLabel('Original Arabic transcripts');
    this.updateTranslationControlsState();
    this.renderRecordingsQueue();
    this.refreshReplicasViewIfActive();
    // Ensure button state is updated after initialization
    this.updateConfiguredCount();
    // Initialize diarization method (default to LLM)
    if (document.getElementById('uploadView')) {
      this.switchDiarizationMethod('llm');
    }

    // Combined mode selection removed - always uses 'hybrid' mode
    // No need for event handlers since there's only one mode now
    
    // Додаємо обробники для audioPipelineMode на вкладці Audio
    document.querySelectorAll('input[name="audioPipelineMode"]').forEach(radio => {
      radio.addEventListener('change', () => this.handleAudioModeChange());
    });
    // Викликаємо один раз при ініціалізації, щоб показати статус тунелю якщо вже вибрано AudioShake на вкладці Audio
    this.handleAudioModeChange();
    
    // Додаємо обробники для diarizationModel на вкладці LLM
    document.querySelectorAll('input[name="diarizationModel"]').forEach(radio => {
      radio.addEventListener('change', () => this.handleLLMModelChange());
    });
    // Викликаємо один раз при ініціалізації
    this.handleLLMModelChange();
    
    const overlapModeSelect = document.getElementById('overlapPipelineMode');
    if (overlapModeSelect) {
      overlapModeSelect.addEventListener('change', () => this.handleOverlapPipelineModeChange());
      this.handleOverlapPipelineModeChange();
    }

    this.pendingFlowIntent = this.loadFlowIntentFromStorage();
    
    // Setup transcript textarea event listener
    const transcriptTextarea = document.getElementById('transcriptText');
    if (transcriptTextarea) {
      transcriptTextarea.addEventListener('input', () => {
        this.updateAnalyzeButtonState();
      });
      transcriptTextarea.addEventListener('paste', () => {
        setTimeout(() => this.updateAnalyzeButtonState(), 10);
      });
      this.updateAnalyzeButtonState();
    }

    // Listen for audio files from Audio Generator
    this.setupAudioGeneratorListener();
    this.checkAudioGeneratorFile();
    
    // Handle hash navigation
    if (window.location.hash === '#upload') {
      this.showView('uploadView');
    }

     // Handle query parameter navigation
    try {
      const params = new URLSearchParams(window.location.search);
      this.requestedView = params.get('view');
      if (this.requestedView === 'overlap') {
        this.showView('configView');
        this.switchDiarizationMethod('overlap');
      }
    } catch (error) {
      console.warn('Failed to parse view query parameter', error);
    }
    
    // Listen for hash changes
    window.addEventListener('hashchange', () => {
      if (window.location.hash === '#upload') {
        this.showView('uploadView');
        this.checkAudioGeneratorFile(); // Check again when navigating to upload
      }
    });
  },

  loadFlowIntentFromStorage() {
    try {
      const raw = localStorage.getItem('diarizationFlowIntent');
      if (!raw) return null;
      localStorage.removeItem('diarizationFlowIntent');
      const intent = JSON.parse(raw);
      if (intent && typeof intent === 'object') {
        return intent;
      }
      return null;
    } catch (error) {
      console.warn('Failed to parse diarization flow intent', error);
      return null;
    }
  },

  renderAudioshakeProgress() {
    const progressContainer = document.getElementById('progressContainer');
    if (!progressContainer) return;
    progressContainer.innerHTML = `
      <div style="margin-bottom: var(--space-24);">
        <h3 style="margin-bottom: var(--space-16);">Overlap Diarization</h3>
        <div id="audioshakeSteps" style="display: flex; flex-direction: column; gap: var(--space-12);">
          ${[
            { step: 1, title: 'Upload & Submit to AudioShake', description: 'Preparing audio file...' },
            { step: 2, title: 'Speaker Separation', description: 'Waiting to start...' },
            { step: 3, title: 'Transcription & Role Classification', description: 'Waiting to start...' }
          ].map(({ step, title, description }) => `
            <div class="step-progress" data-audioshake-step="${step}" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
              <div style="display: flex; align-items: center; gap: var(--space-8);">
                <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">${step}</span>
                <div style="flex: 1;">
                  <div style="font-weight: var(--font-weight-medium);">${title}</div>
                  <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">${description}</div>
                </div>
                <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
              </div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  },

  async checkTunnelStatus(force = false) {
    const now = Date.now();
    if (!force && this.tunnelStatusCache && now - this.lastTunnelStatusCheck < 5000) {
      return this.tunnelStatusCache;
    }

    try {
      const response = await fetch(`/api/tunnel-status?_=${now}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      this.tunnelStatusCache = data;
      this.lastTunnelStatusCheck = now;
      this.currentTunnelUrl = data.url || null;
      return data;
    } catch (error) {
      console.error('Failed to check tunnel status:', error);
      this.tunnelStatusCache = { active: false, error: error.message };
      this.lastTunnelStatusCheck = now;
      return this.tunnelStatusCache;
    }
  },

  async ensureTunnelReady() {
    const status = await this.checkTunnelStatus();
    const isReady = Boolean(status?.active);
    if (!isReady) {
      console.warn('Tunnel not ready:', status?.message || status?.error);
    }
    return isReady;
  },

  updateAudioshakeStepStatus(step, status, description) {
    const stepEl = document.querySelector(`[data-audioshake-step="${step}"]`);
    if (!stepEl) return;
    const statusEl = stepEl.querySelector('.step-status');
    const descriptionEl = stepEl.querySelector('.step-description');

    const states = {
      pending: { label: '⏸️ Pending', color: 'var(--color-border)', opacity: 0.6 },
      processing: { label: '⏳ Processing...', color: 'var(--color-primary)', opacity: 1 },
      completed: { label: '✅ Done', color: 'var(--color-success)', opacity: 1 },
      error: { label: '⚠️ Failed', color: 'var(--color-error)', opacity: 1 }
    };

    const state = states[status] || states.pending;
    if (statusEl) {
      statusEl.textContent = state.label;
    }
    if (description && descriptionEl) {
      descriptionEl.textContent = description;
    }
    stepEl.style.opacity = state.opacity;
    stepEl.style.borderLeft = `4px solid ${state.color}`;
    const indicator = stepEl.querySelector('.step-indicator');
    if (indicator) {
      indicator.style.background = state.color;
    }
  },

  async runAudioshakePipeline(recording) {
    if (!recording) {
      alert('No audio file available for AudioShake processing.');
      return;
    }

    let progressInterval = null;
    let elapsedSeconds = 0;

    // Update progress every 5 seconds while waiting
    const startProgressUpdates = () => {
      if (progressInterval) return; // Already running
      progressInterval = setInterval(() => {
        elapsedSeconds += 5;
        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;
        const timeStr = minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
        this.updateAudioshakeStepStatus(2, 'processing', `Separating speakers... (${timeStr})`);
      }, 5000);
    };

    const stopProgressUpdates = () => {
      if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
      }
    };

    try {
      this.testingInProgress = true;
      this.showView('testingView');
      this.renderAudioshakeProgress();
      this.updateAudioshakeStepStatus(1, 'processing', 'Uploading audio to AudioShake...');

      const tunnelReady = await this.ensureTunnelReady();
      if (!tunnelReady) {
        this.updateAudioshakeStepStatus(1, 'error', 'Public tunnel not ready. Please wait and try again.');
        this.testingInProgress = false;
        this.showView('uploadView');
        alert('Server tunnel is not ready yet. Please wait a few seconds and try again.');
        return;
      }

      const buildFormData = () => {
        const fd = new FormData();
        if (recording.file) {
          fd.append('audio', recording.file);
        } else if (recording.url) {
          fd.append('url', recording.url);
        } else {
          throw new Error('Audio file is required for AudioShake processing.');
        }
        fd.append('variant', 'n_speaker');
        // Перевіряємо, з якої вкладки викликається - Audio або Combined
        const language = document.getElementById('audioLanguageSelect')?.value 
          || document.getElementById('combinedLanguageSelect')?.value 
          || document.getElementById('overlapLanguageSelect')?.value 
          || 'en';
        fd.append('language', language);
        return fd;
      };

      let result = null;
      let usedTasksPipeline = false;

      try {
        this.updateAudioshakeStepStatus(1, 'completed', 'Audio uploaded');
        this.updateAudioshakeStepStatus(2, 'processing', 'Separating speakers...');
        startProgressUpdates();

        const tasksResponse = await fetch('/api/audioshake-tasks', {
          method: 'POST',
          body: buildFormData()
        });

        stopProgressUpdates();

        if (tasksResponse.ok) {
          result = await tasksResponse.json();
          usedTasksPipeline = true;
          console.log('✅ AudioShake Tasks pipeline completed');
        } else if (tasksResponse.status === 501) {
          console.warn('AudioShake Tasks integration not configured on server, falling back to legacy pipeline.');
        } else {
          let errorPayload = {};
          try {
            errorPayload = await tasksResponse.json();
          } catch (e) {
            // ignore parsing
          }
          const baseMessage = errorPayload.error || 'AudioShake tasks pipeline failed';
          throw new Error(baseMessage);
        }
      } catch (tasksError) {
        stopProgressUpdates();
        if (tasksError.message && tasksError.message.includes('tasks pipeline failed')) {
          throw tasksError;
        }
        // Non-fatal errors (like 501) will fall back to legacy pipeline
        if (tasksError.message && !tasksError.message.includes('AudioShake tasks pipeline failed')) {
          console.warn('AudioShake tasks pipeline unavailable:', tasksError.message);
        }
      }

      if (!result) {
        this.updateAudioshakeStepStatus(2, 'processing', 'Separating speakers (legacy pipeline)...');
        startProgressUpdates();
        
        const legacyResponse = await fetch('/api/audioshake-pipeline', {
          method: 'POST',
          body: buildFormData()
        });

        stopProgressUpdates();

        if (!legacyResponse.ok) {
          stopProgressUpdates();
          let errorPayload = {};
          try {
            errorPayload = await legacyResponse.json();
          } catch (e) {
            // ignore parsing error
          }
          const baseMessage = errorPayload.error || 'AudioShake pipeline failed';
          const details = typeof errorPayload.details === 'string' && errorPayload.details.trim().length > 0
            ? ` (${errorPayload.details})`
            : '';
          throw new Error(`${baseMessage}${details}`);
        }

        result = await legacyResponse.json();
      }

      if (!result.success) {
        const baseMessage = result.error || 'AudioShake pipeline returned an error';
        const details = typeof result.details === 'string' && result.details.trim().length > 0
          ? ` (${result.details})`
          : '';
        throw new Error(`${baseMessage}${details}`);
      }

      // Логуємо результати для діагностики
      console.log('📊 AudioShake results received:', {
        success: result.success,
        speakersCount: Array.isArray(result.speakers) ? result.speakers.length : 0,
        resultsCount: Array.isArray(result.results) ? result.results.length : 0,
        usedTasksPipeline,
        firstSpeaker: result.speakers?.[0] || result.results?.[0] || null
      });
      
      this.audioshakeResults = {
        ...result,
        usedTasksPipeline,
        success: result.success !== false // Гарантуємо, що success буде true якщо не false
      };
      this.updateAudioshakeStepStatus(1, 'completed', 'Audio uploaded to AudioShake');
      this.updateAudioshakeStepStatus(2, 'completed', 'Speakers separated into stems');
      this.updateAudioshakeStepStatus(3, 'completed', 'Tracks transcribed & classified');

      let summaryRecord;
      if (usedTasksPipeline) {
        const speakers = Array.isArray(result.speakers) ? result.speakers : [];
        summaryRecord = {
          success: true,
          serviceName: 'Speaker Separation',
          processingTime: result.duration || 0,
          speakerCount: speakers.length,
          cost: result.cost ? `$${result.cost}` : '—',
          segments: speakers.flatMap((speaker) => {
            const diarizationSegments =
              speaker.diarization?.recordings?.flatMap((rec) => rec.results?.speechmatics?.segments || []) || [];
            return diarizationSegments.map((segment) => ({
              ...segment,
              speaker: speaker.speaker || speaker.name || segment.speaker
            }));
          }),
          rawData: {
            source: 'audioshake_tasks',
            taskId: result.taskId,
            speakers
          }
        };
      } else {
        const processingSeconds = result.metrics?.processing_seconds ?? result.metrics?.processingSeconds ?? 0;
        summaryRecord = {
          success: true,
          serviceName: 'Speaker Separation',
          processingTime: processingSeconds || 0,
          speedFactor: result.metrics?.speed_factor || 0,
          speakerCount: Array.isArray(result.results) ? result.results.length : 0,
          cost: '0.0000',
          segments: [],
          rawData: {
            source: 'audioshake',
            jobId: result.jobId || null,
            tracks: result.results || [],
            metrics: result.metrics || {}
          }
        };
      }

      this.testResults = { audioshake: summaryRecord };
      this.testingInProgress = false;
      this.showResults();
    } catch (error) {
      stopProgressUpdates();
      console.error('AudioShake pipeline error:', error);
      this.updateAudioshakeStepStatus(1, 'error', error.message);
      this.updateAudioshakeStepStatus(2, 'error', 'Stopped due to previous error');
      this.updateAudioshakeStepStatus(3, 'error', 'Stopped due to previous error');
      alert(`AudioShake pipeline failed: ${error.message}`);
      this.testingInProgress = false;
      this.showView('uploadView');
    }
  },

  async loadSavedConfig() {
    let validServiceIds = [];

    try {
      const savedKeys = localStorage.getItem(STORAGE_KEYS.apiKeys);
      if (savedKeys) {
        const keys = JSON.parse(savedKeys);
        
        validServiceIds = Object.keys(keys).filter(serviceId => 
          SERVICES.some(s => s.id === serviceId)
        );
        
        const validKeys = {};
        validServiceIds.forEach(serviceId => {
          validKeys[serviceId] = keys[serviceId];
        });
        
        this.config.services = validKeys;
        
        if (Object.keys(keys).length !== validServiceIds.length) {
          const removedServices = Object.keys(keys).filter(id => !validServiceIds.includes(id));
          console.log('Removed invalid services from config:', removedServices);
          this.saveApiKeys();
        }
      }
    } catch (error) {
      console.error('Failed to load saved API keys:', error);
    }

    // Try to load Speechmatics key from server
    await this.loadSpeechmaticsKeyFromServer();

    this.loadEnabledServices(validServiceIds);
  },

  async loadSpeechmaticsKeyFromServer() {
    try {
      const response = await fetch('/api/speechmatics-key');
      if (response.ok) {
        const data = await response.json();
        if (data.apiKey) {
          this.config.services['speechmatics'] = data.apiKey;
          console.log('✅ Loaded Speechmatics API key from server');
          // Update UI if already rendered
          const input = document.getElementById('apikey_speechmatics');
          if (input) {
            input.value = data.apiKey;
            const card = document.getElementById('card_speechmatics');
            if (card) card.classList.add('configured');
            this.updateServiceStatus('speechmatics', '✅');
          }
          this.saveApiKeys();
        }
      } else {
        console.log('Speechmatics API key not configured on server (this is OK if using local keys)');
      }
    } catch (error) {
      console.log('Could not load Speechmatics key from server:', error.message);
      // This is OK - user might be using local keys
    }
  },

  loadEnabledServices(validServiceIds = []) {
    let enabled = [];

    try {
      const savedEnabled = localStorage.getItem(STORAGE_KEYS.enabledServices);
      if (savedEnabled) {
        const parsed = JSON.parse(savedEnabled);
        if (Array.isArray(parsed)) {
          enabled = parsed.filter(id => validServiceIds.includes(id));
        }
      }
    } catch (error) {
      console.error('Failed to load enabled services:', error);
    }

    // Якщо немає збережених сервісів, встановити за замовчуванням тільки Model3 (speechmatics)
    // Це дозволяє використовувати демо-режим навіть без API ключів
    if (enabled.length === 0) {
      enabled = ['speechmatics']; // За замовчуванням тільки Model3
    } else if (validServiceIds.length > 0) {
      const missing = validServiceIds.filter(id => !enabled.includes(id));
      enabled = [...enabled, ...missing];
    }

    // Завжди переконатися, що speechmatics є в обраних (для production-ready режиму)
    if (!enabled.includes('speechmatics')) {
      enabled = ['speechmatics'];
    }

    this.config.selectedServices = enabled;
    this.saveEnabledServices();
  },

  saveApiKeys() {
    // Save API keys to localStorage
    try {
      localStorage.setItem(STORAGE_KEYS.apiKeys, JSON.stringify(this.config.services));
    } catch (error) {
      console.error('Failed to save API keys:', error);
    }
  },

  saveEnabledServices() {
    try {
      localStorage.setItem(STORAGE_KEYS.enabledServices, JSON.stringify(this.config.selectedServices));
    } catch (error) {
      console.error('Failed to save enabled services:', error);
    }
  },

  isServiceEnabled(serviceId) {
    return this.config.selectedServices.includes(serviceId);
  },

  toggleServiceParticipation(serviceId) {
    if (this.testingInProgress) {
      alert('Please wait for the current test to complete before changing the service list.');
      return;
    }

    if (!this.config.services[serviceId]) {
      alert('Please save a valid API key for this service first.');
      return;
    }

    const enabled = this.isServiceEnabled(serviceId);
    if (enabled) {
      this.config.selectedServices = this.config.selectedServices.filter(id => id !== serviceId);
    } else {
      this.config.selectedServices = [...this.config.selectedServices, serviceId];
    }

    this.saveEnabledServices();
    this.updateConfiguredCount();
    this.updateServiceToggleUI(serviceId);
  },

  updateServiceToggleUI(serviceId) {
    const toggleBtn = document.getElementById(`toggle_service_${serviceId}`);
    const card = document.getElementById(`card_${serviceId}`);
    const hasKey = Boolean(this.config.services[serviceId]);
    const enabled = this.isServiceEnabled(serviceId);

    if (toggleBtn) {
      toggleBtn.disabled = !hasKey || this.testingInProgress;
      toggleBtn.textContent = enabled ? '🚫 Disable from test' : '✅ Add to test';
      toggleBtn.title = !hasKey
        ? 'Please save API key first'
        : enabled
          ? 'Service will be skipped in next test'
          : 'Service will be included in next test';
    }

    if (card) {
      card.classList.toggle('disabled', hasKey && !enabled);
    }
  },

  createTranslationState() {
    return {
      currentLanguage: 'original',
      isTranslating: false,
      lastError: null
    };
  },

  createRecordingEntry(file) {
    // Try to get language from specific tab fields, fallback to default
    const language = document.getElementById('audioLanguageSelect')?.value 
      || document.getElementById('combinedLanguageSelect')?.value 
      || document.getElementById('overlapLanguageSelect')?.value 
      || 'ar';
    const speakerCount = document.getElementById('audioSpeakerCount')?.value 
      || document.getElementById('combinedSpeakerCount')?.value 
      || document.getElementById('overlapSpeakerCount')?.value 
      || '';
    const duration = this.estimateDurationFromFile(file);
    return {
      id: `rec_${Date.now()}_${Math.random().toString(16).slice(2)}`,
      file,
      name: file.name,
      size: file.size,
      duration,
      language,
      speakerCount,
      status: 'pending',
      results: {},
      translationState: this.createTranslationState(),
      aggregated: {}
    };
  },

  getActiveRecording() {
    if (!this.activeRecordingId && this.recordings.length > 0) {
      this.activeRecordingId = this.recordings[0].id;
    }
    return this.recordings.find(rec => rec.id === this.activeRecordingId) || null;
  },

  setActiveRecording(recordingId, options = {}) {
    const recording = this.recordings.find(rec => rec.id === recordingId);
    if (!recording) return;
    this.activeRecordingId = recordingId;
    this.uploadedFile = recording.file;
    this.fileDuration = recording.duration;
    if (!recording.results) {
      recording.results = {};
    }
    this.testResults = recording.results;
    if (!recording.translationState) {
      recording.translationState = this.createTranslationState();
    }
    this.translationState = recording.translationState;
    this.renderActiveRecordingInfo();
    this.updateRecordingSelectOptions();
    this.updateTranslationControlsState();
    this.updateTranslationStatusLabel();
    this.renderRecordingsQueue();
    
    // Update Apply Overlap Fixes button visibility based on engine
    const engine = recording.overlapMetadata?.steps?.step1?.engine || 
                  recording.results?.speechmatics?.engine ||
                  recording.results?.['overlap-corrected']?.engine;
    const isAzure = engine && (engine.toLowerCase().includes('azure') || engine.toLowerCase() === 'azure_realtime');
    this.updateApplyOverlapFixesButtonVisibility(recording, isAzure);
    
    // Update separated speakers players when setting active recording
    this.updateSeparatedSpeakersPlayers(recording);
    
    if (!options.skipResultsRender) {
      this.showResults();
    }
  },

  selectRecording(recordingId) {
    if (!recordingId) return;
    this.setActiveRecording(recordingId);
  },

  estimateDurationFromFile(file) {
    return Math.round((file.size / (1024 * 1024)) * 60);
  },

  renderRecordingsQueue() {
    const container = document.getElementById('recordingsQueue');
    const list = document.getElementById('recordingsList');
    if (!container || !list) return;

    // Hide recordings queue for LLM method (text-only, no audio files needed)
    if (this.currentDiarizationMethod === 'llm') {
      container.style.display = 'none';
      return;
    }

    if (this.recordings.length === 0) {
      container.style.display = 'none';
      list.innerHTML = '<p style="color: var(--color-text-secondary);">Queue is empty. Add audio files for batch mode.</p>';
      this.updateStartButtonState();
      return;
    }

    container.style.display = 'block';
    const rows = this.recordings.map((rec, idx) => `
      <tr ${rec.id === this.activeRecordingId ? 'style="background: var(--color-bg-2);"' : ''}>
        <td>${idx + 1}</td>
        <td>
          <strong>${rec.name}</strong>
          <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">
            ${(rec.size / (1024 * 1024)).toFixed(2)} MB • ~${Math.floor(rec.duration / 60)}:${(rec.duration % 60).toString().padStart(2, '0')}
          </div>
        </td>
        <td>${rec.language ? rec.language.toUpperCase() : '—'}</td>
        <td>${rec.speakerCount || '—'}</td>
        <td>${this.formatRecordingStatus(rec.status)}</td>
        <td>
          <div style="display:flex; gap: var(--space-8); flex-wrap: wrap;">
            <button class="btn btn-secondary btn-sm" onclick="app.setActiveRecording('${rec.id}')">Activate</button>
            <button class="btn btn-secondary btn-sm" onclick="app.removeRecording('${rec.id}')" ${this.testingInProgress ? 'disabled' : ''}>🗑️</button>
          </div>
        </td>
      </tr>
    `).join('');

    list.innerHTML = `
      <table class="recordings-table">
        <thead>
          <tr>
            <th>#</th>
            <th>File</th>
            <th>Language</th>
            <th>Speakers</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    this.updateStartButtonState();
    this.updateRecordingSelectOptions();
    this.renderActiveRecordingInfo();
    this.renderReplicaRecordingSelect();
    this.showAggregatedView();
  },

  showAggregatedView() {
    const summary = document.getElementById('recordingsSummary');
    if (!summary) return;
    if (this.recordings.length === 0) {
      summary.innerHTML = '<p style="color: var(--color-text-secondary);">No recordings to display.</p>';
      return;
    }

    const totalServices = Math.max(this.config.selectedServices.length, 1);
    const rows = this.recordings.map((rec, idx) => {
      const successCount = rec.results ? Object.values(rec.results).filter(r => r.success).length : 0;
      const translationLang = rec.translationState?.currentLanguage || 'original';
      let translationLabel = 'Original';
      if (translationLang === 'uk') translationLabel = 'Ukrainian';
      else if (translationLang === 'en') translationLabel = 'English';
      else if (translationLang === 'ar') translationLabel = 'Arabic';

      return `
        <tr>
          <td>${idx + 1}</td>
          <td>${rec.name}</td>
          <td>${rec.status === 'completed' ? '✅ Ready' : rec.status === 'processing' ? '⏳ Processing' : rec.status === 'cancelled' ? '⚠️ Cancelled' : '🕒 Pending'}</td>
          <td>${successCount}/${totalServices}</td>
          <td>${translationLabel}</td>
        </tr>
      `;
    }).join('');

    summary.innerHTML = `
      <table class="recordings-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Recording</th>
            <th>Status</th>
            <th>Successful Services</th>
            <th>Translation</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  },

  updateStartButtonState() {
    const startBtn = document.getElementById('startTestBtn');
    if (!startBtn) return;
    const recordingsCount = this.recordings.length;
    const enabledServices = this.config.selectedServices.length;
    startBtn.disabled = recordingsCount === 0 || enabledServices === 0 || this.testingInProgress;
    if (recordingsCount > 0) {
      startBtn.textContent = `🚀 Start Testing (${recordingsCount} file${recordingsCount === 1 ? '' : 's'})`;
    } else {
      startBtn.textContent = '🚀 Start Testing';
    }
  },

  formatRecordingStatus(status) {
    switch (status) {
      case 'processing':
        return '<span class="status status-processing">Processing…</span>';
      case 'completed':
        return '<span class="status status-success">Ready</span>';
      case 'cancelled':
        return '<span class="status status-warning">Cancelled</span>';
      default:
        return '<span class="status status-pending">Pending</span>';
    }
  },

  updateRecordingSelectOptions() {
    const select = document.getElementById('recordingSelect');
    if (!select) return;
    if (this.recordings.length === 0) {
      select.innerHTML = '<option value="">— No Recordings —</option>';
      return;
    }
    select.innerHTML = this.recordings.map(rec => `
      <option value="${rec.id}" ${rec.id === this.activeRecordingId ? 'selected' : ''}>
        ${rec.name} ${rec.status === 'completed' ? '✓' : ''}
      </option>
    `).join('');
  },

  renderActiveRecordingInfo() {
    const recording = this.getActiveRecording();
    const container = document.getElementById('fileInfoContainer');
    const info = document.getElementById('fileInfo');
    if (!container || !info) return;
    if (this.currentDiarizationMethod === 'combined') {
      container.style.display = 'none';
      return;
    }
    if (this.currentDiarizationMethod === 'combined') {
      container.style.display = 'none';
      info.innerHTML = '';
      return;
    }
    if (!recording) {
      container.style.display = 'none';
      info.innerHTML = '';
      return;
    }
    container.style.display = 'block';
    info.innerHTML = `
      <div class="file-info-row">
        <strong>File Name:</strong>
        <span>${recording.name}</span>
      </div>
      <div class="file-info-row">
        <strong>Size:</strong>
        <span>${(recording.size / (1024 * 1024)).toFixed(2)} MB</span>
      </div>
      <div class="file-info-row">
        <strong>Approx. Duration:</strong>
        <span>~${Math.floor(recording.duration / 60)}:${(recording.duration % 60).toString().padStart(2, '0')}</span>
      </div>
      <div class="file-info-row">
        <strong>Language:</strong>
        <span>${recording.language ? recording.language.toUpperCase() : '—'}</span>
      </div>
      <div class="file-info-row">
        <strong>Expected Speakers:</strong>
        <span>${recording.speakerCount || '—'}</span>
      </div>
      <div class="file-info-row">
        <strong>Status:</strong>
        <span>${this.formatRecordingStatus(recording.status)}</span>
      </div>
    `;
  },

  removeRecording(recordingId) {
    if (this.testingInProgress) {
      alert('Cannot remove recording during testing.');
      return;
    }
    const idx = this.recordings.findIndex(rec => rec.id === recordingId);
    if (idx === -1) return;
    this.recordings.splice(idx, 1);
    if (this.activeRecordingId === recordingId) {
      this.activeRecordingId = this.recordings[0]?.id || null;
      if (this.activeRecordingId) {
        this.setActiveRecording(this.activeRecordingId, { skipResultsRender: true });
      } else {
        this.testResults = {};
      }
    }
    this.renderRecordingsQueue();
    this.showResults();
  },

  clearRecordings() {
    if (this.testingInProgress) {
      alert('Cannot clear queue during testing.');
      return;
    }
    if (confirm('Clear entire recordings queue?')) {
      this.recordings = [];
      this.recordingQueue = [];
      this.activeRecordingId = null;
      this.testResults = {};
      this.uploadedFile = null;
      this.renderRecordingsQueue();
      this.showResults();
    }
  },

  renderRecordingsQueue() {
    const container = document.getElementById('recordingsQueue');
    const list = document.getElementById('recordingsList');
    if (!container || !list) return;

    if (this.recordings.length === 0) {
      container.style.display = 'none';
      list.innerHTML = '<p style="color: var(--color-text-secondary);">Queue is empty. Add audio files for batch mode.</p>';
      this.updateStartButtonState();
      return;
    }

    container.style.display = 'block';
    const rows = this.recordings.map((rec, idx) => `
      <tr ${rec.id === this.activeRecordingId ? 'style="background: var(--color-bg-2);"' : ''}>
        <td>${idx + 1}</td>
        <td>
          <strong>${rec.name}</strong>
          <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">
            ${(rec.size / (1024 * 1024)).toFixed(2)} MB • ~${Math.floor(rec.duration / 60)}:${(rec.duration % 60).toString().padStart(2, '0')}
          </div>
        </td>
        <td>${rec.language?.toUpperCase?.() || '—'}</td>
        <td>${rec.speakerCount || '—'}</td>
        <td>${this.formatRecordingStatus(rec.status)}</td>
        <td>
          <div style="display:flex; gap: var(--space-8); flex-wrap: wrap;">
            <button class="btn btn-secondary btn-sm" onclick="app.setActiveRecording('${rec.id}')">Activate</button>
            <button class="btn btn-secondary btn-sm" onclick="app.removeRecording('${rec.id}')">🗑️</button>
          </div>
        </td>
      </tr>
    `).join('');

    list.innerHTML = `
      <table class="recordings-table">
        <thead>
          <tr>
            <th>#</th>
            <th>File</th>
            <th>Language</th>
            <th>Speakers</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    this.updateStartButtonState();
    this.updateRecordingSelectOptions();
    if (this.currentDiarizationMethod === 'combined') {
      const container = document.getElementById('fileInfoContainer');
      const info = document.getElementById('fileInfo');
      if (container) container.style.display = 'none';
      if (info) info.innerHTML = '';
    } else {
      this.renderActiveRecordingInfo();
    }
  },

  updateStartButtonState() {
    const startBtn = document.getElementById('startTestBtn');
    if (!startBtn) return;
    const recordingsCount = this.recordings.length;
    const enabledServices = this.config.selectedServices.length;
    startBtn.disabled = recordingsCount === 0 || enabledServices === 0 || this.testingInProgress;
    if (recordingsCount > 0) {
      startBtn.textContent = `🚀 Start Testing (${recordingsCount} file${recordingsCount === 1 ? '' : 's'})`;
    } else {
      startBtn.textContent = '🚀 Start Testing';
    }
  },

  formatRecordingStatus(status) {
    switch (status) {
      case 'processing':
        return '<span class="status status-processing">Processing…</span>';
      case 'completed':
        return '<span class="status status-success">Ready</span>';
      case 'cancelled':
        return '<span class="status status-warning">Cancelled</span>';
      default:
        return '<span class="status status-pending">Pending</span>';
    }
  },

  updateRecordingSelectOptions() {
    const select = document.getElementById('recordingSelect');
    if (!select) return;
    if (this.recordings.length === 0) {
      select.innerHTML = '<option value="">— No Recordings —</option>';
      return;
    }
    select.innerHTML = this.recordings.map(rec => `
      <option value="${rec.id}" ${rec.id === this.activeRecordingId ? 'selected' : ''}>
        ${rec.name} ${rec.status === 'completed' ? '✓' : ''}
      </option>
    `).join('');
  },

  renderActiveRecordingInfo() {
    const recording = this.getActiveRecording();
    const container = document.getElementById('fileInfoContainer');
    const info = document.getElementById('fileInfo');
    if (!container || !info) return;
    if (!recording) {
      container.style.display = 'none';
      info.innerHTML = '';
      return;
    }
    container.style.display = 'block';
    info.innerHTML = `
      <div class="file-info-row">
        <strong>File Name:</strong>
        <span>${recording.name}</span>
      </div>
      <div class="file-info-row">
        <strong>Size:</strong>
        <span>${(recording.size / (1024 * 1024)).toFixed(2)} MB</span>
      </div>
      <div class="file-info-row">
        <strong>Approx. Duration:</strong>
        <span>~${Math.floor(recording.duration / 60)}:${(recording.duration % 60).toString().padStart(2, '0')}</span>
      </div>
      <div class="file-info-row">
        <strong>Language:</strong>
        <span>${recording.language?.toUpperCase?.() || '—'}</span>
      </div>
      <div class="file-info-row">
        <strong>Expected Speakers:</strong>
        <span>${recording.speakerCount || '—'}</span>
      </div>
      <div class="file-info-row">
        <strong>Status:</strong>
        <span>${this.formatRecordingStatus(recording.status)}</span>
      </div>
    `;
  },

  removeRecording(recordingId) {
    const idx = this.recordings.findIndex(rec => rec.id === recordingId);
    if (idx === -1) return;
    this.recordings.splice(idx, 1);
    if (this.activeRecordingId === recordingId) {
      this.activeRecordingId = this.recordings[0]?.id || null;
      if (this.activeRecordingId) {
        this.setActiveRecording(this.activeRecordingId, { skipResultsRender: true });
      } else {
        this.testResults = {};
      }
    }
    this.renderRecordingsQueue();
    this.showResults();
  },

  clearRecordings() {
    if (confirm('Clear entire recordings queue?')) {
      this.recordings = [];
      this.recordingQueue = [];
      this.activeRecordingId = null;
      this.testResults = {};
      this.uploadedFile = null;
      this.renderRecordingsQueue();
      this.showResults();
    }
  },

  loadSavedApiKeys() {
    // Load saved keys into UI after rendering
    Object.keys(this.config.services).forEach(serviceId => {
      const input = document.getElementById(`apikey_${serviceId}`);
      const card = document.getElementById(`card_${serviceId}`);
      const testBtn = document.getElementById(`test_${serviceId}`);
      const statusDiv = document.getElementById(`test_status_${serviceId}`);
      
      if (input && this.config.services[serviceId]) {
        input.value = this.config.services[serviceId];
        if (testBtn) testBtn.disabled = false;
        if (statusDiv) {
          statusDiv.textContent = '✅ Saved';
          statusDiv.style.color = 'var(--color-success)';
        }
        if (card) card.classList.add('configured');
        this.updateServiceStatus(serviceId, '✅');
      }
    });
    SERVICES.forEach(service => this.updateServiceToggleUI(service.id));
    this.updateConfiguredCount();
  },

  renderServiceConfig() {
    const container = document.getElementById('servicesContainer');
    if (!container) {
      console.error('Services container not found!');
      return;
    }
    
    console.log('Rendering service cards...');
    
    const cardsHTML = SERVICES.map(service => `
      <div class="service-card" id="card_${service.id}">
        <div class="service-card-header">
          <div>
            <div class="service-card-title">${getServiceDisplayName(service) || ''}</div>
            <div class="service-card-free-tier">🎁 ${service.freeTier}</div>
          </div>
          <div class="service-card-status" id="status_${service.id}">⚪</div>
        </div>
        
        <div class="service-card-body">
          <div class="service-card-input-group">
            <label class="label" for="apikey_${service.id}">API Key</label>
            <div class="service-card-input-wrapper">
              <input type="password" 
                     class="api-key-input" 
                     id="apikey_${service.id}"
                     placeholder="Enter API key..."
                     data-service="${service.id}"
                     oninput="app.onApiKeyInput('${service.id}')">
              <button class="service-card-input-toggle" 
                      onclick="app.togglePasswordVisibility('${service.id}')"
                      title="Show/Hide">
                👁️
              </button>
            </div>
          </div>
          
          <div class="service-card-footer">
            <div style="display: flex; gap: var(--space-8); flex-wrap: wrap;">
              <button class="btn btn-secondary btn-sm" 
                      id="test_${service.id}"
                      onclick="app.testConnection('${service.id}')"
                      disabled>
                🔌 Test Connection
              </button>
              <button class="btn btn-secondary btn-sm" 
                      id="toggle_service_${service.id}"
                      onclick="app.toggleServiceParticipation('${service.id}')">
                ⚙️ Manage
              </button>
            </div>
            <div id="test_status_${service.id}" style="font-size: var(--font-size-sm); color: var(--color-text-secondary); min-height: 20px; margin-top: var(--space-8);"></div>
          </div>
        </div>
        
        <div class="service-card-links">
          <a href="${service.signupUrl}" target="_blank">Get API Key ↗</a>
          <a href="${service.docsUrl}" target="_blank">Documentation ↗</a>
        </div>
      </div>
    `).join('');
    
    container.innerHTML = cardsHTML;
    console.log('Service cards rendered:', SERVICES.length);
    
    SERVICES.forEach(service => this.updateServiceToggleUI(service.id));
    this.updateConfiguredCount();
  },

  onApiKeyInput(serviceId) {
    const input = document.getElementById(`apikey_${serviceId}`);
    const testBtn = document.getElementById(`test_${serviceId}`);
    const statusDiv = document.getElementById(`test_status_${serviceId}`);
    
    if (input.value.trim().length > 0) {
      testBtn.disabled = false;
      statusDiv.textContent = '';
      this.updateServiceStatus(serviceId, '⚪');
    } else {
      testBtn.disabled = true;
      statusDiv.textContent = '';
      this.updateServiceStatus(serviceId, '⚪');
      delete this.config.services[serviceId];
      this.config.selectedServices = this.config.selectedServices.filter(id => id !== serviceId);
      this.saveApiKeys(); // Save after deletion
      this.saveEnabledServices();
      this.updateConfiguredCount();
      this.updateServiceToggleUI(serviceId);
    }
  },

  togglePasswordVisibility(serviceId) {
    const input = document.getElementById(`apikey_${serviceId}`);
    if (input.type === 'password') {
      input.type = 'text';
    } else {
      input.type = 'password';
    }
  },

  async testConnection(serviceId) {
    const input = document.getElementById(`apikey_${serviceId}`);
    const testBtn = document.getElementById(`test_${serviceId}`);
    const statusDiv = document.getElementById(`test_status_${serviceId}`);
    const card = document.getElementById(`card_${serviceId}`);
    const apiKey = input.value.trim();
    
    if (!apiKey) {
      statusDiv.textContent = '❌ Enter API key';
      statusDiv.style.color = 'var(--color-error)';
      return;
    }
    
    testBtn.disabled = true;
    statusDiv.textContent = '🔄 Checking...';
    statusDiv.style.color = 'var(--color-text-secondary)';
    this.updateServiceStatus(serviceId, '🟡');
    
    // Simulate API connection test (2-3 seconds)
    await this.sleep(2000 + Math.random() * 1000);
    
    // 90% success rate for demo
    const success = Math.random() < 0.9;
    
    if (success) {
      this.config.services[serviceId] = apiKey;
      if (!this.config.selectedServices.includes(serviceId)) {
        this.config.selectedServices.push(serviceId);
      }
      this.saveApiKeys(); // Save to localStorage
      this.saveEnabledServices();
      this.updateServiceStatus(serviceId, '✅');
      statusDiv.textContent = '✅ Connection successful and saved';
      statusDiv.style.color = 'var(--color-success)';
      card.classList.add('configured');
      this.updateServiceToggleUI(serviceId);
    } else {
      this.updateServiceStatus(serviceId, '❌');
      statusDiv.textContent = '❌ Invalid API key';
      statusDiv.style.color = 'var(--color-error)';
      card.classList.remove('configured');
    }
    
    testBtn.disabled = false;
    this.updateConfiguredCount();
  },

  updateConfiguredCount() {
    const count = this.config.selectedServices.length;
    const countEl = document.getElementById('configuredCount');
    const continueBtn = document.getElementById('continueToUploadBtn');
    
    if (countEl) {
      // Show "Ready" status - always ready now since LLM diarization doesn't require service configuration
      countEl.textContent = 'Ready';
    }
    
    if (continueBtn) {
      // Always enable the button - LLM diarization works without service configuration
      continueBtn.disabled = false;
      continueBtn.title = 'Click to proceed to file upload';
    }
  },

  continueToUpload() {
    console.log('continueToUpload called');
    // No need to check for services - LLM diarization works without service configuration
    try {
      if (!this.showView) {
        console.error('showView method not found');
        alert('Error: showView method not available');
        return;
      }
      
      this.showView('uploadView');
      
      // Also initialize the LLM tab when switching to upload view
      setTimeout(() => {
        const uploadView = document.getElementById('uploadView');
        if (uploadView && this.switchDiarizationMethod) {
          this.switchDiarizationMethod('llm');
        }
      }, 100);
    } catch (error) {
      console.error('Error in continueToUpload:', error);
      alert('Error switching to upload view: ' + error.message);
    }
  },

  saveConfig() {
    // Deprecated - configuration is saved automatically on test
    const count = this.config.selectedServices.length;
    if (count > 0) {
      alert(`Configured ${count} service(s)`);
      this.showView('uploadView');
    } else {
      alert('Please test connection with at least one service first.');
    }
  },

  clearAllData() {
    if (confirm('Are you sure you want to clear all configurations and results?')) {
      this.config.services = {};
      this.config.selectedServices = [];
      this.testResults = {};
      this.uploadedFile = null;
      this.recordings = [];
      this.recordingQueue = [];
      this.activeRecordingId = null;
      this.renderRecordingsQueue();
      
      // Clear localStorage
      try {
        localStorage.removeItem(STORAGE_KEYS.apiKeys);
        localStorage.removeItem(STORAGE_KEYS.enabledServices);
      } catch (error) {
        console.error('Failed to clear localStorage:', error);
      }
      
      location.reload();
    }
  },

  updateServiceStatus(serviceId, icon) {
    const statusIcon = document.getElementById(`status_${serviceId}`);
    if (statusIcon) {
      statusIcon.textContent = icon;
    }
  },

  showView(viewId) {
    console.log('showView called with:', viewId);
    const targetView = document.getElementById(viewId);
    if (!targetView) {
      console.error('View not found:', viewId);
      return;
    }
    
    document.querySelectorAll('.view').forEach(view => {
      view.classList.remove('active');
      view.style.display = 'none';
    });
    
    targetView.classList.add('active');
    targetView.style.display = 'block';
    console.log('View switched to:', viewId);
  },

  setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      
      if (e.dataTransfer.files.length > 0) {
        const files = Array.from(e.dataTransfer.files);
        this.handleIncomingFiles(files);
      }
    });
  },

  handleFileSelect(event) {
    const files = Array.from(event.target.files || []);
    if (files.length > 0) {
      this.handleIncomingFiles(files);
      event.target.value = '';
    }
  },

  handleIncomingFiles(files) {
    files.forEach(file => this.processFile(file));
  },

  processFile(file) {
    // Validate file type
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/x-m4a', 'audio/ogg'];
    const validExtensions = ['.mp3', '.wav', '.m4a', '.ogg'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
      alert('Unsupported file format. Please use MP3, WAV, M4A, or OGG.');
      return;
    }

    // Validate file size (100MB max)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      alert('File is too large. Maximum size: 100 MB.');
      return;
    }

    const recording = this.createRecordingEntry(file);
    this.recordings.push(recording);
    if (!this.activeRecordingId) {
      this.setActiveRecording(recording.id, { skipResultsRender: true });
    }
    this.renderRecordingsQueue();
  },

  async startTesting() {
    this.audioshakeResults = null;
    // Determine which method to use based on active tab
    const method = this.currentDiarizationMethod || 'llm';

    // Route to appropriate method handler
    if (method === 'llm') {
      await this.startLLMDiarization();
    } else if (method === 'audio') {
      await this.startAudioDiarization();
    } else if (method === 'combined') {
      await this.startCombinedDiarization();
    } else if (method === 'overlap') {
      await this.startOverlapDiarization();
    }
  },

  async startLLMDiarization() {
    const selectedModel = document.querySelector('input[name="diarizationModel"]:checked')?.value || 'fast-tier';
    const isMultimodalMode = selectedModel === 'multimodal';
    
    // Check if transcript is provided
    const transcriptText = document.getElementById('transcriptText')?.value?.trim();
    if (!transcriptText) {
      alert('Please provide a transcript text for LLM diarization.');
      return;
    }
    
    // For multimodal mode, check if audio file is provided
    if (isMultimodalMode) {
      if (!this.llmMultimodalAudioFile && this.recordings.length === 0) {
        alert('Multimodal mode requires both a transcript and an audio file. Please upload an audio file.');
        return;
      }
      
      // Use combined diarization with multimodal mode
      if (this.recordings.length === 0) {
        alert('Please add an audio file to the recordings queue for multimodal processing.');
        return;
      }
      
      const recording = this.recordings[0];
      if (!recording.file && !recording.url) {
        alert('No audio file available for multimodal processing.');
        return;
      }
      
      try {
        this.testingInProgress = true;
        this.showView('testingView');
        
        // Try to get language from specific tab fields, fallback to default
        const language = document.getElementById('audioLanguageSelect')?.value 
          || document.getElementById('combinedLanguageSelect')?.value 
          || document.getElementById('overlapLanguageSelect')?.value 
          || 'ar';
        const speakerCount = document.getElementById('audioSpeakerCount')?.value 
          || document.getElementById('combinedSpeakerCount')?.value 
          || document.getElementById('overlapSpeakerCount')?.value 
          || null;
        
        const formData = new FormData();
        if (recording.file) {
          formData.append('audio', recording.file);
        } else if (recording.url) {
          formData.append('url', recording.url);
        }
        formData.append('language', language);
        formData.append('mode', 'gemini-2.5-pro');
        formData.append('pipelineMode', 'multimodal');
        if (speakerCount) {
          formData.append('speakerCount', speakerCount);
        }
        formData.append('plainTranscript', transcriptText);
        
        const response = await fetch('/api/combined-diarization', {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || 'Multimodal diarization failed');
        }
        
        const result = await response.json();
        this.testResults = { combined: result };
        this.testingInProgress = false;
        this.showResults();
        
      } catch (error) {
        console.error('Multimodal diarization error:', error);
        alert(`Multimodal diarization failed: ${error.message}`);
        this.testingInProgress = false;
        this.showView('uploadView');
      }
      return;
    }

    // Use existing analyzeTranscript functionality for text-only modes
    await this.analyzeTranscript();
  },

  async startAudioDiarization() {
    if (this.recordings.length === 0) {
      alert('Please add at least one audio file.');
      return;
    }

    // Get settings from Audio tab
    const language = document.getElementById('audioLanguageSelect')?.value || 'ar';
    const speakerCount = document.getElementById('audioSpeakerCount')?.value || null;
    const audioPipelineMode = document.querySelector('input[name="audioPipelineMode"]:checked')?.value || 'speechmatics';

    // Use the first recording for audio diarization
    const recording = this.recordings[0];
    if (!recording.file && !recording.url) {
      alert('No audio file available for processing.');
      return;
    }

    // Якщо вибрано AudioShake, викликаємо runAudioshakePipeline
    if (audioPipelineMode === 'audioshake') {
      await this.runAudioshakePipeline(recording);
      return;
    }

    try {
      this.testingInProgress = true;
      
      // Calculate estimated time
      const fileDuration = recording.duration || this.estimateDurationFromFile(recording.file || { size: recording.size || 0 });
      const avgProcessingTime = fileDuration * 0.3;
      const estimatedMinutes = Math.ceil(avgProcessingTime / 60);
      
      this.showView('testingView');
      
      // Update estimated time
      setTimeout(() => {
        const estimatedEl = document.getElementById('estimatedTime');
        if (estimatedEl) {
          estimatedEl.textContent = `${estimatedMinutes}-${estimatedMinutes + 2} minutes`;
          console.log(`✅ Estimated time set: ${estimatedMinutes}-${estimatedMinutes + 2} minutes`);
        }
      }, 100);

      const formData = new FormData();
      if (recording.file) {
        formData.append('audio', recording.file);
      } else if (recording.url) {
        formData.append('url', recording.url);
      }
      formData.append('language', language);
      if (speakerCount) {
        formData.append('speakerCount', speakerCount);
      }
      const response = await fetch('/api/diarize-audio', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Audio diarization failed');
      }

      const result = await response.json();
      
      // Convert result to expected format for showResults()
      const recordingData = result?.recordings?.[0];
      const speechmaticsResult = recordingData?.results?.speechmatics;
      const segments = speechmaticsResult?.segments || [];
      
      const formattedResult = {
        success: !!speechmaticsResult?.success,
        serviceName: speechmaticsResult?.serviceName || 'Model3',
        processingTime: speechmaticsResult?.processingTime || 0,
        speakerCount: recordingData?.speakerCount || new Set(segments.map(s => s.speaker)).size,
        segments: segments,
        rawData: result
      };
      
      // Store result and show it
      this.testResults = { audio: formattedResult };
      this.testingInProgress = false;
      this.showResults();
      
    } catch (error) {
      console.error('Audio diarization error:', error);
      alert(`Audio diarization failed: ${error.message}`);
      this.testingInProgress = false;
      this.showView('uploadView');
    }
  },

  async startCombinedDiarization() {
    if (this.recordings.length === 0) {
      alert('Please add at least one audio file for combined diarization.');
      return;
    }

    try {
      this.testingInProgress = true;
      this.showView('testingView');
      
      const recording = this.recordings[0];
      if (!recording.file && !recording.url) {
        alert('No audio file available for processing.');
        return;
      }

      // Get settings from Combined tab
      const language = document.getElementById('combinedLanguageSelect')?.value || 'ar';
      const speakerCount = document.getElementById('combinedSpeakerCount')?.value || null;
      const pipelineMode = document.querySelector('input[name="combinedPipelineMode"]:checked')?.value || 'hybrid';
      // Multimodal та AudioShake більше не на цій вкладці
      const mode = document.querySelector('input[name="combinedDiarizationModel"]:checked')?.value || 'smart-tier';
      const modeMap = {
        'fast-tier': 'fast',
        'smart-tier': 'smart',
        'smart-2-tier': 'smart-2',
        'local-tier': 'local'
      };
      const apiMode = modeMap[mode] || 'smart';

      const formData = new FormData();
      if (recording.file) {
        formData.append('audio', recording.file);
      } else if (recording.url) {
        formData.append('url', recording.url);
      }
      formData.append('language', language);
      formData.append('mode', apiMode);
      formData.append('pipelineMode', pipelineMode);
      if (speakerCount) {
        formData.append('speakerCount', speakerCount);
      }
      const manualTranscript = document.getElementById('transcriptText')?.value?.trim();
      if (manualTranscript) {
        formData.append('plainTranscript', manualTranscript);
      }

      // Show initial progress with all steps
      const progressContainer = document.getElementById('progressContainer');
      if (progressContainer) {
        progressContainer.innerHTML = `
          <div style="margin-bottom: var(--space-24);">
            <h3 style="margin-bottom: var(--space-16);">Combined Diarization Progress</h3>
            <div id="combinedStepsProgress" style="display: flex; flex-direction: column; gap: var(--space-12);">
              <div class="step-progress" data-step="1" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-primary);">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-primary); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">1</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 1: Audio Transcription</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Extracting transcript from audio...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏳ Processing...</span>
                </div>
              </div>
              <div class="step-progress" data-step="2" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">2</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 2: LLM Diarization</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Waiting...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
                </div>
              </div>
              <div class="step-progress" data-step="3" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">3</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 3: Comparison Analysis</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Waiting...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
                </div>
              </div>
              <div class="step-progress" data-step="4" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">4</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 4: Correction & Merging</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Waiting...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
                </div>
              </div>
            </div>
          </div>
        `;
      }

      // Update step statuses
      // Multimodal режим тепер обробляється на вкладці LLM
      this.updateCombinedStepProgress(1, 'processing', 'Extracting transcript from audio...');

      const response = await fetch('/api/diarize-combined', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const error = await response.json();
        this.updateCombinedStepProgress(1, 'error', 'Failed: ' + (error.error || 'Unknown error'));
        throw new Error(error.error || 'Combined diarization failed');
      }

      const result = await response.json();
      const llmSource = result?.llm?.source || 'unknown';
      const llmProvider = result?.llm?.provider || (llmSource.includes('openrouter') ? 'openrouter' : 'google');
      const llmModeLabel = result?.llm?.multimodal ? 'multimodal (audio+text)' : 'text-only fallback';
      console.info(`🎧 Combined LLM provider: ${llmProvider} (${llmModeLabel})`, { source: llmSource, step2: result?.steps?.step2?.status });
      
      // Update all steps with results
      const applyStepInfo = (stepNumber, stepInfo) => {
        if (!stepInfo) return;
        const status = stepInfo.status || 'completed';
        if (status === 'skipped') {
          this.updateCombinedStepProgress(stepNumber, 'skipped', stepInfo.reason || 'Skipped');
        } else if (status === 'completed_with_fallback') {
          const message = stepInfo.reason || `Completed in ${stepInfo.duration || 'N/A'}`;
          this.updateCombinedStepProgress(stepNumber, 'completed-warning', message);
        } else if (status === 'error') {
          this.updateCombinedStepProgress(stepNumber, 'error', stepInfo.reason || 'Failed');
        } else {
          const durationMessage = stepInfo.duration ? `Completed in ${stepInfo.duration}` : (stepInfo.reason || 'Completed');
          this.updateCombinedStepProgress(stepNumber, 'completed', durationMessage);
        }
      };

      if (result.steps) {
        applyStepInfo(1, result.steps.step1);
        applyStepInfo(2, result.steps.step2);
        applyStepInfo(3, result.steps.step3);
        applyStepInfo(4, result.steps.step4);
      }
      
      // Persist best-known transcript result on the recording so replicas view can use it
      this.attachCombinedResultToRecording(recording, result);
      this.renderRecordingsQueue();
      this.renderReplicaRecordingSelect();

      // Store result and show it
      this.testResults = { combined: result };
      this.testingInProgress = false;
      this.showResults();
      
    } catch (error) {
      console.error('Combined diarization error:', error);
      alert(`Combined diarization failed: ${error.message}`);
      this.testingInProgress = false;
      this.showView('uploadView');
    }
  },

  updateCombinedStepProgress(stepNumber, status, message) {
    const stepElement = document.querySelector(`.step-progress[data-step="${stepNumber}"]`);
    if (!stepElement) return;

    const stepIndicator = stepElement.querySelector('.step-indicator');
    const stepStatus = stepElement.querySelector('.step-status');
    const stepMessage = stepElement.querySelector('.step-description');

    // Update status based on step state
    if (status === 'processing') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeftColor = 'var(--color-primary)';
      stepIndicator.style.background = 'var(--color-primary)';
      stepIndicator.textContent = '⏳';
      stepStatus.textContent = '⏳ Processing...';
      stepStatus.style.color = 'var(--color-primary)';
    } else if (status === 'completed') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeftColor = 'var(--color-success)';
      stepIndicator.style.background = 'var(--color-success)';
      stepIndicator.textContent = '✓';
      stepStatus.textContent = '✓ Completed';
      stepStatus.style.color = 'var(--color-success)';
    } else if (status === 'completed-warning') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeftColor = 'var(--color-warning)';
      stepIndicator.style.background = 'var(--color-warning)';
      stepIndicator.textContent = '⚠️';
      stepStatus.textContent = '⚠️ Completed with fallback';
      stepStatus.style.color = 'var(--color-warning)';
    } else if (status === 'skipped') {
      stepElement.style.opacity = '0.7';
      stepElement.style.borderLeftColor = 'var(--color-border)';
      stepIndicator.style.background = 'var(--color-border)';
      stepIndicator.textContent = '⏭️';
      stepStatus.textContent = '⏭️ Skipped';
      stepStatus.style.color = 'var(--color-text-secondary)';
    } else if (status === 'error') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeftColor = 'var(--color-error)';
      stepIndicator.style.background = 'var(--color-error)';
      stepIndicator.textContent = '✗';
      stepStatus.textContent = '✗ Failed';
      stepStatus.style.color = 'var(--color-error)';
    } else {
      stepElement.style.opacity = '0.6';
      stepElement.style.borderLeftColor = 'var(--color-border)';
      stepIndicator.style.background = 'var(--color-border)';
      stepIndicator.textContent = stepNumber;
      stepStatus.textContent = '⏸️ Pending';
      stepStatus.style.color = 'var(--color-text-secondary)';
    }

    // Update message
    if (stepMessage && message) {
      stepMessage.textContent = message;
    }

    // Add animation
    stepElement.style.transition = 'all 0.3s ease';
  },

  async startOverlapDiarization() {
    if (this.recordings.length === 0) {
      alert('Please add at least one audio file for overlap diarization.');
      return;
    }

    try {
      this.testingInProgress = true;
      
      const recording = this.recordings[0];
      if (!recording.file && !recording.url) {
        alert('No audio file available for processing.');
        return;
      }

      // Get settings from Overlap tab
      const language = document.getElementById('overlapLanguageSelect')?.value || 'ar';
      const speakerCount = document.getElementById('overlapSpeakerCount')?.value || null;
      const mode = document.querySelector('input[name="overlapDiarizationModel"]:checked')?.value || 'fast';
      const pipelineMode = document.getElementById('overlapPipelineMode')?.value || 'mode1';
      const transcriptionEngine = document.querySelector('input[name="overlapAudioModel"]:checked')?.value || 'speechmatics';
      
      // Calculate estimated time BEFORE showing view
      const fileDuration = recording.duration || this.estimateDurationFromFile(recording.file || { size: recording.size || 0 });
      // Overlap diarization takes longer: Step 1 (1x) + Step 2 (2x) + Step 3-4 (2x per speaker) + Step 5 (0.5x)
      // Rough estimate: base time * (1 + 2 + 2*2 + 0.5) = base * 7.5 for 2 speakers
      const estimatedMultiplier = 5 + (speakerCount ? parseInt(speakerCount) * 1.5 : 3);
      const avgProcessingTime = fileDuration * 0.3 * estimatedMultiplier;
      const estimatedMinutes = Math.ceil(avgProcessingTime / 60);
      
      // Show view first
      this.showView('testingView');
      
      // Show progress UI first
      const progressContainer = document.getElementById('progressContainer');
      if (progressContainer) {
        progressContainer.innerHTML = `
          <div style="margin-bottom: var(--space-24);">
            <h3 style="margin-bottom: var(--space-16);">Overlap Diarization Progress</h3>
            <div id="overlapStepsProgress" style="display: flex; flex-direction: column; gap: var(--space-12);">
              <div class="step-progress" data-step="1" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-primary); transition: all 0.3s ease;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-primary); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: all 0.3s ease;">1</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 1: Initial Audio Analysis</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary); min-height: 20px;">Analyzing audio...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm); transition: all 0.3s ease;">⏳ Processing...</span>
                </div>
              </div>
              <div class="step-progress" data-step="2" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">2</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 2: Speaker Separation</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Waiting...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
                </div>
              </div>
              <div class="step-progress" data-step="3" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">3</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 3-4: Voice Track Transcription & Role Analysis</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Waiting...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
                </div>
              </div>
              <div class="step-progress" data-step="5" style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); border-left: 4px solid var(--color-border); opacity: 0.6;">
                <div style="display: flex; align-items: center; gap: var(--space-8);">
                  <span class="step-indicator" style="width: 24px; height: 24px; border-radius: 50%; background: var(--color-border); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">5</span>
                  <div style="flex: 1;">
                    <div style="font-weight: var(--font-weight-medium);">Step 5: Overlap Correction</div>
                    <div class="step-description" style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Waiting...</div>
                  </div>
                  <span class="step-status" style="font-size: var(--font-size-sm);">⏸️ Pending</span>
                </div>
              </div>
            </div>
          </div>
        `;
      }
      
      // Update estimated time AFTER progress UI is set up (use setTimeout to ensure DOM is ready)
      setTimeout(() => {
        const estimatedEl = document.getElementById('estimatedTime');
        if (estimatedEl) {
          estimatedEl.textContent = `${estimatedMinutes}-${estimatedMinutes + 3} minutes`;
          console.log(`✅ Estimated time set: ${estimatedMinutes}-${estimatedMinutes + 3} minutes (fileDuration: ${fileDuration}s, multiplier: ${estimatedMultiplier})`);
        } else {
          console.warn('⚠️ estimatedTime element not found after progressContainer setup');
        }
      }, 100);

      const formData = new FormData();
      if (recording.file) {
        formData.append('audio', recording.file);
      } else if (recording.url) {
        formData.append('url', recording.url);
      }
      formData.append('language', language);
      formData.append('mode', mode);
      formData.append('pipelineMode', pipelineMode);
      formData.append('engine', transcriptionEngine);
      if (speakerCount) {
        formData.append('speakerCount', speakerCount);
      }

      // Update step 1 status immediately
      this.updateOverlapStepProgress(1, 'processing', 'Analyzing audio...');
      
      // Show that we're starting
      console.log('🔵 Starting overlap diarization, pipeline mode:', pipelineMode, 'engine:', transcriptionEngine);
      
      // Start SSE connection for real-time progress updates (for mode 3 and mode 1)
      // Mode 1 also needs SSE to prevent 504 Gateway Timeout
      // For mode 3 and mode 1, we use SSE for detailed real-time updates
      if (pipelineMode === 'mode3' || pipelineMode === 'mode1') {
        console.log(`🔵 Setting up SSE connection for ${pipelineMode.toUpperCase()} real-time updates`);
        
        // Create detailed info panel immediately
        this.createDetailedInfoPanel();
        
        this.setupOverlapSSE(formData, pipelineMode);
        return; // SSE handler will manage the response
      }
      
      // For other modes, use polling (but still show updates)
      console.log('🔵 Using polling for progress updates');
      this.overlapProgressInterval = this.startOverlapProgressPolling();

      const response = await fetch('/api/diarize-overlap', {
        method: 'POST',
        body: formData
      });

      // Clear polling interval when request completes
      if (this.overlapProgressInterval) {
        clearInterval(this.overlapProgressInterval);
        this.overlapProgressInterval = null;
      }

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Overlap diarization failed');
      }

      const result = await response.json();
      
      // Update all step statuses
      if (result.steps) {
        if (result.steps.step1) {
          this.updateOverlapStepProgress(1, result.steps.step1.status, `Completed in ${result.steps.step1.duration}`, result.steps.step1);
        }
        if (result.steps.step2) {
          this.updateOverlapStepProgress(2, result.steps.step2.status, `Completed in ${result.steps.step2.duration}`, result.steps.step2);
        }
        if (result.steps.step3) {
          this.updateOverlapStepProgress(3, result.steps.step3.status, `Completed in ${result.steps.step3.duration}`, result.steps.step3);
        }
        if (result.steps.step5) {
          this.updateOverlapStepProgress(5, result.steps.step5.status, `Completed in ${result.steps.step5.duration}`, result.steps.step5);
        }
      }
      
      // Store result
      this.testResults = { overlap: result };
      
      // Attach result to recording
      this.attachOverlapResultToRecording(recording, result);
      this.renderRecordingsQueue();
      
      // Update separated speakers players if we're in replica view
      if (this.currentView === 'replicaView') {
        this.updateSeparatedSpeakersPlayers(recording);
      }
      
      this.testingInProgress = false;
      
      // Show results
      this.showResults();
      
    } catch (error) {
      // Clear polling interval if it exists
      if (this.overlapProgressInterval) {
        clearInterval(this.overlapProgressInterval);
        this.overlapProgressInterval = null;
      }
      
      console.error('Overlap diarization error:', error);
      alert(`Overlap diarization failed: ${error.message}`);
      this.testingInProgress = false;
      this.showView('uploadView');
    }
  },

  setupOverlapSSE(formData, pipelineMode) {
    console.log('🔵 Setting up SSE connection for real-time updates');
    
    // Create a new request with streaming support
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/diarize-overlap');
    
    let lastProcessedLength = 0;
    let buffer = '';
    
    xhr.onprogress = () => {
      // Process SSE data chunks as they arrive in real-time
      const currentLength = xhr.responseText.length;
      if (currentLength > lastProcessedLength) {
        const newData = xhr.responseText.substring(lastProcessedLength);
        lastProcessedLength = currentLength;
        buffer += newData;
        
        // Process all complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.substring(6).trim();
            if (jsonStr) {
              try {
                const data = JSON.parse(jsonStr);
                console.log('[SSE] ✅ Received update:', {
                  type: data.type,
                  step: data.step,
                  status: data.status,
                  description: data.description?.substring(0, 50)
                });
                
                if (data.type === 'step-progress') {
                  // Update immediately for real-time display
                  this.handleSSEProgressUpdate(data);
                } else if (data.type === 'final-result') {
                  // Handle final result immediately
                  console.log('[SSE] ✅ Final result received');
                  delete data.type;
                  this.handleOverlapDiarizationResult(data);
                  return;
                } else if (data.type === 'pipeline-error') {
                  // Handle pipeline error immediately
                  console.error('[SSE] ❌ Pipeline error received:', data);
                  const errorMsg = data.error || data.details || 'Overlap diarization failed';
                  alert(`Overlap diarization failed: ${errorMsg}`);
                  this.testingInProgress = false;
                  this.showView('uploadView');
                  return;
                }
              } catch (e) {
                console.warn('[SSE] ❌ Failed to parse:', e, jsonStr.substring(0, 100));
              }
            }
          }
        }
      }
    };
    
    xhr.onload = () => {
      if (xhr.status === 200) {
        try {
          // Process all SSE events to find final result
          const lines = xhr.responseText.split('\n');
          let finalJson = null;
          let pipelineError = null;
          
          // First pass: look for final-result or pipeline-error
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                if (data.type === 'final-result') {
                  finalJson = data;
                  break;
                } else if (data.type === 'pipeline-error') {
                  pipelineError = data;
                  break;
                } else if (data.type === 'step-progress') {
                  // Already handled in onprogress
                  continue;
                }
              } catch (e) {
                // Skip invalid JSON lines
                console.warn('Failed to parse SSE line:', line.substring(0, 100));
              }
            }
          }
          
          // If pipeline error found, show it
          if (pipelineError) {
            const errorMsg = pipelineError.error || pipelineError.details || 'Overlap diarization failed';
            alert(`Overlap diarization failed: ${errorMsg}`);
            this.testingInProgress = false;
            this.showView('uploadView');
            return;
          }
          
          // Fallback: try to find any JSON object in response (including partial)
          if (!finalJson) {
            // Try to find last complete JSON object
            let jsonBuffer = '';
            for (let i = lines.length - 1; i >= 0; i--) {
              const trimmed = lines[i].trim();
              if (trimmed.startsWith('data: ')) {
                try {
                  const data = JSON.parse(trimmed.substring(6));
                  if (data.success !== undefined || data.type === 'final-result') {
                    finalJson = data;
                    break;
                  }
                } catch (e) {
                  // Continue
                }
              } else if (trimmed.startsWith('{')) {
                jsonBuffer = trimmed + jsonBuffer;
                try {
                  finalJson = JSON.parse(jsonBuffer);
                  if (finalJson.success !== undefined || finalJson.type === 'final-result') {
                    break;
                  }
                } catch (e) {
                  // Try to find complete JSON by searching backwards
                  if (i > 0) {
                    jsonBuffer = lines[i - 1].trim() + jsonBuffer;
                    try {
                      finalJson = JSON.parse(jsonBuffer);
                      if (finalJson.success !== undefined || finalJson.type === 'final-result') {
                        break;
                      }
                    } catch (e2) {
                      // Continue
                    }
                  }
                }
              }
            }
          }
          
          if (finalJson && (finalJson.success || finalJson.type === 'final-result')) {
            // Remove type field if present
            if (finalJson.type === 'final-result') {
              delete finalJson.type;
            }
            this.handleOverlapDiarizationResult(finalJson);
          } else {
            // Last resort: show response for debugging
            console.error('No final result found in response. Full response:', xhr.responseText);
            throw new Error('Invalid response format - no final result found. Check console for details.');
          }
        } catch (error) {
          console.error('Failed to parse final response:', error);
          console.error('Response text:', xhr.responseText.substring(0, 2000));
          alert(`Overlap diarization failed: ${error.message}`);
          this.testingInProgress = false;
          this.showView('uploadView');
        }
      } else {
        try {
          const error = JSON.parse(xhr.responseText);
          throw new Error(error.error || 'Overlap diarization failed');
        } catch (parseError) {
          throw new Error(`Server error: ${xhr.status} ${xhr.statusText}`);
        }
      }
    };
    
    xhr.onerror = () => {
      console.error('❌ SSE request failed');
      alert('Network error during overlap diarization');
      this.testingInProgress = false;
      this.showView('uploadView');
    };
    
    xhr.onloadstart = () => {
      console.log('🔵 SSE connection started');
      // Remove initial waiting message
      const contentDiv = document.getElementById('overlapDetailedContent');
      if (contentDiv) {
        const initMsg = contentDiv.querySelector('div[style*="background: var(--color-primary)"]');
        if (initMsg) {
          initMsg.remove();
        }
      }
    };
    
    xhr.send(formData);
    this.overlapXHR = xhr;
    
    console.log('✅ SSE request sent, waiting for updates...');
  },
  
  handleSSEProgressUpdate(data) {
    const { step, status, description, details } = data;
    
    console.log(`[SSE Progress] ✅ Step ${step}: ${description}`, details);
    
    // Force immediate UI update
    requestAnimationFrame(() => {
      // Update step progress immediately for real-time display
      this.updateOverlapStepProgress(step, status, description, details);
      
      // Always add detailed info for mode 3 (even without stage)
      // This ensures all updates are visible in real-time
      if (details) {
        this.updateDetailedProgressInfo(step, details, description);
      } else if (description) {
        // Even if no details, show the description update
        this.updateDetailedProgressInfo(step, { stage: status }, description);
      }
    });
  },
  
  createDetailedInfoPanel() {
    // Find or create detailed info container
    let infoContainer = document.getElementById('overlapDetailedInfo');
    if (!infoContainer) {
      const progressContainer = document.getElementById('progressContainer');
      if (progressContainer) {
        infoContainer = document.createElement('div');
        infoContainer.id = 'overlapDetailedInfo';
        infoContainer.style.cssText = 'margin-top: var(--space-16); padding: var(--space-16); background: var(--color-bg-1); border-radius: var(--radius-base); border: 1px solid var(--color-border); max-height: 500px; overflow-y: auto;';
        infoContainer.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-12);">
            <h4 style="margin: 0;">📊 Детальна інформація про обробку (Mode 3) - Режим реального часу</h4>
            <button id="clearDetailedInfo" style="padding: var(--space-4) var(--space-8); font-size: var(--font-size-sm); background: var(--color-bg-2); border: 1px solid var(--color-border); border-radius: var(--radius-sm); cursor: pointer;">Очистити</button>
          </div>
          <div id="overlapDetailedContent" style="display: flex; flex-direction: column; gap: var(--space-8);"></div>
        `;
        progressContainer.appendChild(infoContainer);
        
        // Add clear button handler
        const clearBtn = document.getElementById('clearDetailedInfo');
        if (clearBtn) {
          clearBtn.addEventListener('click', () => {
            const contentDiv = document.getElementById('overlapDetailedContent');
            if (contentDiv) {
              contentDiv.innerHTML = '';
            }
          });
        }
        
        // Add initial message
        const contentDiv = document.getElementById('overlapDetailedContent');
        if (contentDiv) {
          const initMsg = document.createElement('div');
          initMsg.style.cssText = 'padding: var(--space-12); background: var(--color-primary); color: white; border-radius: var(--radius-sm); text-align: center; font-weight: var(--font-weight-medium);';
          initMsg.textContent = '⏳ Очікування оновлень від сервера...';
          contentDiv.appendChild(initMsg);
        }
      }
    }
    return infoContainer;
  },
  
  updateDetailedProgressInfo(step, details, description = '') {
    // Ensure panel exists
    let infoContainer = this.createDetailedInfoPanel();
    
    if (infoContainer) {
      const contentDiv = document.getElementById('overlapDetailedContent');
      if (contentDiv) {
        // Remove initial waiting message if present
        const initMsg = contentDiv.querySelector('div[style*="background: var(--color-primary)"]');
        if (initMsg) {
          initMsg.remove();
        }
        
        const timestamp = new Date().toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const infoEntry = document.createElement('div');
        infoEntry.style.cssText = 'padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-sm); font-size: var(--font-size-sm); border-left: 3px solid var(--color-primary); animation: slideIn 0.3s ease-out;';
        infoEntry.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: var(--space-8);">
            <div style="flex: 1;">
              <strong style="color: var(--color-primary);">Крок ${step}</strong>
              ${description ? `<div style="margin-top: var(--space-4); color: var(--color-text); font-weight: var(--font-weight-medium);">${description}</div>` : ''}
            </div>
            <span style="color: var(--color-text-secondary); font-size: var(--font-size-xs); white-space: nowrap; margin-left: var(--space-8);">${timestamp}</span>
          </div>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-8); margin-top: var(--space-8);">
            ${details.stage ? `<div><strong>Етап:</strong> <span style="color: var(--color-text-secondary);">${details.stage}</span></div>` : ''}
            ${details.segmentsCount !== undefined ? `<div><strong>Сегментів:</strong> <span style="color: var(--color-text-secondary);">${details.segmentsCount}</span></div>` : ''}
            ${details.speakersCount !== undefined ? `<div><strong>Спікерів:</strong> <span style="color: var(--color-text-secondary);">${details.speakersCount}</span></div>` : ''}
            ${details.transcriptLength !== undefined ? `<div><strong>Транскрипт:</strong> <span style="color: var(--color-text-secondary);">${details.transcriptLength} символів</span></div>` : ''}
            ${details.trackNumber !== undefined ? `<div><strong>Трек:</strong> <span style="color: var(--color-text-secondary);">${details.trackNumber}/${details.totalTracks || '?'}</span></div>` : ''}
            ${details.role ? `<div><strong>Роль:</strong> <span style="color: var(--color-text-secondary);">${details.role} (${(details.confidence * 100).toFixed(1)}%)</span></div>` : ''}
            ${details.model ? `<div><strong>Модель LLM:</strong> <span style="color: var(--color-text-secondary);">${details.model}</span></div>` : ''}
            ${details.duration ? `<div><strong>Тривалість:</strong> <span style="color: var(--color-text-secondary);">${details.duration}</span></div>` : ''}
            ${details.promptLength !== undefined ? `<div><strong>Промпт:</strong> <span style="color: var(--color-text-secondary);">${details.promptLength} символів</span></div>` : ''}
            ${details.outputLength !== undefined ? `<div><strong>Відповідь LLM:</strong> <span style="color: var(--color-text-secondary);">${details.outputLength} символів</span></div>` : ''}
            ${details.speakers ? `<div><strong>Спікери:</strong> <span style="color: var(--color-text-secondary);">${Array.isArray(details.speakers) ? details.speakers.join(', ') : details.speakers}</span></div>` : ''}
            ${details.fileSize ? `<div><strong>Розмір файлу:</strong> <span style="color: var(--color-text-secondary);">${details.fileSize}</span></div>` : ''}
            ${details.azureStatus ? `<div><strong>Azure статус:</strong> <span style="color: var(--color-text-secondary);">${details.azureStatus}</span></div>` : ''}
            ${details.azureAttempt !== undefined ? `<div><strong>Azure прогрес:</strong> <span style="color: var(--color-text-secondary);">${details.azureAttempt}/${details.azureTotalAttempts || '?'}</span></div>` : ''}
            ${details.azureJobId ? `<div><strong>Azure job:</strong> <span style="color: var(--color-text-secondary); font-size: var(--font-size-xs); word-break: break-all;">${details.azureJobId}</span></div>` : ''}
            ${details.azureFilesCount !== undefined ? `<div><strong>Azure файлів:</strong> <span style="color: var(--color-text-secondary);">${details.azureFilesCount}</span></div>` : ''}
            ${details.azurePhrasesCount !== undefined ? `<div><strong>Фраз Azure:</strong> <span style="color: var(--color-text-secondary);">${details.azurePhrasesCount}</span></div>` : ''}
            ${details.azureSegmentsCount !== undefined ? `<div><strong>Сегментів Azure:</strong> <span style="color: var(--color-text-secondary);">${details.azureSegmentsCount}</span></div>` : ''}
            ${details.audioPath ? `<div style="grid-column: 1 / -1;"><strong>Шлях:</strong> <span style="color: var(--color-text-secondary); font-size: var(--font-size-xs); word-break: break-all;">${details.audioPath}</span></div>` : ''}
          </div>
          ${details.input ? `
            <div style="margin-top: var(--space-12); padding: var(--space-12); background: var(--color-bg-3); border-radius: var(--radius-sm); border-left: 3px solid var(--color-teal-500);">
              <strong style="color: var(--color-teal-500); display: block; margin-bottom: var(--space-8);">📥 Вхідні дані:</strong>
              <pre style="margin: 0; font-size: var(--font-size-xs); overflow-x: auto; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto;">${JSON.stringify(details.input, null, 2)}</pre>
            </div>
          ` : ''}
          ${details.output ? `
            <div style="margin-top: var(--space-12); padding: var(--space-12); background: var(--color-bg-3); border-radius: var(--radius-sm); border-left: 3px solid var(--color-success);">
              <strong style="color: var(--color-success); display: block; margin-bottom: var(--space-8);">📤 Вихідні дані:</strong>
              <pre style="margin: 0; font-size: var(--font-size-xs); overflow-x: auto; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto;">${JSON.stringify(details.output, null, 2)}</pre>
            </div>
          ` : ''}
          ${details.error ? `
            <div style="margin-top: var(--space-12); padding: var(--space-12); background: rgba(var(--color-error-rgb), 0.1); border-radius: var(--radius-sm); border-left: 3px solid var(--color-error);">
              <strong style="color: var(--color-error); display: block; margin-bottom: var(--space-8);">❌ Помилка:</strong>
              <pre style="margin: 0; font-size: var(--font-size-xs); overflow-x: auto; white-space: pre-wrap; word-break: break-word;">${JSON.stringify(details.error, null, 2)}</pre>
            </div>
          ` : ''}
        `;
        contentDiv.appendChild(infoEntry);
        
        // Auto-scroll to bottom with smooth animation
        requestAnimationFrame(() => {
          infoContainer.scrollTo({
            top: infoContainer.scrollHeight,
            behavior: 'smooth'
          });
        });
      }
    }
  },
  
  handleOverlapDiarizationResult(result) {
    // Update all step statuses
    if (result.steps) {
      if (result.steps.step1) {
        this.updateOverlapStepProgress(1, result.steps.step1.status, `Completed in ${result.steps.step1.duration}`, result.steps.step1);
      }
      if (result.steps.step2) {
        this.updateOverlapStepProgress(2, result.steps.step2.status, `Completed in ${result.steps.step2.duration}`, result.steps.step2);
      }
      if (result.steps.step3) {
        this.updateOverlapStepProgress(3, result.steps.step3.status, `Completed in ${result.steps.step3.duration}`, result.steps.step3);
      }
      if (result.steps.step5) {
        this.updateOverlapStepProgress(5, result.steps.step5.status, `Completed in ${result.steps.step5.duration}`, result.steps.step5);
      }
    }
    
    // Store result
    this.testResults = { overlap: result };
    
    // Attach result to recording
    const recording = this.recordings[0];
    this.attachOverlapResultToRecording(recording, result);
    this.renderRecordingsQueue();
    
    // Update separated speakers players if we're in replica view
    if (this.currentView === 'replicaView') {
      this.updateSeparatedSpeakersPlayers(recording);
    }
    
    this.testingInProgress = false;
    
    // Show results
    this.showResults();
  },

  updateOverlapStepProgress(stepNumber, status, description, stepData = null) {
    const stepElement = document.querySelector(`#overlapStepsProgress .step-progress[data-step="${stepNumber}"]`);
    if (!stepElement) return;

    const stepIndicator = stepElement.querySelector('.step-indicator');
    const stepDescription = stepElement.querySelector('.step-description');
    const stepStatus = stepElement.querySelector('.step-status');
    const borderLeft = stepElement.style.borderLeft;

    if (status === 'completed') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeft = '4px solid var(--color-success)';
      if (stepIndicator) {
        stepIndicator.style.background = 'var(--color-success)';
        stepIndicator.textContent = '✓';
      }
      if (stepStatus) {
        stepStatus.textContent = '✅ Completed';
        stepStatus.style.color = 'var(--color-success)';
      }
    } else if (status === 'processing') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeft = '4px solid var(--color-primary)';
      if (stepIndicator) {
        stepIndicator.style.background = 'var(--color-primary)';
        stepIndicator.textContent = stepNumber;
        // Add pulsing animation
        stepIndicator.style.animation = 'pulse 2s infinite';
      }
      if (stepStatus) {
        stepStatus.textContent = '⏳ Processing...';
        stepStatus.style.color = 'var(--color-primary)';
      }
    } else if (status === 'failed') {
      stepElement.style.opacity = '1';
      stepElement.style.borderLeft = '4px solid var(--color-error)';
      if (stepIndicator) {
        stepIndicator.style.background = 'var(--color-error)';
        stepIndicator.textContent = '✗';
        stepIndicator.style.animation = 'none';
      }
      if (stepStatus) {
        stepStatus.textContent = '❌ Failed';
        stepStatus.style.color = 'var(--color-error)';
      }
    } else if (status === 'pending' || status === 'waiting') {
      stepElement.style.opacity = '0.6';
      stepElement.style.borderLeft = '4px solid var(--color-border)';
      if (stepIndicator) {
        stepIndicator.style.background = 'var(--color-border)';
        stepIndicator.textContent = stepNumber;
        stepIndicator.style.animation = 'none';
      }
      if (stepStatus) {
        stepStatus.textContent = '⏸️ Pending';
        stepStatus.style.color = 'var(--color-text-secondary)';
      }
    }

    if (stepDescription) {
      stepDescription.textContent = description;
    }

    // Add additional info if available
    if (stepData && stepData.segmentsCount !== undefined) {
      if (stepDescription) {
        stepDescription.textContent += ` (${stepData.segmentsCount} segments)`;
      }
    }
    if (stepData && stepData.speakersCount !== undefined) {
      if (stepDescription) {
        stepDescription.textContent += ` (${stepData.speakersCount} speakers)`;
      }
    }
    if (stepData && stepData.processedTracks !== undefined) {
      if (stepDescription) {
        stepDescription.textContent += ` (${stepData.processedTracks}/${stepData.totalTracks} tracks)`;
      }
    }
  },

  startOverlapProgressPolling() {
    // Estimate step durations based on typical processing times
    const stepDurations = {
      1: 15, // Step 1: Initial analysis (15 seconds typical)
      2: 300, // Step 2: Speaker separation (5 minutes for PyAnnote, less for AudioShake)
      3: 120, // Step 3-4: Transcription (2 minutes per speaker)
      5: 30  // Step 5: Overlap correction (30 seconds)
    };
    
    const startTime = Date.now();
    let currentStep = 1;
    
    const updateProgress = () => {
      const elapsed = (Date.now() - startTime) / 1000; // seconds
      
      // Update step 1
      if (elapsed < stepDurations[1]) {
        if (currentStep !== 1) {
          currentStep = 1;
          this.updateOverlapStepProgress(1, 'processing', `Analyzing audio... (${Math.floor(elapsed)}s)`);
        } else {
          this.updateOverlapStepProgress(1, 'processing', `Analyzing audio... (${Math.floor(elapsed)}s)`);
        }
        this.updateOverlapStepProgress(2, 'pending', 'Waiting for Step 1 to complete...');
        this.updateOverlapStepProgress(3, 'pending', 'Waiting...');
        this.updateOverlapStepProgress(5, 'pending', 'Waiting...');
      }
      // Update step 2
      else if (elapsed < stepDurations[1] + stepDurations[2]) {
        if (currentStep !== 2) {
          currentStep = 2;
          this.updateOverlapStepProgress(1, 'completed', 'Completed');
          this.updateOverlapStepProgress(2, 'processing', `Separating speakers... (${Math.floor(elapsed - stepDurations[1])}s)`);
        } else {
          this.updateOverlapStepProgress(2, 'processing', `Separating speakers... (${Math.floor(elapsed - stepDurations[1])}s)`);
        }
        this.updateOverlapStepProgress(3, 'pending', 'Waiting for Step 2 to complete...');
        this.updateOverlapStepProgress(5, 'pending', 'Waiting...');
      }
      // Update step 3
      else if (elapsed < stepDurations[1] + stepDurations[2] + stepDurations[3]) {
        if (currentStep !== 3) {
          currentStep = 3;
          this.updateOverlapStepProgress(2, 'completed', 'Completed');
          this.updateOverlapStepProgress(3, 'processing', `Transcribing and analyzing... (${Math.floor(elapsed - stepDurations[1] - stepDurations[2])}s)`);
        } else {
          this.updateOverlapStepProgress(3, 'processing', `Transcribing and analyzing... (${Math.floor(elapsed - stepDurations[1] - stepDurations[2])}s)`);
        }
        this.updateOverlapStepProgress(5, 'pending', 'Waiting for Step 3 to complete...');
      }
      // Update step 5
      else if (elapsed < stepDurations[1] + stepDurations[2] + stepDurations[3] + stepDurations[5]) {
        if (currentStep !== 5) {
          currentStep = 5;
          this.updateOverlapStepProgress(3, 'completed', 'Completed');
          this.updateOverlapStepProgress(5, 'processing', `Correcting overlaps... (${Math.floor(elapsed - stepDurations[1] - stepDurations[2] - stepDurations[3])}s)`);
        } else {
          this.updateOverlapStepProgress(5, 'processing', `Correcting overlaps... (${Math.floor(elapsed - stepDurations[1] - stepDurations[2] - stepDurations[3])}s)`);
        }
      }
    };
    
    // Update immediately
    updateProgress();
    
    // Update every 2 seconds
    return setInterval(updateProgress, 2000);
  },

  async processRecordingQueue() {
    if (this.testingCancelled) {
      this.finishBatchProcessing();
      return;
    }

    if (this.recordingQueue.length === 0) {
      this.finishBatchProcessing();
      return;
    }

    const recording = this.recordingQueue.shift();
    this.setActiveRecording(recording.id, { skipResultsRender: true });
    recording.status = 'processing';
    this.renderRecordingsQueue();

    await this.runRecordingTests(recording);

    if (this.testingCancelled) {
      recording.status = 'cancelled';
      this.renderRecordingsQueue();
      this.finishBatchProcessing();
      return;
    }

    recording.status = 'completed';
    this.renderRecordingsQueue();
    await this.processRecordingQueue();
  },

  finishBatchProcessing() {
    this.testingInProgress = false;
    SERVICES.forEach(service => this.updateServiceToggleUI(service.id));
    this.updateStartButtonState();
    if (!this.testingCancelled) {
      this.showResults();
    } else {
      this.showView('uploadView');
    }
  },

  async runRecordingTests(recording) {
    this.uploadedFile = recording.file;
    this.fileDuration = recording.duration;
    recording.results = {};
    this.testResults = recording.results;
    this.currentRecordingLanguage = recording.language;
    this.currentRecordingSpeakerCount = recording.speakerCount;
    this.translationState = recording.translationState || this.createTranslationState();

    const avgProcessingTime = this.fileDuration * 0.3;
    const estimatedMinutes = Math.ceil(avgProcessingTime / 60);
    const estimatedEl = document.getElementById('estimatedTime');
    if (estimatedEl) {
      estimatedEl.textContent = `${estimatedMinutes}-${estimatedMinutes + 2} minutes`;
    }

    const progressContainer = document.getElementById('progressContainer');
    const validServices = this.config.selectedServices.filter(serviceId => 
      SERVICES.some(s => s.id === serviceId)
    );
    this.config.selectedServices = validServices;
    this.saveEnabledServices();
    this.updateConfiguredCount();

    if (progressContainer) {
      progressContainer.innerHTML = validServices.map(serviceId => {
        const service = SERVICES.find(s => s.id === serviceId);
        if (!service) return '';
        return `
          <div class="progress-item" id="progress_${serviceId}">
            <div class="progress-header">
              <strong>${getServiceDisplayName(service) || ''}</strong>
              <span class="status status-pending" id="status_text_${serviceId}">Waiting...</span>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" id="progress_bar_${serviceId}" style="width: 0%"></div>
            </div>
            <div style="margin-top: var(--space-8); color: var(--color-text-secondary); font-size: var(--font-size-sm);" id="progress_detail_${serviceId}"></div>
          </div>
        `;
      }).filter(Boolean).join('');
    }

    const promises = validServices.map(serviceId => this.testService(serviceId, recording));
    await Promise.allSettled(promises);
  },

  async testService(serviceId, recording) {
    const service = SERVICES.find(s => s.id === serviceId);
    const statusEl = document.getElementById(`status_text_${serviceId}`);
    const progressBar = document.getElementById(`progress_bar_${serviceId}`);
    const detailEl = document.getElementById(`progress_detail_${serviceId}`);

    const startTime = Date.now();
    let apiKey = this.config.services[serviceId];
    
    // For Speechmatics, try to load from server if not in local config
    if (serviceId === 'speechmatics' && !apiKey) {
      try {
        const response = await fetch('/api/speechmatics-key');
        if (response.ok) {
          const data = await response.json();
          if (data.apiKey) {
            apiKey = data.apiKey;
            this.config.services[serviceId] = apiKey;
            this.saveApiKeys();
            console.log('✅ Loaded Speechmatics API key from server');
          }
        }
      } catch (error) {
        console.error('Failed to load Speechmatics key from server:', error);
      }
    }
    
    // Prioritize current select value over stored recording language
    // Try to get from specific tab fields, fallback to recording data or default
    const language = document.getElementById('audioLanguageSelect')?.value 
      || document.getElementById('combinedLanguageSelect')?.value 
      || document.getElementById('overlapLanguageSelect')?.value 
      || this.currentRecordingLanguage 
      || recording?.language 
      || 'ar';
    const speakerCount = document.getElementById('audioSpeakerCount')?.value 
      || document.getElementById('combinedSpeakerCount')?.value 
      || document.getElementById('overlapSpeakerCount')?.value 
      || this.currentRecordingSpeakerCount 
      || recording?.speakerCount;

    if (!apiKey) {
      throw new Error('API key not configured');
    }

    try {
      // Stage 1: Uploading
      statusEl.textContent = 'Uploading...';
      statusEl.className = 'status status-processing';
      detailEl.textContent = 'Uploading file to server...';
      await this.simulateProgress(progressBar, 0, 20, 500);
      
      if (this.testingCancelled) throw new Error('Cancelled');

      // Stage 2: Processing - Real API call
      statusEl.textContent = 'Processing...';
      detailEl.textContent = 'Analyzing audio and diarization...';
      await this.simulateProgress(progressBar, 20, 60, 1000);
      
      if (this.testingCancelled) throw new Error('Cancelled');

      // Make real API call
      const segments = await this.callDiarizationAPI(serviceId, apiKey, language, speakerCount);
      
      if (this.testingCancelled) throw new Error('Cancelled');

      // Stage 3: Retrieving results
      statusEl.textContent = 'Fetching results...';
      detailEl.textContent = 'Forming transcript...';
      await this.simulateProgress(progressBar, 60, 100, 500);

      const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);
      const speedFactor = (this.fileDuration / parseFloat(processingTime)).toFixed(1);
      const uniqueSpeakers = new Set(segments.map(s => s.speaker));
      const cost = (this.fileDuration / 3600) * service.pricingPerHour;

      // Ensure API key is saved
      if (this.config.services[serviceId] !== apiKey) {
        this.config.services[serviceId] = apiKey;
        this.saveApiKeys();
        if (!this.config.selectedServices.includes(serviceId)) {
          this.config.selectedServices.push(serviceId);
        }
        this.saveEnabledServices();
        this.updateServiceToggleUI(serviceId);
      }

      const displayName = getServiceDisplayName(service) || service.name;

      this.testResults[serviceId] = {
        success: true,
        serviceName: displayName,
        processingTime: parseFloat(processingTime),
        speedFactor: parseFloat(speedFactor),
        speakerCount: uniqueSpeakers.size,
        cost: cost.toFixed(4),
        segments: segments.sort((a, b) => a.start - b.start),
        rawData: {
          duration: this.fileDuration,
          language: language
        }
      };

      const expectedSpeakers = parseInt(speakerCount || '', 10) || 2;
      if (uniqueSpeakers.size <= 1 && expectedSpeakers > 1) {
        this.handleSingleSpeakerWarning(serviceId, expectedSpeakers, uniqueSpeakers.size, statusEl, detailEl, processingTime);
      } else {
        statusEl.textContent = 'Success ✓';
        statusEl.className = 'status status-success';
        detailEl.textContent = `Completed in ${processingTime}s`;
      }

    } catch (error) {
      if (error.message === 'Cancelled') {
        statusEl.textContent = 'Cancelled';
        statusEl.className = 'status status-pending';
        detailEl.textContent = '';
        return;
      }

      // Log detailed error information
      const errorLog = this.createErrorLog(error, serviceId, service.name);
      this.errorLogs[serviceId] = errorLog;
      console.error(`[${service.name}] Error:`, errorLog);

      const displayName = getServiceDisplayName(service) || service.name;

      this.testResults[serviceId] = {
        success: false,
        serviceName: displayName,
        error: error.message || 'Unknown error',
        errorDetails: errorLog
      };

      statusEl.textContent = 'Error ✗';
      statusEl.className = 'status status-error';
      detailEl.textContent = `Error: ${error.message}`;
    }
  },

  handleSingleSpeakerWarning(serviceId, expected, actual, statusEl, detailEl, processingTime) {
    console.warn(`[${serviceId}] Only ${actual} speaker detected (expected ${expected}).`);
    if (statusEl) {
      statusEl.textContent = '⚠️ Only one speaker detected';
      statusEl.className = 'status status-warning';
    }
    if (detailEl) {
      const timingInfo = processingTime ? `Completed in ${processingTime}s. ` : '';
      detailEl.textContent = `${timingInfo}Service returned a single speaker. Adjust speaker settings or sensitivity for better diarization.`;
    }
    if (this.testResults[serviceId]) {
      this.testResults[serviceId].warning = {
        type: 'single-speaker',
        expected,
        actual,
        message: 'Service returned only one speaker'
      };
    }
  },

  /**
   * Create detailed error log
   */
  createErrorLog(error, serviceId, serviceName) {
    const log = {
      timestamp: new Date().toISOString(),
      service: serviceName,
      serviceId: serviceId,
      message: error.message || 'Unknown error',
      stack: error.stack || null,
      name: error.name || 'Error',
      type: 'API_ERROR'
    };

    // Add response details if available
    if (error.response) {
      log.response = {
        status: error.response.status,
        statusText: error.response.statusText,
        headers: Object.fromEntries(error.response.headers || []),
        body: error.response.body || null
      };
    }

    // Add fetch error details
    if (error.fetchError) {
      log.fetchError = {
        url: error.fetchError.url || null,
        method: error.fetchError.method || null,
        status: error.fetchError.status || null,
        statusText: error.fetchError.statusText || null,
        body: error.fetchError.body || null
      };
    }

    // Add any additional error data
    if (error.data) {
      log.data = error.data;
    }

    // Add original error if available
    if (error.originalError) {
      log.originalError = error.originalError;
    }

    // Add console error for debugging
    console.error(`[${serviceName}] Full error details:`, error);
    console.error(`[${serviceName}] Error log:`, log);

    return log;
  },

  /**
   * Call real diarization API
   */
  async callDiarizationAPI(serviceId, apiKey, language, speakerCount) {
    const file = this.uploadedFile;
    if (!file) {
      throw new Error('File not uploaded');
    }

    // Convert file to base64 or FormData
    const formData = new FormData();
    formData.append('file', file);
    if (language) formData.append('language', language);
    if (speakerCount) formData.append('speaker_count', speakerCount);

    let response;
    let uploadUrl;
    let transcriptId;

    switch (serviceId) {
      case 'assemblyai':
        // Model1: Upload file first, then transcribe with diarization
        uploadUrl = 'https://api.assemblyai.com/v2/upload';
        const uploadResponse = await fetch(uploadUrl, {
          method: 'POST',
          headers: {
            'authorization': apiKey
          },
          body: file
        });
        
        if (!uploadResponse.ok) {
          let errorText = '';
          try {
            errorText = await uploadResponse.text();
          } catch (e) {
            errorText = `Failed to read response: ${e.message}`;
          }
          
          const error = new Error(`Model1 upload failed: ${uploadResponse.status} ${uploadResponse.statusText || ''}`);
          error.fetchError = {
            url: uploadUrl,
            method: 'POST',
            status: uploadResponse.status,
            statusText: uploadResponse.statusText || '',
            body: errorText || 'Empty response'
          };
          throw error;
        }
        
        let uploadData;
        try {
          uploadData = await uploadResponse.json();
        } catch (e) {
          const error = new Error(`Model1: Failed to parse upload response`);
          error.fetchError = {
            url: uploadUrl,
            method: 'POST',
            status: uploadResponse.status,
            statusText: uploadResponse.statusText,
            body: await uploadResponse.text().catch(() => 'Unable to read response')
          };
          error.originalError = e.message;
          throw error;
        }
        const audioUrl = uploadData.upload_url;

        // Start transcription with speaker_labels
        console.log('[Model1] Creating transcript with diarization...');
        const transcriptRequest = {
          audio_url: audioUrl,
          speaker_labels: true,  // ОБОВ'ЯЗКОВО для діаризації
          speakers_expected: speakerCount ? parseInt(speakerCount) : 2,  // За замовчуванням 2
          language_code: language || 'ar'
        };
        console.log('[Model1] Request body:', JSON.stringify(transcriptRequest, null, 2));
        
        const transcriptResponse = await fetch('https://api.assemblyai.com/v2/transcript', {
          method: 'POST',
          headers: {
            'authorization': apiKey,
            'content-type': 'application/json'
          },
          body: JSON.stringify(transcriptRequest)
        });

        if (!transcriptResponse.ok) {
          let errorText = '';
          try {
            errorText = await transcriptResponse.text();
          } catch (e) {
            errorText = `Failed to read response: ${e.message}`;
          }
          
          const error = new Error(`Model1 transcription failed: ${transcriptResponse.status} ${transcriptResponse.statusText || ''}`);
          error.fetchError = {
            url: 'https://api.assemblyai.com/v2/transcript',
            method: 'POST',
            status: transcriptResponse.status,
            statusText: transcriptResponse.statusText || '',
            body: errorText || 'Empty response'
          };
          throw error;
        }

        let transcriptData;
        try {
          transcriptData = await transcriptResponse.json();
          console.log('[Model1] Transcript created. Response:', JSON.stringify(transcriptData, null, 2));
        } catch (e) {
          const error = new Error(`Model1: Failed to parse transcript response`);
          error.fetchError = {
            url: 'https://api.assemblyai.com/v2/transcript',
            method: 'POST',
            status: transcriptResponse.status,
            statusText: transcriptResponse.statusText,
            body: await transcriptResponse.text().catch(() => 'Unable to read response')
          };
          error.originalError = e.message;
          throw error;
        }
        
        if (!transcriptData.id) {
          const error = new Error('Model1: No transcript ID in response');
          error.data = transcriptData;
          throw error;
        }
        
        transcriptId = transcriptData.id;
        console.log('[Model1] Transcript ID:', transcriptId);
        console.log('[Model1] Starting polling...');

        // Poll for results - збільшено інтервал до 5 секунд
        let transcriptResult;
        let attempts = 0;
        const maxAttempts = 60;
        
        do {
          // Чекати 5 секунд між запитами (як в інструкції)
          await this.sleep(5000);
          
          const statusResponse = await fetch(`https://api.assemblyai.com/v2/transcript/${transcriptId}`, {
            method: 'GET',
            headers: { 'authorization': apiKey }
          });
          
          if (!statusResponse.ok) {
            let errorText = '';
            try {
              errorText = await statusResponse.text();
            } catch (e) {
              errorText = `Failed to read response: ${e.message}`;
            }
            
            const error = new Error(`Model1 status check failed: ${statusResponse.status} ${statusResponse.statusText || ''}`);
            error.fetchError = {
              url: `https://api.assemblyai.com/v2/transcript/${transcriptId}`,
              method: 'GET',
              status: statusResponse.status,
              statusText: statusResponse.statusText || '',
              body: errorText || 'Empty response'
            };
            throw error;
          }
          
          let parsedResult;
          try {
            parsedResult = await statusResponse.json();
          } catch (e) {
            const error = new Error(`Model1: Failed to parse status response`);
            error.fetchError = {
              url: `https://api.assemblyai.com/v2/transcript/${transcriptId}`,
              method: 'GET',
              status: statusResponse.status,
              statusText: statusResponse.statusText,
              body: await statusResponse.text().catch(() => 'Unable to read response')
            };
            error.originalError = e.message;
            throw error;
          }
          transcriptResult = parsedResult;
          attempts++;
          
          console.log(`[Model1] Polling attempt ${attempts}: status=${transcriptResult.status}`);
          
          // Перевірка на помилку
          if (transcriptResult.status === 'error') {
            const error = new Error(`Model1 transcription error: ${transcriptResult.error || 'Unknown error'}`);
            error.data = transcriptResult;
            throw error;
          }
          
        } while (
          transcriptResult.status !== 'completed' && 
          transcriptResult.status !== 'error' &&
          attempts < maxAttempts
        );

        // Перевірка що досягнуто completed
        if (transcriptResult.status !== 'completed') {
          const error = new Error(`Model1 timeout after ${attempts} attempts. Last status: ${transcriptResult.status}`);
          error.data = transcriptResult;
          throw error;
        }
        
        console.log('[Model1] Transcription completed successfully');
        console.log('[Model1] Utterances count:', transcriptResult.utterances?.length || 0);

        // Log full response before parsing
        console.log('===== ASSEMBLYAI RAW RESPONSE =====');
        console.log(JSON.stringify(transcriptResult, null, 2));
        console.log('===== END ASSEMBLYAI RAW RESPONSE =====');

        // Parse Model1 response
        return this.parseModel1Response(transcriptResult);

      case 'deepgram':
        // Model2: Direct transcription with diarization
        const deepgramParams = new URLSearchParams({
          diarize: 'true',
          punctuate: 'true',
          model: 'nova-2',
          language: language || 'ar'
        });
        
        response = await fetch(`https://api.deepgram.com/v1/listen?${deepgramParams}`, {
          method: 'POST',
          headers: {
            'Authorization': `Token ${apiKey}`,
            'Content-Type': file.type || 'audio/*'
          },
          body: file
        });

        if (!response.ok) {
          let errorText = '';
          try {
            errorText = await response.text();
          } catch (e) {
            errorText = `Failed to read response: ${e.message}`;
          }
          
          const error = new Error(`Model2 API failed: ${response.status} ${response.statusText || ''} - ${errorText}`);
          error.fetchError = {
            url: `https://api.deepgram.com/v1/listen?${deepgramParams}`,
            method: 'POST',
            status: response.status,
            statusText: response.statusText || '',
            body: errorText || 'Empty response'
          };
          throw error;
        }

        let deepgramData;
        try {
          deepgramData = await response.json();
        } catch (e) {
          const error = new Error(`Model2: Failed to parse response`);
          error.fetchError = {
            url: `https://api.deepgram.com/v1/listen?${deepgramParams}`,
            method: 'POST',
            status: response.status,
            statusText: response.statusText,
            body: await response.text().catch(() => 'Unable to read response')
          };
          error.originalError = e.message;
          throw error;
        }
        return this.parseModel2Response(deepgramData);

      case 'speechmatics':
        // Model3: Upload file directly with transcription config
        const smFormData = new FormData();
        smFormData.append('data_file', file);
        
        const smConfig = {
          type: 'transcription',
          transcription_config: {
            language: language || 'ar',
            diarization: 'speaker',
            operating_point: 'enhanced',
            speaker_diarization_config: {
              get_speakers: true
            }
          }
        };
        
        smFormData.append('config', JSON.stringify(smConfig));

        const smUpload = await fetch('https://asr.api.speechmatics.com/v2/jobs', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${apiKey}`
            // Content-Type will be set automatically by browser for FormData
          },
          body: smFormData
        });

        if (!smUpload.ok) {
          let errorText = '';
          try {
            errorText = await smUpload.text();
          } catch (e) {
            errorText = `Failed to read response: ${e.message}`;
          }
          
          const error = new Error(`Model3 upload failed: ${smUpload.status} ${smUpload.statusText || ''}`);
          error.fetchError = {
            url: 'https://asr.api.speechmatics.com/v2/jobs',
            method: 'POST',
            status: smUpload.status,
            statusText: smUpload.statusText || '',
            body: errorText || 'Empty response'
          };
          throw error;
        }

        let smData;
        try {
          smData = await smUpload.json();
        } catch (e) {
          const error = new Error(`Model3: Failed to parse upload response`);
          error.fetchError = {
            url: 'https://asr.api.speechmatics.com/v2/jobs',
            method: 'POST',
            status: smUpload.status,
            statusText: smUpload.statusText || '',
            body: await smUpload.text().catch(() => 'Unable to read response')
          };
          error.originalError = e.message;
          throw error;
        }
        
        if (!smData.id) {
          const error = new Error('Model3: No job ID in response');
          error.data = smData;
          throw error;
        }
        
        const smJobId = smData.id;

        // Poll for results
        let smResult;
        let smAttempts = 0;
        while (smAttempts < 60) {
          await this.sleep(2000);
          const smStatus = await fetch(`https://asr.api.speechmatics.com/v2/jobs/${smJobId}`, {
            headers: { 'Authorization': `Bearer ${apiKey}` }
          });
          
          if (!smStatus.ok) {
            let errorText = '';
            try {
              errorText = await smStatus.text();
            } catch (e) {
              errorText = `Failed to read response: ${e.message}`;
            }
            
            const error = new Error(`Model3 status check failed: ${smStatus.status} ${smStatus.statusText || ''}`);
            error.fetchError = {
              url: `https://asr.api.speechmatics.com/v2/jobs/${smJobId}`,
              method: 'GET',
              status: smStatus.status,
              statusText: smStatus.statusText || '',
              body: errorText || 'Empty response'
            };
            throw error;
          }
          
          let parsedSmResult;
          try {
            parsedSmResult = await smStatus.json();
          } catch (e) {
            const error = new Error(`Model3: Failed to parse status response`);
            error.fetchError = {
              url: `https://asr.api.speechmatics.com/v2/jobs/${smJobId}`,
              method: 'GET',
              status: smStatus.status,
              statusText: smStatus.statusText || '',
              body: await smStatus.text().catch(() => 'Unable to read response')
            };
            error.originalError = e.message;
            throw error;
          }
          smResult = parsedSmResult;
          
          if (smResult.job && smResult.job.status === 'done') break;
          if (smResult.job && smResult.job.status === 'rejected') {
            const error = new Error(smResult.job.failure_reason || 'Transcription failed');
            error.data = smResult;
            throw error;
          }
          smAttempts++;
        }

        if (!smResult.job || smResult.job.status !== 'done') {
          const error = new Error('Transcription timeout');
          error.data = smResult;
          throw error;
        }

        // Get transcript
        const smTranscript = await fetch(`https://asr.api.speechmatics.com/v2/jobs/${smJobId}/transcript`, {
          headers: { 
            'Authorization': `Bearer ${apiKey}`,
            'Accept': 'application/json'
          }
        });
        
        if (!smTranscript.ok) {
          let errorText = '';
          try {
            errorText = await smTranscript.text();
          } catch (e) {
            errorText = `Failed to read response: ${e.message}`;
          }
          
          const error = new Error(`Model3 transcript fetch failed: ${smTranscript.status} ${smTranscript.statusText || ''}`);
          error.fetchError = {
            url: `https://asr.api.speechmatics.com/v2/jobs/${smJobId}/transcript`,
            method: 'GET',
            status: smTranscript.status,
            statusText: smTranscript.statusText || '',
            body: errorText || 'Empty response'
          };
          throw error;
        }
        
        let smTranscriptData;
        try {
          smTranscriptData = await smTranscript.json();
        } catch (e) {
          const error = new Error(`Model3: Failed to parse transcript response`);
          error.fetchError = {
            url: `https://asr.api.speechmatics.com/v2/jobs/${smJobId}/transcript`,
            method: 'GET',
            status: smTranscript.status,
            statusText: smTranscript.statusText || '',
            body: await smTranscript.text().catch(() => 'Unable to read response')
          };
          error.originalError = e.message;
          throw error;
        }
        return this.parseModel3Response(smTranscriptData);

      default:
        throw new Error(`Unknown service: ${serviceId}`);
    }
  },

  /**
   * Parse Model1 response
   * КРИТИЧНО: Перевіряємо наявність utterances та додаємо детальне логування
   */
  parseModel1Response(data) {
    console.log('===== ASSEMBLYAI PARSING =====');
    console.log('Full data object:', data);
    console.log('Data type:', typeof data);
    console.log('Data is null?:', data === null);
    console.log('Data is undefined?:', data === undefined);
    
    if (!data) {
      console.error('Model1: data is null or undefined!');
      console.error('===== END ASSEMBLYAI PARSING (ERROR - NO DATA) =====');
      return [];
    }
    
    console.log('Data keys:', Object.keys(data || {}));
    console.log('Has utterances?:', !!data?.utterances);
    console.log('Utterances count:', data?.utterances?.length || 0);
    console.log('Has words?:', !!data?.words);
    console.log('Words count:', data?.words?.length || 0);
    console.log('Has text?:', !!data?.text);
    console.log('Status:', data?.status);
    
    const segments = [];
    
    // Спробувати різні формати відповіді
    // Format 1: utterances (найкращий варіант)
    if (data.utterances && Array.isArray(data.utterances) && data.utterances.length > 0) {
      console.log('[Model1] Using utterances format');
      console.log('[Model1] Utterances count:', data.utterances.length);
      console.log('[Model1] First utterance sample:', JSON.stringify(data.utterances[0], null, 2));
      
      data.utterances.forEach(utterance => {
        let speaker = utterance.speaker;
        
        // Model1 повертає speaker як "A", "B", "C" або числа 0, 1, 2
        // Конвертуємо формат спікера в стандартний
        if (speaker === undefined || speaker === null) {
          speaker = 0;
        } else if (typeof speaker === 'string') {
          // Якщо "A", "B", "C" - конвертуємо в числа: A=0, B=1, C=2
          const charCode = speaker.charCodeAt(0);
          if (charCode >= 65 && charCode <= 90) { // A-Z
            speaker = charCode - 65; // A=0, B=1, C=2
          } else {
            // Спробувати парсити як число
            speaker = parseInt(speaker) || 0;
          }
        } else {
          speaker = parseInt(speaker) || 0;
        }
        
        const speakerLabel = `SPEAKER_${speaker.toString().padStart(2, '0')}`;
        
        segments.push({
          speaker: speakerLabel,
          start: utterance.start / 1000, // Convert ms to seconds
          end: utterance.end / 1000,
          text: utterance.text || ''
        });
      });
      
      const uniqueSpeakers = [...new Set(segments.map(s => s.speaker))];
      console.log(`[Model1] Parsed ${segments.length} utterances from ${uniqueSpeakers.length} speakers`);
      console.log(`[Model1] Speakers:`, uniqueSpeakers);
      console.log(`[Model1] Final result:`, {
        speakerCount: uniqueSpeakers.length,
        speakers: uniqueSpeakers,
        utterancesCount: segments.length
      });
      console.log('===== END ASSEMBLYAI PARSED =====');
      return segments;
    }
    
    // Format 2: words array (fallback)
    if (data.words && Array.isArray(data.words) && data.words.length > 0) {
      console.log('Model1: Using words format (fallback)');
      let currentSpeaker = null;
      let currentStart = null;
      let currentText = [];
      
      data.words.forEach(word => {
        const speaker = word.speaker !== undefined ? word.speaker : 0;
        const speakerLabel = `SPEAKER_${speaker.toString().padStart(2, '0')}`;
        const wordStart = (word.start || 0) / 1000; // ms to seconds
        const wordEnd = (word.end || wordStart * 1000) / 1000;
        
        if (currentSpeaker !== speakerLabel) {
          if (currentSpeaker !== null && currentStart !== null) {
            segments.push({
              speaker: currentSpeaker,
              start: currentStart,
              end: wordStart,
              text: currentText.join(' ').trim()
            });
          }
          currentSpeaker = speakerLabel;
          currentStart = wordStart;
          currentText = [word.text || ''];
        } else {
          currentText.push(word.text || '');
        }
      });
      
      if (currentSpeaker !== null && currentStart !== null && data.words.length > 0) {
        const lastWord = data.words[data.words.length - 1];
        segments.push({
          speaker: currentSpeaker,
          start: currentStart,
          end: (lastWord.end || lastWord.start || currentStart * 1000) / 1000,
          text: currentText.join(' ').trim()
        });
      }
      
      console.log(`Model1 parsed (from words): ${segments.length} segments`);
      console.log('===== END ASSEMBLYAI PARSED =====');
      return segments;
    }
    
    // Format 3: results.utterances (можливий вкладений формат)
    if (data.results && data.results.utterances && Array.isArray(data.results.utterances)) {
      console.log('Model1: Using results.utterances format');
      data.results.utterances.forEach(utterance => {
        const speaker = utterance.speaker !== undefined ? utterance.speaker : 0;
        segments.push({
          speaker: `SPEAKER_${speaker.toString().padStart(2, '0')}`,
          start: (utterance.start || 0) / 1000,
          end: (utterance.end || utterance.start || 0) / 1000,
          text: utterance.text || ''
        });
      });
      console.log(`Model1 parsed (from results.utterances): ${segments.length} segments`);
      console.log('===== END ASSEMBLYAI PARSED =====');
      return segments;
    }
    
    // Якщо нічого не знайдено
    console.error('Model1: No utterances or words found!');
    console.error('Full response structure:', JSON.stringify(data, null, 2));
    console.error('===== END ASSEMBLYAI PARSING (ERROR - NO DATA FOUND) =====');
    return segments;
  },

  /**
   * Parse Model2 response
   */
  parseModel2Response(data) {
    const segments = [];
    if (data.results && data.results.channels && data.results.channels[0]) {
      const channel = data.results.channels[0];
      if (channel.alternatives && channel.alternatives[0]) {
        const words = channel.alternatives[0].words || [];
        if (words.length === 0) return segments;

        let currentSpeaker = null;
        let currentStart = null;
        let currentText = [];

        words.forEach((word, idx) => {
          const speaker = word.speaker !== undefined ? word.speaker : 0;
          const speakerLabel = `SPEAKER_${speaker.toString().padStart(2, '0')}`;
          const wordStart = word.start || 0;
          const wordEnd = word.end || wordStart;

          if (currentSpeaker !== speakerLabel) {
            if (currentSpeaker !== null && currentStart !== null) {
              segments.push({
                speaker: currentSpeaker,
                start: currentStart,
                end: wordStart,
                text: currentText.join(' ').trim()
              });
            }
            currentSpeaker = speakerLabel;
            currentStart = wordStart;
            currentText = [word.word || word.punctuated_word || ''];
          } else {
            currentText.push(word.word || word.punctuated_word || '');
          }
        });

        // Add last segment
        if (currentSpeaker !== null && currentStart !== null && words.length > 0) {
          const lastWord = words[words.length - 1];
          segments.push({
            speaker: currentSpeaker,
            start: currentStart,
            end: lastWord.end || lastWord.start || currentStart,
            text: currentText.join(' ').trim()
          });
        }
      }
    }
    return segments;
  },

  /**
   * Parse Model3 response (v2 API format)
   * КРИТИЧНО: Спікери в data.results[].alternatives[0].speaker (формат "S1", "S2")
   * Треба групувати words в utterances самостійно
   */
  parseModel3Response(data) {
    console.log('===== SPEECHMATICS RAW RESPONSE =====');
    console.log(JSON.stringify(data, null, 2));
    console.log('===== END SPEECHMATICS RAW RESPONSE =====');
    
    const segments = [];
    const words = [];
    
    // Model3 повертає масив results з words
    if (!data?.results || !Array.isArray(data.results)) {
      console.error('Model3: results array not found');
      console.error('Available paths:', Object.keys(data || {}));
      
      // Fallback: спробувати інші формати
      if (data.words && Array.isArray(data.words)) {
        // Direct words array
        data.words.forEach(word => {
          const speakerId = word.speaker || word.speaker_id || 0;
          const speakerLabel = `SPEAKER_${speakerId.toString().padStart(2, '0')}`;
          words.push({
            word: word.word || word.text || '',
            start: word.start || word.start_time || 0,
            end: word.end || word.end_time || word.start || 0,
            speaker: speakerId
          });
        });
      } else if (data.job && data.job.transcript) {
        // v1 API format
        const transcript = data.job.transcript;
        if (transcript.speakers) {
          transcript.speakers.forEach(speaker => {
            if (speaker.segments) {
              speaker.segments.forEach(segment => {
                segments.push({
                  speaker: `SPEAKER_${speaker.id.toString().padStart(2, '0')}`,
                  start: segment.start_time || 0,
                  end: segment.end_time || segment.start_time || 0,
                  text: segment.text || ''
                });
              });
            }
          });
        }
        console.log(`Model3 parsed (v1 format): ${segments.length} segments`);
        return segments;
      } else {
        console.warn('Model3: No results array and no fallback format found');
        return segments;
      }
    } else {
      // Збираємо всі слова зі спікерами з results масиву
      data.results.forEach(result => {
        // Пропускаємо punctuation
        if (result.type === 'punctuation') return;
        
        if (result.type === 'word' && result.alternatives?.[0]) {
          const alt = result.alternatives[0];
          
          // speaker формат: "S1", "S2", etc.
          const speakerLabel = alt.speaker || 'S1';
          
          // Конвертуємо "S1" -> 0, "S2" -> 1, "S3" -> 2, etc.
          let speakerNum = 0;
          if (speakerLabel.startsWith('S')) {
            const numStr = speakerLabel.substring(1);
            speakerNum = parseInt(numStr) - 1;
            if (isNaN(speakerNum) || speakerNum < 0) speakerNum = 0;
          } else {
            // Якщо вже число
            speakerNum = parseInt(speakerLabel) || 0;
          }
          
          words.push({
            word: alt.content || '',
            start: result.start_time || 0,
            end: result.end_time || result.start_time || 0,
            speaker: speakerNum,
            confidence: alt.confidence
          });
        }
      });
    }
    
    if (words.length === 0) {
      console.warn('Model3: No words found in response');
      return segments;
    }
    
    // Групуємо слова в utterances по спікерах
    let currentUtterance = null;
    
    words.forEach(word => {
      const speakerLabel = `SPEAKER_${word.speaker.toString().padStart(2, '0')}`;
      
      if (!currentUtterance || currentUtterance.speaker !== speakerLabel) {
        // Зберегти попередній
        if (currentUtterance) {
          segments.push(currentUtterance);
        }
        // Почати новий
        currentUtterance = {
          speaker: speakerLabel,
          text: word.word,
          start: word.start,
          end: word.end,
          words: [word]
        };
      } else {
        // Додати до поточного
        currentUtterance.text += ' ' + word.word;
        currentUtterance.end = word.end;
        currentUtterance.words.push(word);
      }
    });
    
    // Додати останній utterance
    if (currentUtterance) {
      segments.push(currentUtterance);
    }
    
    console.log(`Model3 parsed: ${segments.length} segments, speakers:`, [...new Set(segments.map(s => s.speaker))]);
    console.log('===== END SPEECHMATICS PARSED =====');
    
    return segments;
  },

  /**
   * Get JSON schema for structured diarization output
   * Формат точно відповідає diarization_test_results.json structure
   */
  getTextServiceDiarizationSchema() {
    return {
      "type": "object",
      "properties": {
        "version": {
          "type": "string",
          "description": "Version of the output format, must be '2.0'"
        },
        "exportedAt": {
          "type": "string",
          "description": "ISO 8601 timestamp of export"
        },
        "activeRecordingId": {
          "type": "string",
          "description": "ID of the active recording"
        },
        "recordings": {
          "type": "array",
          "description": "Array of recording results. Each recording contains segments with speaker diarization",
          "items": {
            "type": "object",
            "properties": {
              "id": {
                "type": "string",
                "description": "Unique recording identifier"
              },
              "name": {
                "type": "string",
                "description": "Recording name or title"
              },
              "fileName": {
                "type": "string",
                "description": "Original file name"
              },
              "size": {
                "type": "integer",
                "description": "File size in bytes (0 for text input)"
              },
              "duration": {
                "type": "number",
                "description": "Total duration in seconds (calculated from segments)"
              },
              "language": {
                "type": "string",
                "description": "Language code (e.g., 'ar' for Arabic)"
              },
              "speakerCount": {
                "type": "string",
                "description": "Number of speakers detected (as string)"
              },
              "status": {
                "type": "string",
                "description": "Status, must be 'completed'"
              },
              "addedAt": {
                "type": ["string", "null"],
                "description": "ISO 8601 timestamp or null"
              },
              "translationState": {
                "type": "object",
                "properties": {
                  "currentLanguage": {
                    "type": "string",
                    "description": "Current language, default 'original'"
                  },
                  "lastError": {
                    "type": ["string", "null"],
                    "description": "Last error message or null"
                  }
                },
                "required": ["currentLanguage", "lastError"]
              },
              "results": {
                "type": "object",
                "description": "Results object containing service results",
                "properties": {
                  [TEXT_SERVICE_KEY]: {
                    "type": "object",
                    "properties": {
                      "success": {
                        "type": "boolean",
                        "description": "Whether processing was successful"
                      },
                      "serviceName": {
                        "type": "string",
                        "description": "Service name, must be 'Text Mode 📝'"
                      },
                      "processingTime": {
                        "type": "number",
                        "description": "Processing time in seconds (0 for text mode)"
                      },
                      "speedFactor": {
                        "type": "number",
                        "description": "Speed factor (0 for text mode)"
                      },
                      "speakerCount": {
                        "type": "integer",
                        "description": "Number of unique speakers detected"
                      },
                      "cost": {
                        "type": "string",
                        "description": "Cost as string, must be '0.0000' for text mode"
                      },
                      "segments": {
                        "type": "array",
                        "description": "Array of diarization segments with speaker labels, timestamps, and words",
                        "items": {
                          "type": "object",
                          "properties": {
                            "speaker": {
                              "type": "string",
                              "description": "Speaker label in format SPEAKER_00, SPEAKER_01, etc.",
                              "pattern": "^SPEAKER_\\d{2}$"
                            },
                            "text": {
                              "type": "string",
                              "description": "Full text of the segment"
                            },
                            "start": {
                              "type": "number",
                              "description": "Start time in seconds (float)"
                            },
                            "end": {
                              "type": "number",
                              "description": "End time in seconds (float)"
                            },
                            "words": {
                              "type": "array",
                              "description": "Array of individual words with timestamps",
                              "items": {
                                "type": "object",
                                "properties": {
                                  "word": {
                                    "type": "string",
                                    "description": "The word text"
                                  },
                                  "start": {
                                    "type": "number",
                                    "description": "Word start time in seconds"
                                  },
                                  "end": {
                                    "type": "number",
                                    "description": "Word end time in seconds"
                                  },
                                  "speaker": {
                                    "type": "integer",
                                    "description": "Speaker number (0, 1, 2, etc.)"
                                  },
                                  "confidence": {
                                    "type": "number",
                                    "description": "Confidence score between 0 and 1",
                                    "minimum": 0,
                                    "maximum": 1
                                  }
                                },
                                "required": ["word", "start", "end", "speaker"]
                              }
                            }
                          },
                          "required": ["speaker", "text", "start", "end"]
                        }
                      },
                      "rawData": {
                        "type": "object",
                        "properties": {
                          "duration": {
                            "type": "number",
                            "description": "Total duration in seconds"
                          },
                          "language": {
                            "type": "string",
                            "description": "Language code"
                          },
                          "source": {
                            "type": "string",
                            "description": "Source type, must be 'text'"
                          }
                        },
                        "required": ["duration", "language", "source"]
                      }
                    },
                    "required": ["success", "serviceName", "processingTime", "speedFactor", "speakerCount", "cost", "segments", "rawData"]
                  }
                },
                "required": [TEXT_SERVICE_KEY]
              },
              "aggregated": {
                "type": "object",
                "description": "Aggregated metrics (empty object for text mode)"
              },
              "servicesTested": {
                "type": "array",
                "description": "Array of tested service IDs",
                "items": {
                  "type": "string"
                },
                "contains": {
                  "const": TEXT_SERVICE_KEY
                }
              }
            },
            "required": ["id", "name", "fileName", "size", "duration", "language", "speakerCount", "status", "translationState", "results", "aggregated", "servicesTested"]
          }
        }
      },
      "required": ["version", "exportedAt", "activeRecordingId", "recordings"]
    };
  },

  /**
   * Parse structured text-based diarization response
   * Очікує JSON у форматі diarization_test_results.json (з Structured Output)
   * Обробляє так само як parseModel3Response для Speechmatics
   */
  parseTextServiceResponse(data) {
    console.log('===== TEXT SERVICE RAW RESPONSE =====');
    console.log(JSON.stringify(data, null, 2));
    console.log('===== END TEXT SERVICE RAW RESPONSE =====');
    
    const segments = [];
    
    // Очікуємо формат: { version, exportedAt, activeRecordingId, recordings: [...] }
    let segmentsArray = null;
    let actualData = data;
    
    if (typeof data === 'string') {
      try {
        const parsed = JSON.parse(data);
        return this.parseTextServiceResponse(parsed);
      } catch (e) {
        console.error('Failed to parse string response as JSON:', e);
        return segments;
      }
    }
    
    // Обробка відповіді від n8n: масив з об'єктом, що містить поле "output"
    if (Array.isArray(data) && data.length > 0) {
      const firstItem = data[0];
      if (firstItem && firstItem.output) {
        // n8n формат: [{ output: { recordings: [...] } }]
        actualData = firstItem.output;
        console.log('Detected n8n response format, extracted output:', actualData);
      } else if (firstItem && firstItem.recordings) {
        // Альтернативний формат: [{ recordings: [...] }]
        actualData = firstItem;
      }
    } else if (data.output) {
      // Прямий об'єкт з полем output
      actualData = data.output;
      console.log('Detected output field, extracted:', actualData);
    }
    
    // New format: recordings[0].results[text-service].segments
    // Обробляємо всі recordings і збираємо segments з усіх
    if (actualData.recordings && Array.isArray(actualData.recordings) && actualData.recordings.length > 0) {
      // Збираємо segments з усіх recordings
      const allSegments = [];
      actualData.recordings.forEach(recording => {
        if (recording.results && this.getTextServiceResult(recording)) {
          const textResult = this.getTextServiceResult(recording);
          if (textResult.segments) {
            allSegments.push(...textResult.segments);
          }
        }
      });
      if (allSegments.length > 0) {
        segmentsArray = allSegments;
        console.log(`Found ${allSegments.length} segments across ${actualData.recordings.length} recordings`);
      }
    }
    // Старий формат (fallback): { segments: [...] } або прямий масив segments
    else if (Array.isArray(actualData)) {
      segmentsArray = actualData;
    } else if (actualData.segments && Array.isArray(actualData.segments)) {
      segmentsArray = actualData.segments;
    } else if (actualData.response_format && actualData.response_format.segments) {
      segmentsArray = actualData.response_format.segments;
    }
    
    if (!segmentsArray) {
      console.warn('Text service: No segments found in response', Object.keys(data || {}));
      return segments;
    }
    
    // Обробляємо segments так само як Speechmatics
    segmentsArray.forEach(segment => {
      // Перевіряємо формат segment
      if (!segment.speaker || !segment.text || segment.start === undefined || segment.end === undefined) {
        console.warn('Text service: Invalid segment format', segment);
        return;
      }
      
      // Нормалізуємо speaker label
      let speakerLabel = segment.speaker;
      if (typeof speakerLabel === 'number') {
        speakerLabel = `SPEAKER_${speakerLabel.toString().padStart(2, '0')}`;
      } else if (speakerLabel.startsWith('S') && !speakerLabel.startsWith('SPEAKER_')) {
        // Конвертуємо "S1" -> "SPEAKER_00", "S2" -> "SPEAKER_01"
        const num = parseInt(speakerLabel.substring(1)) - 1;
        speakerLabel = `SPEAKER_${Math.max(0, num).toString().padStart(2, '0')}`;
      } else if (!speakerLabel.startsWith('SPEAKER_')) {
        const num = parseInt(speakerLabel) || 0;
        speakerLabel = `SPEAKER_${num.toString().padStart(2, '0')}`;
      }
      
      // Нормалізуємо words
      let words = [];
      if (segment.words && Array.isArray(segment.words)) {
        words = segment.words.map(word => {
          // Визначаємо speaker number з speaker label сегменту
          const speakerMatch = speakerLabel.match(/SPEAKER_(\d+)/);
          const speakerNum = speakerMatch ? parseInt(speakerMatch[1]) : 0;
          
          return {
            word: word.word || word.text || '',
            start: parseFloat(word.start || word.start_time || segment.start || 0),
            end: parseFloat(word.end || word.end_time || word.start || segment.start || 0),
            speaker: typeof word.speaker === 'number' ? word.speaker : speakerNum,
            confidence: typeof word.confidence === 'number' ? word.confidence : 1
          };
        });
      }
      
      segments.push({
        speaker: speakerLabel,
        text: segment.text || '',
        start: parseFloat(segment.start),
        end: parseFloat(segment.end),
        words: words,
        role: segment.role || null, // Preserve role if present (operator/client)
        overlap: segment.overlap !== undefined ? segment.overlap : false // Will be calculated after all segments are collected
      });
    });
    
    // Helper function to detect overlaps
    function detectOverlap(segments, start, end, currentSpeaker) {
      return segments.some(seg => {
        if (seg.speaker === currentSpeaker) return false; // Same speaker doesn't count as overlap
        // Check if time ranges overlap
        return (start < seg.end && end > seg.start);
      });
    }
    
    function isAbruptEnding(text = '') {
      const trimmed = (text || '').trim();
      if (!trimmed) return false;
      const abruptPattern = /(—|-|–|\.\.\.|…|--|-)$/;
      if (abruptPattern.test(trimmed)) {
        return true;
      }
      // Ends with comma or conjunctions often means interruption
      if (/[,؛、，]$/.test(trimmed)) {
        return true;
      }
      // Missing strong punctuation when sentence seems incomplete
      if (!/[.!?؟!…]$/.test(trimmed) && trimmed.split(/\s+/).length >= 5) {
        return true;
      }
      return false;
    }
    
    function markContextualOverlaps(sortedSegments) {
      if (!Array.isArray(sortedSegments) || sortedSegments.length < 2) return;
      for (let i = 0; i < sortedSegments.length - 1; i++) {
        const current = sortedSegments[i];
        const next = sortedSegments[i + 1];
        if (!current || !next || current.speaker === next.speaker) continue;
        
        const timeGap = (next.start ?? 0) - (current.end ?? 0);
        const overlapsInTime = (current.end ?? 0) > (next.start ?? 0);
        const abruptEnding = isAbruptEnding(current.text);
        const quickInterruption = timeGap < 0.35;
        
        if ((overlapsInTime || (abruptEnding && quickInterruption)) && !current.overlap && !next.overlap) {
          current.overlap = true;
          next.overlap = true;
          current.overlapReason = overlapsInTime ? 'timeline_overlap' : 'contextual_abrupt';
          next.overlapReason = overlapsInTime ? 'timeline_overlap' : 'contextual_abrupt';
        }
      }
    }
    
    // After all segments are collected, detect overlaps
    segments.forEach(seg => {
      if (seg.overlap === false) {
        seg.overlap = detectOverlap(segments, seg.start, seg.end, seg.speaker);
        if (seg.overlap && !seg.overlapReason) {
          seg.overlapReason = 'timeline_overlap';
        }
      }
    });
    
    // Сортуємо segments за часом (як у Speechmatics)
    segments.sort((a, b) => a.start - b.start);
    markContextualOverlaps(segments);
    
    console.log(`Text service parsed: ${segments.length} segments, speakers:`, [...new Set(segments.map(s => s.speaker))]);
    console.log('===== END OPENAI TEXT PARSED =====');
    
    return segments;
  },

  async simulateProgress(progressBar, from, to, duration) {
    const steps = 20;
    const stepDuration = duration / steps;
    const increment = (to - from) / steps;

    for (let i = 0; i <= steps; i++) {
      if (this.testingCancelled) break;
      progressBar.style.width = `${from + (increment * i)}%`;
      await this.sleep(stepDuration);
    }
  },

  generateMockSegments(speakerCount) {
    const segments = [];
    const speakerLabels = Array.from({length: speakerCount}, (_, i) => `SPEAKER_${i.toString().padStart(2, '0')}`);
    
    let currentTime = 0;
    const numSegments = Math.floor(Math.random() * 5) + 8; // 8-12 segments

    for (let i = 0; i < numSegments; i++) {
      const speaker = speakerLabels[Math.floor(Math.random() * speakerCount)];
      const duration = Math.floor(Math.random() * 15) + 5; // 5-20 seconds
      const text = SAMPLE_PHRASES[Math.floor(Math.random() * SAMPLE_PHRASES.length)];

      segments.push({
        speaker,
        start: currentTime,
        end: currentTime + duration,
        text
      });

      currentTime += duration + Math.floor(Math.random() * 3) + 1; // Add small gap
    }

    return segments;
  },

  cancelTesting() {
    this.testingCancelled = true;
    this.testingInProgress = false;
    this.showView('uploadView');
    SERVICES.forEach(service => this.updateServiceToggleUI(service.id));
  },

  ensureTranslationMetadata(recording = this.getActiveRecording()) {
    const results = recording?.results || this.testResults || {};
    Object.values(results).forEach(result => {
      if (!result || !result.success || !Array.isArray(result.segments)) {
        return;
      }
      result.segments.forEach(segment => {
        if (typeof segment.originalText !== 'string') {
          segment.originalText = segment.text || '';
        }
        if (!segment.translations || typeof segment.translations !== 'object') {
          segment.translations = {};
        }
      });
    });
  },

  showResults() {
    this.showView('resultsView');
    this.ensureTranslationMetadata();
    
    // Переконатися, що вкладка Summary видима
    try {
      const summaryTab = document.getElementById('tab-summary');
      if (summaryTab) {
        summaryTab.style.display = 'block';
        summaryTab.classList.add('active');
      }
      
      // Показати всі вкладки контенту
      document.querySelectorAll('.tab-content').forEach(tab => {
        if (tab.id === 'tab-summary') {
          tab.style.display = 'block';
        } else {
          tab.style.display = 'none';
        }
      });
    } catch (error) {
      console.warn('Error showing summary tab:', error);
    }
    
    // Calculate comprehensive metrics if ground truth available
    if (this.groundTruth) {
      this.calculateComprehensiveMetrics();
    }
    
    // Render executive summary
    // this.renderExecutiveSummary(); // Removed - no longer showing best performance banner

    // Check if we have Combined or Overlap results
    const hasCombinedResults = this.testResults.combined && this.testResults.combined.status === 'completed';
    const hasOverlapResults = this.testResults.overlap && this.testResults.overlap.success;
    
    // Populate results table
    const tbody = document.getElementById('resultsTableBody');
    
    // If we have Overlap results, show them first
    if (hasOverlapResults) {
      this.renderOverlapDiarizationResults(this.testResults.overlap, tbody);
    }
    
    // If we have Combined results, show them with detailed steps
    if (hasCombinedResults) {
      const combined = this.testResults.combined;
      const finalResult = combined.finalResult || combined.corrected;
      const pipelineMode = combined.pipelineMode || 'hybrid';
      const llmSource = combined.llm?.source || 'unknown';
      const llmProvider = combined.llm?.provider || (llmSource.includes('openrouter') ? 'openrouter' : 'google');
      const llmIsMultimodal = typeof combined.llm?.multimodal === 'boolean'
        ? combined.llm.multimodal
        : (pipelineMode === 'multimodal' && llmProvider === 'google');
      const pipelineLabel = pipelineMode === 'multimodal'
        ? (llmIsMultimodal ? 'Multimodal' : 'Multimodal (Fallback text-only)')
        : 'Audio + Text LLM';
      const llmProviderLabel = llmProvider === 'google'
        ? 'Multimodal'
        : llmProvider === 'openrouter'
          ? 'Multimodal'
          : llmSource;
      const llmModeText = llmIsMultimodal ? 'Multimodal (audio + text)' : 'Fallback (text-only)';
      const llmModeColor = llmIsMultimodal ? 'var(--color-success)' : 'var(--color-warning)';
      const llmModeBadge = llmIsMultimodal ? '🟢' : '⚠️';
      const recording = finalResult?.recordings?.[0];
      const combinedSegments = recording?.results?.combined?.segments;
      const textSegments = recording?.results?.[TEXT_SERVICE_KEY]?.segments;
      const segments = combinedSegments || textSegments || [];
      const speakerCount = new Set(segments.map(s => s.speaker)).size;
      
      tbody.innerHTML = `
        <tr style="background: var(--color-bg-2);">
          <td colspan="6" style="padding: var(--space-16);">
            <h3 style="margin-bottom: var(--space-16);">Combined Diarization Results</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-16);">
              <div>
                <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Total Duration</div>
                <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${combined.totalDuration || 'N/A'}</div>
              </div>
              <div>
                <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Detected Speakers</div>
                <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${speakerCount}</div>
              </div>
              <div>
                <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Segments</div>
                <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${segments.length}</div>
              </div>
              <div>
                <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Pipeline Mode</div>
                <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${pipelineLabel}</div>
              </div>
              <div>
                <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">LLM Provider</div>
                <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${llmProviderLabel}</div>
                <div style="font-size: var(--font-size-sm); color: ${llmModeColor}; margin-top: var(--space-4);">${llmModeBadge} ${llmModeText}</div>
              </div>
            </div>
            <div style="margin-bottom: var(--space-16);">
              <h4 style="margin-bottom: var(--space-12);">Processing Steps</h4>
              <div style="display: flex; flex-direction: column; gap: var(--space-8);">
                ${combined.steps ? Object.entries(combined.steps).map(([stepKey, step]) => {
                  const stepNum = stepKey.replace('step', '');
                  const stepName = step.name || `Step ${stepNum}`;
                  const stepStatusIcon = step.status === 'completed'
                    ? '✓'
                    : step.status === 'completed_with_fallback'
                      ? '⚠️'
                      : step.status === 'skipped'
                        ? '⏭️'
                        : '✗';
                  const stepColor = step.status === 'completed'
                    ? 'var(--color-success)'
                    : step.status === 'completed_with_fallback'
                      ? 'var(--color-warning)'
                      : step.status === 'skipped'
                        ? 'var(--color-border)'
                        : 'var(--color-error)';
                  const stepDetail = step.status === 'skipped'
                    ? (step.reason || 'Skipped for current pipeline')
                    : `Duration: ${step.duration || 'N/A'}`;
                  return `
                    <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid ${stepColor};">
                      <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                          <strong>${stepName}</strong>
                          <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                            ${stepDetail}
                          </div>
                        </div>
                        <span style="color: ${stepColor}; font-weight: bold;">${stepStatusIcon}</span>
                      </div>
                    </div>
                  `;
                }).join('') : ''}
              </div>
            </div>
            <div style="display: flex; gap: var(--space-8);">
          <button class="action-btn" onclick="app.showCombinedDetails()">👁️ View Final Result</button>
          <button class="action-btn" onclick="app.showCombinedJSON()">🔎 View LLM JSON</button>
          <button class="action-btn" onclick="app.downloadCombinedJSON()">💾 Download JSON</button>
              ${combined.comparison && combined.comparison.status !== 'skipped' ? `<button class="action-btn" onclick="app.showComparisonAnalysis()">📊 View Comparison</button>` : ''}
            </div>
          </td>
        </tr>
      ` + Object.entries(this.testResults).filter(([serviceId]) => serviceId !== 'combined' && serviceId !== 'overlap').map(([serviceId, result]) => {
        if (result.success) {
          const isTextMode = serviceId === TEXT_SERVICE_KEY || result.rawData?.source === 'text';
          const processingTimeDisplay = isTextMode ? '—' : `${result.processingTime}s`;
          const costDisplay = isTextMode ? '—' : `$${result.cost}`;
          
          return `
            <tr>
              <td><strong>${result.serviceName}</strong>${isTextMode ? ' <span style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">📝 text mode</span>' : ''}</td>
              <td>${processingTimeDisplay}</td>
              <td>${result.speakerCount}</td>
              <td>${costDisplay}</td>
              <td><span class="status status-success">Success ✓</span></td>
              <td>
                <button class="action-btn" onclick="app.showDetails('${serviceId}')">👁️ Details</button>
                <button class="action-btn" onclick="app.downloadJSON('${serviceId}')">💾 JSON</button>
              </td>
            </tr>
          `;
        } else {
          const hasLogs = this.errorLogs[serviceId] ? 'true' : 'false';
          return `
            <tr>
              <td><strong>${result.serviceName}</strong></td>
              <td>—</td>
              <td>—</td>
              <td>—</td>
              <td><span class="status status-error">Error ✗</span></td>
              <td>
                <small style="color: var(--color-error);">${result.error}</small>
                ${hasLogs === 'true' ? `<button class="action-btn" onclick="app.showErrorLogs('${serviceId}')" style="margin-left: var(--space-8);">📋 Logs</button>` : ''}
              </td>
            </tr>
          `;
        }
      }).join('');
    } else {
      // Regular results display
      tbody.innerHTML = Object.entries(this.testResults).map(([serviceId, result]) => {
        if (result.success) {
          const isTextMode = serviceId === TEXT_SERVICE_KEY || result.rawData?.source === 'text';
          const processingTimeDisplay = isTextMode ? '—' : `${result.processingTime}s`;
          const costDisplay = isTextMode ? '—' : `$${result.cost}`;
          
          return `
            <tr>
              <td><strong>${result.serviceName}</strong>${isTextMode ? ' <span style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">📝 text mode</span>' : ''}</td>
              <td>${processingTimeDisplay}</td>
              <td>${result.speakerCount}</td>
              <td>${costDisplay}</td>
              <td><span class="status status-success">Success ✓</span></td>
              <td>
                <button class="action-btn" onclick="app.showDetails('${serviceId}')">👁️ Details</button>
                <button class="action-btn" onclick="app.downloadJSON('${serviceId}')">💾 JSON</button>
              </td>
            </tr>
          `;
        } else {
          const hasLogs = this.errorLogs[serviceId] ? 'true' : 'false';
          return `
            <tr>
              <td><strong>${result.serviceName}</strong></td>
              <td>—</td>
              <td>—</td>
              <td>—</td>
              <td><span class="status status-error">Error ✗</span></td>
              <td>
                <small style="color: var(--color-error);">${result.error}</small>
                ${hasLogs === 'true' ? `<button class="action-btn" onclick="app.showErrorLogs('${serviceId}')" style="margin-left: var(--space-8);">📋 Logs</button>` : ''}
              </td>
            </tr>
          `;
        }
      }).join('');
    }

    const audioshakeCard = document.getElementById('audioshakeResultsCard');
    console.log('🔍 Checking AudioShake results card:', {
      cardExists: !!audioshakeCard,
      hasAudioshakeResults: !!this.audioshakeResults,
      audioshakeResults: this.audioshakeResults
    });
    
    if (audioshakeCard) {
      // Перевіряємо наявність результатів - більш гнучка перевірка
      const hasSpeakers = Array.isArray(this.audioshakeResults?.speakers) && this.audioshakeResults.speakers.length > 0;
      const hasResults = Array.isArray(this.audioshakeResults?.results) && this.audioshakeResults.results.length > 0;
      const hasResultsData = hasSpeakers || hasResults;
      const successNotFalse = this.audioshakeResults?.success !== false;
      
      const finalHasResults = hasResultsData && (successNotFalse || this.audioshakeResults?.success === true);
      
      console.log('🔍 AudioShake results check:', {
        hasSpeakers,
        hasResultsArray: hasResults,
        successNotFalse,
        hasResultsData,
        finalHasResults,
        speakersCount: this.audioshakeResults?.speakers?.length || 0,
        resultsCount: this.audioshakeResults?.results?.length || 0
      });
      
      if (finalHasResults) {
        const tracks = Array.isArray(this.audioshakeResults.speakers)
          ? this.audioshakeResults.speakers
          : (Array.isArray(this.audioshakeResults.results) ? this.audioshakeResults.results : []);
        
        console.log('📋 Rendering AudioShake tracks:', {
          tracksCount: tracks.length,
          firstTrack: tracks[0] || null
        });
        const metrics = this.audioshakeResults.metrics || {};
        const processingSeconds = this.audioshakeResults.duration
          ?? metrics.processing_seconds
          ?? metrics.processingSeconds
          ?? 0;
        const jobId = this.audioshakeResults.taskId || this.audioshakeResults.jobId || 'n/a';
        const pipelineLabel = this.audioshakeResults.usedTasksPipeline ? 'AudioShake Tasks' : 'AudioShake Legacy';

        audioshakeCard.style.display = 'block';
        audioshakeCard.innerHTML = `
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-16);">
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Separated Tracks</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${tracks.length}</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Processing Time</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${processingSeconds}s</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Pipeline</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${pipelineLabel}</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Job ID</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold); overflow-wrap: anywhere;">${jobId}</div>
            </div>
          </div>
          <div style="overflow-x: auto;">
            <table class="results-table">
              <thead>
                <tr>
                  <th>Track</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Summary / Transcript</th>
                  <th>Download</th>
                </tr>
              </thead>
              <tbody>
                ${tracks.map(track => {
                  const roleLabel = track.role || track.speaker || track.name || 'Unknown';
                  const summaryText = track.summary
                    ? this.escapeHtml(track.summary)
                    : '—';
                  
                  // Отримуємо діаризовані сегменти з правильної структури
                  const diarizationSegments = track.diarization?.recordings?.flatMap(rec => {
                    // Спробувати різні можливі структури
                    return rec.results?.speechmatics?.segments 
                      || rec.results?.segments
                      || rec.segments
                      || [];
                  }) || [];
                  
                  // Формуємо діаризований текст
                  const diarizationText = diarizationSegments.length > 0
                    ? diarizationSegments.map(seg => {
                        const start = seg.start !== undefined ? this.formatTime(seg.start) : '';
                        const end = seg.end !== undefined ? this.formatTime(seg.end) : '';
                        const speaker = seg.speaker || seg.speaker_label || '';
                        const text = seg.text || seg.word || '';
                        return `[${start}-${end}] ${speaker}: ${text}`;
                      }).join('<br>')
                    : '';
                  
                  const transcript = track.transcript || '';
                  const transcriptPreview = transcript
                    ? `${this.escapeHtml(transcript.slice(0, 160))}${transcript.length > 160 ? '…' : ''}`
                    : '';
                  
                  // Комбінуємо транскрипт та діаризацію
                  const fullText = transcriptPreview || diarizationText || '—';
                  
                  // Кнопка скачування
                  const downloadUrl = track.downloadUrl || track.audioUrl;
                  const downloadLink = downloadUrl
                    ? `<a href="${downloadUrl}" download class="action-btn" style="display: inline-block; padding: var(--space-4) var(--space-8); text-decoration: none; background: var(--color-primary); color: white; border-radius: var(--radius-base); font-size: var(--font-size-sm);">⬇️ Download</a>`
                    : '—';
                  
                  const trackLabel = track.track_id || track.file_name || track.speaker || track.name || 'Track';
                  const statusBadge = track.error
                    ? `<span class="status status-error">Error</span>`
                    : (track.skipped ? `<span class="status status-warning">Skipped</span>` : `<span class="status status-success">OK</span>`);
                  const statusDetails = track.error || (track.skipped ? track.reason : (typeof track.confidence === 'number' ? `${Math.round(track.confidence * 100)}% confidence` : 'Processed'));
                  
                  return `
                    <tr>
                      <td><strong>${this.escapeHtml(trackLabel)}</strong></td>
                      <td>${this.escapeHtml(roleLabel)}</td>
                      <td>
                        ${statusBadge}
                        <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">${this.escapeHtml(statusDetails || '')}</div>
                      </td>
                      <td style="max-width: 400px; font-size: var(--font-size-sm);">
                        ${summaryText !== '—' ? `<div style="margin-bottom: var(--space-4); font-weight: var(--font-weight-semibold);">${summaryText}</div>` : ''}
                        <div style="max-height: 200px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word;">${fullText}</div>
                        ${diarizationSegments.length > 0 ? `<div style="margin-top: var(--space-4); font-size: var(--font-size-xs); color: var(--color-text-secondary);">${diarizationSegments.length} segments</div>` : ''}
                      </td>
                      <td>${downloadLink}</td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        `;
      } else {
        console.log('⚠️ AudioShake results card hidden - no valid results');
        audioshakeCard.style.display = 'none';
        audioshakeCard.innerHTML = '';
      }
    } else {
      console.warn('⚠️ AudioShake results card element not found in DOM');
    }

    this.updateTranslationControlsState();
    this.updateTranslationStatusLabel();
  },

  showCombinedDetails() {
    const combined = this.testResults.combined;
    if (!combined) return;

    const finalResult = combined.finalResult || combined.corrected;
    const segments = finalResult?.recordings?.[0]?.results?.combined?.segments || [];
    
    // Use existing showDetails function but with combined data
    const container = document.getElementById('detailsContainer');
    if (!container) {
      alert('Details container not found');
      return;
    }

    const detailHTML = `
      <div class="detail-view" id="detail_combined">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-16);">
          <h3>Combined Diarization - Final Result</h3>
          <button class="action-btn" onclick="document.getElementById('detail_combined').remove()">✕ Close</button>
        </div>

        <div style="margin-bottom: var(--space-24);">
          <h4>Processing Summary</h4>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16);">
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Total Duration</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${combined.totalDuration || 'N/A'}</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Segments</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${segments.length}</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Speakers</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${new Set(segments.map(s => s.speaker)).size}</div>
            </div>
          </div>
        </div>

        <h4 style="margin-bottom: var(--space-12);">Transcript with timestamps</h4>
        <div class="timeline">
          ${segments.map(segment => {
            const overlapClass = segment.overlap ? 'timeline-segment-overlap' : '';
            const overlapStyle = segment.overlap ? 'border-left: 3px solid var(--color-warning); background: rgba(255, 193, 7, 0.1);' : '';
            return `
            <div class="timeline-segment ${overlapClass}" style="${overlapStyle}">
              <div class="timeline-time">${this.formatTime(segment.start)} → ${this.formatTime(segment.end)}</div>
              <div class="timeline-speaker">
                ${segment.role ? (segment.role === 'operator' ? 'Agent' : 'Client') : segment.speaker}
                ${segment.role ? ` <span style="font-size: 0.85em; color: ${segment.role === 'operator' ? 'var(--color-success)' : 'var(--color-primary)'}; font-weight: normal;">(${segment.role === 'operator' ? '👨‍💼 Agent' : '👤 Client'})</span>` : ''}
                ${segment.overlap ? ` <span style="font-size: 0.8em; color: var(--color-warning); font-weight: bold;">⚠️ Overlap</span>` : ''}
              </div>
              <div class="timeline-text">${segment.text}</div>
            </div>
          `;
          }).join('')}
        </div>
      </div>
    `;

    container.innerHTML = detailHTML;
  },

  showCombinedJSON() {
    const combined = this.testResults.combined;
    if (!combined) return;

    const finalResult = combined.finalResult || combined.corrected;
    if (!finalResult) {
      alert('Combined JSON is not available for this run.');
      return;
    }

    const container = document.getElementById('detailsContainer');
    if (!container) {
      alert('Details container not found');
      return;
    }

    const jsonString = JSON.stringify(finalResult, null, 2);
    const detailHTML = `
      <div class="detail-view" id="detail_combined_json">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-16);">
          <h3>LLM JSON Output (Combined)</h3>
          <div style="display: flex; gap: var(--space-8);">
            <button class="action-btn" onclick="navigator.clipboard.writeText(document.querySelector('#detail_combined_json pre').innerText)">📋 Copy</button>
            <button class="action-btn" onclick="document.getElementById('detail_combined_json').remove()">✕ Close</button>
          </div>
        </div>
        <pre style="max-height: 60vh; overflow: auto; background: #111216; color: #c9f0ff; padding: var(--space-16); border-radius: var(--radius-base); font-size: 12px; border: 1px solid var(--color-border);">${this.escapeHtml(jsonString)}</pre>
      </div>
    `;

    container.innerHTML = detailHTML;
  },

  downloadCombinedJSON() {
    const combined = this.testResults.combined;
    if (!combined) return;

    const finalResult = combined.finalResult || combined.corrected;
    const dataStr = JSON.stringify(finalResult, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `combined-diarization-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  },

  showComparisonAnalysis() {
    const combined = this.testResults.combined;
    if (!combined || !combined.comparison || combined.comparison.status === 'skipped') {
      alert('Comparison analysis is not available for this pipeline run.');
      return;
    }

    const comparison = combined.comparison;
    const container = document.getElementById('detailsContainer');
    if (!container) {
      alert('Details container not found');
      return;
    }

    const analysis = comparison.comparison || {};
    const assessment = analysis.overallAssessment || {};
    const roleAnalysis = analysis.roleAnalysis || {};
    const guidance = analysis.correctionGuidance || {};

    const detailHTML = `
      <div class="detail-view" id="detail_comparison">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-16);">
          <h3>Comparison Analysis</h3>
          <button class="action-btn" onclick="document.getElementById('detail_comparison').remove()">✕ Close</button>
        </div>

        <div style="margin-bottom: var(--space-24);">
          <h4>Overall Assessment</h4>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-16);">
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Audio Method Score</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${assessment.audioMethodScore || 'N/A'}/100</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">LLM Method Score</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${assessment.llmMethodScore || 'N/A'}/100</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">Recommended Method</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${assessment.recommendedMethod || 'N/A'}</div>
            </div>
          </div>
          <div style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base);">
            <strong>Reasoning:</strong> ${assessment.reasoning || 'No reasoning provided'}
          </div>
        </div>

        ${roleAnalysis.recommendedMethodForRoles ? `
        <div style="margin-bottom: var(--space-24);">
          <h4>Role Identification Analysis</h4>
          <div style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base);">
            <div style="margin-bottom: var(--space-8);">
              <strong>Recommended Method for Roles:</strong> ${roleAnalysis.recommendedMethodForRoles}
            </div>
            <div>
              <strong>Reasoning:</strong> ${roleAnalysis.reasoning || 'No reasoning provided'}
            </div>
          </div>
        </div>
        ` : ''}

        ${guidance.mergeStrategy ? `
        <div style="margin-bottom: var(--space-24);">
          <h4>Correction Guidance</h4>
          <div style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base);">
            <div style="margin-bottom: var(--space-8);">
              <strong>Merge Strategy:</strong> ${guidance.mergeStrategy}
            </div>
            ${guidance.useAudioFor && guidance.useAudioFor.length > 0 ? `
            <div style="margin-bottom: var(--space-8);">
              <strong>Use Audio Method For:</strong> ${guidance.useAudioFor.join(', ')}
            </div>
            ` : ''}
            ${guidance.useLLMFor && guidance.useLLMFor.length > 0 ? `
            <div>
              <strong>Use LLM Method For:</strong> ${guidance.useLLMFor.join(', ')}
            </div>
            ` : ''}
          </div>
        </div>
        ` : ''}

        ${analysis.disagreements && analysis.disagreements.length > 0 ? `
        <div style="margin-bottom: var(--space-24);">
          <h4>Disagreements (${analysis.disagreements.length})</h4>
          <div style="max-height: 400px; overflow-y: auto;">
            ${analysis.disagreements.map((disagreement, idx) => `
              <div style="padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); margin-bottom: var(--space-8);">
                <div style="font-weight: var(--font-weight-medium); margin-bottom: var(--space-8);">Disagreement #${idx + 1}</div>
                <div style="font-size: var(--font-size-sm); margin-bottom: var(--space-4);">
                  <strong>Issue:</strong> ${disagreement.issue || 'Unknown'}
                </div>
                <div style="font-size: var(--font-size-sm); margin-bottom: var(--space-4);">
                  <strong>Recommended Resolution:</strong> ${disagreement.recommendedResolution || 'Unknown'}
                </div>
                <div style="font-size: var(--font-size-sm);">
                  <strong>Reasoning:</strong> ${disagreement.reasoning || 'No reasoning provided'}
                </div>
              </div>
            `).join('')}
          </div>
        </div>
        ` : ''}
      </div>
    `;

    container.innerHTML = detailHTML;
  },

  /**
   * Calculate comprehensive metrics for all services
   */
  calculateComprehensiveMetrics() {
    Object.entries(this.testResults).forEach(([serviceId, result]) => {
      if (result.success && this.groundTruth) {
        const metrics = new DiarizationMetrics(this.groundTruth.segments, result.segments);
        this.comprehensiveMetrics[serviceId] = metrics.calculateAllMetrics();
      }
    });
  },

  /**
   * Render executive summary with winner
   */
  renderExecutiveSummary() {
    const container = document.getElementById('executiveSummary');
    if (!container) return;

    const successfulResults = Object.entries(this.testResults).filter(([_, r]) => r.success);
    
    if (successfulResults.length === 0) {
      container.innerHTML = '<p>No successful results to display.</p>';
      return;
    }

    // Determine winner based on multiple criteria
    let bestService = null;
    let bestScore = -1;

    successfulResults.forEach(([serviceId, result]) => {
      let score = 0;
      
      // Score based on metrics if available
      if (this.comprehensiveMetrics[serviceId]) {
        const metrics = this.comprehensiveMetrics[serviceId];
        const derScore = Math.max(0, 100 - parseFloat(metrics.der.derPercent));
        const speakerScore = metrics.speakerCount.correct ? 100 : 70;
        score = (derScore * 0.7) + (speakerScore * 0.3);
      } else {
        // Score based on speed and speaker count
        score = (result.speedFactor || 1) * 10;
      }

      if (score > bestScore) {
        bestScore = score;
        bestService = { id: serviceId, ...result };
      }
    });

    const bestMetrics = this.comprehensiveMetrics[bestService.id];
    const bestDER = bestMetrics ? bestMetrics.der.derPercent + '%' : 'N/A';
    const bestSpeed = bestService.speedFactor ? bestService.speedFactor + 'x' : 'N/A';

    container.innerHTML = `
      <div class="winner-banner">
        <h2>🏆 Best Performance: ${bestService.serviceName}</h2>
        <div class="winner-stats">
          <div class="winner-stat">
            <div class="winner-stat-value">${bestDER}</div>
            <div class="winner-stat-label">Diarization Error Rate</div>
          </div>
          <div class="winner-stat">
            <div class="winner-stat-value">${bestSpeed}</div>
            <div class="winner-stat-label">Processing Speed</div>
          </div>
          <div class="winner-stat">
            <div class="winner-stat-value">${bestService.speakerCount}</div>
            <div class="winner-stat-label">Speakers Detected</div>
          </div>
          <div class="winner-stat">
            <div class="winner-stat-value">$${bestService.cost}</div>
            <div class="winner-stat-label">Cost</div>
          </div>
        </div>
      </div>
    `;
  },

  /**
   * Render Overlap Diarization Results
   */
  renderOverlapDiarizationResults(overlapResult, tbody) {
    const primaryDiarization = overlapResult.primaryDiarization;
    const separation = overlapResult.separation;
    const voiceTracks = overlapResult.voiceTracks || [];
    const correctedDiarization = overlapResult.correctedDiarization;
    const steps = overlapResult.steps || {};
    const diagnostics = overlapResult.diagnostics || {};
    const voiceTranscript = diagnostics.voiceTranscript;
    const missingReplicas = diagnostics.missingReplicas || [];
    
    // Extract segments from corrected diarization
    const recording = correctedDiarization?.recordings?.[0];
    const correctedSegments = recording?.results?.['overlap-corrected']?.segments || [];
    const overlapSegmentCount = correctedSegments.filter(segment => segment.overlap).length;
    const speakerCount = new Set(correctedSegments.map(s => s.speaker)).size;
    
    // Extract segments from primary diarization for comparison
    const primaryRecording = primaryDiarization?.recordings?.[0];
    const primarySegments = primaryRecording?.results?.speechmatics?.segments || [];
    
    const row = document.createElement('tr');
    row.style.background = 'var(--color-bg-2)';
    row.innerHTML = `
      <td colspan="6" style="padding: var(--space-16);">
        <h3 style="margin-bottom: var(--space-16);">Overlap Diarization Results</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-16);">
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Total Duration</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${overlapResult.totalDuration || 'N/A'}</div>
          </div>
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Detected Speakers</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${speakerCount}</div>
          </div>
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Corrected Segments</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${correctedSegments.length}</div>
          </div>
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Voice Tracks</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${voiceTracks.filter(vt => !vt.error).length}</div>
          </div>
        </div>
        ${voiceTranscript?.downloadUrl ? `
          <div style="margin-bottom: var(--space-16); display: flex; align-items: center; gap: var(--space-12);">
            <a class="action-btn" href="${voiceTranscript.downloadUrl}" download target="_blank">💾 Download Voice Transcript</a>
            <span style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">
              ${(voiceTranscript.size / 1024).toFixed(1)} KB • created ${new Date(voiceTranscript.createdAt).toLocaleString()}
            </span>
          </div>
        ` : ''}
        
        <div style="margin-bottom: var(--space-16);">
          <h4 style="margin-bottom: var(--space-12);">Processing Steps</h4>
          <div style="display: flex; flex-direction: column; gap: var(--space-8);">
            ${steps.step1 ? `
              <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-success);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <strong>${steps.step1.name}</strong>
                    <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                      Duration: ${steps.step1.duration || 'N/A'} • ${steps.step1.segmentsCount || 0} segments
                    </div>
                  </div>
                  <span style="color: var(--color-success); font-weight: bold;">✓</span>
                </div>
              </div>
            ` : ''}
            ${steps.step2 ? `
              <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-success);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <strong>${steps.step2.name}</strong>
                    <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                      Duration: ${steps.step2.duration || 'N/A'} • ${steps.step2.speakersCount || 0} speakers
                    </div>
                  </div>
                  <span style="color: var(--color-success); font-weight: bold;">✓</span>
                </div>
              </div>
            ` : ''}
            ${steps.step3 ? `
              <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-success);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <strong>${steps.step3.name}</strong>
                    <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                      Duration: ${steps.step3.duration || 'N/A'} • ${steps.step3.processedTracks || 0}/${steps.step3.totalTracks || 0} tracks processed
                    </div>
                  </div>
                  <span style="color: var(--color-success); font-weight: bold;">✓</span>
                </div>
              </div>
            ` : ''}
            ${steps.step5 ? `
              <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid ${steps.step5.status === 'failed' ? 'var(--color-error)' : 'var(--color-success)'};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <strong>${steps.step5.name}</strong>
                    <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                      Duration: ${steps.step5.duration || 'N/A'} • ${steps.step5.segmentsCount || 0} segments
                      ${steps.step5.error ? `<div style="color: var(--color-error); margin-top: var(--space-4);">Error: ${steps.step5.error}</div>` : ''}
                    </div>
                  </div>
                  <span style="color: ${steps.step5.status === 'failed' ? 'var(--color-error)' : 'var(--color-success)'}; font-weight: bold;">${steps.step5.status === 'failed' ? '✗' : '✓'}</span>
                </div>
              </div>
            ` : ''}
          </div>
        </div>

        <div style="margin-bottom: var(--space-16);">
          <h4 style="margin-bottom: var(--space-8);">Overlap & Missing Replica Summary</h4>
          <div style="display: flex; flex-direction: column; gap: var(--space-8);">
            <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-primary);">
              <strong>Overlap segments flagged:</strong> ${overlapSegmentCount}
            </div>
            ${missingReplicas.length ? `
              <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-warning);">
                <strong>Missing replicas detected: ${missingReplicas.length}</strong>
                <div style="margin-top: var(--space-8); display: flex; flex-direction: column; gap: var(--space-8);">
                  ${missingReplicas.slice(0, 8).map(rep => `
                    <div style="padding: var(--space-8); background: var(--color-bg-2); border-radius: var(--radius-base);">
                      <div style="font-weight: var(--font-weight-medium);">
                        ${this.escapeHtml(rep.speaker || 'UNKNOWN')}
                        ${rep.role ? `<span style="margin-left: var(--space-8); color: var(--color-text-secondary);">${this.escapeHtml(rep.role)}</span>` : ''}
                      </div>
                      <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-top: var(--space-4);">
                        ${rep.start !== undefined ? `${rep.start.toFixed(2)}s → ${rep.end.toFixed(2)}s` : ''}
                      </div>
                      <div style="margin-top: var(--space-4);">${this.escapeHtml(rep.text || '')}</div>
                    </div>
                  `).join('')}
                  ${missingReplicas.length > 8 ? `<div style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">+${missingReplicas.length - 8} more…</div>` : ''}
                </div>
              </div>
            ` : `
              <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-success);">
                <strong>No missing replicas detected.</strong>
              </div>
            `}
          </div>
        </div>
        
        ${separation && separation.speakers && separation.speakers.length > 0 ? `
        <div style="margin-bottom: var(--space-16);">
          <h4 style="margin-bottom: var(--space-12);">🎤 Розділені аудіо доріжки</h4>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: var(--space-16);">
            ${separation.speakers.map((speaker, idx) => {
              const speakerUrl = speaker.downloadUrl || speaker.url;
              const speakerName = speaker.name || `Speaker ${idx + 1}`;
              const speakerNum = speakerName.replace('SPEAKER_', '');
              const speakerColor = speakerNum === '00' ? 'var(--color-primary)' : 'var(--color-warning)';
              
              if (!speakerUrl) {
                return `
                  <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid ${speakerColor};">
                    <div style="font-weight: bold; margin-bottom: var(--space-8); color: ${speakerColor};">
                      ${speakerName}
                    </div>
                    <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">
                      URL не знайдено
                    </div>
                  </div>
                `;
              }
              
              // Конвертуємо URL якщо потрібно
              let audioUrl = speakerUrl;
              
              // Якщо це localtunnel/ngrok URL, конвертуємо в localhost
              if (speakerUrl.includes('loca.lt') || speakerUrl.includes('ngrok') || speakerUrl.includes('tunnel')) {
                try {
                  const url = new URL(speakerUrl);
                  // Використовуємо localhost:3000 замість window.location.origin для uploads
                  audioUrl = `http://localhost:3000${url.pathname}`;
                } catch (e) {
                  const pathMatch = speakerUrl.match(/\/uploads\/[^\/]+$/);
                  if (pathMatch) {
                    audioUrl = `http://localhost:3000${pathMatch[0]}`;
                  }
                }
              }
              // Якщо це відносний шлях, додаємо origin
              else if (speakerUrl.startsWith('/uploads/') || speakerUrl.startsWith('/')) {
                // Використовуємо localhost:3000 для uploads
                if (speakerUrl.startsWith('/uploads/')) {
                  audioUrl = `http://localhost:3000${speakerUrl}`;
                } else {
                  audioUrl = `${window.location.origin}${speakerUrl}`;
                }
              }
              // Якщо це вже повний URL, залишаємо як є
              else if (speakerUrl.startsWith('http://') || speakerUrl.startsWith('https://')) {
                audioUrl = speakerUrl;
              }
              
              return `
                <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid ${speakerColor};">
                  <div style="font-weight: bold; margin-bottom: var(--space-8); color: ${speakerColor};">
                    ${speakerName}
                  </div>
                  <audio controls style="width: 100%; margin-top: var(--space-8);">
                    <source src="${audioUrl}" type="audio/${speaker.format || 'wav'}">
                  </audio>
                  ${speaker.downloadUrl ? `
                    <a href="${audioUrl}" download style="display: inline-block; margin-top: var(--space-8); font-size: var(--font-size-sm); color: var(--color-primary); text-decoration: none;">
                      ⬇️ Завантажити
                    </a>
                  ` : ''}
                </div>
              `;
            }).join('')}
          </div>
        </div>
        ` : ''}
        
        <div style="margin-bottom: var(--space-16);">
          <h4 style="margin-bottom: var(--space-12);">Voice Tracks Analysis</h4>
          <div style="display: flex; flex-direction: column; gap: var(--space-8);">
            ${voiceTracks.map((vt, idx) => {
              if (vt.error) {
                return `
                  <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-error);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                      <div>
                        <strong>${vt.speaker || `Track ${idx + 1}`}</strong>
                        <div style="font-size: var(--font-size-sm); color: var(--color-error); margin-top: var(--space-4);">
                          Error: ${vt.error}
                        </div>
                      </div>
                      <span style="color: var(--color-error); font-weight: bold;">✗</span>
                    </div>
                  </div>
                `;
              }
              const role = vt.roleAnalysis?.role || 'Unknown';
              const confidence = vt.roleAnalysis?.confidence || 0;
              const roleColor = role === 'operator' ? 'var(--color-primary)' : 'var(--color-warning)';
              return `
                <div style="padding: var(--space-12); background: var(--color-bg-1); border-radius: var(--radius-base); border-left: 4px solid var(--color-success);">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                      <strong>${vt.speaker || `Track ${idx + 1}`}</strong>
                      <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                        <div>Role: <span style="color: ${roleColor}; font-weight: bold;">${role}</span> (confidence: ${(confidence * 100).toFixed(0)}%)</div>
                        <div style="margin-top: var(--space-4);">Transcript: ${vt.transcriptText ? (vt.transcriptText.substring(0, 100) + (vt.transcriptText.length > 100 ? '...' : '')) : 'N/A'}</div>
                      </div>
                    </div>
                    <span style="color: var(--color-success); font-weight: bold;">✓</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
        
        <div style="display: flex; gap: var(--space-8); flex-wrap: wrap;">
          <button class="action-btn" onclick="app.showOverlapDetails()">👁️ View Corrected Result</button>
          <button class="action-btn" onclick="app.downloadOverlapJSON()">💾 Download JSON</button>
          <button class="action-btn" onclick="app.showOverlapPrimaryComparison()">📊 Compare Primary vs Corrected</button>
        </div>
      </td>
    `;
    tbody.appendChild(row);
  },

  showOverlapDetails() {
    const overlapResult = this.testResults.overlap;
    if (!overlapResult) return;
    
    const correctedDiarization = overlapResult.correctedDiarization;
    const recording = correctedDiarization?.recordings?.[0];
    const segments = recording?.results?.['overlap-corrected']?.segments || [];
    
    // Create a temporary recording to display
    const tempRecording = {
      id: 'overlap-corrected',
      name: 'Overlap Corrected Diarization',
      results: {
        'overlap-corrected': {
          segments: segments,
          success: true,
          serviceName: 'Overlap Corrected'
        }
      }
    };
    
    // Add to recordings if not exists
    const existingIndex = this.recordings.findIndex(r => r.id === 'overlap-corrected');
    if (existingIndex >= 0) {
      this.recordings[existingIndex] = tempRecording;
    } else {
      this.recordings.push(tempRecording);
    }
    
    this.setActiveRecording('overlap-corrected');
    this.showResults();
  },

  downloadOverlapJSON() {
    const overlapResult = this.testResults.overlap;
    if (!overlapResult) return;
    
    const dataStr = JSON.stringify(overlapResult, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `overlap-diarization-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  },

  showOverlapPrimaryComparison() {
    const overlapResult = this.testResults.overlap;
    if (!overlapResult) return;
    
    const primaryDiarization = overlapResult.primaryDiarization;
    const correctedDiarization = overlapResult.correctedDiarization;
    
    // Show both in results
    const primaryRecording = primaryDiarization?.recordings?.[0];
    const correctedRecording = correctedDiarization?.recordings?.[0];
    
    const primarySegments = primaryRecording?.results?.speechmatics?.segments || [];
    const correctedSegments = correctedRecording?.results?.['overlap-corrected']?.segments || [];
    
    // Create temporary recordings for comparison
    const primaryTemp = {
      id: 'overlap-primary',
      name: 'Initial Diarization',
      results: {
        speechmatics: {
          segments: primarySegments,
          success: true,
          serviceName: 'Initial Analysis'
        }
      }
    };
    
    const correctedTemp = {
      id: 'overlap-corrected',
      name: 'Overlap Corrected Diarization',
      results: {
        'overlap-corrected': {
          segments: correctedSegments,
          success: true,
          serviceName: 'Overlap Corrected'
        }
      }
    };
    
    // Add to recordings
    const primaryIndex = this.recordings.findIndex(r => r.id === 'overlap-primary');
    if (primaryIndex >= 0) {
      this.recordings[primaryIndex] = primaryTemp;
    } else {
      this.recordings.push(primaryTemp);
    }
    
    const correctedIndex = this.recordings.findIndex(r => r.id === 'overlap-corrected');
    if (correctedIndex >= 0) {
      this.recordings[correctedIndex] = correctedTemp;
    } else {
      this.recordings.push(correctedTemp);
    }
    
    this.setActiveRecording('overlap-corrected');
    this.showResults();
  },

  /**
   * Switch between result tabs
   */
  switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    this.currentTab = tabName;

    // Render tab-specific content
    if (tabName === 'detailed') {
      this.renderDetailedMetrics();
    } else if (tabName === 'errors') {
      this.renderErrorAnalysis();
    } else if (tabName === 'comparison') {
      this.renderCrossServiceComparison();
    } else if (tabName === 'timeline') {
      this.renderTimelineVisualization();
    }
  },

  handleCombinedModeChange() {
    // Combined mode always uses 'hybrid' - no mode selection needed
    // This function is kept for compatibility but does nothing now
    const textModelBlock = document.getElementById('combinedTextModelBlock');
    if (textModelBlock) {
      textModelBlock.style.display = 'block';
    }
  },

  handleLLMModelChange() {
    const selectedModel = document.querySelector('input[name="diarizationModel"]:checked')?.value || 'fast-tier';
    const isMultimodal = selectedModel === 'multimodal';
    const multimodalAudioBlock = document.getElementById('llmMultimodalAudioBlock');
    
    if (multimodalAudioBlock) {
      if (isMultimodal) {
        multimodalAudioBlock.style.display = 'block';
      } else {
        multimodalAudioBlock.style.display = 'none';
        this.llmMultimodalAudioFile = null;
        const audioInfo = document.getElementById('llmMultimodalAudioInfo');
        if (audioInfo) {
          audioInfo.style.display = 'none';
        }
      }
    }
  },

  handleLLMMultimodalAudioUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    this.llmMultimodalAudioFile = file;
    const audioInfo = document.getElementById('llmMultimodalAudioInfo');
    if (audioInfo) {
      const fileSize = (file.size / 1024 / 1024).toFixed(2);
      audioInfo.innerHTML = `
        <div style="display: flex; align-items: center; gap: var(--space-8);">
          <span>✅</span>
          <span><strong>${file.name}</strong> (${fileSize} MB)</span>
          <button onclick="app.clearLLMMultimodalAudio()" style="margin-left: auto; padding: var(--space-4) var(--space-8); background: var(--color-error); color: white; border: none; border-radius: var(--radius-base); cursor: pointer; font-size: var(--font-size-sm);">Remove</button>
        </div>
      `;
      audioInfo.style.display = 'block';
    }
    
    // Додаємо файл до черги записів, якщо його там немає
    const existingRecording = this.recordings.find(r => r.file === file || r.file?.name === file.name);
    if (!existingRecording) {
      this.recordings.push({
        id: Date.now().toString(),
        file: file,
        name: file.name,
        size: file.size
      });
      this.renderRecordingsQueue();
    }
  },

  clearLLMMultimodalAudio() {
    this.llmMultimodalAudioFile = null;
    const audioInput = document.getElementById('llmMultimodalAudioInput');
    if (audioInput) {
      audioInput.value = '';
    }
    const audioInfo = document.getElementById('llmMultimodalAudioInfo');
    if (audioInfo) {
      audioInfo.style.display = 'none';
    }
  },

  handleAudioModeChange() {
    const mode = document.querySelector('input[name="audioPipelineMode"]:checked')?.value || 'speechmatics';
    const tunnelStatusBlock = document.getElementById('tunnelStatusBlock');
    
    if (tunnelStatusBlock) {
      if (mode === 'audioshake') {
        tunnelStatusBlock.style.display = 'block';
        this.updateTunnelStatusDisplay();
        // Оновлюємо статус кожні 5 секунд
        if (this.tunnelStatusInterval) {
          clearInterval(this.tunnelStatusInterval);
        }
        this.tunnelStatusInterval = setInterval(() => {
          this.updateTunnelStatusDisplay();
        }, 5000);
      } else {
        tunnelStatusBlock.style.display = 'none';
        if (this.tunnelStatusInterval) {
          clearInterval(this.tunnelStatusInterval);
          this.tunnelStatusInterval = null;
        }
      }
    }
  },

  handleOverlapPipelineModeChange() {
    const modeSelect = document.getElementById('overlapPipelineMode');
    const modeDetails = {
      mode1: document.getElementById('overlapMode1Details'),
      mode2: document.getElementById('overlapMode2Details'),
      mode3: document.getElementById('overlapMode3Details')
    };

    Object.entries(modeDetails).forEach(([mode, element]) => {
      if (!element) return;
      element.style.display = modeSelect?.value === mode ? 'block' : 'none';
    });
  },

  async openSpeechbrainSamples() {
    this.showView('speechbrainView');
    await this.loadSpeechbrainSamples();
  },

  async reloadSpeechbrainSamples() {
    await this.loadSpeechbrainSamples(true);
  },

  async loadSpeechbrainSamples(forceReload = false) {
    if (!this.speechbrainSamples) {
      this.speechbrainSamples = { data: [], lastLoaded: null, loading: false, error: null };
    }

    if (this.speechbrainSamples.loading) {
      return;
    }

    const now = Date.now();
    if (!forceReload && this.speechbrainSamples.data.length > 0 && now - (this.speechbrainSamples.lastLoaded || 0) < 30000) {
      this.renderSpeechbrainSamples();
      return;
    }

    const loadingEl = document.getElementById('speechbrainSamplesLoading');
    const errorEl = document.getElementById('speechbrainSamplesError');
    const container = document.getElementById('speechbrainSamplesContainer');

    if (loadingEl) loadingEl.style.display = 'block';
    if (errorEl) {
      errorEl.style.display = 'none';
      errorEl.textContent = '';
    }
    if (container) container.style.display = 'none';

    this.speechbrainSamples.loading = true;
    try {
      const data = await this.fetchSpeechbrainSamples();
      this.speechbrainSamples.data = data;
      this.speechbrainSamples.lastLoaded = Date.now();
      this.speechbrainSamples.error = null;
      this.renderSpeechbrainSamples();
    } catch (error) {
      console.error('SpeechBrain samples load error:', error);
      this.speechbrainSamples.error = error.message;
      if (errorEl) {
        errorEl.style.display = 'block';
        errorEl.textContent = `Failed to load samples: ${error.message}`;
      }
    } finally {
      this.speechbrainSamples.loading = false;
      if (loadingEl) loadingEl.style.display = 'none';
    }
  },

  async fetchSpeechbrainSamples() {
    const endpoints = [
      '/speechbrain-dashboard/data',
      '/speechbrain_test_results.json',
      '/Debug/speechbrain_test_results.json'
    ];

    let lastError = null;
    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, { headers: { 'Accept': 'application/json' } });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status} for ${endpoint}`);
        }
        const data = await response.json();
        if (Array.isArray(data)) {
          return data;
        }
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error('SpeechBrain samples are unavailable');
  },

  renderSpeechbrainSamples() {
    const container = document.getElementById('speechbrainSamplesContainer');
    const loadingEl = document.getElementById('speechbrainSamplesLoading');
    if (!container) return;

    const samples = this.speechbrainSamples?.data || [];
    if (loadingEl) {
      loadingEl.style.display = this.speechbrainSamples?.loading ? 'block' : 'none';
    }

    if (samples.length === 0) {
      container.innerHTML = `
        <div class="alert alert-warning">
          SpeechBrain demo samples are not available yet. Run Mode 3 once to populate this dashboard.
        </div>
      `;
      container.style.display = 'block';
      return;
    }

    container.innerHTML = samples.map(sample => this.buildSpeechbrainSampleCard(sample)).join('');
    container.style.display = 'block';
  },

  buildSpeechbrainSampleCard(sample) {
    const speakers = sample?.separation?.speakers || [];
    const timeline = sample?.separation?.timeline || [];
    const transcripts = this.extractSpeechbrainSegments(sample);
    const sourceAudioUrl = this.normalizeSpeechbrainUrl(sample?.sourceAudio);
    const badgeStyle = 'background: var(--color-bg-1); border-radius: 999px; padding: 4px 12px; font-size: var(--font-size-sm); color: var(--color-text-secondary); border: 1px solid var(--color-border);';

    const speakerCards = speakers.length
      ? speakers.map((speaker, idx) => {
          const url = this.normalizeSpeechbrainUrl(speaker.downloadUrl || speaker.url);
          const label = speaker.name || `SPEAKER_${idx.toString().padStart(2, '0')}`;
          const range = this.formatSpeechbrainRange(timeline.find(entry => entry.speaker === speaker.name));
          return `
            <div class="card" style="background: var(--color-bg-1); padding: var(--space-16); border-left: 4px solid ${idx % 2 === 0 ? 'var(--color-primary)' : 'var(--color-warning)'};">
              <div style="font-weight: var(--font-weight-semibold); margin-bottom: var(--space-8); display:flex; justify-content:space-between; gap:var(--space-8); flex-wrap:wrap;">
                <span>${this.escapeHtml(label)}</span>
                <span style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">${range}</span>
              </div>
              ${url ? `<audio controls style="width: 100%;" src="${url}"></audio>` : '<div class="alert alert-warning">URL not available</div>'}
            </div>
          `;
        }).join('')
      : '<div class="alert alert-warning">No separated stems attached to this sample.</div>';

    const transcriptSection = transcripts.length
      ? transcripts.map(seg => `
          <div style="padding: var(--space-12); border-left: 4px solid var(--color-primary); background: var(--color-bg-2); border-radius: var(--radius-base); margin-bottom: var(--space-8);">
            <div style="font-weight: var(--font-weight-semibold); display:flex; justify-content:space-between; gap:var(--space-8); flex-wrap:wrap;">
              <span>${this.escapeHtml(seg.speaker || 'Unknown')}</span>
              <span style="color: var(--color-text-secondary); font-size: var(--font-size-sm);">${this.formatTime(seg.start || 0)} – ${this.formatTime(seg.end || 0)}</span>
            </div>
            <div style="margin-top: var(--space-8); color: var(--color-text); line-height: 1.5;">
              ${this.escapeHtml(seg.text || seg.originalText || '')}
            </div>
          </div>
        `).join('')
      : '<div class="alert alert-info">Transcripts are not attached to this sample.</div>';

    return `
      <div class="card" style="margin-bottom: var(--space-24); background: var(--color-bg-3);">
        <div style="display:flex; flex-wrap:wrap; gap:var(--space-12); justify-content:space-between; align-items:center; margin-bottom: var(--space-16);">
          <div>
            <h3 style="margin:0; color: var(--color-primary);">${this.escapeHtml(sample.name || 'SpeechBrain sample')}</h3>
            <p style="color: var(--color-text-secondary); margin-top: var(--space-4);">${this.escapeHtml(sample.pipelineMode || 'mode3').toUpperCase()}</p>
          </div>
          <div style="display:flex; gap: var(--space-8); flex-wrap:wrap;">
            <span style="${badgeStyle}">${sample.success ? '✅ Successful' : '❌ Failed'}</span>
            <span style="${badgeStyle}">Speakers: ${speakers.length}</span>
          </div>
        </div>

        ${sourceAudioUrl ? `
          <div style="margin-bottom: var(--space-16);">
            <div style="font-weight: var(--font-weight-semibold); margin-bottom: var(--space-8);">🎵 Original audio</div>
            <audio controls style="width: 100%;" src="${sourceAudioUrl}"></audio>
          </div>
        ` : ''}

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: var(--space-12);">
          ${speakerCards}
        </div>

        <div style="margin-top: var(--space-24);">
          <h4 style="margin-bottom: var(--space-12);">📝 Transcripts</h4>
          ${transcriptSection}
        </div>
      </div>
    `;
  },

  extractSpeechbrainSegments(sample) {
    const recordings = sample?.correctedDiarization?.recordings
      || sample?.primaryDiarization?.recordings
      || sample?.recordings
      || [];
    const segments = [];
    recordings.forEach(recording => {
      const results = recording.results || {};
      Object.values(results).forEach(service => {
        if (Array.isArray(service?.segments)) {
          segments.push(...service.segments);
        }
      });
    });
    return segments;
  },

  normalizeSpeechbrainUrl(path) {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) {
      return path;
    }
    if (path.startsWith('/')) {
      return path;
    }
    return `/${path.replace(/^\.?\/+/, '')}`;
  },

  formatSpeechbrainRange(entry) {
    if (!entry) return '';
    const start = this.formatTime(entry.start || 0);
    const end = this.formatTime(entry.end || entry.duration || 0);
    return `${start} – ${end}`;
  },

  async updateTunnelStatusDisplay() {
    const statusBlock = document.getElementById('tunnelStatusBlock');
    const statusIcon = document.getElementById('tunnelStatusIcon');
    const statusText = document.getElementById('tunnelStatusText');
    const urlBlock = document.getElementById('tunnelUrlText');
    const urlValue = document.getElementById('tunnelUrlValue');
    const startButtonContainer = document.getElementById('tunnelStartButtonContainer');
    const startButton = document.getElementById('tunnelStartButton');
    
    if (!statusBlock || !statusIcon || !statusText) return;
    
    try {
      const status = await this.checkTunnelStatus(true); // Force refresh
      
      if (status.active && status.url) {
        // Тунель активний
        statusIcon.textContent = '✅';
        statusIcon.style.color = 'var(--color-success)';
        statusText.textContent = 'Tunnel is active and ready';
        statusText.style.color = 'var(--color-success)';
        
        if (urlBlock && urlValue) {
          urlBlock.style.display = 'block';
          urlValue.textContent = status.url;
          urlValue.style.color = 'var(--color-primary)';
          urlValue.style.textDecoration = 'underline';
          urlValue.style.cursor = 'pointer';
          urlValue.onclick = () => {
            window.open(status.url, '_blank');
          };
        }
        
        // Приховати кнопку запуску
        if (startButtonContainer) startButtonContainer.style.display = 'none';
      } else {
        // Тунель не активний
        statusIcon.textContent = '⚠️';
        statusIcon.style.color = 'var(--color-warning)';
        statusText.textContent = status.message || status.error || 'Tunnel is not ready.';
        statusText.style.color = 'var(--color-warning)';
        
        if (urlBlock) {
          urlBlock.style.display = 'none';
        }
        
        // Показати кнопку запуску
        if (startButtonContainer) startButtonContainer.style.display = 'block';
        if (startButton) {
          startButton.disabled = false;
          startButton.textContent = '🚀 Start Tunnel';
        }
      }
    } catch (error) {
      statusIcon.textContent = '❌';
      statusIcon.style.color = 'var(--color-error)';
      statusText.textContent = `Failed to check tunnel status: ${error.message}`;
      statusText.style.color = 'var(--color-error)';
      
      if (urlBlock) {
        urlBlock.style.display = 'none';
      }
      
      // Показати кнопку запуску при помилці
      if (startButtonContainer) startButtonContainer.style.display = 'block';
      if (startButton) {
        startButton.disabled = false;
        startButton.textContent = '🚀 Start Tunnel';
      }
    }
  },

  async startTunnel() {
    const startButton = document.getElementById('tunnelStartButton');
    const statusText = document.getElementById('tunnelStatusText');
    const statusIcon = document.getElementById('tunnelStatusIcon');

    if (startButton) {
      startButton.disabled = true;
      startButton.textContent = '⏳ Starting...';
    }
    if (statusIcon) {
      statusIcon.textContent = '🔄';
      statusIcon.style.color = 'var(--color-primary)';
    }
    if (statusText) {
      statusText.textContent = 'Starting tunnel...';
      statusText.style.color = 'var(--color-text-secondary)';
    }

    try {
      const response = await fetch('/api/tunnel-start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText.substring(0, 100)}`);
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 200)}`);
      }

      const data = await response.json();

      if (data.success && data.active) {
        if (statusIcon) {
          statusIcon.textContent = '✅';
          statusIcon.style.color = 'var(--color-success)';
        }
        if (statusText) {
          statusText.textContent = 'Tunnel started successfully!';
          statusText.style.color = 'var(--color-success)';
        }
        // Оновити відображення через 1 секунду
        setTimeout(() => this.updateTunnelStatusDisplay(), 1000);
      } else {
        if (statusIcon) {
          statusIcon.textContent = '❌';
          statusIcon.style.color = 'var(--color-error)';
        }
        if (statusText) {
          statusText.textContent = data.message || 'Failed to start tunnel';
          statusText.style.color = 'var(--color-error)';
        }
        if (startButton) {
          startButton.disabled = false;
          startButton.textContent = '🚀 Start Tunnel';
        }
      }
    } catch (error) {
      console.error('Error starting tunnel:', error);
      if (statusIcon) {
        statusIcon.textContent = '❌';
        statusIcon.style.color = 'var(--color-error)';
      }
      if (statusText) {
        statusText.textContent = `Error: ${error.message}`;
        statusText.style.color = 'var(--color-error)';
      }
      if (startButton) {
        startButton.disabled = false;
        startButton.textContent = '🚀 Start Tunnel';
      }
    }
  },

  /**
   * Switch between diarization methods (LLM, Audio, Combined)
   */
  switchDiarizationMethod(method) {
    // Update current method
    this.currentDiarizationMethod = method;

    // Update tab buttons in diarization tabs
    const tabsNav = document.getElementById('diarizationTabsNav');
    if (tabsNav) {
      tabsNav.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-method') === method) {
          btn.classList.add('active');
        }
      });
    }

    // Update tab content
    document.querySelectorAll('#llmTabContent, #audioTabContent, #combinedTabContent, #overlapTabContent').forEach(content => {
      content.classList.remove('active');
    });

    const activeTabContent = document.getElementById(`${method}TabContent`);
    if (activeTabContent) {
      activeTabContent.classList.add('active');
    }
    
    // Якщо переключилися на вкладку Audio, перевіряємо режим обробки
    if (method === 'audio') {
      this.handleAudioModeChange();
    }

    // Show/hide audio upload area based on method
    const uploadArea = document.getElementById('uploadArea');
    const fileInfoContainer = document.getElementById('fileInfoContainer');
    const recordingsQueue = document.getElementById('recordingsQueue');
    
    if (uploadArea) {
      if (method === 'llm') {
        // Hide upload area for LLM method (text-only)
        uploadArea.style.display = 'none';
        if (fileInfoContainer) {
          fileInfoContainer.style.display = 'none';
        }
        // Hide recordings queue for LLM method (text-only, no audio files needed)
        if (recordingsQueue) {
          recordingsQueue.style.display = 'none';
        }
      } else {
        // Show upload area for Audio, Combined, and Overlap methods
        uploadArea.style.display = 'flex';
        // Show recordings queue for Audio and Combined methods (if there are recordings)
        if (recordingsQueue && this.recordings.length > 0) {
          recordingsQueue.style.display = 'block';
        }
        if (fileInfoContainer) {
          fileInfoContainer.style.display = method === 'combined' ? 'none' : 'block';
        }
      }
    }


    // Sync settings between tabs (removed main language/speakerCount sync since fields were removed)
    // Each tab now has its own independent settings
  },

  /**
   * Render detailed metrics for all services
   */
  renderDetailedMetrics() {
    const container = document.getElementById('detailedMetricsContainer');
    if (!container) return;

    let html = '';

    Object.entries(this.testResults).forEach(([serviceId, result]) => {
      if (!result.success) return;

      const metrics = this.comprehensiveMetrics[serviceId];
      
      html += `
        <div class="card" style="margin-bottom: var(--space-24); background: var(--color-bg-1);">
          <h3 style="color: var(--color-primary); margin-bottom: var(--space-16);">
            ${result.serviceName} - Comprehensive Metrics
          </h3>
      `;

      if (metrics) {
        html += `
          <h4>Primary Metrics</h4>
          <div class="metrics-grid" style="margin-bottom: var(--space-24);">
            <div class="metric-card">
              <div class="metric-label">🎯 Diarization Error Rate (DER)</div>
              <div class="metric-value ${parseFloat(metrics.der.derPercent) < 10 ? 'good' : parseFloat(metrics.der.derPercent) < 20 ? 'acceptable' : 'poor'}">
                ${metrics.der.derPercent}%
              </div>
              <div class="metric-subtitle">
                FA: ${metrics.der.faPercent}% | MISS: ${metrics.der.missPercent}% | CONF: ${metrics.der.confPercent}%
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-label">🔶 Jaccard Error Rate (JER)</div>
              <div class="metric-value ${parseFloat(metrics.jer.jerPercent) < 10 ? 'good' : 'acceptable'}">
                ${metrics.jer.jerPercent}%
              </div>
              <div class="metric-subtitle">Avg Jaccard: ${metrics.jer.avgJaccard}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">🗣️ Speaker Count</div>
              <div class="metric-value ${metrics.speakerCount.correct ? 'good' : 'poor'}">
                ${metrics.speakerCount.hypothesis} / ${metrics.speakerCount.reference}
              </div>
              <div class="metric-subtitle">
                ${metrics.speakerCount.correct ? '✅ Correct' : `❌ Off by ${Math.abs(metrics.speakerCount.difference)}`}
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-label">⚡ Speed Factor</div>
              <div class="metric-value good">${result.speedFactor}x</div>
              <div class="metric-subtitle">${result.processingTime}s processing</div>
            </div>
          </div>

          <h4>Per-Speaker Performance</h4>
          <div style="overflow-x: auto; margin-bottom: var(--space-24);">
            <table class="results-table">
              <thead>
                <tr>
                  <th>Speaker</th>
                  <th>Speaking Time</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                </tr>
              </thead>
              <tbody>
        `;

        Object.entries(metrics.perSpeaker).forEach(([speaker, spMetrics]) => {
          html += `
            <tr>
              <td><strong>${speaker}</strong></td>
              <td>${spMetrics.speakingTime}s (${spMetrics.speakingPercent}%)</td>
              <td>${spMetrics.precision}%</td>
              <td>${spMetrics.recall}%</td>
              <td>${spMetrics.f1}%</td>
            </tr>
          `;
        });

        html += `
              </tbody>
            </table>
          </div>

          <h4>Error Summary</h4>
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">🚨 Total Errors</div>
              <div class="metric-value">${metrics.summary.totalErrors}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">🔴 False Alarms</div>
              <div class="metric-value">${metrics.summary.falseAlarms}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">🟡 Missed Detections</div>
              <div class="metric-value">${metrics.summary.missedDetections}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">🟣 Speaker Confusions</div>
              <div class="metric-value">${metrics.summary.speakerConfusions}</div>
            </div>
          </div>
        `;
      } else {
        html += `
          <div class="alert alert-info">
            📄 Ground truth not available. Upload ground truth to see comprehensive metrics.
          </div>
        `;
      }

      html += '</div>';
    });

    container.innerHTML = html;
  },

  /**
   * Render error analysis
   */
  renderErrorAnalysis() {
    const container = document.getElementById('errorAnalysisContainer');
    if (!container) return;

    if (!this.groundTruth || Object.keys(this.comprehensiveMetrics).length === 0) {
      container.innerHTML = `
        <div class="alert alert-warning">
          ⚠️ Ground truth required for error analysis. Please upload ground truth data.
        </div>
      `;
      return;
    }

    let html = '';

    Object.entries(this.comprehensiveMetrics).forEach(([serviceId, metrics]) => {
      const result = this.testResults[serviceId];
      
      html += `
        <div class="error-category">
          <h4>
            ${result.serviceName}
            <span class="error-badge">${metrics.summary.totalErrors} errors</span>
          </h4>
          
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-16);">
            <div>
              <strong style="color: var(--color-error);">🔴 False Alarms:</strong> ${metrics.summary.falseAlarms}
              <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Non-speech detected as speech</div>
            </div>
            <div>
              <strong style="color: var(--color-warning);">🟡 Missed Detections:</strong> ${metrics.summary.missedDetections}
              <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Speech not detected</div>
            </div>
            <div>
              <strong style="color: #8B5CF6;">🟣 Speaker Confusions:</strong> ${metrics.summary.speakerConfusions}
              <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Wrong speaker assigned</div>
            </div>
          </div>

          <h5>Error Timeline</h5>
          <div class="error-timeline" style="margin-top: var(--space-12);">
      `;

      // Show first few error segments
      const errorSamples = metrics.errors.slice(0, 5);
      errorSamples.forEach(error => {
        const errorType = {
          fa: { label: 'False Alarm', color: 'var(--color-error)' },
          miss: { label: 'Missed Detection', color: 'var(--color-warning)' },
          conf: { label: 'Speaker Confusion', color: '#8B5CF6' }
        }[error.type];

        html += `
          <div style="padding: var(--space-12); margin-bottom: var(--space-8); background: var(--color-surface); border-left: 3px solid ${errorType.color}; border-radius: var(--radius-sm);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div>
                <strong style="color: ${errorType.color};">${errorType.label}</strong>
                <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-top: var(--space-4);">
                  Time: ${error.start.toFixed(1)}s - ${error.end.toFixed(1)}s (${error.duration.toFixed(1)}s)
                </div>
              </div>
            </div>
          </div>
        `;
      });

      if (metrics.errors.length > 5) {
        html += `<p style="text-align: center; color: var(--color-text-secondary); font-size: var(--font-size-sm);">...and ${metrics.errors.length - 5} more errors</p>`;
      }

      html += '</div></div>';
    });

    container.innerHTML = html;
  },

  /**
   * Render cross-service comparison
   */
  renderCrossServiceComparison() {
    const container = document.getElementById('comparisonContainer');
    if (!container) return;

    const successfulResults = Object.entries(this.testResults)
      .filter(([_, r]) => r.success)
      .map(([id, r]) => ({ id, ...r }));

    if (successfulResults.length < 2) {
      container.innerHTML = '<div class="alert alert-info">Need at least 2 successful results for comparison.</div>';
      return;
    }

    // Calculate agreement matrix
    const servicesForMatrix = successfulResults.map(r => ({
      id: r.id,
      name: r.serviceName,
      segments: r.segments
    }));

    const crossMetrics = new CrossServiceMetrics(servicesForMatrix);
    const agreementMatrix = crossMetrics.calculateAgreementMatrix();
    const serviceNames = {};
    servicesForMatrix.forEach(s => serviceNames[s.id] = s.serviceName);

    let html = `
      <h3>Agreement Matrix</h3>
      <p style="color: var(--color-text-secondary); margin-bottom: var(--space-16);">Percentage of time where services agree on speaker assignment</p>
    `;

    container.innerHTML = html;
    
    // Render matrix
    AgreementMatrix.render('comparisonContainer', agreementMatrix, serviceNames);

    // Add comparison bars for key metrics
    if (this.groundTruth) {
      const metricsComparison = successfulResults.map(r => {
        const metrics = this.comprehensiveMetrics[r.id];
        return {
          name: r.serviceName,
          der: metrics ? metrics.der.derPercent : 'N/A',
          jer: metrics ? metrics.jer.jerPercent : 'N/A',
          speed: r.speedFactor
        };
      });

      container.innerHTML += `
        <div style="margin-top: var(--space-32);">
          <h3>Metric Comparison</h3>
          <div id="derComparisonBars"></div>
        </div>
      `;

      // Render DER comparison
      const derData = metricsComparison.map(m => ({
        name: m.name,
        value: m.der,
        unit: '%'
      }));
      
      MetricsChart.renderComparisonBars('derComparisonBars', derData, 'DER');
    }
  },

  /**
   * Render timeline visualization
   */
  renderTimelineVisualization() {
    const container = document.getElementById('timelineVizContainer');
    if (!container) return;

    container.innerHTML = '<div id="timelineCanvas"></div>';

    const successfulResults = Object.entries(this.testResults)
      .filter(([_, r]) => r.success)
      .map(([id, r]) => ({
        id,
        name: r.serviceName,
        segments: r.segments
      }));

    const viz = new TimelineVisualization('timelineCanvas');
    
    if (this.groundTruth) {
      viz.renderStackedTimeline(successfulResults, {
        name: 'Ground Truth',
        segments: this.groundTruth.segments
      });
    } else {
      viz.renderStackedTimeline(successfulResults);
    }
  },

  /**
   * Ground truth handling
   */
  handleGroundTruthUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      this.groundTruth = this.parseGroundTruth(content, file.name);
      
      const infoText = `✅ Loaded: ${file.name} (${this.groundTruth.segments.length} segments, ${this.groundTruth.speakerCount} speakers)`;
      
      // Update both ground truth info displays if they exist
      const groundTruthInfo = document.getElementById('groundTruthInfo');
      const audioGroundTruthInfo = document.getElementById('audioGroundTruthInfo');
      
      if (groundTruthInfo) {
        groundTruthInfo.innerHTML = infoText;
        groundTruthInfo.style.display = 'block';
      }
      if (audioGroundTruthInfo) {
        audioGroundTruthInfo.innerHTML = infoText;
        audioGroundTruthInfo.style.display = 'block';
      }
    };
    reader.readAsText(file);
  },

  parseGroundTruth(content, filename) {
    const segments = [];
    const lines = content.trim().split('\n');

    if (filename.endsWith('.rttm')) {
      // Parse RTTM format
      lines.forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 8 && parts[0] === 'SPEAKER') {
          segments.push({
            start: parseFloat(parts[3]),
            end: parseFloat(parts[3]) + parseFloat(parts[4]),
            speaker: parts[7]
          });
        }
      });
    } else {
      // Parse simple format: start end speaker [text]
      lines.forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 3) {
          segments.push({
            start: parseFloat(parts[0]),
            end: parseFloat(parts[1]),
            speaker: parts[2],
            text: parts.slice(3).join(' ')
          });
        }
      });
    }

    const speakers = new Set(segments.map(s => s.speaker));
    
    return {
      segments: segments.sort((a, b) => a.start - b.start),
      speakerCount: speakers.size,
      format: filename.endsWith('.rttm') ? 'RTTM' : 'Simple'
    };
  },

  showGroundTruthEditor() {
    document.getElementById('groundTruthEditor').style.display = 'flex';
    this.renderGroundTruthSegments();
  },

  closeGroundTruthEditor() {
    document.getElementById('groundTruthEditor').style.display = 'none';
  },

  renderGroundTruthSegments() {
    const container = document.getElementById('groundTruthSegments');
    const segments = this.groundTruth ? this.groundTruth.segments : [];

    let html = '<div style="margin-bottom: var(--space-16);">';
    
    segments.forEach((seg, idx) => {
      html += `
        <div class="gt-segment">
          <input type="number" step="0.1" value="${seg.start}" placeholder="Start" data-idx="${idx}" data-field="start">
          <input type="number" step="0.1" value="${seg.end}" placeholder="End" data-idx="${idx}" data-field="end">
          <select data-idx="${idx}" data-field="speaker">
            <option value="SPEAKER_A" ${seg.speaker === 'SPEAKER_A' ? 'selected' : ''}>Speaker A</option>
            <option value="SPEAKER_B" ${seg.speaker === 'SPEAKER_B' ? 'selected' : ''}>Speaker B</option>
            <option value="SPEAKER_C" ${seg.speaker === 'SPEAKER_C' ? 'selected' : ''}>Speaker C</option>
            <option value="SPEAKER_D" ${seg.speaker === 'SPEAKER_D' ? 'selected' : ''}>Speaker D</option>
          </select>
          <input type="text" value="${seg.text || ''}" placeholder="Text (optional)" data-idx="${idx}" data-field="text">
          <button class="btn btn-secondary btn-sm" onclick="app.removeGroundTruthSegment(${idx})">✖</button>
        </div>
      `;
    });

    html += '</div>';
    container.innerHTML = html;
  },

  addGroundTruthSegment() {
    if (!this.groundTruth) {
      this.groundTruth = { segments: [], speakerCount: 0, format: 'Simple' };
    }
    
    this.groundTruth.segments.push({
      start: 0,
      end: 1,
      speaker: 'SPEAKER_A',
      text: ''
    });
    
    this.renderGroundTruthSegments();
  },

  removeGroundTruthSegment(idx) {
    if (this.groundTruth && this.groundTruth.segments[idx]) {
      this.groundTruth.segments.splice(idx, 1);
      this.renderGroundTruthSegments();
    }
  },

  saveGroundTruth() {
    // Collect values from inputs
    const inputs = document.querySelectorAll('#groundTruthSegments input, #groundTruthSegments select');
    const segments = [];
    const segmentMap = {};

    inputs.forEach(input => {
      const idx = input.dataset.idx;
      const field = input.dataset.field;
      
      if (!segmentMap[idx]) {
        segmentMap[idx] = {};
      }
      
      segmentMap[idx][field] = field === 'start' || field === 'end' 
        ? parseFloat(input.value) 
        : input.value;
    });

    Object.values(segmentMap).forEach(seg => {
      segments.push(seg);
    });

    this.groundTruth = {
      segments: segments.sort((a, b) => a.start - b.start),
      speakerCount: new Set(segments.map(s => s.speaker)).size,
      format: 'Simple'
    };

    const infoText = `✅ Created: ${this.groundTruth.segments.length} segments, ${this.groundTruth.speakerCount} speakers`;
    
    // Update both ground truth info displays if they exist
    const groundTruthInfo = document.getElementById('groundTruthInfo');
    const audioGroundTruthInfo = document.getElementById('audioGroundTruthInfo');
    
    if (groundTruthInfo) {
      groundTruthInfo.innerHTML = infoText;
      groundTruthInfo.style.display = 'block';
    }
    if (audioGroundTruthInfo) {
      audioGroundTruthInfo.innerHTML = infoText;
      audioGroundTruthInfo.style.display = 'block';
    }
    
    this.closeGroundTruthEditor();
  },

  /**
   * Transcript Context Diarization handling
   */
  handleTranscriptFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Check file extension
    const fileName = file.name.toLowerCase();
    const isMarkdown = fileName.endsWith('.md') || fileName.endsWith('.markdown');
    const fileType = isMarkdown ? 'Markdown' : 'Text';

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      document.getElementById('transcriptText').value = content;
      document.getElementById('transcriptFileInfo').innerHTML = `
        ✅ Loaded: ${file.name} (${fileType}, ${(file.size / 1024).toFixed(2)} KB)
      `;
      document.getElementById('transcriptFileInfo').style.display = 'block';
      this.updateAnalyzeButtonState();
    };
    reader.onerror = () => {
      alert('Error reading the file. Please make sure it is a text document.');
    };
    reader.readAsText(file, 'UTF-8');
  },

  updateAnalyzeButtonState() {
    const textarea = document.getElementById('transcriptText');
    const button = document.getElementById('analyzeTranscriptBtn');
    if (textarea && button) {
      button.disabled = !textarea.value.trim();
    }
  },

  // Helper function to clean service messages from transcript text
  cleanServiceMessages(text) {
    const lines = text.split('\n');
    const cleanedLines = lines.filter(line => {
      const trimmed = line.trim();
      // Skip lines that are service markers (-- ServiceName --)
      if (SERVICE_MARKER_REGEX.test(trimmed)) return false;
      // Skip lines that are just separators
      if (SEPARATOR_REGEX.test(trimmed)) return false;
      return true;
    });
    return cleanedLines.join('\n').trim();
  },

  // Helper function to extract recording name from header
  extractRecordingName(headerLine) {
    const match = headerLine.match(RECORDING_NAME_EXTRACT_PATTERN);
    if (match) {
      const content = match[1].trim();
      // Remove "Запис X:" or "Recording X:" prefix (support multiple languages: Ukrainian, English, Arabic)
      const nameMatch = content.match(RECORDING_NAME_PREFIX_PATTERN);
      if (nameMatch) {
        return nameMatch[1].trim();
      }
      return content;
    }
    return null;
  },

  // Split transcript into multiple recordings on client side
  splitTranscriptIntoRecordings(transcript) {
    const lines = transcript.split('\n');
    const recordings = [];
    let currentRecording = [];
    let currentRecordingName = null;
    let recordingIndex = 0;
    let inRecording = false;
    
    console.log(`🔍 Splitting transcript: ${lines.length} lines`);
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();
      
      // Check if this is a recording header
      const isHeader = RECORDING_HEADER_PATTERN.test(trimmed);
      if (isHeader) {
        console.log(`  📌 Found recording header at line ${i + 1}: "${trimmed.substring(0, 80)}..."`);
        
        // Save previous recording if exists
        if (inRecording && currentRecording.length > 0) {
          const recordingText = this.cleanServiceMessages(currentRecording.join('\n'));
          if (recordingText.length > 0) {
            console.log(`  💾 Saving recording ${recordingIndex + 1}: "${currentRecordingName}" (${recordingText.length} chars)`);
            recordings.push({
              id: `rec_${String(recordingIndex + 1).padStart(3, '0')}`,
              name: currentRecordingName || `Recording ${recordingIndex + 1}`,
              text: recordingText
            });
            recordingIndex++;
          } else {
            console.log(`  ⚠️ Skipping empty recording "${currentRecordingName}"`);
          }
        }
        
        // Start new recording
        currentRecording = [];
        currentRecordingName = this.extractRecordingName(trimmed) || `Recording ${recordingIndex + 1}`;
        console.log(`  🆕 Starting new recording: "${currentRecordingName}"`);
        inRecording = true;
        continue;
      } else if (trimmed.startsWith('===') && trimmed.endsWith('===') && trimmed.length > 6) {
        // Log potential headers that didn't match (for debugging)
        console.log(`  ⚠️ Potential header that didn't match pattern: "${trimmed.substring(0, 80)}..."`);
      }
      
      // Skip header lines (=== ... ===) that are not recording headers
      if (HEADER_SEPARATOR_PATTERN.test(trimmed)) {
        continue;
      }
      
      // Skip metadata lines at the start (like "AI SUMMARY GENERATED AT...")
      if (!inRecording && METADATA_PATTERN.test(trimmed)) {
        continue;
      }
      
      // If we're in a recording, add the line (we'll clean service messages later)
      if (inRecording) {
        currentRecording.push(line);
      } else if (trimmed.length > 0) {
        // If we haven't found a header yet but have content, start collecting
        currentRecording.push(line);
      }
    }
    
    // Add last recording if any
    if (inRecording && currentRecording.length > 0) {
      const recordingText = this.cleanServiceMessages(currentRecording.join('\n'));
      if (recordingText.length > 0) {
        console.log(`  💾 Saving final recording ${recordingIndex + 1}: "${currentRecordingName}" (${recordingText.length} chars)`);
        recordings.push({
          id: `rec_${String(recordingIndex + 1).padStart(3, '0')}`,
          name: currentRecordingName || `Recording ${recordingIndex + 1}`,
          text: recordingText
        });
      }
    }
    
    // If no recordings were found (no headers), treat entire transcript as one recording
    if (recordings.length === 0) {
      console.log(`  ⚠️ No recording headers found, treating as single recording`);
      const cleanedText = this.cleanServiceMessages(transcript);
      recordings.push({
        id: 'rec_001',
        name: 'Transcript Analysis',
        text: cleanedText
      });
    }
    
    console.log(`📝 Split transcript into ${recordings.length} recording(s):`);
    recordings.forEach((rec, idx) => {
      const preview = rec.text.substring(0, 100).replace(/\n/g, ' ');
      console.log(`  ${idx + 1}. ${rec.id} - "${rec.name}" (${rec.text.length} chars, preview: "${preview}...")`);
    });
    
    return recordings;
  },

  async analyzeTranscript() {
    const textarea = document.getElementById('transcriptText');
    const transcriptText = textarea.value.trim();
    
    if (!transcriptText) {
      alert('Please enter the transcript text or upload a file.');
      return;
    }

    const button = document.getElementById('analyzeTranscriptBtn');
    const resultDiv = document.getElementById('diarizationResult');
    const resultContent = document.getElementById('diarizationResultContent');

    // Disable button and show loading
    button.disabled = true;
    button.textContent = '⏳ Analyzing...';
    resultDiv.style.display = 'none';

    try {
      // Split transcript into multiple recordings
      const recordings = this.splitTranscriptIntoRecordings(transcriptText);
      console.log(`📝 Processing ${recordings.length} recording(s) separately...`);

      // Get selected model and determine mode
      const modelRadio = document.querySelector('input[name="diarizationModel"]:checked');
      const selectedModel = modelRadio ? modelRadio.value : 'fast-tier';
      
      // Map model to mode for backend
      let mode = 'fast';
      if (selectedModel === 'smart-tier') {
        mode = 'smart';
      } else if (selectedModel === 'smart-2-tier') {
        mode = 'smart-2';
      } else if (selectedModel === 'gemini-2.5-pro-text') {
        // Gemini 2.5 Pro in text-only mode (no multimodal)
        mode = 'smart-2'; // Use Smart 2 model for text-only processing
      } else if (selectedModel === 'local-tier') {
        mode = 'local';
      }
      
      // Process all recordings in parallel
      button.textContent = `⏳ Analyzing ${recordings.length} recordings...`;
      
      const processingPromises = recordings.map(async (recording, idx) => {
        console.log(`🔄 Processing recording ${idx + 1}/${recordings.length}: ${recording.name}`);
        
        // LLM method doesn't use multimodal - only text processing
        // Standard JSON request
        const response = await fetch('/api/diarize', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            transcript: recording.text,
            mode: mode
          })
        });

        if (!response.ok) {
          throw new Error(`Request failed for recording ${recording.name}: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
        
        // Extract recording from result
        let recordingData = null;
        if (result.recordings && Array.isArray(result.recordings) && result.recordings.length > 0) {
          recordingData = result.recordings[0];
          
          // Generate ID based on recording name (like rec_001_gulf_appointment)
          const nameSlug = recording.name
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '_')
            .replace(/^_+|_+$/g, '')
            .substring(0, 30);
          recordingData.id = `rec_${String(idx + 1).padStart(3, '0')}_${nameSlug}`;
          recordingData.name = recording.name;
          
          // Ensure all required fields
          if (!recordingData.fileName) recordingData.fileName = 'transcript.txt';
          if (recordingData.size === undefined) recordingData.size = 0;
          if (!recordingData.language) recordingData.language = 'ar';
          if (!recordingData.status) recordingData.status = 'completed';
          if (recordingData.addedAt === undefined) recordingData.addedAt = null;
          if (!recordingData.translationState) {
            recordingData.translationState = { currentLanguage: 'original', lastError: null };
          }
          if (!recordingData.aggregated) recordingData.aggregated = {};
          if (!recordingData.servicesTested) recordingData.servicesTested = [TEXT_SERVICE_KEY];
          else if (!recordingData.servicesTested.includes(TEXT_SERVICE_KEY)) recordingData.servicesTested.push(TEXT_SERVICE_KEY);
        } else {
          throw new Error(`Recording ${recording.name} does not contain any data`);
        }
        
        console.log(`✅ Completed recording ${idx + 1}/${recordings.length}: ${recording.name} (ID: ${recordingData.id})`);
        return recordingData;
      });
      
      // Wait for all recordings to complete
      const processedRecordings = await Promise.all(processingPromises);
      
      console.log(`✅ All ${processedRecordings.length} recordings processed`);
      
      // Combine into the required format: [{ output: { recordings: [...] } }]
      const combinedResult = [{
        output: {
          version: '2.0',
          exportedAt: new Date().toISOString(),
          activeRecordingId: processedRecordings[0]?.id || 'rec_001',
          recordings: processedRecordings
        }
      }];
      
      console.log('📦 Combined result:', {
        format: '[{ output: { recordings: [...] } }]',
        recordingsCount: processedRecordings.length,
        activeRecordingId: combinedResult[0].output.activeRecordingId,
        recordingIds: processedRecordings.map(r => r.id)
      });
      
      let finalResults = combinedResult[0].output;
      
      // Perform LLM post-processing (verification) - always enabled
      {
        console.log('🔄 Performing LLM post-processing (verification)...');
        button.textContent = '⏳ Post-processing...';
        
        try {
          // Load verification prompt template
          const verificationPromptResponse = await fetch('prompts/diarization_verification_prompt.txt');
          if (!verificationPromptResponse.ok) {
            throw new Error('Unable to load the verification prompt template');
          }
          let verificationPrompt = await verificationPromptResponse.text();
          
          // Replace placeholder with current recordings JSON
          const jsonPlaceholder = '[PASTE THE JSON OUTPUT FROM THE FIRST WEBHOOK HERE]';
          const jsonString = JSON.stringify(finalResults, null, 2);
          verificationPrompt = verificationPrompt.replace(jsonPlaceholder, jsonString);
          
          // Send to verification API
          const verificationResponse = await fetch('/api/diarize', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              transcript: JSON.stringify(finalResults),
              mode: mode,
              isVerification: true,
              verificationPrompt: verificationPrompt
            })
          });
          
          if (!verificationResponse.ok) {
            throw new Error(`Verification request failed: ${verificationResponse.status} ${verificationResponse.statusText}`);
          }
          
          // Parse verified response
          const verifiedResult = await verificationResponse.json();
          console.log('✅ Post-processing complete');
          
          // Check for error
          if (verifiedResult.error) {
            throw new Error(`Verification error: ${verifiedResult.error}`);
          }
          
          // Extract verified data
          let verifiedData = verifiedResult;
          
          // Check if verified data is in wrapped format
          if (Array.isArray(verifiedData) && verifiedData.length > 0) {
            const firstItem = verifiedData[0];
            if (firstItem && firstItem.output && firstItem.output.recordings) {
              verifiedData = firstItem.output;
            } else if (firstItem && firstItem.recordings) {
              verifiedData = firstItem;
            }
          }
          
          // Fix: If recordings is an object instead of array, convert it to array
          if (verifiedData.recordings && !Array.isArray(verifiedData.recordings) && typeof verifiedData.recordings === 'object') {
            verifiedData.recordings = [verifiedData.recordings];
          }
          
          // Use verified results if valid
          if (verifiedData.recordings && Array.isArray(verifiedData.recordings) && verifiedData.recordings.length > 0) {
            finalResults = verifiedData;
            console.log('✅ Using verified (post-processed) results');
          } else {
            console.warn('⚠️ Verified data invalid, using original results');
          }
        } catch (verificationError) {
          console.warn('⚠️ Post-processing failed, using original results:', verificationError);
          // Continue with original results if verification fails
        }
      }
      
      // Load the results (pass the output object, not the wrapped array)
      this.loadJSONResults(finalResults);
      
      // Display result in JSON format for debugging
      const displayResult = [{
        output: finalResults
      }];
      resultContent.textContent = JSON.stringify(displayResult, null, 2);
      resultDiv.style.display = 'block';
      
      // Show success message and navigate to replicas view
      alert(`✅ Analysis complete! Processed ${processedRecordings.length} recordings.\n\nOpening the replicas comparison view...`);
      
      // Navigate to replicas view
      this.showReplicasComparison();
      
      // Scroll to result (if still on upload view)
      resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
      console.error('Error analyzing transcript:', error);
      resultContent.textContent = `Error: ${error.message}\n\nDetails:\n${error.stack || ''}`;
      resultDiv.style.display = 'block';
      resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } finally {
      button.disabled = false;
      button.textContent = '🔍 Analyze Transcript';
    }
  },

  clearTranscript() {
    document.getElementById('transcriptText').value = '';
    document.getElementById('transcriptFileInput').value = '';
    document.getElementById('transcriptFileInfo').style.display = 'none';
    document.getElementById('diarizationResult').style.display = 'none';
    this.updateAnalyzeButtonState();
  },

  async fixMistakes() {
    if (this.recordings.length === 0) {
      alert('No recordings are available for fixes. Please analyze the transcript first.');
      return;
    }

    const button = document.getElementById('fixMistakesBtn');
    if (!button) return;

    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '⏳ Preparing transcripts...';

    const modelRadio = document.querySelector('input[name="diarizationModel"]:checked');
    const selectedModel = modelRadio ? modelRadio.value : 'fast-tier';
    let mode = 'fast';
    if (selectedModel === 'smart-tier') {
      mode = 'smart';
    } else if (selectedModel === 'smart-2-tier') {
      mode = 'smart-2';
    } else if (selectedModel === 'gemini-2.5-pro-text') {
      mode = 'smart-2';
    } else if (selectedModel === 'local-tier') {
      mode = 'local';
    }

    try {
      const updatedRecordings = [];

      for (let i = 0; i < this.recordings.length; i++) {
        const recording = this.recordings[i];
        const transcriptSection = this.buildRecordingTextForFix(recording, i);
        if (!transcriptSection) {
          console.warn(`Skipping recording ${recording.id}: no transcript data available`);
          continue;
        }

        const transcriptText = [
          `AI SUMMARY GENERATED AT ${new Date().toISOString()}`,
          '',
          transcriptSection
        ].join('\n');

        button.textContent = `⏳ Fixing ${recording.name || recording.id} (${i + 1}/${this.recordings.length})...`;

        // Load verification prompt with operator/client identification
        let verificationPrompt;
        try {
          const promptTemplateResponse = await fetch('prompts/diarization_verification_prompt.txt');
          if (promptTemplateResponse.ok) {
            const promptTemplate = await promptTemplateResponse.text();
            verificationPrompt = promptTemplate.replace(
              '[PASTE THE JSON OUTPUT FROM THE FIRST WEBHOOK HERE]',
              transcriptText
            );
          }
        } catch (error) {
          console.warn('Failed to load verification prompt template:', error);
        }

        const response = await fetch('/api/diarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            transcript: transcriptText,
            mode,
            isFixRequest: true,
            verificationPrompt: verificationPrompt
          })
        });

        if (!response.ok) {
          const errorText = await response.text().catch(() => response.statusText);
          throw new Error(`Fix request failed (${response.status}): ${errorText}`);
        }

        let resultData = await response.json();
        if (resultData.error) {
          throw new Error(resultData.error);
        }

        if (Array.isArray(resultData) && resultData.length > 0) {
          const firstItem = resultData[0];
          if (firstItem?.output?.recordings) {
            resultData = firstItem.output;
          } else if (firstItem?.recordings) {
            resultData = firstItem;
          }
        }

        if (resultData.recordings && !Array.isArray(resultData.recordings)) {
          resultData.recordings = [resultData.recordings];
        }

        if (!resultData.recordings || resultData.recordings.length === 0) {
          throw new Error('Fixer response does not include recordings.');
        }

        const correctedRecording = resultData.recordings[0];
        correctedRecording.id = recording.id;
        correctedRecording.name = recording.name || correctedRecording.name;
        correctedRecording.fileName = recording.fileName || correctedRecording.fileName || 'transcript.txt';
        correctedRecording.size = recording.size ?? correctedRecording.size ?? 0;
        correctedRecording.duration = recording.duration || correctedRecording.duration || 0;
        correctedRecording.language = recording.language || correctedRecording.language || 'ar';
        correctedRecording.status = 'completed';
        correctedRecording.translationState = recording.translationState || this.createTranslationState();
        correctedRecording.aggregated = correctedRecording.aggregated || {};

        const hydrated = this.hydrateRecordingFromExport(correctedRecording);
        updatedRecordings.push(hydrated);
      }

      if (updatedRecordings.length === 0) {
        throw new Error('No transcript data available for verification.');
      }

      this.recordings = updatedRecordings;
      this.activeRecordingId = updatedRecordings[0]?.id || null;
      if (this.activeRecordingId) {
        this.setActiveRecording(this.activeRecordingId, { skipResultsRender: true });
      }
      this.renderRecordingsQueue();
      this.showResults();

      alert(`✅ Issues resolved! Updated ${updatedRecordings.length} recording${updatedRecordings.length === 1 ? '' : 's'}.`);
    } catch (error) {
      console.error('Error fixing mistakes:', error);
      alert(`Error while fixing results: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
  },

  /**
   * Generate comprehensive report
   */
  generateReport() {
    let report = '='.repeat(60) + '\n';
    report += 'SPEAKER DIARIZATION BENCHMARK REPORT\n';
    report += '='.repeat(60) + '\n\n';
    report += `Generated: ${new Date().toLocaleString()}\n`;
    report += `Audio File: ${this.uploadedFile ? this.uploadedFile.name : 'N/A'}\n`;
    report += `Duration: ${Math.floor(this.fileDuration / 60)}:${(this.fileDuration % 60).toString().padStart(2, '0')}\n`;
    report += `Services Tested: ${Object.keys(this.testResults).length}\n`;
    report += `Ground Truth: ${this.groundTruth ? 'Yes' : 'No'}\n\n`;

    Object.entries(this.testResults).forEach(([serviceId, result]) => {
      if (!result.success) return;

      report += '-'.repeat(60) + '\n';
      report += `${result.serviceName}\n`;
      report += '-'.repeat(60) + '\n';
      report += `Processing Time: ${result.processingTime}s\n`;
      report += `Speed Factor: ${result.speedFactor}x real-time\n`;
      report += `Speakers Detected: ${result.speakerCount}\n`;
      report += `Cost: $${result.cost}\n`;

      if (this.comprehensiveMetrics[serviceId]) {
        const metrics = this.comprehensiveMetrics[serviceId];
        report += `\nMetrics:\n`;
        report += `  DER: ${metrics.der.derPercent}%\n`;
        report += `    - False Alarm: ${metrics.der.faPercent}%\n`;
        report += `    - Missed Detection: ${metrics.der.missPercent}%\n`;
        report += `    - Speaker Confusion: ${metrics.der.confPercent}%\n`;
        report += `  JER: ${metrics.jer.jerPercent}%\n`;
        report += `  Speaker Count Accuracy: ${metrics.speakerCount.correct ? 'Correct' : 'Incorrect'}\n`;
        report += `  Total Errors: ${metrics.summary.totalErrors}\n`;
      }

      report += '\n';
    });

    // Download as text file
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'diarization_benchmark_report.txt';
    link.click();
    URL.revokeObjectURL(url);
  },

  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  },

  downloadJSON(serviceId) {
    const result = this.testResults[serviceId];
    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${serviceId}_results.json`;
    link.click();
    URL.revokeObjectURL(url);
  },

  cloneResults(results) {
    if (!results) return {};
    try {
      return JSON.parse(JSON.stringify(results));
    } catch (err) {
      console.warn('Failed to deep-clone results, falling back to shallow copy', err);
      return { ...results };
    }
  },

  attachCombinedResultToRecording(recording, combinedResult) {
    if (!recording || !combinedResult) {
      return;
    }

    const finalRecording = combinedResult.finalResult?.recordings?.[0];
    if (!finalRecording || !finalRecording.results) {
      console.warn('Combined result is missing final recordings payload');
      return;
    }

    recording.results = this.cloneResults(finalRecording.results);
    recording.duration = finalRecording.duration || recording.duration || 0;
    recording.language = finalRecording.language || recording.language || 'ar';
    recording.speakerCount = finalRecording.speakerCount ?? recording.speakerCount ?? '';
    recording.translationState = {
      ...(recording.translationState || this.createTranslationState()),
      ...(finalRecording.translationState || {})
    };
    recording.aggregated = finalRecording.aggregated || recording.aggregated || {};
    recording.servicesTested = finalRecording.servicesTested || recording.servicesTested || [TEXT_SERVICE_KEY];

    this.normalizeTextServiceResult(recording);

    const segmentsCount = this.getTextServiceResult(recording)?.segments?.length || 0;
    console.log(`✅ Attached combined result to recording ${recording.id}: ${segmentsCount} segments available`);
  },

  attachOverlapResultToRecording(recording, overlapResult) {
    if (!recording || !overlapResult) {
      return;
    }

    const correctedRecording = overlapResult.correctedDiarization?.recordings?.[0];
    if (!correctedRecording || !correctedRecording.results) {
      console.warn('Overlap result is missing corrected recordings payload');
      return;
    }

    recording.results = this.cloneResults(correctedRecording.results);
    recording.duration = correctedRecording.duration || recording.duration || 0;
    recording.language = correctedRecording.language || recording.language || 'ar';
    recording.speakerCount = correctedRecording.speakerCount ?? recording.speakerCount ?? '';
    recording.translationState = {
      ...(recording.translationState || this.createTranslationState()),
      ...(correctedRecording.translationState || {})
    };
    recording.aggregated = correctedRecording.aggregated || recording.aggregated || {};
    recording.servicesTested = correctedRecording.servicesTested || recording.servicesTested || ['overlap-corrected'];

    // Store overlap metadata for reference
    recording.overlapMetadata = {
      primaryDiarization: overlapResult.primaryDiarization,
      separation: overlapResult.separation,
      voiceTracks: overlapResult.voiceTracks,
      steps: overlapResult.steps,
      diagnostics: overlapResult.diagnostics
    };
    recording.missingReplicas = overlapResult.diagnostics?.missingReplicas || [];
    
    // Store text analysis results for color highlighting
    recording.textAnalysis = overlapResult.textAnalysis || null;

    // Normalize text service result - also handle overlap-corrected
    this.normalizeTextServiceResult(recording);
    
    // If overlap-corrected exists, also make it available as text-service for compatibility
    if (recording.results['overlap-corrected'] && !recording.results[TEXT_SERVICE_KEY]) {
      recording.results[TEXT_SERVICE_KEY] = recording.results['overlap-corrected'];
    }

    const segmentsCount = this.getTextServiceResult(recording)?.segments?.length || 
                         recording.results['overlap-corrected']?.segments?.length || 0;
    console.log(`✅ Attached overlap result to recording ${recording.id}: ${segmentsCount} segments available`);
    
    // Update Apply Overlap Fixes button visibility based on engine
    const engine = recording.overlapMetadata?.steps?.step1?.engine || 
                  recording.results?.speechmatics?.engine ||
                  recording.results?.['overlap-corrected']?.engine;
    const isAzure = engine && (engine.toLowerCase().includes('azure') || engine.toLowerCase() === 'azure_realtime');
    this.updateApplyOverlapFixesButtonVisibility(recording, isAzure);
  },

  normalizeTextServiceResult(recording) {
    if (!recording) return recording;
    if (!recording.results) recording.results = {};

    LEGACY_TEXT_SERVICE_KEYS.forEach(key => {
      if (recording.results[key]) {
        if (!recording.results[TEXT_SERVICE_KEY]) {
          recording.results[TEXT_SERVICE_KEY] = recording.results[key];
        }
        delete recording.results[key];
      }
    });

    // If overlap-corrected exists and no text-service, copy it for compatibility
    if (recording.results['overlap-corrected'] && !recording.results[TEXT_SERVICE_KEY]) {
      recording.results[TEXT_SERVICE_KEY] = { ...recording.results['overlap-corrected'] };
    }

    if (!recording.servicesTested || recording.servicesTested.length === 0) {
      recording.servicesTested = [TEXT_SERVICE_KEY];
    } else {
      recording.servicesTested = recording.servicesTested.map(id =>
        LEGACY_TEXT_SERVICE_KEYS.includes(id) ? TEXT_SERVICE_KEY : id
      );
      if (!recording.servicesTested.includes(TEXT_SERVICE_KEY)) {
        recording.servicesTested.push(TEXT_SERVICE_KEY);
      }
    }

    const textResult = recording.results[TEXT_SERVICE_KEY];
    if (textResult && textResult.serviceName && textResult.serviceName !== TEXT_SERVICE_NAME) {
      textResult.serviceName = TEXT_SERVICE_NAME;
    }
    return recording;
  },

  /**
   * Parse Azure native segments format to table-compatible format
   * Azure format: { type: "final", speakerId: "Guest-1", text: "...", offset: 0.71, duration: 7.6 }
   * Table format: { speaker: "SPEAKER_01", text: "...", start: 0.71, end: 8.31 }
   */
  parseAzureSegments(azureRawData) {
    if (!azureRawData || !azureRawData.segments) {
      return [];
    }
    
    const speakerMap = azureRawData.speakerMap || {};
    const segments = [];
    
    azureRawData.segments.forEach(seg => {
      if (seg.type !== 'final' || !seg.text) {
        return;
      }
      
      const start = seg.offset || 0;
      const end = start + (seg.duration || 0);
      const azureSpeakerId = seg.speakerId || 'Unknown';
      
      // Use speakerMap if available, otherwise try to extract number from Guest-X format
      let speaker = speakerMap[azureSpeakerId];
      if (!speaker && azureSpeakerId.startsWith('Guest-')) {
        try {
          const guestNum = parseInt(azureSpeakerId.split('-')[1]);
          // Format as SPEAKER_01, SPEAKER_02, etc.
          speaker = `SPEAKER_${String(guestNum).padStart(2, '0')}`;
        } catch (e) {
          speaker = `SPEAKER_${azureSpeakerId}`;
        }
      } else if (!speaker) {
        speaker = `SPEAKER_${azureSpeakerId}`;
      }
      
      segments.push({
        speaker: speaker,
        text: seg.text,
        start: start,
        end: end,
        words: [], // Azure Realtime doesn't provide word-level timestamps
        confidence: 0.9, // Default confidence
        pauses: [],
        // Keep original Azure data for reference
        _azure: {
          speakerId: azureSpeakerId,
          type: seg.type,
          offset: seg.offset,
          duration: seg.duration
        }
      });
    });
    
    return segments;
  },

  /**
   * Remove duplicate and similar segments (80% similarity threshold)
   */
  deduplicateSegments(segments) {
    if (!segments || !Array.isArray(segments) || segments.length === 0) {
      return segments;
    }
    
    // Check if TextSimilarityUtils is available
    if (typeof window === 'undefined' || !window.TextSimilarityUtils) {
      console.warn('TextSimilarityUtils not available, skipping deduplication');
      return segments;
    }
    
    const areTextsSimilar = window.TextSimilarityUtils.areTextsSimilar;
    const deduplicated = [];
    const SIMILARITY_THRESHOLD = 0.80; // 80% similarity
    
    // Sort segments by start time for consistent processing
    const sortedSegments = [...segments].sort((a, b) => {
      const startA = parseFloat(a.start) || 0;
      const startB = parseFloat(b.start) || 0;
      if (startA !== startB) return startA - startB;
      const endA = parseFloat(a.end) || 0;
      const endB = parseFloat(b.end) || 0;
      return endA - endB;
    });
    
    for (const current of sortedSegments) {
      const currentText = (current.text || '').trim();
      if (!currentText) {
        continue; // Skip empty segments
      }
      
      let isDuplicate = false;
      
      // Check against already processed segments
      for (let i = 0; i < deduplicated.length; i++) {
        const existing = deduplicated[i];
        const existingText = (existing.text || '').trim();
        
        if (!existingText) continue;
        
        // Check text similarity (80% threshold)
        const textSimilar = areTextsSimilar(currentText, existingText, {
          minLevenshteinSim: SIMILARITY_THRESHOLD,
          minJaccardSim: SIMILARITY_THRESHOLD,
          minSubstringMatch: SIMILARITY_THRESHOLD
        });
        
        if (textSimilar) {
          // Check temporal overlap (segments should be close in time)
          const currentStart = parseFloat(current.start) || 0;
          const currentEnd = parseFloat(current.end) || currentStart;
          const existingStart = parseFloat(existing.start) || 0;
          const existingEnd = parseFloat(existing.end) || existingStart;
          
          // Calculate overlap
          const overlapStart = Math.max(currentStart, existingStart);
          const overlapEnd = Math.min(currentEnd, existingEnd);
          const overlapDuration = Math.max(0, overlapEnd - overlapStart);
          
          // Consider duplicate if there's temporal overlap or segments are close (within 5 seconds)
          const timeDistance = Math.abs(currentStart - existingStart);
          const hasTemporalOverlap = overlapDuration > 0.1 || timeDistance < 5.0;
          
          if (hasTemporalOverlap) {
            // Prefer longer, more complete text
            if (currentText.length > existingText.length) {
              deduplicated[i] = current;
            }
            isDuplicate = true;
            break;
          }
        }
      }
      
      if (!isDuplicate) {
        deduplicated.push(current);
      }
    }
    
    const removedCount = segments.length - deduplicated.length;
    if (removedCount > 0) {
      console.log(`🧹 Removed ${removedCount} duplicate/similar segments (80% similarity threshold)`);
    }
    
    // IMPORTANT: Sort by start time after deduplication to ensure correct chronological order
    // This fixes issues where timestamps get mixed up after deduplication
    const finalSorted = deduplicated.sort((a, b) => {
      const startA = parseFloat(a.start) || 0;
      const startB = parseFloat(b.start) || 0;
      if (startA !== startB) return startA - startB;
      // If start times are equal, sort by end time
      const endA = parseFloat(a.end) || startA;
      const endB = parseFloat(b.end) || startB;
      return endA - endB;
    });
    
    return finalSorted;
  },

  /**
   * Merge consecutive segments from the same speaker when no other speaker intervenes
   * Useful for Azure realtime which may emit micro segments per sentence
   */
  mergeConsecutiveSpeakerSegments(segments, maxGapSeconds = 2.0) {
    if (!Array.isArray(segments) || segments.length === 0) {
      return segments;
    }

    const merged = [];
    const MAX_GAP = typeof maxGapSeconds === 'number' ? maxGapSeconds : 2.0;

    const normalizeText = (text) => (typeof text === 'string' ? text.trim() : '');

    segments.forEach(segment => {
      if (!segment) return;
      const currentStart = parseFloat(segment.start) || 0;
      const currentEnd = parseFloat(segment.end) || currentStart;
      const text = normalizeText(segment.text || segment.originalText);

      const last = merged[merged.length - 1];
      if (last && last.speaker === segment.speaker) {
        const lastEnd = parseFloat(last.end) || parseFloat(last.start) || 0;
        const gap = currentStart - lastEnd;

        if (gap <= MAX_GAP) {
          // Extend timing
          last.end = Math.max(lastEnd, currentEnd);
          // Merge text
          const combinedText = [normalizeText(last.text), text]
            .filter(Boolean)
            .join(gap > 0.15 ? ' ' : ' ')
            .replace(/\s+/g, ' ')
            .trim();
          if (combinedText) {
            last.text = combinedText;
          }
          // Merge metadata
          if (segment.words && segment.words.length) {
            last.words = [...(last.words || []), ...segment.words];
          }
          if (segment.pauses && segment.pauses.length) {
            last.pauses = [...(last.pauses || []), ...segment.pauses];
          }
          if (!last.role && segment.role) {
            last.role = segment.role;
          }
          if (!last.source && segment.source) {
            last.source = segment.source;
          }
          if (segment.overlap) {
            last.overlap = last.overlap || segment.overlap;
          }
          if (!last.confidence && segment.confidence) {
            last.confidence = segment.confidence;
          }
          last.mergedSegmentsCount = (last.mergedSegmentsCount || 1) + 1;
          return;
        }
      }

      // Push copy to avoid mutating original object references
      merged.push({
        ...segment,
        words: segment.words ? [...segment.words] : (segment.words === undefined ? undefined : []),
        pauses: segment.pauses ? [...segment.pauses] : (segment.pauses === undefined ? undefined : []),
        mergedSegmentsCount: 1
      });
    });

    if (merged.length !== segments.length) {
      console.log(`🔗 Merged ${segments.length - merged.length} adjacent segments for speaker continuity`);
    }

    return merged;
  },

  /**
   * Full normalization pipeline for displaying or exporting segments.
   * Keeps logic centralized so UI, exports, and downstream merges stay in sync.
   */
  prepareSegmentsForDisplay(segments, options = {}) {
    if (!Array.isArray(segments) || segments.length === 0) {
      return [];
    }
    const maxGap = typeof options.maxGapSeconds === 'number' ? options.maxGapSeconds : undefined;
    let normalized = this.deduplicateSegments(segments);
    normalized = this.mergeConsecutiveSpeakerSegments(normalized, maxGap);
    return normalized;
  },

  /**
   * Align speaker replicas chronologically for side-by-side comparison tables.
   * Consecutive segments from the same speaker (before the other speaker replies)
   * are merged into a single turn to keep question/answer pairs on one row.
   * Returns rows [{ speaker1Segment, speaker1Indices, speaker2Segment, speaker2Indices }]
   */
  alignSpeakerSegmentsForTable(speaker1Segments, speaker2Segments, options = {}) {
    const rows = [];
    const list1 = Array.isArray(speaker1Segments)
      ? [...speaker1Segments].sort((a, b) => (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0))
      : [];
    const list2 = Array.isArray(speaker2Segments)
      ? [...speaker2Segments].sort((a, b) => (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0))
      : [];

    const maxSameSpeakerGap = typeof options.maxSameSpeakerGapSeconds === 'number'
      ? options.maxSameSpeakerGapSeconds
      : 3.0;

    const getStart = (segment) => parseFloat(segment?.start) || 0;
    const getEnd = (segment) => parseFloat(segment?.end) || getStart(segment);

    const cloneSegment = (segment) => ({
      ...segment,
      text: (segment?.text || '').trim(),
      words: Array.isArray(segment?.words) ? [...segment.words] : [],
      pauses: Array.isArray(segment?.pauses) ? [...segment.pauses] : []
    });

    const mergeSegmentContent = (target, addition) => {
      if (!addition) return target;
      target.end = Math.max(getEnd(target), getEnd(addition));
      target.text = [target.text, addition.text].filter(Boolean).join(' ').replace(/\s+/g, ' ').trim();
      if (addition.words && addition.words.length) {
        target.words = [...(target.words || []), ...addition.words];
      }
      if (addition.pauses && addition.pauses.length) {
        target.pauses = [...(target.pauses || []), ...addition.pauses];
      }
      target.confidence = Math.max(
        typeof target.confidence === 'number' ? target.confidence : 0,
        typeof addition.confidence === 'number' ? addition.confidence : 0
      );
      target.role = target.role || addition.role || null;
      target.source = target.source || addition.source || null;
      target.overlap = target.overlap || addition.overlap || false;
      return target;
    };

    const consumeSpeakerRun = (segments, pointer, stopBeforeTime = Infinity) => {
      if (pointer >= segments.length) return null;
      let consumed = 0;
      let current = cloneSegment(segments[pointer]);
      const indices = [pointer];
      let lastEnd = getEnd(current);
      consumed++;
      let idx = pointer + 1;
      while (idx < segments.length) {
        const seg = segments[idx];
        const segStart = getStart(seg);
        if (segStart >= stopBeforeTime) break;
        if (segStart - lastEnd > maxSameSpeakerGap) break;
        current = mergeSegmentContent(current, seg);
        lastEnd = getEnd(current);
        indices.push(idx);
        consumed++;
        idx++;
      }
      return { segment: current, consumedCount: consumed, indices };
    };

    let i = 0;
    let j = 0;

    while (i < list1.length || j < list2.length) {
      const nextSeg1 = list1[i] || null;
      const nextSeg2 = list2[j] || null;
      if (!nextSeg1 && !nextSeg2) break;

      const speaker1StartsFirst = !nextSeg2 || (nextSeg1 && getStart(nextSeg1) <= getStart(nextSeg2));
      const row = {
        speaker1Segment: null,
        speaker1Index: null,
        speaker1Indices: [],
        speaker2Segment: null,
        speaker2Index: null,
        speaker2Indices: []
      };

      if (speaker1StartsFirst && nextSeg1) {
        const stopBefore = nextSeg2 ? getStart(nextSeg2) : Infinity;
        const run = consumeSpeakerRun(list1, i, stopBefore);
        if (run) {
          row.speaker1Segment = run.segment;
          row.speaker1Indices = run.indices;
          row.speaker1Index = run.indices[0];
          i += run.consumedCount;
        }
      } else if (!speaker1StartsFirst && nextSeg2) {
        const stopBefore = nextSeg1 ? getStart(nextSeg1) : Infinity;
        const run = consumeSpeakerRun(list2, j, stopBefore);
        if (run) {
          row.speaker2Segment = run.segment;
          row.speaker2Indices = run.indices;
          row.speaker2Index = run.indices[0];
          j += run.consumedCount;
        }
      }

      if (speaker1StartsFirst) {
        if (j < list2.length) {
          const nextPrimaryStart = i < list1.length ? getStart(list1[i]) : Infinity;
          const run = consumeSpeakerRun(list2, j, nextPrimaryStart);
          if (run) {
            row.speaker2Segment = run.segment;
            row.speaker2Indices = run.indices;
            row.speaker2Index = run.indices[0];
            j += run.consumedCount;
          }
        }
      } else {
        if (i < list1.length) {
          const nextPrimaryStart = j < list2.length ? getStart(list2[j]) : Infinity;
          const run = consumeSpeakerRun(list1, i, nextPrimaryStart);
          if (run) {
            row.speaker1Segment = run.segment;
            row.speaker1Indices = run.indices;
            row.speaker1Index = run.indices[0];
            i += run.consumedCount;
          }
        }
      }

      rows.push(row);
    }

    return rows;
  },

  getTextServiceResult(recording) {
    if (!recording || !recording.results) return null;
    // Prioritize overlap-corrected results (they contain all replicas from separated tracks)
    let result = null;
    if (recording.results['overlap-corrected']) {
      result = recording.results['overlap-corrected'];
    } else if (recording.results[TEXT_SERVICE_KEY]) {
      result = recording.results[TEXT_SERVICE_KEY];
    } else {
      for (const legacyKey of LEGACY_TEXT_SERVICE_KEYS) {
        if (recording.results[legacyKey]) {
          result = recording.results[legacyKey];
          break;
        }
      }
    }
    
    // Check for Azure native format
    if (!result) {
      const azureResult = recording.results.azure;
      if (azureResult && azureResult.rawData) {
        // Parse Azure native format to table-compatible format
        const parsedSegments = this.parseAzureSegments(azureResult.rawData);
        result = {
          ...azureResult,
          segments: parsedSegments,
          _isAzureNative: true
        };
      }
    }
    
    // Fallback to speechmatics (for backward compatibility with Azure results stored in speechmatics key)
    if (!result) {
      const speechmaticsResult = recording.results.speechmatics;
      if (speechmaticsResult && speechmaticsResult.rawData && speechmaticsResult.engine && speechmaticsResult.engine.includes('azure')) {
        // This is Azure result stored in speechmatics key (legacy format)
        const parsedSegments = this.parseAzureSegments(speechmaticsResult.rawData);
        result = {
          ...speechmaticsResult,
          segments: parsedSegments,
          _isAzureNative: true
        };
      }
    }
    
    // Apply full normalization pipeline to segments before returning
    if (result && result.segments && Array.isArray(result.segments)) {
      const normalizedSegments = this.prepareSegmentsForDisplay(result.segments);
      result = {
        ...result,
        segments: normalizedSegments
      };
    }
    
    return result;
  },

  getRecordingsForExport() {
    if (Array.isArray(this.recordings) && this.recordings.length > 0) {
      return this.recordings;
    }

    const hasResults = this.testResults && Object.keys(this.testResults).length > 0;
    if (!hasResults) return [];

    return [{
      id: `standalone_${Date.now().toString(16)}_${Math.random().toString(16).slice(2)}`,
      name: this.uploadedFile?.name || 'single_recording',
      file: this.uploadedFile || null,
      size: this.uploadedFile?.size || null,
      duration: this.fileDuration || this.audioDuration || 0,
      language: document.getElementById('audioLanguageSelect')?.value 
        || document.getElementById('combinedLanguageSelect')?.value 
        || document.getElementById('overlapLanguageSelect')?.value 
        || 'ar',
      speakerCount: document.getElementById('audioSpeakerCount')?.value 
        || document.getElementById('combinedSpeakerCount')?.value 
        || document.getElementById('overlapSpeakerCount')?.value 
        || '',
      status: 'completed',
      results: this.testResults,
      translationState: this.translationState
    }];
  },

  buildRecordingExportSnapshot(recording) {
    if (!recording) return null;
    const translationState = recording.translationState || this.createTranslationState();
    
    // Extract audio URLs
    const audioUrl = recording.url || recording.file?.name || null;
    const audioFile = recording.file ? {
      name: recording.file.name,
      size: recording.file.size,
      type: recording.file.type
    } : null;
    
    // Extract voice tracks with their URLs and transcriptions
    const voiceTracks = (recording.overlapMetadata?.voiceTracks || []).map(track => ({
      speaker: track.speaker,
      url: track.url || null,
      downloadUrl: track.downloadUrl || null,
      audioUrl: track.audioUrl || null,
      local_path: track.local_path || null,
      transcription: track.transcription || null,
      transcriptText: track.transcriptText || null,
      roleAnalysis: track.roleAnalysis || null,
      error: track.error || null
    }));
    
    // Extract separated speakers (from separation step)
    const separatedSpeakers = (recording.overlapMetadata?.separation?.speakers || []).map(speaker => ({
      name: speaker.name,
      url: speaker.url || null,
      downloadUrl: speaker.downloadUrl || null,
      local_path: speaker.local_path || null
    }));
    
    // Build snapshot with all metadata
    const snapshot = {
      id: recording.id,
      name: recording.name,
      fileName: recording.file?.name || recording.name || null,
      size: recording.size ?? recording.file?.size ?? null,
      duration: recording.duration || 0,
      language: recording.language || 'ar',
      speakerCount: recording.speakerCount ?? '',
      status: recording.status || 'pending',
      addedAt: recording.addedAt || recording.createdAt || recording.importedAt || null,
      translationState: {
        currentLanguage: translationState.currentLanguage || 'original',
        lastError: translationState.lastError || null
      },
      results: this.cloneResults(recording.results || {}),
      aggregated: recording.aggregated || {},
      servicesTested: Object.keys(recording.results || {}),
      // Audio URLs for playback
      audio: {
        url: audioUrl,
        file: audioFile
      },
      // Voice tracks with their audio URLs and transcriptions
      voiceTracks: voiceTracks.length > 0 ? voiceTracks : undefined,
      // Separated speakers audio files
      separatedSpeakers: separatedSpeakers.length > 0 ? separatedSpeakers : undefined,
      // Overlap metadata (includes steps, diagnostics, etc.)
      overlapMetadata: recording.overlapMetadata ? {
        primaryDiarization: recording.overlapMetadata.primaryDiarization || null,
        voiceTracks: voiceTracks.length > 0 ? voiceTracks : (recording.overlapMetadata.voiceTracks || []), // Preserve voice tracks in overlapMetadata
        separation: recording.overlapMetadata.separation ? {
          engine: recording.overlapMetadata.separation.engine,
          speakers: recording.overlapMetadata.separation.speakers?.map(s => ({
            name: s.name,
            url: s.url || null,
            downloadUrl: s.downloadUrl || null,
            local_path: s.local_path || null
          })) || []
        } : null,
        steps: recording.overlapMetadata.steps || null,
        diagnostics: recording.overlapMetadata.diagnostics || null
      } : undefined
    };
    
    // Remove undefined fields to keep JSON clean
    Object.keys(snapshot).forEach(key => {
      if (snapshot[key] === undefined) {
        delete snapshot[key];
      }
    });
    
    return snapshot;
  },

  getFirstSuccessfulRecordingResult(recording) {
    if (!recording || !recording.results) return null;
    return Object.values(recording.results).find(result =>
      result && result.success && Array.isArray(result.segments) && result.segments.length > 0
    ) || null;
  },

  getFirstSuccessfulTestResult() {
    return Object.values(this.testResults || {}).find(result =>
      result && result.success && Array.isArray(result.segments) && result.segments.length > 0
    ) || null;
  },

  normalizeSpeakerLabelForFix(rawSpeaker, fallbackIndex = 0) {
    if (typeof rawSpeaker === 'number' && Number.isFinite(rawSpeaker)) {
      return `SPEAKER_${rawSpeaker.toString().padStart(2, '0')}`;
    }

    if (typeof rawSpeaker === 'string') {
      const trimmed = rawSpeaker.trim().toUpperCase();
      if (/^SPEAKER_\d+$/.test(trimmed)) {
        return trimmed;
      }
      const digits = trimmed.match(/(\d+)/);
      if (digits) {
        return `SPEAKER_${digits[1].padStart(2, '0')}`;
      }
      if (trimmed === 'A') return 'SPEAKER_00';
      if (trimmed === 'B') return 'SPEAKER_01';
    }

    return `SPEAKER_${(fallbackIndex % 10).toString().padStart(2, '0')}`;
  },

  parseSegmentTime(value) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    const parsed = parseFloat(value);
    return Number.isFinite(parsed) ? parsed : 0;
  },

  normalizeSegmentsForFix(segments) {
    return segments
      .filter(segment => segment && (segment.text || segment.originalText))
      .map((segment, index) => {
        const start = this.parseSegmentTime(segment.start);
        const end = Math.max(start, this.parseSegmentTime(segment.end));
        return {
          speaker: this.normalizeSpeakerLabelForFix(segment.speaker, index),
          text: (segment.text || segment.originalText || '').trim(),
          start,
          end,
          words: Array.isArray(segment.words) ? segment.words.map(word => ({
            word: word.word || word.text || '',
            start: this.parseSegmentTime(word.start ?? word.start_time ?? start),
            end: this.parseSegmentTime(word.end ?? word.end_time ?? word.start ?? start),
            speaker: typeof word.speaker === 'number' ? word.speaker : undefined,
            confidence: typeof word.confidence === 'number' ? word.confidence : 1
          })) : []
        };
      })
      .filter(segment => segment.text.length > 0)
      // CRITICAL: Sort by timestamp before grouping to ensure correct chronological order
      .sort((a, b) => {
        const startDiff = (a.start || 0) - (b.start || 0);
        if (startDiff !== 0) return startDiff;
        return (a.end || 0) - (b.end || 0);
      });
  },

  collectSegmentsForFix(recording) {
    const textResult = this.getTextServiceResult(recording);
    if (textResult?.segments?.length) {
      return this.normalizeSegmentsForFix(textResult.segments);
    }

    const fallbackRecordingResult = this.getFirstSuccessfulRecordingResult(recording);
    if (fallbackRecordingResult?.segments?.length) {
      return this.normalizeSegmentsForFix(fallbackRecordingResult.segments);
    }

    const testResult = this.getFirstSuccessfulTestResult();
    if (testResult?.segments?.length) {
      return this.normalizeSegmentsForFix(testResult.segments);
    }

    return null;
  },

  buildRecordingSnapshotForFixes(recording) {
    if (!recording) return null;
    const segments = this.collectSegmentsForFix(recording);
    if (!segments || segments.length === 0) return null;

    const sourceResult = this.getTextServiceResult(recording)
      || this.getFirstSuccessfulRecordingResult(recording)
      || this.getFirstSuccessfulTestResult()
      || {};

    const uniqueSpeakers = new Set(segments.map(seg => seg.speaker));
    const durationFromSegments = segments.reduce((max, seg) => Math.max(max, seg.end || 0), 0);

    return {
      id: recording.id,
      name: recording.name,
      fileName: recording.fileName || recording.name || 'transcript.txt',
      size: recording.size ?? 0,
      duration: recording.duration || sourceResult.rawData?.duration || durationFromSegments || 0,
      language: recording.language || sourceResult.rawData?.language || 'ar',
      speakerCount: recording.speakerCount || `${uniqueSpeakers.size || 2}`,
      status: 'completed',
      addedAt: recording.addedAt || null,
      translationState: { currentLanguage: 'original', lastError: null },
      results: {
        [TEXT_SERVICE_KEY]: {
          success: true,
          serviceName: TEXT_SERVICE_NAME,
          processingTime: sourceResult.processingTime || 0,
          speedFactor: sourceResult.speedFactor || 0,
          speakerCount: sourceResult.speakerCount || uniqueSpeakers.size || 0,
          cost: typeof sourceResult.cost === 'string' ? sourceResult.cost : '0.0000',
          segments,
          rawData: {
            duration: recording.duration || sourceResult.rawData?.duration || durationFromSegments || 0,
            language: recording.language || sourceResult.rawData?.language || 'ar',
            source: sourceResult.rawData?.source || 'text'
          }
        }
      },
      aggregated: {},
      servicesTested: [TEXT_SERVICE_KEY]
    };
  },

  buildRecordingTextForFix(recording, index) {
    if (!recording) return null;
    const segments = this.collectSegmentsForFix(recording);
    if (!segments || segments.length === 0) return null;

    const speakerSet = new Set(segments.map(seg => seg.speaker));
    const lines = segments.map(segment => {
      const start = this.formatTime(Math.max(0, Math.round(segment.start || 0)));
      const end = this.formatTime(Math.max(0, Math.round(segment.end ?? segment.start ?? 0)));
      const speakerLabel = segment.role ? (segment.role === 'operator' ? 'Agent' : 'Client') : segment.speaker;
      const roleLabel = segment.role ? ` [${segment.role === 'operator' ? 'Agent' : 'Client'}]` : '';
      const overlapLabel = segment.overlap ? ' [⚠️ Overlap]' : '';
      return `[${start} → ${end}] ${speakerLabel}${roleLabel}${overlapLabel}: ${segment.text}`;
    });

    return [
      `=== Record ${index + 1}: ${recording.name || 'Recording'} ===`,
      `Speakers detected: ${speakerSet.size || 'N/A'}`,
      '',
      ...lines
    ].join('\n');
  },

  exportJSON() {
    const timestamp = new Date().toISOString();
    const recordingsForExport = this.getRecordingsForExport();

    if (recordingsForExport.length === 0) {
      alert('No data to export.');
      return;
    }

    const snapshots = recordingsForExport
      .map(rec => this.buildRecordingExportSnapshot(rec))
      .filter(Boolean);

    const hasActive = snapshots.some(snap => snap.id === this.activeRecordingId);
    const effectiveActiveId = hasActive
      ? this.activeRecordingId
      : snapshots[0]?.id || null;

    const payload = {
      version: '2.0',
      exportedAt: timestamp,
      activeRecordingId: effectiveActiveId,
      recordings: snapshots
    };

    if (snapshots.length === 1) {
      payload.legacyTestResults = snapshots[0].results;
    }

    const dataStr = JSON.stringify(payload, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'diarization_test_results.json';
    link.click();
    URL.revokeObjectURL(url);
  },

  exportCSV() {
    let csv = 'Service,Processing Time (s),Speaker Count,Cost ($),Status\n';
    
    Object.entries(this.testResults).forEach(([serviceId, result]) => {
      if (result.success) {
        csv += `${result.serviceName},${result.processingTime},${result.speakerCount},${result.cost},Success\n`;
      } else {
        csv += `${result.serviceName},—,—,—,Error: ${result.error}\n`;
      }
    });

    const dataBlob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'diarization_test_results.csv';
    link.click();
    URL.revokeObjectURL(url);
  },

  exportForAIAnalysis() {
    const timestamp = new Date().toISOString();
    const recordingsForExport = this.getRecordingsForExport();

    if (recordingsForExport.length === 0) {
      alert('No data to export.');
      return;
    }

    const snapshots = recordingsForExport
      .map(rec => this.buildRecordingExportSnapshot(rec))
      .filter(Boolean);

    if (snapshots.length === 0) {
      alert('Nothing to export.');
      return;
    }

    const hasActive = snapshots.some(snap => snap.id === this.activeRecordingId);
    const effectiveActiveId = hasActive
      ? this.activeRecordingId
      : snapshots[0]?.id || null;

    const sections = snapshots.map((snapshot, index) =>
      this.buildAISummarySection(snapshot, index)
    );

    const header = `AI SUMMARY GENERATED AT ${timestamp}`;
    const body = sections.filter(Boolean).join('\n\n\n');
    const fileContents = `${header}\n\n\n${body}`.trimEnd() + '\n';

    const blob = new Blob([fileContents], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const safeTimestamp = timestamp.replace(/[:.]/g, '-');
    link.download = `ai_summary_${safeTimestamp}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  },

  getDialogueExportPayload() {
    if (!Array.isArray(this.recordings) || this.recordings.length === 0) {
      return { error: 'No recordings available for export.' };
    }

    const recording = this.getActiveRecording() || this.recordings[0];
    if (!recording) {
      return { error: 'No active recording selected.' };
    }

    const segments = this.collectSegmentsForFix(recording);
    if (!segments || segments.length === 0) {
      return { error: 'No transcript data available for dialogue export.' };
    }

    const dialogueLines = this.buildDialogueLinesForExport(segments);
    if (dialogueLines.length === 0) {
      return { error: 'Unable to build dialogue text for export.' };
    }

    const recordingName = recording.name || recording.fileName || 'Recording';
    const now = new Date();
    const headerLines = [
      `Dialogue Export — ${recordingName}`,
      `Generated: ${now.toLocaleString()}`
    ];

    const text = [...headerLines, '', ...dialogueLines].join('\n').trimEnd();
    return {
      text,
      timestampIso: now.toISOString()
    };
  },

  exportDialogueToText() {
    const payload = this.getDialogueExportPayload();
    if (!payload || payload.error) {
      alert(payload?.error || 'Unable to export dialogue.');
      return;
    }

    const fileContents = `${payload.text}\n`;
    const blob = new Blob([fileContents], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const safeTimestamp = payload.timestampIso.replace(/[:.]/g, '-');
    link.download = `dialogue_${safeTimestamp}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  },

  copyDialogueToClipboard() {
    const payload = this.getDialogueExportPayload();
    if (!payload || payload.error) {
      alert(payload?.error || 'Unable to copy dialogue text.');
      return;
    }

    navigator.clipboard.writeText(payload.text).then(() => {
      alert('Dialogue copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy dialogue text:', err);
      alert('Failed to copy dialogue text.');
    });
  },

  getStoredOriginalDialogue() {
    try {
      const raw = localStorage.getItem('diarizationOriginalScript');
      if (!raw) return null;
      const payload = JSON.parse(raw);
      if (!payload || !payload.text) {
        return null;
      }
      return payload;
    } catch (error) {
      console.warn('Failed to load stored original dialogue', error);
      return null;
    }
  },

  async openEfficiencyAnalyzer() {
    const originalPayload = this.getStoredOriginalDialogue();
    if (!originalPayload) {
      alert('Original dialogue not found. Please generate audio via the Audio Generator before running efficiency analysis.');
      return;
    }

    if (!Array.isArray(this.recordings) || this.recordings.length === 0) {
      alert('No diarization recordings available. Run a diarization job first.');
      return;
    }

    const recording = this.getActiveRecording() || this.recordings[0];
    if (!recording) {
      alert('No active recording selected.');
      return;
    }

    const segments = this.collectSegmentsForFix(recording);
    if (!segments || !segments.length) {
      alert('No transcript data available for the selected recording.');
      return;
    }

    const recognizedLines = this.buildDialogueLinesForExport(segments);
    if (!recognizedLines.length) {
      alert('Unable to build recognized dialogue for analysis.');
      return;
    }

    const payload = {
      original: originalPayload,
      recognized: {
        text: recognizedLines.join('\n'),
        recordingId: recording.id,
        recordingName: recording.name || recording.fileName || 'Recording',
        segments: segments.length,
        generatedAt: new Date().toISOString()
      }
    };

    try {
      localStorage.setItem('diarizationAnalyzerPayload', JSON.stringify(payload));
    } catch (error) {
      console.warn('Failed to cache analyzer payload locally', error);
    }

    try {
      const response = await fetch('/api/analyzer-payload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        console.warn('Failed to sync analyzer payload to server:', await response.text());
      }
    } catch (error) {
      console.warn('Analyzer payload sync failed', error);
    }

    window.open('/Features/diarization-efficiency.html#auto', '_blank');
  },

  buildDialogueLinesForExport(segments) {
    if (!Array.isArray(segments) || segments.length === 0) {
      return [];
    }

    // Ensure segments are sorted by timestamp (should already be sorted by normalizeSegmentsForFix, but double-check)
    const sortedSegments = [...segments].sort((a, b) => {
      const startDiff = (a.start || 0) - (b.start || 0);
      if (startDiff !== 0) return startDiff;
      return (a.end || 0) - (b.end || 0);
    });

    // Light de-duplication: collapse near-duplicate segments that occur close in time
    const dedupedSegments = [];
    sortedSegments.forEach((segment) => {
      const rawText = segment?.text || segment?.originalText || '';
      const text = typeof rawText === 'string' ? rawText.trim() : '';
      if (!text) return;

      const normText = text.toLowerCase().replace(/\s+/g, ' ').trim();
      const start = parseFloat(segment.start) || 0;
      const end = parseFloat(segment.end) || start;

      if (!dedupedSegments.length) {
        dedupedSegments.push(segment);
        return;
      }

      const prev = dedupedSegments[dedupedSegments.length - 1];
      const prevRaw = prev?.text || prev?.originalText || '';
      const prevText = typeof prevRaw === 'string' ? prevRaw.trim() : '';
      const prevNorm = prevText.toLowerCase().replace(/\s+/g, ' ').trim();
      const prevStart = parseFloat(prev.start) || 0;
      const prevEnd = parseFloat(prev.end) || prevStart;
      const timeDiff = Math.abs(start - prevStart);

      // Consider texts similar if equal, or one largely contains the other
      const similarText =
        normText === prevNorm ||
        (normText.length > 20 &&
          prevNorm.length > 20 &&
          (normText.includes(prevNorm) || prevNorm.includes(normText)));

      if (similarText && timeDiff < 3.0) {
        // Prefer the longer, more complete text
        if (text.length > prevText.length) {
          dedupedSegments[dedupedSegments.length - 1] = segment;
        }
      } else {
        dedupedSegments.push(segment);
      }
    });

    // IMPORTANT: Re-sort after deduplication to ensure correct chronological order
    // This fixes issues where timestamps get mixed up after deduplication
    const finalSorted = dedupedSegments.sort((a, b) => {
      const startA = parseFloat(a.start) || 0;
      const startB = parseFloat(b.start) || 0;
      if (startA !== startB) return startA - startB;
      const endA = parseFloat(a.end) || startA;
      const endB = parseFloat(b.end) || startB;
      return endA - endB;
    });

    const lines = [];
    const speakerOrder = new Map();
    const speakerRoles = new Map(); // Map to store roles for each speaker
    let currentLabel = null;
    let buffer = [];
    let lastSegmentEnd = 0;

    const flush = () => {
      if (!buffer.length) return;
      const speakerName = this.getDialogueSpeakerDisplayName(currentLabel, speakerOrder, speakerRoles);
      const combined = buffer.join(' ').replace(/\s+/g, ' ').trim();
      if (combined) {
        lines.push(`${speakerName}: ${combined}`);
      }
      buffer = [];
    };

    finalSorted.forEach((segment, index) => {
      const rawText = segment?.text || segment?.originalText || '';
      const text = typeof rawText === 'string' ? rawText.trim() : '';
      if (!text) {
        return;
      }

      const label = segment?.speaker || `speaker_${index}`;
      const segmentStart = parseFloat(segment.start) || 0;
      const segmentEnd = parseFloat(segment.end) || segmentStart;
      
      // Store role for this speaker if available
      if (segment?.role && !speakerRoles.has(label)) {
        speakerRoles.set(label, segment.role);
      }
      
      // If speaker changed OR there's a significant time gap (> 2 seconds), flush buffer
      // This prevents merging segments that are chronologically far apart
      const timeGap = segmentStart - lastSegmentEnd;
      if (label !== currentLabel || timeGap > 2.0) {
        flush();
        currentLabel = label;
      }

      buffer.push(text);
      lastSegmentEnd = Math.max(lastSegmentEnd, segmentEnd);
    });

    flush();
    return lines;
  },

  getDialogueSpeakerDisplayName(label, speakerOrder, speakerRoles) {
    const key = (label && label.toString().trim()) || `speaker_${speakerOrder.size}`;
    
    // Check if we have a role for this speaker
    const role = speakerRoles?.get(label);
    if (role) {
      const roleLabel = role === 'operator' ? 'Agent' : 'Client';
      return roleLabel;
    }
    
    // Fallback to Speaker X if no role
    if (!speakerOrder.has(key)) {
      speakerOrder.set(key, speakerOrder.size + 1);
    }
    const number = speakerOrder.get(key);
    return `Speaker ${number}`;
  },

  copyCombinedSeparatedTranscript() {
    const recording = this.getActiveRecording() || this.recordings[0];
    if (!recording) {
      alert('No recording selected.');
      return;
    }

    // Helper function to get voice tracks from multiple possible locations
    const getVoiceTracksFromRecording = (rec) => {
      let tracks = rec.overlapMetadata?.voiceTracks || [];
      if (tracks.length === 0 && rec.voiceTracks && Array.isArray(rec.voiceTracks)) {
        tracks = rec.voiceTracks;
      }
      return tracks;
    };
    
    const voiceTracks = getVoiceTracksFromRecording(recording);
    if (!voiceTracks.length) {
      alert('No separated voice tracks available. Run overlap diarization first.');
      return;
    }

    const speakerKeys = ['SPEAKER_00', 'SPEAKER_01'];
    const normalizedSegments = this.collectSegmentsForFix(recording) || [];
    const findTrack = (speakerKey) => {
      const exact = voiceTracks.find(vt => vt.speaker === speakerKey);
      if (exact) return exact;
      const normalizedSuffix = speakerKey.replace('SPEAKER_', '');
      return voiceTracks.find(vt => vt.speaker?.endsWith(normalizedSuffix)) || null;
    };

    const speakerPayloads = speakerKeys.map(speakerKey => {
      const track = findTrack(speakerKey);
      const transcriptPayload = this.buildVoiceTrackTranscriptPayload(track, speakerKey, normalizedSegments);
      if (!transcriptPayload) {
        return null;
      }
      return {
        speaker: speakerKey,
        role: track?.roleAnalysis?.role || null,
        roleConfidence: typeof track?.roleAnalysis?.confidence === 'number'
          ? track.roleAnalysis.confidence
          : null,
        transcript: transcriptPayload
      };
    }).filter(Boolean);

    if (!speakerPayloads.length) {
      alert('Unable to build combined transcript JSON. No transcripts available.');
      return;
    }

    const combinedPayload = {
      version: '1.0',
      generatedAt: new Date().toISOString(),
      recordingId: recording.id,
      speakers: speakerPayloads
    };

    const combinedText = JSON.stringify(combinedPayload, null, 2);

    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(combinedText).then(() => {
        alert('Combined speaker transcript JSON copied to clipboard.');
      }).catch(err => {
        console.error('Clipboard copy failed:', err);
        this.downloadJSONFallback(combinedText, 'combined_voice_tracks.json');
      });
    } else {
      this.downloadJSONFallback(combinedText, 'combined_voice_tracks.json');
    }
  },

  copyInitialDiarizationJSON() {
    const recording = this.getActiveRecording() || this.recordings[0];
    if (!recording) {
      alert('No recording data available.');
      return;
    }

    const speechmaticsResult = recording.results?.speechmatics;
    if (!speechmaticsResult) {
      alert('Initial diarization JSON is not available for this recording.');
      return;
    }

    const payload = {
      version: '2.0',
      exportedAt: new Date().toISOString(),
      activeRecordingId: recording.id,
      recordings: [
        {
          id: recording.id,
          name: recording.name || 'Recording',
          fileName: recording.fileName || 'audio.wav',
          size: recording.size ?? null,
          duration: recording.duration ?? 0,
          language: recording.language || 'ar',
          speakerCount: recording.speakerCount || '0',
          status: 'completed',
          addedAt: recording.addedAt ?? null,
          translationState: recording.translationState || this.createTranslationState(),
          results: {
            speechmatics: speechmaticsResult
          },
          aggregated: recording.aggregated || {},
          servicesTested: recording.servicesTested || ['speechmatics']
        }
      ]
    };

    const jsonText = JSON.stringify(payload, null, 2);
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(jsonText)
        .then(() => alert('Initial diarization JSON copied to clipboard.'))
        .catch(err => {
          console.error('Clipboard copy failed:', err);
          this.downloadJSONFallback(jsonText, 'initial_diarization.json');
        });
    } else {
      this.downloadJSONFallback(jsonText, 'initial_diarization.json');
    }
  },

  downloadJSONFallback(content, filename) {
    try {
      // Ensure content is a string
      let contentStr = content;
      if (typeof content !== 'string') {
        if (content instanceof Blob) {
          // If it's already a Blob, use it directly
          const url = URL.createObjectURL(content);
          const link = document.createElement('a');
          link.href = url;
          link.download = filename || `export_${Date.now()}.json`;
          link.click();
          URL.revokeObjectURL(url);
          alert('Clipboard unavailable. JSON file downloaded instead.');
          return;
        } else {
          // Try to stringify if it's an object
          contentStr = JSON.stringify(content, null, 2);
        }
      }
      
      // Validate that we have valid content
      if (!contentStr || (typeof contentStr !== 'string')) {
        throw new Error('Invalid content type for download');
      }
      
      const blob = new Blob([contentStr], { type: 'application/json' });
      if (!blob || blob.size === 0) {
        throw new Error('Failed to create blob from content');
      }
      
      const url = URL.createObjectURL(blob);
      if (!url) {
        throw new Error('Failed to create object URL');
      }
      
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `export_${Date.now()}.json`;
      link.click();
      URL.revokeObjectURL(url);
      alert('Clipboard unavailable. JSON file downloaded instead.');
    } catch (error) {
      console.error('Failed to download JSON fallback:', error);
      alert('Failed to copy initial JSON: ' + error.message);
    }
  },

  safeCloneTranscription(transcription) {
    if (!transcription) return null;
    try {
      if (typeof structuredClone === 'function') {
        return structuredClone(transcription);
      }
      return JSON.parse(JSON.stringify(transcription));
    } catch (error) {
      console.error('Failed to clone transcription:', error);
      return null;
    }
  },

  countTranscriptionSegments(transcription) {
    if (!transcription) return 0;
    const recordings = Array.isArray(transcription.recordings) ? transcription.recordings : [];
    return recordings.reduce((recordingTotal, recording) => {
      if (!recording?.results) {
        return recordingTotal;
      }
      const resultSegments = Object.values(recording.results).reduce((resultTotal, result) => {
        if (result && Array.isArray(result.segments)) {
          return resultTotal + result.segments.length;
        }
        return resultTotal;
      }, 0);
      return recordingTotal + resultSegments;
    }, 0);
  },

  sanitizeTranscriptionForSpeaker(transcription, speakerKey) {
    if (!transcription || !speakerKey) return null;
    const targetSpeaker = this.normalizeSpeakerLabelForFix(speakerKey);
    const cloned = this.safeCloneTranscription(transcription);
    if (!cloned) return null;

    const matchesTarget = (value) => {
      if (!value) return false;
      const normalized = this.normalizeSpeakerLabelForFix(value);
      return normalized === targetSpeaker;
    };

    const filterSegmentsArray = (segments = []) => {
      if (!Array.isArray(segments)) return [];
      return segments
        .filter(segment => {
          if (!segment) return false;
          if (!segment.speaker) return true;
          return matchesTarget(segment.speaker);
        })
        .map(segment => ({
          ...segment,
          speaker: targetSpeaker
        }));
    };

    const filterRawDataSegments = (rawData) => {
      if (!rawData || !Array.isArray(rawData.segments)) return;
      const speakerMap = rawData.speakerMap || {};
      rawData.segments = rawData.segments.filter(rawSegment => {
        if (!rawSegment) return false;
        const mappedSpeaker = speakerMap[rawSegment.speakerId]
          || rawSegment.speaker
          || rawSegment.speaker_label
          || rawSegment.speakerName
          || rawSegment.speakerTag;
        if (!mappedSpeaker) return false;
        return matchesTarget(mappedSpeaker);
      });
    };

    if (Array.isArray(cloned.speakers)) {
      cloned.speakers = cloned.speakers.filter(entry => matchesTarget(entry?.speaker));
    }

    if (Array.isArray(cloned.recordings)) {
      cloned.recordings = cloned.recordings.map(recording => {
        if (!recording) return recording;
        const updatedRecording = {
          ...recording,
          speakerCount: '1'
        };
        if (!updatedRecording.results) {
          return updatedRecording;
        }

        ['azure', 'speechmatics', 'overlap-corrected'].forEach(resultKey => {
          const result = updatedRecording.results[resultKey];
          if (!result) return;
          updatedRecording.results[resultKey] = {
            ...result,
            speakerCount: '1',
            segments: filterSegmentsArray(result.segments)
          };
          if (result.rawData) {
            updatedRecording.results[resultKey].rawData = {
              ...result.rawData
            };
            filterRawDataSegments(updatedRecording.results[resultKey].rawData);
          }
        });

        return updatedRecording;
      });
    }

    const remainingSegments = this.countTranscriptionSegments(cloned);
    if (remainingSegments === 0) {
      return null;
    }

    return cloned;
  },

  buildVoiceTrackTranscriptPayload(track, speakerKey, normalizedSegments = []) {
    // First try: use transcription object
    if (track?.transcription) {
      const sanitized = this.sanitizeTranscriptionForSpeaker(track.transcription, speakerKey);
      if (sanitized && sanitized.segments && sanitized.segments.length > 0) {
        return sanitized;
      }
    }

    // Second try: use transcriptText directly
    if (track?.transcriptText && track.transcriptText.trim().length > 0) {
      // Extract segments from transcription if available
      let segments = [];
      if (track.transcription?.recordings?.[0]?.results?.speechmatics?.segments) {
        segments = track.transcription.recordings[0].results.speechmatics.segments
          .filter(seg => seg.text && seg.text.trim())
          .map(seg => ({
            speaker: speakerKey,
            text: seg.text,
            start: seg.start,
            end: seg.end,
            words: seg.words || [],
            role: track.roleAnalysis?.role || null,
            source: 'voice-track'
          }));
      } else {
        // If no segments, create one segment from transcriptText
        segments = [{
          speaker: speakerKey,
          text: track.transcriptText.trim(),
          start: 0,
          end: 0,
          words: [],
          role: track.roleAnalysis?.role || null,
          source: 'voice-track-text'
        }];
      }

      if (segments.length > 0) {
        return {
          speaker: speakerKey,
          source: 'voice-track-transcript',
          generatedAt: new Date().toISOString(),
          transcriptText: track.transcriptText.trim(),
          segments: segments,
          role: track.roleAnalysis?.role || null
        };
      }
    }

    // Third try: use normalized segments as fallback
    if (Array.isArray(normalizedSegments) && normalizedSegments.length > 0) {
      const fallbackSegments = normalizedSegments
        .filter(segment => (segment.speaker || '').toUpperCase() === (speakerKey || '').toUpperCase())
        .map(segment => ({
          speaker: segment.speaker,
          text: segment.text,
          start: segment.start,
          end: segment.end,
          role: segment.role || null,
          overlap: !!segment.overlap,
          source: 'fallback'
        }));

      if (fallbackSegments.length > 0) {
        return {
          speaker: speakerKey,
          source: 'fallback-segments',
          generatedAt: new Date().toISOString(),
          segments: fallbackSegments
        };
      }
    }

    return null;
  },

  buildAISummarySection(recordingSnapshot, index) {
    if (!recordingSnapshot) return '';

    const recordingCopy = this.safeClone(recordingSnapshot);
    const normalizedRecording = this.normalizeTextServiceResult(recordingCopy);
    const transcriptText = this.buildTranscriptTextForRecording(normalizedRecording);
    const displayName = normalizedRecording.fileName || normalizedRecording.name || `Recording ${index + 1}`;
    const safeTranscript = transcriptText || '[No transcript data available]';

    return [
      `=== Record ${index + 1}: ${displayName} ===`,
      '',
      safeTranscript
    ].join('\n\n');
  },

  buildTranscriptTextForRecording(recording) {
    if (!recording) return '';
    const segments = this.extractSegmentsForAISummary(recording);
    let transcriptText = this.buildReadableTranscriptFromSegments(segments);

    if (!transcriptText) {
      transcriptText = this.extractFallbackTranscript(recording);
    }

    return transcriptText.trim();
  },

  extractSegmentsForAISummary(recording) {
    if (!recording || !recording.results) return [];

    const textResult = this.getTextServiceResult(recording);
    if (textResult && Array.isArray(textResult.segments) && textResult.segments.length > 0) {
      return textResult.segments;
    }

    for (const result of Object.values(recording.results)) {
      if (result && result.success && Array.isArray(result.segments) && result.segments.length > 0) {
        return result.segments;
      }
    }

    return [];
  },

  buildReadableTranscriptFromSegments(segments) {
    if (!Array.isArray(segments) || segments.length === 0) return '';

    const blocks = [];
    let currentSpeaker = null;
    let buffer = [];

    segments.forEach(segment => {
      const rawText = segment?.text || segment?.originalText || '';
      const text = typeof rawText === 'string' ? rawText.trim() : '';
      if (!text) {
        return;
      }

      const speaker = (segment?.speaker || '').toString();
      if (speaker && speaker !== currentSpeaker) {
        if (buffer.length) {
          blocks.push(buffer.join(' ').replace(/\s+/g, ' ').trim());
          buffer = [];
        }
        currentSpeaker = speaker;
      }

      buffer.push(text);
    });

    if (buffer.length) {
      blocks.push(buffer.join(' ').replace(/\s+/g, ' ').trim());
    }

    const cleaned = blocks.filter(Boolean);
    if (cleaned.length === 0) {
      return segments
        .map(seg => (seg?.text || seg?.originalText || '').trim())
        .filter(Boolean)
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();
    }

    return cleaned.join('\n\n');
  },

  extractFallbackTranscript(recording) {
    if (!recording) return '';

    const candidates = [];
    if (recording.aggregated && typeof recording.aggregated === 'object') {
      Object.values(recording.aggregated).forEach(value => {
        if (typeof value === 'string') {
          candidates.push(value);
        }
      });
    }

    ['transcript', 'text', 'content', 'rawTranscript'].forEach(key => {
      const value = recording[key];
      if (typeof value === 'string') {
        candidates.push(value);
      }
    });

    const winner = candidates
      .map(value => value.trim())
      .find(value => value.length > 0);

    return winner || '';
  },

  safeClone(value) {
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      console.warn('Failed to deep clone value, returning shallow copy', error);
      return { ...value };
    }
  },

  copyResults() {
    let text = 'AUDIO DIARIZATION TEST RESULTS\n';
    text += '================================\n\n';
    
    Object.entries(this.testResults).forEach(([serviceId, result]) => {
      text += `${result.serviceName}\n`;
      if (result.success) {
        text += `  Processing Time: ${result.processingTime}s\n`;
        text += `  Speakers: ${result.speakerCount}\n`;
        text += `  Cost: $${result.cost}\n`;
        text += `  Status: Success\n`;
      } else {
        text += `  Status: Error - ${result.error}\n`;
      }
      text += '\n';
    });

    navigator.clipboard.writeText(text).then(() => {
      alert('Results copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy:', err);
      alert('Failed to copy results.');
    });
  },

  hasSuccessfulSegments() {
    const results = this.testResults || {};
    return Object.values(results).some(result =>
      result && result.success && Array.isArray(result.segments) && result.segments.length > 0
    );
  },

  allSegmentsHaveTranslation(targetLang) {
    const successful = Object.values(this.testResults).filter(result =>
      result && result.success && Array.isArray(result.segments) && result.segments.length > 0
    );
    if (successful.length === 0) return false;
    return successful.every(result =>
      result.segments.every(segment => segment.translations && segment.translations[targetLang])
    );
  },

  updateTranslationControlsState() {
    const hasSegments = this.hasSuccessfulSegments();
    const disablePrimary = !hasSegments || this.translationState.isTranslating;

    const ukBtn = document.getElementById('translateToUkBtn');
    const enBtn = document.getElementById('translateToEnBtn');
    const restoreBtn = document.getElementById('restoreOriginalBtn');
    const translateAllUkBtn = document.getElementById('translateAllUkBtn');
    const translateAllEnBtn = document.getElementById('translateAllEnBtn');
    const replicaUkBtn = document.getElementById('replicaTranslateUkBtn');
    const replicaEnBtn = document.getElementById('replicaTranslateEnBtn');
    const replicaRestoreBtn = document.getElementById('replicaRestoreBtn');

    if (ukBtn) ukBtn.disabled = disablePrimary;
    if (enBtn) enBtn.disabled = disablePrimary;
    if (restoreBtn) {
      restoreBtn.disabled = !hasSegments || this.translationState.isTranslating || this.translationState.currentLanguage === 'original';
    }
    if (replicaUkBtn) replicaUkBtn.disabled = disablePrimary;
    if (replicaEnBtn) replicaEnBtn.disabled = disablePrimary;
    if (replicaRestoreBtn) {
      replicaRestoreBtn.disabled = !hasSegments || this.translationState.isTranslating || this.translationState.currentLanguage === 'original';
    }

    const disableAll = this.recordings.length === 0 || this.translationState.isTranslating;
    if (translateAllUkBtn) translateAllUkBtn.disabled = disableAll;
    if (translateAllEnBtn) translateAllEnBtn.disabled = disableAll;
  },

  setTranslationStatus(message, variant = 'info') {
    const statusEl = document.getElementById('translationStatus');
    const replicaStatusEl = document.getElementById('replicaTranslationStatus');

    const variantClass = {
      info: 'status status-info',
      success: 'status status-success',
      error: 'status status-error',
      warning: 'status status-warning',
      processing: 'status status-processing'
    }[variant] || 'status status-info';

    if (statusEl) {
      statusEl.className = variantClass;
      statusEl.textContent = message;
    }
    if (replicaStatusEl) {
      replicaStatusEl.className = variantClass;
      replicaStatusEl.textContent = message;
    }
  },

  refreshReplicasViewIfActive() {
    const replicasView = document.getElementById('replicasView');
    if (replicasView && replicasView.classList.contains('active')) {
      this.updateReplicasComparison();
    }
  },

  updateTranslationStatusLabel(message) {
    if (message) {
      this.setTranslationStatus(message, this.translationState.isTranslating ? 'processing' : 'info');
      return;
    }

    const lang = this.translationState.currentLanguage;
    if (lang === 'uk') {
      this.setTranslationStatus('Current translation: Ukrainian', 'success');
    } else if (lang === 'en') {
      this.setTranslationStatus('Current translation: English', 'success');
    } else if (lang === 'ar') {
      this.setTranslationStatus('Current translation: Arabic', 'success');
    } else {
      this.setTranslationStatus('Original Arabic transcripts', 'info');
    }
  },

  getTranslationLanguageMeta(targetLang) {
    const map = {
      uk: { label: 'Ukrainian', display: 'Ukrainian', english: 'Ukrainian' },
      en: { label: 'English', display: 'English', english: 'English' }
    };
    return map[targetLang];
  },

  buildTranslationInput() {
    const services = [];
    Object.entries(this.testResults).forEach(([serviceId, result]) => {
      if (!result || !result.success || !Array.isArray(result.segments) || result.segments.length === 0) {
        return;
      }
      services.push({
        serviceId,
        serviceName: result.serviceName,
        segments: result.segments.map((segment, index) => ({
          index,
          speaker: segment.speaker,
          text: segment.originalText || segment.text || '',
          role: segment.role || null, // Preserve role if present
          overlap: segment.overlap !== undefined ? segment.overlap : false // Preserve overlap if present
        }))
      });
    });
    return services;
  },

  async buildTranslationPrompt(meta, services) {
    try {
      const response = await fetch('prompts/translation_prompt_template.txt');
      if (!response.ok) {
        throw new Error('Unable to load translation prompt template');
      }
      let template = await response.text();
      
      // Replace placeholders
      template = template.replace(/{TARGET_LANGUAGE}/g, meta.english);
      template = template.replace(/{INPUT_DATA}/g, JSON.stringify({ services }, null, 2));
      
      return template.trim();
    } catch (error) {
      console.warn('Failed to load translation prompt template, using fallback:', error);
      // Fallback to original hardcoded prompt
      return `
You are a professional Arabic-to-${meta.english} translator. Translate every diarization segment exactly and keep punctuation.
Return ONLY valid JSON using this schema:
{
  "services": [
    {
      "serviceId": "assemblyai",
      "translations": [
        { "segmentIndex": 0, "translatedText": "..." }
      ]
    }
  ]
}

Do not include explanations or Markdown.

Input data (translate each "text" field):
${JSON.stringify({ services }, null, 2)}
`.trim();
    }
  },

  /**
   * Нормалізує та парсить відповідь LLM для перекладу
   * Гарантує стабільний парсинг незалежно від формату відповіді
   */
  parseTranslationResponse(rawContent) {
    if (!rawContent) {
      throw new Error('Empty response from model');
    }

    // Крок 0: Якщо вже об'єкт - спробуємо валідувати напряму
    if (typeof rawContent === 'object' && rawContent !== null) {
      const directValidation = this.parseAndValidateTranslationJson(JSON.stringify(rawContent));
      if (directValidation) {
        return directValidation;
      }
    }

    // Крок 1: Нормалізація вхідних даних
    let normalized = this.normalizeLLMResponse(rawContent);
    
    // Крок 2: Витягнення JSON з різних форматів
    const jsonCandidates = this.extractAllJsonCandidates(normalized);
    
    // Крок 3: Парсинг та валідація
    for (const candidate of jsonCandidates) {
      try {
        const parsed = this.parseAndValidateTranslationJson(candidate);
        if (parsed) {
          return parsed;
        }
      } catch (err) {
        continue;
      }
    }

    // Крок 4: Детальне логування помилки
    console.error('Failed to parse translation JSON', {
      rawType: typeof rawContent,
      rawPreview: String(rawContent).slice(0, 800),
      candidatesCount: jsonCandidates.length,
      candidates: jsonCandidates.slice(0, 3).map(c => c.slice(0, 200))
    });
    throw new Error('Failed to parse JSON response from model. Check translation service settings.');
  },

  /**
   * Нормалізує відповідь LLM до рядка
   */
  normalizeLLMResponse(rawContent) {
    // Якщо вже об'єкт - спочатку перевіряємо, чи це вже правильний формат
    if (typeof rawContent === 'object' && rawContent !== null) {
      // Якщо це масив сервісів - повертаємо як є (буде оброблено в parseAndValidateTranslationJson)
      if (Array.isArray(rawContent) && rawContent.length > 0 && typeof rawContent[0] === 'object') {
        return JSON.stringify(rawContent);
      }
      // Якщо це об'єкт з services
      if (rawContent.services && Array.isArray(rawContent.services)) {
        return JSON.stringify(rawContent.services);
      }
      // Якщо це об'єкт з translations
      if (rawContent.translations) {
        return JSON.stringify([rawContent]);
      }
      // Інші об'єкти - серіалізуємо
      try {
        return JSON.stringify(rawContent);
      } catch (error) {
        return String(rawContent);
      }
    }
    
    if (typeof rawContent !== 'string') {
      return String(rawContent);
    }
    
    return rawContent;
  },

  /**
   * Витягує всі можливі JSON кандидати з тексту
   */
  extractAllJsonCandidates(text) {
    const candidates = [];
    const addUnique = (value) => {
      if (!value) return;
      // Перетворюємо на рядок, якщо потрібно
      if (typeof value !== 'string') {
        value = String(value);
      }
      const trimmed = value.trim();
      if (trimmed && trimmed.length > 2 && !candidates.includes(trimmed)) {
        candidates.push(trimmed);
      }
    };

    // Переконуємося, що text - це рядок
    if (typeof text !== 'string') {
      text = String(text);
    }

    // 1. Видаляємо шум LLM
    let cleaned = this.stripLLMNoise(text);
    addUnique(cleaned);

    // 2. Витягуємо з markdown блоків
    const fencedMatches = cleaned.matchAll(/```(?:json|)?\s*([\s\S]*?)```/gi);
    for (const match of fencedMatches) {
      addUnique(match[1]);
    }

    // 3. Витягуємо з JSON-рядків (екранованих)
    const unescaped = this.safeJsonStringUnescape(cleaned);
    addUnique(unescaped);

    // 4. Витягуємо збалансований JSON
    const balanced = this.extractJsonCandidate(cleaned);
    addUnique(balanced);
    
    const balancedUnescaped = this.extractJsonCandidate(unescaped);
    addUnique(balancedUnescaped);

    // 5. Спроба знайти JSON після видалення всіх markdown блоків
    const noMarkdown = cleaned.replace(/```[\s\S]*?```/g, '').trim();
    if (noMarkdown !== cleaned) {
      addUnique(noMarkdown);
      const balancedNoMarkdown = this.extractJsonCandidate(noMarkdown);
      addUnique(balancedNoMarkdown);
    }

    // 6. Спроба знайти JSON об'єкт або масив в тексті
    const jsonObjectMatch = cleaned.match(/\{[\s\S]*\}/);
    if (jsonObjectMatch) {
      addUnique(jsonObjectMatch[0]);
    }
    
    const jsonArrayMatch = cleaned.match(/\[[\s\S]*\]/);
    if (jsonArrayMatch) {
      addUnique(jsonArrayMatch[0]);
    }

    return candidates;
  },

  /**
   * Парсить та валідує JSON перекладу
   * Повертає нормалізований масив сервісів або null
   */
  parseAndValidateTranslationJson(jsonString) {
    if (!jsonString || typeof jsonString !== 'string') {
      return null;
    }

    let parsed;
    try {
      parsed = JSON.parse(jsonString);
    } catch (err) {
      return null;
    }

    // Валідація структури
    if (Array.isArray(parsed)) {
      // Перевіряємо, чи це масив сервісів
      if (parsed.length > 0 && parsed[0] && typeof parsed[0] === 'object') {
        return parsed;
      }
      return null;
    }

    if (typeof parsed === 'object' && parsed !== null) {
      // Формат: { services: [...] }
      if (parsed.services && Array.isArray(parsed.services)) {
        return parsed.services;
      }
      
      // Формат: { translations: [...] }
      if (parsed.translations && Array.isArray(parsed.translations)) {
        return [{ translations: parsed.translations }];
      }
      
      // Формат: { serviceId: "...", translations: [...] }
      if (parsed.serviceId || parsed.translations) {
        return [parsed];
      }
      
      // Формат: { id: "...", segments: [...] }
      if (parsed.id || parsed.segments) {
        return [parsed];
      }
    }

    return null;
  },

  stripLLMNoise(text) {
    if (!text) return '';
    // Переконуємося, що це рядок
    if (typeof text !== 'string') {
      text = String(text);
    }
    return text
      .replace(/<think>[\s\S]*?<\/think>/gi, '')
      .replace(/<think>[\s\S]*?<\/redacted_reasoning>/gi, '')
      .replace(/^[^\[{]*?(?=\{|\[)/s, '')
      .replace(/(?:Response|Answer|Output):/gi, '')
      .trim();
  },

  extractJsonCandidate(text) {
    if (!text) return '';
    // Переконуємося, що це рядок
    if (typeof text !== 'string') {
      text = String(text);
    }
    const firstBrace = text.indexOf('{');
    const firstBracket = text.indexOf('[');
    let start = -1;
    if (firstBrace === -1) start = firstBracket;
    else if (firstBracket === -1) start = firstBrace;
    else start = Math.min(firstBrace, firstBracket);
    if (start === -1) return text.trim();

    const stack = [];
    const startChar = text[start];
    const pushClosing = (ch) => {
      stack.push(ch);
    };

    const pairs = {
      '{': '}',
      '[': ']'
    };

    if (pairs[startChar]) {
      pushClosing(pairs[startChar]);
    } else {
      return text.trim();
    }

    let inString = false;
    let escapeNext = false;
    for (let i = start + 1; i < text.length; i++) {
      const ch = text[i];

      if (inString) {
        if (escapeNext) {
          escapeNext = false;
          continue;
        }
        if (ch === '\\') {
          escapeNext = true;
          continue;
        }
        if (ch === '"') {
          inString = false;
        }
        continue;
      } else {
        if (ch === '"') {
          inString = true;
          continue;
        }
      }

      if (pairs[ch]) {
        pushClosing(pairs[ch]);
        continue;
      }

      if (stack.length && ch === stack[stack.length - 1]) {
        stack.pop();
        if (stack.length === 0) {
          return text.slice(start, i + 1).trim();
        }
      }
    }

    return text.slice(start).trim();
  },

  safeJsonStringUnescape(text) {
    if (!text) return '';
    let trimmed = text.trim();
    
    // Видаляємо зовнішні лапки (одинарні або подвійні)
    const isWrappedInQuotes = (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
      (trimmed.startsWith('\'') && trimmed.endsWith('\''));
    
    if (isWrappedInQuotes) {
      trimmed = trimmed.slice(1, -1);
    }
    
    // Спроба розпарсити як JSON рядок (екранований)
    try {
      const parsed = JSON.parse(`"${trimmed}"`);
      // Якщо це JSON рядок, спробуємо розпарсити його знову
      if (typeof parsed === 'string') {
        try {
          return JSON.parse(parsed);
        } catch (err) {
          // Якщо не вийшло, повертаємо розпарсений рядок
          return parsed;
        }
      }
      return parsed;
    } catch (error) {
      // Якщо не вдалося розпарсити як екранований рядок, спробуємо як звичайний JSON
      try {
        return JSON.parse(trimmed);
      } catch (err2) {
        // Якщо і це не вийшло, повертаємо оригінальний текст
        return trimmed;
      }
    }
  },

  resolveServiceResult(serviceId) {
    if (!serviceId || !this.testResults) return null;
    const trimmedId = serviceId.trim();
    if (this.testResults[trimmedId]) {
      return { key: trimmedId, result: this.testResults[trimmedId] };
    }
    const lower = trimmedId.toLowerCase();
    const matchKey = Object.keys(this.testResults).find(key => key.toLowerCase() === lower);
    if (matchKey) {
      return { key: matchKey, result: this.testResults[matchKey] };
    }
    return null;
  },

  getFirstSuccessfulServiceResult() {
    if (!this.testResults) return null;
    const entry = Object.entries(this.testResults).find(([_, result]) =>
      result && result.success && Array.isArray(result.segments) && result.segments.length > 0
    );
    return entry ? { key: entry[0], result: entry[1] } : null;
  },

  /**
   * Застосовує результат перекладу до сегментів
   * Підтримує різні формати відповіді LLM
   */
  applyTranslationResult(servicesTranslations, targetLang) {
    if (!Array.isArray(servicesTranslations) || servicesTranslations.length === 0) {
      throw new Error('Empty translation result');
    }

    let updated = 0;
    const errors = [];

    servicesTranslations.forEach((entry, entryIdx) => {
      try {
        // Нормалізація serviceId
        const serviceId = this.normalizeServiceId(entry.serviceId || entry.id || entry.service || '');
        
        // Знаходимо відповідний сервіс
        let resolved = this.resolveServiceResult(serviceId);
        if (!resolved || !resolved.result || !resolved.result.success) {
          // Fallback: використовуємо перший успішний сервіс
          resolved = this.getFirstSuccessfulServiceResult();
        }
        
        if (!resolved) {
          errors.push(`Entry ${entryIdx}: service not found for "${serviceId}"`);
          return;
        }

        // Нормалізація масиву перекладів
        const translations = this.normalizeTranslationsArray(entry);
        if (!Array.isArray(translations) || translations.length === 0) {
          errors.push(`Entry ${entryIdx}: no translations`);
          return;
        }

        const segments = resolved.result.segments || [];
        if (segments.length === 0) {
          errors.push(`Entry ${entryIdx}: no segments in service`);
          return;
        }

        // Застосовуємо переклади
        translations.forEach((item, itemIdx) => {
          try {
            const index = this.normalizeSegmentIndex(item, itemIdx, segments.length);
            const text = this.normalizeTranslationText(item);
            
            if (!text) {
              return; // Пропускаємо порожні переклади
            }

            const segment = segments[index];
            if (!segment) {
              errors.push(`Entry ${entryIdx}, item ${itemIdx}: segment ${index} does not exist`);
              return;
            }

            // Зберігаємо оригінал, якщо ще не збережено
            if (!segment.originalText && segment.text) {
              segment.originalText = segment.text;
            }

            // Зберігаємо переклад
            if (!segment.translations) {
              segment.translations = {};
            }
            segment.translations[targetLang] = text;
            updated += 1;
          } catch (err) {
            errors.push(`Entry ${entryIdx}, item ${itemIdx}: ${err.message}`);
          }
        });
      } catch (err) {
        errors.push(`Entry ${entryIdx}: ${err.message}`);
      }
    });

    if (updated === 0) {
      const serviceNames = Object.keys(this.testResults || {}).filter(key =>
        this.testResults[key]?.success && Array.isArray(this.testResults[key]?.segments) && this.testResults[key].segments.length > 0
      ).join(', ') || 'no active services with segments';
      
      const errorDetails = errors.length > 0 ? `\nDetails: ${errors.slice(0, 5).join('; ')}` : '';
      throw new Error(`Model did not return translated segments or failed to apply them (available services: ${serviceNames})${errorDetails}`);
    }

    // Логуємо попередження, якщо є помилки, але є й успішні оновлення
    if (errors.length > 0) {
      console.warn('Translation applied with some errors:', errors.slice(0, 10));
    }

    this.applyCachedTranslation(targetLang);
  },

  /**
   * Нормалізує serviceId для пошуку
   */
  normalizeServiceId(serviceId) {
    if (!serviceId) return '';
    return String(serviceId).trim().toLowerCase();
  },

  /**
   * Нормалізує масив перекладів з різних форматів
   */
  normalizeTranslationsArray(entry) {
    // Формат 1: { translations: [...] }
    if (entry.translations && Array.isArray(entry.translations)) {
      return entry.translations;
    }
    
    // Формат 2: { segments: [...] }
    if (entry.segments && Array.isArray(entry.segments)) {
      return entry.segments;
    }
    
    // Формат 3: Прямий масив
    if (Array.isArray(entry)) {
      return entry;
    }
    
    return [];
  },

  /**
   * Нормалізує індекс сегменту
   */
  normalizeSegmentIndex(item, fallbackIdx, maxLength) {
    // Спробуємо знайти індекс з різних полів
    let index = null;
    
    if (typeof item.segmentIndex === 'number') {
      index = item.segmentIndex;
    } else if (typeof item.index === 'number') {
      index = item.index;
    } else if (typeof item.segment === 'number') {
      index = item.segment;
    } else if (typeof item.id === 'number') {
      index = item.id;
    }
    
    // Якщо індекс не знайдено, використовуємо fallback
    if (index === null || isNaN(index)) {
      index = fallbackIdx;
    }
    
    // Обмежуємо індекс діапазоном
    return Math.max(0, Math.min(Math.floor(index), maxLength - 1));
  },

  /**
   * Нормалізує текст перекладу
   */
  normalizeTranslationText(item) {
    // Спробуємо знайти текст з різних полів
    const text = item.translatedText || item.text || item.translation || item.content || '';
    
    if (typeof text !== 'string') {
      return '';
    }
    
    return text.trim();
  },

  applyCachedTranslation(targetLang) {
    let applied = 0;
    Object.values(this.testResults).forEach(result => {
      if (!result || !result.success || !Array.isArray(result.segments)) return;
      result.segments.forEach(segment => {
        if (segment.translations && segment.translations[targetLang]) {
          segment.text = segment.translations[targetLang];
          applied += 1;
        }
      });
    });

    if (applied === 0) {
      throw new Error('No saved translations for this language');
    }

    this.translationState.currentLanguage = targetLang;
    const detailsContainer = document.getElementById('detailsContainer');
    if (detailsContainer) {
      detailsContainer.innerHTML = '';
    }
    this.showResults();
    this.refreshReplicasViewIfActive();
  },

  async translateTranscripts(targetLang, options = {}) {
    const meta = this.getTranslationLanguageMeta(targetLang);
    if (!meta) {
      if (!options.silent) alert('Unsupported translation language');
      return;
    }

    const recording = options.recording || this.getActiveRecording();
    if (!recording) {
      if (!options.silent) alert('No recording to translate.');
      return;
    }

    if (this.activeRecordingId !== recording.id) {
      this.setActiveRecording(recording.id, { skipResultsRender: options.silent });
    }

    if (!this.hasSuccessfulSegments()) {
      if (!options.silent) alert('Please run testing first or load results with transcripts.');
      return;
    }

    if (this.translationState.isTranslating) {
      return;
    }

    this.ensureTranslationMetadata(recording);
    this.translationState.lastError = null;

    if (this.translationState.currentLanguage === targetLang && this.allSegmentsHaveTranslation(targetLang)) {
      this.setTranslationStatus(`[${recording.name}] Transcript already translated to ${meta.display.toLowerCase()}`, 'info');
      return;
    }

    if (this.allSegmentsHaveTranslation(targetLang)) {
      try {
        this.applyCachedTranslation(targetLang);
        this.setTranslationStatus(`[${recording.name}] Applied saved translation to ${meta.display.toLowerCase()}`, 'success');
      } catch (error) {
        this.setTranslationStatus(error.message, 'warning');
      }
      return;
    }

    const services = this.buildTranslationInput();
    if (services.length === 0) {
      this.setTranslationStatus('No segments to translate', 'warning');
      return;
    }

    this.translationState.isTranslating = true;
    this.updateTranslationControlsState();
    this.setTranslationStatus(`[${recording.name}] Translating to ${meta.display.toLowerCase()}...`, 'processing');

    try {
      // Load translation system message
      let systemMessage;
      try {
        const systemResponse = await fetch('prompts/system_translation.txt');
        if (systemResponse.ok) {
          systemMessage = (await systemResponse.text()).trim().replace(/{TARGET_LANGUAGE}/g, meta.english);
        } else {
          throw new Error('Unable to load system message');
        }
      } catch (error) {
        console.warn('Failed to load translation system message, using fallback:', error);
        systemMessage = `You are an expert translator that converts Arabic diarization transcripts into ${meta.english}. Respond with valid JSON only.`;
      }
      
      const userPrompt = await this.buildTranslationPrompt(meta, services);
      
      const response = await fetch(this.translationConfig.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: this.translationConfig.model,
          temperature: 0,
          messages: [
            {
              role: 'system',
              content: systemMessage
            },
            {
              role: 'user',
              content: userPrompt
            }
          ]
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const content = data?.choices?.[0]?.message?.content;
      const parsed = this.parseTranslationResponse(content);
      this.applyTranslationResult(parsed, targetLang);
      this.setTranslationStatus(`[${recording.name}] Translated to ${meta.display.toLowerCase()}`, 'success');
    } catch (error) {
      console.error('Translation error:', error);
      this.translationState.lastError = error.message || 'Unknown error';
      this.setTranslationStatus(`Translation error: ${error.message}`, 'error');
    } finally {
      this.translationState.isTranslating = false;
      this.updateTranslationControlsState();
      this.renderRecordingsQueue();
    }
  },

  async translateAllRecordings(targetLang = 'uk') {
    if (this.recordings.length === 0) {
      alert('No recordings to translate.');
      return;
    }
    const meta = this.getTranslationLanguageMeta(targetLang);
    if (!meta) {
      alert('Unsupported translation language');
      return;
    }

    const originalRecordingId = this.activeRecordingId;
    for (const recording of this.recordings) {
      await this.translateTranscripts(targetLang, { recording, silent: true });
    }

    if (originalRecordingId) {
      this.setActiveRecording(originalRecordingId);
    }
    this.setTranslationStatus(`All recordings translated to ${meta.display.toLowerCase()}`, 'success');
    this.renderRecordingsQueue();
    this.refreshReplicasViewIfActive();
  },

  restoreOriginalTranscripts() {
    if (!this.hasSuccessfulSegments()) {
      this.setTranslationStatus('No transcripts to restore', 'warning');
      return;
    }

    this.ensureTranslationMetadata();
    Object.values(this.testResults).forEach(result => {
      if (!result || !result.success || !Array.isArray(result.segments)) return;
      result.segments.forEach(segment => {
        if (typeof segment.originalText === 'string') {
          segment.text = segment.originalText;
        }
      });
    });

    this.translationState.currentLanguage = 'original';
    this.translationState.lastError = null;

    const detailsContainer = document.getElementById('detailsContainer');
    if (detailsContainer) {
      detailsContainer.innerHTML = '';
    }

    this.showResults();
    this.refreshReplicasViewIfActive();
    this.setTranslationStatus('Original Arabic transcripts', 'info');
  },

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  },

  // AI Comparison functionality
  showAIComparison() {
    this.showView('aiComparisonView');
  },

  async runAIAnalysis() {
    const apiKey = (document.getElementById('openrouterApiKey')?.value || '').trim();
    const model = document.getElementById('aiModelSelect').value;

    if (!apiKey) {
      alert('Please enter OpenRouter API key');
      return;
    }

    const statusEl = document.getElementById('aiAnalysisStatus');
    const resultEl = document.getElementById('aiAnalysisResult');
    const btnEl = document.getElementById('startAIAnalysisBtn');

    btnEl.disabled = true;
    statusEl.style.display = 'block';
    statusEl.textContent = '🤖 Analyzing transcripts with AI...';
    resultEl.innerHTML = '';

    try {
      // Prepare transcripts for comparison
      const transcriptsData = Object.entries(this.testResults)
        .filter(([_, result]) => result.success)
        .map(([serviceId, result]) => ({
          service: result.serviceName,
          speakers: result.speakerCount,
          segments: result.segments
        }));

      const prompt = await this.buildAIPrompt(transcriptsData);

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': window.location.href,
          'X-Title': 'Diarization Tester'
        },
        body: JSON.stringify({
          model: model,
          messages: [{
            role: 'user',
            content: prompt
          }]
        })
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const analysis = data.choices[0].message.content;

      statusEl.style.display = 'none';
      this.displayAIAnalysis(analysis);

    } catch (error) {
      statusEl.textContent = `❌ Error: ${error.message}`;
      statusEl.style.color = 'var(--color-error)';
    } finally {
      btnEl.disabled = false;
    }
  },

  async buildAIPrompt(transcriptsData) {
    try {
      const response = await fetch('prompts/ai_analysis_prompt_template.txt');
      if (!response.ok) {
        throw new Error('Unable to load AI analysis prompt template');
      }
      let template = await response.text();
      
      // Build service data section
      let serviceData = '';
      transcriptsData.forEach((data, idx) => {
        serviceData += `\n=== ${data.service} (${data.speakers} speakers) ===\n`;
        data.segments.forEach(seg => {
          serviceData += `[${this.formatTime(seg.start)}-${this.formatTime(seg.end)}] ${seg.speaker}: ${seg.text}\n`;
        });
      });
      
      // Replace placeholders
      template = template.replace(/{SERVICE_COUNT}/g, transcriptsData.length.toString());
      template = template.replace(/{SERVICE_DATA}/g, serviceData);
      
      return template.trim();
    } catch (error) {
      console.warn('Failed to load AI analysis prompt template, using fallback:', error);
      // Fallback to original hardcoded prompt
      let prompt = `Analyze the diarization results from ${transcriptsData.length} different services. Compare how accurately each service separates speakers and how well it preserves transcript quality.\n\n`;

      transcriptsData.forEach((data, idx) => {
        prompt += `\n=== ${data.service} (${data.speakers} speakers) ===\n`;
        data.segments.forEach(seg => {
          prompt += `[${this.formatTime(seg.start)}-${this.formatTime(seg.end)}] ${seg.speaker}: ${seg.text}\n`;
        });
      });

      prompt += `\n\nProvide a detailed analysis:\n1. Which service identified speakers most accurately?\n2. Where do disagreements between services appear?\n3. Evaluate the transcript quality produced by each service.\n4. Give recommendations for practical use.\n\nFormat the response with headings and concise summaries.`;

      return prompt;
    }
  },

  displayAIAnalysis(analysis) {
    const resultEl = document.getElementById('aiAnalysisResult');
    resultEl.innerHTML = `
      <div class="card" style="background: var(--color-bg-5); margin-top: var(--space-16);">
        <h3 style="color: var(--color-primary); margin-bottom: var(--space-16);">🤖 AI Analysis</h3>
        <div style="white-space: pre-wrap; line-height: 1.8;">${this.escapeHtml(analysis)}</div>
      </div>
    `;
  },

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  // Manual Review functionality
  showManualReview() {
    this.showView('manualReviewView');
    this.initAudioPlayer();
    this.renderTranscriptComparison();
  },

  initAudioPlayer() {
    if (!this.uploadedFile) return;
    
    // Validate that uploadedFile is a valid File or Blob
    if (!(this.uploadedFile instanceof File) && !(this.uploadedFile instanceof Blob)) {
      console.error('Invalid uploadedFile type:', typeof this.uploadedFile);
      return;
    }

    const audioPlayer = document.getElementById('audioPlayer');
    if (!audioPlayer) return;
    
    try {
      const url = URL.createObjectURL(this.uploadedFile);
      if (!url) {
        throw new Error('Failed to create object URL');
      }
      audioPlayer.src = url;
    } catch (error) {
      console.error('Failed to create audio player URL:', error);
    }

    // Setup playback speed controls
    const speedBtns = document.querySelectorAll('.speed-btn');
    speedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const speed = parseFloat(btn.dataset.speed);
        audioPlayer.playbackRate = speed;
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Time update handler
    audioPlayer.addEventListener('timeupdate', () => {
      this.syncTranscriptWithAudio(audioPlayer.currentTime);
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (document.getElementById('manualReviewView').classList.contains('active')) {
        if (e.code === 'Space') {
          e.preventDefault();
          audioPlayer.paused ? audioPlayer.play() : audioPlayer.pause();
        } else if (e.code === 'ArrowLeft') {
          audioPlayer.currentTime = Math.max(0, audioPlayer.currentTime - 5);
        } else if (e.code === 'ArrowRight') {
          audioPlayer.currentTime = Math.min(audioPlayer.duration, audioPlayer.currentTime + 5);
        }
      }
    });
  },

  renderTranscriptComparison() {
    const container = document.getElementById('transcriptComparison');
    const successfulResults = Object.entries(this.testResults).filter(([_, r]) => r.success);

    // Initialize role filter state if not exists
    if (!this.roleFilter) {
      this.roleFilter = { agent: true, client: true, none: true };
    }

    // Role filter controls
    const filterHTML = `
      <div class="role-filter-controls" style="margin-bottom: var(--space-16); padding: var(--space-12); background: var(--color-bg-2); border-radius: var(--radius-base); display: flex; gap: var(--space-16); align-items: center; flex-wrap: wrap;">
        <span style="font-weight: var(--font-weight-medium);">Filter by Role:</span>
        <label style="display: flex; align-items: center; gap: var(--space-8); cursor: pointer;">
          <input type="checkbox" id="filterAgent" ${this.roleFilter.agent ? 'checked' : ''} onchange="app.updateRoleFilter('agent', this.checked)" style="cursor: pointer;">
          <span>Agent</span>
        </label>
        <label style="display: flex; align-items: center; gap: var(--space-8); cursor: pointer;">
          <input type="checkbox" id="filterClient" ${this.roleFilter.client ? 'checked' : ''} onchange="app.updateRoleFilter('client', this.checked)" style="cursor: pointer;">
          <span>Client</span>
        </label>
        <label style="display: flex; align-items: center; gap: var(--space-8); cursor: pointer;">
          <input type="checkbox" id="filterNone" ${this.roleFilter.none ? 'checked' : ''} onchange="app.updateRoleFilter('none', this.checked)" style="cursor: pointer;">
          <span>No Role</span>
        </label>
      </div>
    `;

    container.innerHTML = filterHTML + successfulResults.map(([serviceId, result]) => {
      const preparedSegments = this.prepareSegmentsForDisplay(result.segments);
      // Filter segments based on role filter
      let filteredSegments = preparedSegments.filter(seg => {
        if (!seg.role) return this.roleFilter.none;
        if (seg.role === 'operator') return this.roleFilter.agent;
        if (seg.role === 'client') return this.roleFilter.client;
        return true;
      });

      return `
      <div class="transcript-panel" data-service="${serviceId}">
        <h3 style="margin-bottom: var(--space-16); padding-bottom: var(--space-12); border-bottom: 2px solid var(--color-border);">
          ${result.serviceName} (${result.speakerCount} speakers) - Showing ${filteredSegments.length} of ${preparedSegments.length} segments
        </h3>
        <div class="transcript-segments" id="transcript_${serviceId}">
          ${filteredSegments.map((seg, idx) => `
            <div class="transcript-segment" data-start="${seg.start}" data-end="${seg.end}" data-idx="${idx}">
              <div class="segment-time">[${this.formatTime(seg.start)} - ${this.formatTime(seg.end)}]</div>
              <div class="segment-speaker">
                ${seg.role ? (seg.role === 'operator' ? 'Agent' : 'Client') : seg.speaker}
                ${seg.role ? ` <span style="font-size: 0.9em; color: ${seg.role === 'operator' ? 'var(--color-success)' : 'var(--color-primary)'}; font-weight: normal; margin-left: var(--space-8);">(${seg.role === 'operator' ? '👨‍💼 Agent' : '👤 Client'})</span>` : ''}
              </div>
              <div class="segment-text">${seg.text}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `}).join('');

    // Add click handlers for segments
    document.querySelectorAll('.transcript-segment').forEach(segment => {
      segment.addEventListener('click', () => {
        const startTime = parseFloat(segment.dataset.start);
        document.getElementById('audioPlayer').currentTime = startTime;
      });
    });
  },

  updateRoleFilter(role, enabled) {
    if (role === 'agent') {
      this.roleFilter.agent = enabled;
    } else if (role === 'client') {
      this.roleFilter.client = enabled;
    } else if (role === 'none') {
      this.roleFilter.none = enabled;
    }
    // Re-render transcript comparison with new filter
    this.renderTranscriptComparison();
  },

  syncTranscriptWithAudio(currentTime) {
    document.querySelectorAll('.transcript-segment').forEach(segment => {
      const start = parseFloat(segment.dataset.start);
      const end = parseFloat(segment.dataset.end);
      
      if (currentTime >= start && currentTime <= end) {
        segment.classList.add('active');
        // Auto-scroll to active segment
        segment.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        segment.classList.remove('active');
      }
    });
  },

  /**
   * Enhanced demo mode with realistic metrics
   */
  useDemoMode() {
    if (confirm('Start demo mode with sample data? (No live API calls)')) {
      // Mock all 4 services as configured
      SERVICES.forEach(service => {
        this.config.services[service.id] = 'demo_key_' + service.id;
        if (!this.config.selectedServices.includes(service.id)) {
          this.config.selectedServices.push(service.id);
        }
      });
      this.saveApiKeys();
      this.saveEnabledServices();
      this.updateConfiguredCount();
      SERVICES.forEach(service => this.updateServiceToggleUI(service.id));
      this.renderServiceConfig();
      this.loadSavedApiKeys();
      
      // Create demo ground truth
      this.groundTruth = {
        segments: [
          { start: 0, end: 5.2, speaker: 'SPEAKER_A', text: 'Hello, how are you today?' },
          { start: 5.2, end: 9.0, speaker: 'SPEAKER_B', text: 'I am doing great, thanks for asking.' },
          { start: 9.0, end: 15.1, speaker: 'SPEAKER_A', text: 'That is wonderful to hear. Lets discuss the project.' },
          { start: 15.1, end: 19.4, speaker: 'SPEAKER_C', text: 'Yes, I have some ideas to share.' }
        ],
        speakerCount: 3,
        format: 'Demo'
      };
      
      alert('Demo mode activated! Now upload a file for testing.');
      this.showView('uploadView');
    }
  },

  showDetails(serviceId) {
    if (serviceId === 'audioshake') {
      this.showAudioshakeDetails();
      return;
    }
    if (serviceId === 'audio') {
      // Handle audio diarization results
      const result = this.testResults[serviceId];
      if (!result) {
        console.error('No audio result found');
        return;
      }
      const segments = result?.segments || [];
      const container = document.getElementById('detailsContainer');
      
      const detailHTML = `
        <div class="detail-view" id="detail_audio">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-16);">
            <h3>${result.serviceName || 'Audio Diarization'} - Detailed Results</h3>
            <button class="action-btn" onclick="document.getElementById('detail_audio').remove()">✕ Close</button>
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-24);">
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Processing Time</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${result.processingTime || 0}s</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Detected Speakers</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${result.speakerCount || 0}</div>
            </div>
            <div>
              <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Segments</div>
              <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${segments.length}</div>
            </div>
          </div>

          <h4 style="margin-bottom: var(--space-12);">Transcript with timestamps</h4>
          <div class="timeline">
            ${segments.map(segment => {
              return `
              <div class="timeline-segment">
                <div class="timeline-time">${this.formatTime(segment.start)} → ${this.formatTime(segment.end)}</div>
                <div class="timeline-speaker">${segment.speaker || 'Unknown'}</div>
                <div class="timeline-text">${segment.text || ''}</div>
              </div>
            `;
            }).join('')}
          </div>
        </div>
      `;

      container.insertAdjacentHTML('beforeend', detailHTML);
      document.getElementById('detail_audio').scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    
    const result = this.testResults[serviceId];
    if (!result) {
      console.error(`No result found for service: ${serviceId}`);
      return;
    }
    const container = document.getElementById('detailsContainer');

    const detailHTML = `
      <div class="detail-view" id="detail_${serviceId}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-16);">
          <h3>${result.serviceName} - Detailed Results</h3>
          <button class="action-btn" onclick="document.getElementById('detail_${serviceId}').remove()">✕ Close</button>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-16); margin-bottom: var(--space-24);">
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Processing Time</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${result.processingTime}s</div>
          </div>
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Detected Speakers</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold);">${result.speakerCount}</div>
          </div>
          <div>
            <div style="color: var(--color-text-secondary); font-size: var(--font-size-sm); margin-bottom: var(--space-4);">Cost</div>
            <div style="font-size: var(--font-size-xl); font-weight: var(--font-weight-semibold); color: var(--color-primary);">$${result.cost}</div>
          </div>
        </div>

        <h4 style="margin-bottom: var(--space-12);">Transcript with timestamps</h4>
        <div class="timeline">
          ${result.segments.map(segment => {
            const overlapClass = segment.overlap ? 'timeline-segment-overlap' : '';
            const overlapStyle = segment.overlap ? 'border-left: 3px solid var(--color-warning); background: rgba(255, 193, 7, 0.1);' : '';
            return `
            <div class="timeline-segment ${overlapClass}" style="${overlapStyle}">
              <div class="timeline-time">${this.formatTime(segment.start)} → ${this.formatTime(segment.end)}</div>
              <div class="timeline-speaker">
                ${segment.role ? (segment.role === 'operator' ? 'Agent' : 'Client') : segment.speaker}
                ${segment.role ? ` <span style="font-size: 0.85em; color: ${segment.role === 'operator' ? 'var(--color-success)' : 'var(--color-primary)'}; font-weight: normal;">(${segment.role === 'operator' ? '👨‍💼 Agent' : '👤 Client'})</span>` : ''}
                ${segment.overlap ? ` <span style="font-size: 0.8em; color: var(--color-warning); font-weight: bold;" title="Перекриття: спікери говорять одночасно">⚠️ Overlap</span>` : ''}
              </div>
              <div class="timeline-text">${segment.text}</div>
            </div>
          `;
          }).join('')}
        </div>
      </div>
    `;

    container.insertAdjacentHTML('beforeend', detailHTML);
    document.getElementById(`detail_${serviceId}`).scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  showAudioshakeDetails() {
    if (!this.audioshakeResults) {
      alert('AudioShake results are not available.');
      return;
    }

    const container = document.getElementById('detailsContainer');
    if (!container) {
      alert('Details container not found');
      return;
    }

    const tracks = Array.isArray(this.audioshakeResults.speakers)
      ? this.audioshakeResults.speakers
      : (Array.isArray(this.audioshakeResults.results) ? this.audioshakeResults.results : []);

    if (tracks.length === 0) {
      alert('AudioShake tracks are not available.');
      return;
    }

    const detailHTML = `
      <div class="detail-view" id="detail_audioshake">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--space-16);">
          <h3>Overlap Diarization</h3>
          <button class="action-btn" onclick="document.getElementById('detail_audioshake').remove()">✕ Close</button>
        </div>
        <div class="timeline">
          ${tracks.map(track => {
            const transcript = track.transcript || '';
            const diarizationSegments = track.diarization?.recordings?.flatMap(rec => rec.results?.speechmatics?.segments || []) || [];
            const diarizationPreview = diarizationSegments.slice(0, 3).map(seg => {
              return `[${this.formatTime(seg.start)}-${this.formatTime(seg.end)}] ${seg.speaker || ''}: ${seg.text || ''}`;
            }).join('\\n');
            const previewText = transcript || diarizationPreview;
            const downloadLink = track.downloadUrl || track.audioUrl
              ? `<a href="${track.downloadUrl || track.audioUrl}" target="_blank" rel="noopener">Download stem</a>`
              : 'No download link';
            const confidence = typeof track.confidence === 'number'
              ? `${Math.round(track.confidence * 100)}%`
              : '—';
            const label = track.track_id || track.file_name || track.speaker || track.name || 'Track';
            const role = track.role || track.speaker || track.name || 'Unknown';
            return `
              <div class="timeline-segment" style="border-left: 3px solid var(--color-primary);">
                <div class="timeline-speaker" style="display: flex; justify-content: space-between; align-items: center;">
                  <div>
                    <strong>${this.escapeHtml(label)}</strong>
                    <span style="margin-left: var(--space-8); color: var(--color-text-secondary);">${this.escapeHtml(role)}</span>
                  </div>
                  <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">Confidence: ${confidence}</div>
                </div>
                <div class="timeline-text" style="margin-top: var(--space-8);">
                  ${track.summary ? this.escapeHtml(track.summary) : 'No summary'}
                </div>
                <details style="margin-top: var(--space-8);">
                  <summary style="cursor: pointer;">Full transcript preview</summary>
                  <div style="margin-top: var(--space-8); white-space: pre-wrap;">
                    ${previewText ? this.escapeHtml(previewText) : 'No transcript available'}
                  </div>
                  ${diarizationSegments.length > 0 ? `
                    <div style="margin-top: var(--space-8); font-size: var(--font-size-sm); color: var(--color-text-secondary);">
                      Showing ${Math.min(3, diarizationSegments.length)} of ${diarizationSegments.length} diarization segments.
                    </div>
                  ` : ''}
                </details>
                <div style="margin-top: var(--space-8); font-size: var(--font-size-sm);">
                  ${downloadLink}
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    container.insertAdjacentHTML('beforeend', detailHTML);
    document.getElementById('detail_audioshake').scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  /**
   * Show error logs for a service
   */
  showErrorLogs(serviceId) {
    const errorLog = this.errorLogs[serviceId];
    if (!errorLog) {
      alert('Error logs not found');
      return;
    }

    // Create modal for error logs
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'flex';
    modal.id = 'errorLogModal';
    
    const logText = this.formatErrorLog(errorLog);
    
    modal.innerHTML = `
      <div class="modal-content" style="max-width: 90%; max-height: 90%;">
        <div class="modal-header">
          <h2>Error Logs: ${errorLog.service}</h2>
          <button class="btn-close" onclick="document.getElementById('errorLogModal').remove()">✕</button>
        </div>
        <div class="modal-body">
          <div style="margin-bottom: var(--space-16);">
            <button class="btn btn-secondary" onclick="app.copyErrorLog('${serviceId}')">📋 Copy Logs</button>
            <button class="btn btn-secondary" onclick="app.downloadErrorLog('${serviceId}')" style="margin-left: var(--space-8);">💾 Download</button>
          </div>
          <div class="error-log-container">
            <pre class="error-log-content">${this.escapeHtml(logText)}</pre>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" onclick="document.getElementById('errorLogModal').remove()">Close</button>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
  },

  /**
   * Format error log for display (matching console format)
   */
  formatErrorLog(errorLog) {
    let log = '';
    
    // Main error message (like console.error)
    log += `❌ [${errorLog.service}] ${errorLog.name}: ${errorLog.message}\n`;
    log += `   at ${errorLog.timestamp}\n\n`;

    // Stack trace (if available)
    if (errorLog.stack) {
      log += `Stack Trace:\n`;
      const stackLines = errorLog.stack.split('\n');
      stackLines.forEach((line, idx) => {
        if (idx === 0) {
          log += `   ${line}\n`;
        } else {
          log += `   ${line.trim()}\n`;
        }
      });
      log += `\n`;
    }

    // Fetch error details (most common)
    if (errorLog.fetchError) {
      log += `Fetch Request Details:\n`;
      log += `   URL: ${errorLog.fetchError.url || 'N/A'}\n`;
      log += `   Method: ${errorLog.fetchError.method || 'N/A'}\n`;
      log += `   Status: ${errorLog.fetchError.status || 'N/A'} ${errorLog.fetchError.statusText || ''}\n`;
      
      if (errorLog.fetchError.body) {
        log += `   Response Body:\n`;
        try {
          // Try to parse as JSON for better formatting
          const parsed = typeof errorLog.fetchError.body === 'string' 
            ? JSON.parse(errorLog.fetchError.body) 
            : errorLog.fetchError.body;
          log += `   ${JSON.stringify(parsed, null, 6).split('\n').join('\n   ')}\n`;
        } catch (e) {
          // If not JSON, show as text
          const bodyLines = String(errorLog.fetchError.body).split('\n');
          bodyLines.forEach(line => {
            log += `   ${line}\n`;
          });
        }
      }
      log += `\n`;
    }

    // HTTP Response details
    if (errorLog.response) {
      log += `HTTP Response:\n`;
      log += `   Status: ${errorLog.response.status} ${errorLog.response.statusText}\n`;
      if (errorLog.response.headers) {
        log += `   Headers:\n`;
        Object.entries(errorLog.response.headers).forEach(([key, value]) => {
          log += `      ${key}: ${value}\n`;
        });
      }
      if (errorLog.response.body) {
        log += `   Body:\n`;
        try {
          const parsed = typeof errorLog.response.body === 'string'
            ? JSON.parse(errorLog.response.body)
            : errorLog.response.body;
          log += `   ${JSON.stringify(parsed, null, 6).split('\n').join('\n   ')}\n`;
        } catch (e) {
          log += `   ${errorLog.response.body}\n`;
        }
      }
      log += `\n`;
    }

    // Additional data
    if (errorLog.data) {
      log += `Additional Error Data:\n`;
      log += `${JSON.stringify(errorLog.data, null, 2).split('\n').join('\n   ')}\n\n`;
    }

    // Original error (for parsing errors)
    if (errorLog.originalError) {
      log += `Original Error: ${errorLog.originalError}\n\n`;
    }

    // Full error object (for complete debugging)
    log += `\n--- Full Error Object (for debugging) ---\n`;
    log += `${JSON.stringify(errorLog, null, 2)}\n`;

    return log;
  },

  /**
   * Copy error log to clipboard
   */
  copyErrorLog(serviceId) {
    const errorLog = this.errorLogs[serviceId];
    if (!errorLog) return;

    const logText = this.formatErrorLog(errorLog);
    navigator.clipboard.writeText(logText).then(() => {
      alert('✅ Logs copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy:', err);
      alert('❌ Failed to copy logs');
    });
  },

  /**
   * Download error log as file
   */
  downloadErrorLog(serviceId) {
    const errorLog = this.errorLogs[serviceId];
    if (!errorLog) return;

    const logText = this.formatErrorLog(errorLog);
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `error_log_${errorLog.serviceId}_${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  },

  /**
   * Escape HTML for safe display
   */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  /**
   * Handle JSON file upload
   */
  handleJSONUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const jsonData = JSON.parse(e.target.result);
        this.loadJSONResults(jsonData);
        alert('✅ JSON file loaded successfully! Go to results to view.');
        this.showView('resultsView');
      } catch (error) {
        alert('❌ Error loading JSON: ' + error.message);
        console.error('JSON parse error:', error);
      }
    };
    reader.readAsText(file);
  },

  /**
   * Load results from JSON
   */
  loadJSONResults(jsonData) {
    this.recordings = [];
    this.activeRecordingId = null;
    this.recordingQueue = [];

    const finishImport = () => {
      if (this.recordings.length === 0) {
        console.warn('⚠️ No recordings loaded from JSON data');
        console.warn('JSON data structure:', Object.keys(jsonData || {}));
        this.testResults = {};
        this.translationState = this.createTranslationState();
        this.showResults();
        return;
      }
      console.log(`✅ Loaded ${this.recordings.length} recording(s) from JSON`);
      const targetId = this.activeRecordingId || this.recordings[0].id;
      this.setActiveRecording(targetId);
      this.renderRecordingsQueue();
      this.updateStartButtonState();
      this.renderReplicaRecordingSelect(); // Update replica recording select
      
      // Update separated speakers players after import
      const activeRecording = this.getActiveRecording();
      if (activeRecording) {
        console.log('🔄 Updating separated speakers players after JSON import...');
        this.updateSeparatedSpeakersPlayers(activeRecording);
        // Also update replica audio player
        this.updateReplicaAudioPlayer(activeRecording.id);
      }
      
      // Log segments count for debugging
      this.recordings.forEach((rec, idx) => {
        const textResult = this.getTextServiceResult(rec);
        const segments = textResult?.segments || [];
        const voiceTracksCount = this.getVoiceTracksFromRecording(rec).length;
        const separatedSpeakersCount = rec.overlapMetadata?.separation?.speakers?.length || 0;
        console.log(`Recording ${idx + 1} (${rec.id}): ${segments.length} segments, ${voiceTracksCount} voice tracks, ${separatedSpeakersCount} separated speakers`);
      });
    };

    // Check if data is in wrapped format: [{ output: { recordings: [...] } }]
    if (Array.isArray(jsonData) && jsonData.length > 0) {
      const firstItem = jsonData[0];
      if (firstItem && firstItem.output && firstItem.output.recordings) {
        console.log('✅ Found wrapped format: [{ output: { recordings: [...] } }]');
        jsonData = firstItem.output;
      } else if (firstItem && firstItem.recordings) {
        // Alternative format: [{ recordings: [...] }]
        console.log('✅ Found alternative wrapped format: [{ recordings: [...] }]');
        jsonData = firstItem;
      }
    }

    if (jsonData && Array.isArray(jsonData.recordings)) {
      console.log('✅ Found recordings array, count:', jsonData.recordings.length);
      
      // Log to server
      try {
        fetch('/api/log', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            level: 'info',
            message: 'Loading recordings from JSON',
            data: { recordingsCount: jsonData.recordings.length }
          })
        }).catch(() => {});
      } catch (e) {}
      
      this.recordings = jsonData.recordings.map(recording => {
        const hydrated = this.hydrateRecordingFromExport(recording);
        const textResult = this.getTextServiceResult(hydrated);
        const segments = textResult?.segments || [];
        console.log(`Recording ${hydrated.id}: ${segments.length} segments`);
        
        // Log each recording to server
        if (segments.length === 0) {
          try {
            fetch('/api/log', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                level: 'warn',
                message: `Recording ${hydrated.id} has 0 segments`,
                data: {
                  recordingId: hydrated.id,
                  hasResults: !!recording.results,
                  hasTextResult: !!textResult,
                  resultsKeys: Object.keys(recording.results || {})
                }
              })
            }).catch(() => {});
          } catch (e) {}
        }
        
        return hydrated;
      });
      this.activeRecordingId = jsonData.activeRecordingId || (this.recordings[0]?.id || null);
      finishImport();
      return;
    }
    
    console.warn('⚠️ JSON data does not have recordings array');
    console.warn('JSON data keys:', Object.keys(jsonData || {}));

    let parsedResults = {};
    if (jsonData && jsonData.legacyTestResults) {
      parsedResults = jsonData.legacyTestResults;
    } else if (Array.isArray(jsonData)) {
      jsonData.forEach((result, idx) => {
        const serviceId = result.serviceId || `service_${idx}`;
        parsedResults[serviceId] = result;
      });
    } else if (jsonData && (jsonData.serviceName || jsonData.segments)) {
      parsedResults['uploaded_service'] = jsonData;
    } else if (jsonData && typeof jsonData === 'object') {
      Object.entries(jsonData).forEach(([serviceId, result]) => {
        parsedResults[serviceId] = result;
      });
    }

    const importedRecording = this.createImportedRecordingFromResults(parsedResults, jsonData);
    this.recordings = importedRecording ? [importedRecording] : [];
    this.activeRecordingId = importedRecording?.id || null;
    finishImport();
  },

  hydrateRecordingFromExport(recording) {
    this.normalizeTextServiceResult(recording);
    const baseState = this.createTranslationState();
    const stateFromExport = recording.translationState || {};
    
    // Log for debugging
    const textResult = this.getTextServiceResult(recording);
    const segments = textResult?.segments || [];
    console.log(`Hydrating recording ${recording.id}: ${segments.length} segments in text-service`);
    
    // Restore audio URL if available
    const audioUrl = recording.audio?.url || recording.url || null;
    const audioFile = recording.audio?.file || recording.file || null;
    
    // Restore overlap metadata (voice tracks, separated speakers, etc.)
    let overlapMetadata = recording.overlapMetadata || null;
    
    // If voiceTracks or separatedSpeakers are at top level, merge into overlapMetadata
    if (recording.voiceTracks || recording.separatedSpeakers) {
      if (!overlapMetadata) {
        overlapMetadata = {};
      }
      // Merge voice tracks from top level into overlapMetadata
      if (recording.voiceTracks && Array.isArray(recording.voiceTracks)) {
        overlapMetadata.voiceTracks = recording.voiceTracks;
        console.log(`✅ Restored ${recording.voiceTracks.length} voice tracks from top-level field`);
      }
      // Merge separated speakers from top level into overlapMetadata
      if (recording.separatedSpeakers && Array.isArray(recording.separatedSpeakers)) {
        if (!overlapMetadata.separation) {
          overlapMetadata.separation = {};
        }
        overlapMetadata.separation.speakers = recording.separatedSpeakers;
        console.log(`✅ Restored ${recording.separatedSpeakers.length} separated speakers from top-level field`);
      }
    }
    
    // If overlapMetadata exists but voiceTracks is missing, try to restore from nested structure
    if (overlapMetadata && !overlapMetadata.voiceTracks) {
      // Check if voice tracks are in a nested structure
      if (overlapMetadata.separation?.speakers) {
        // If we have separated speakers but no voice tracks, log a warning
        console.warn(`⚠️ overlapMetadata exists but voiceTracks array is missing`);
      }
      // Try to restore from top-level if not already done
      if (recording.voiceTracks && Array.isArray(recording.voiceTracks)) {
        overlapMetadata.voiceTracks = recording.voiceTracks;
        console.log(`✅ Restored ${recording.voiceTracks.length} voice tracks from top-level (fallback)`);
      }
    }
    
    // Log overlap metadata restoration
    if (overlapMetadata) {
      const voiceTracksCount = overlapMetadata.voiceTracks?.length || 0;
      const separatedSpeakersCount = overlapMetadata.separation?.speakers?.length || 0;
      console.log(`✅ Restored overlapMetadata: ${voiceTracksCount} voice tracks, ${separatedSpeakersCount} separated speakers`);
      
      // Validate voice tracks structure
      if (voiceTracksCount > 0) {
        const firstTrack = overlapMetadata.voiceTracks[0];
        console.log(`📊 First voice track sample:`, {
          speaker: firstTrack?.speaker,
          hasTranscription: !!firstTrack?.transcription,
          hasUrl: !!firstTrack?.url,
          hasDownloadUrl: !!firstTrack?.downloadUrl,
          keys: Object.keys(firstTrack || {})
        });
      }
    } else {
      console.warn(`⚠️ No overlapMetadata found in imported recording`);
      console.warn(`📋 Available top-level keys:`, Object.keys(recording).filter(k => 
        k.includes('voice') || k.includes('track') || k.includes('overlap') || k.includes('separation')
      ));
    }
    
    // Build audio object if URL or file is available
    const audio = (audioUrl || audioFile) ? {
      url: audioUrl,
      file: audioFile
    } : null;
    
    // Also preserve top-level voiceTracks and separatedSpeakers for compatibility
    const voiceTracks = overlapMetadata?.voiceTracks || recording.voiceTracks || null;
    const separatedSpeakers = overlapMetadata?.separation?.speakers || recording.separatedSpeakers || null;
    
    const hydrated = {
      id: recording.id || `imported_${Date.now().toString(16)}_${Math.random().toString(16).slice(2)}`,
      name: recording.name || recording.fileName || 'Imported recording',
      file: audioFile,
      url: audioUrl, // Restore audio URL
      size: recording.size || null,
      duration: recording.duration || 0,
      language: recording.language || 'ar',
      speakerCount: recording.speakerCount ?? '',
      status: recording.status || 'completed',
      results: recording.results || {},
      translationState: {
        ...baseState,
        ...stateFromExport,
        isTranslating: false
      },
      aggregated: recording.aggregated || {},
      addedAt: recording.addedAt || recording.importedAt || null,
      overlapMetadata: overlapMetadata, // Restore overlap metadata with voice tracks and separated speakers
      // Also preserve at top level for compatibility
      voiceTracks: voiceTracks,
      separatedSpeakers: separatedSpeakers
    };
    
    // Add audio object if available
    if (audio) {
      hydrated.audio = audio;
    }
    
    return hydrated;
  },

  createImportedRecordingFromResults(results, sourceMeta = {}) {
    if (!results || Object.keys(results).length === 0) {
      return null;
    }
    return {
      id: `imported_${Date.now().toString(16)}_${Math.random().toString(16).slice(2)}`,
      name: sourceMeta.audioFile || sourceMeta.name || sourceMeta.fileName || 'Imported recording',
      file: null,
      size: sourceMeta.size || null,
      duration: sourceMeta.duration || sourceMeta.audioDuration || 0,
      language: sourceMeta.language || 'ar',
      speakerCount: sourceMeta.speakerCount || sourceMeta.expectedSpeakers || '',
      status: 'completed',
      results,
      translationState: this.createTranslationState(),
      aggregated: sourceMeta.aggregated || {}
    };
  },

  /**
   * Show replicas comparison view
   */
  showReplicasComparison() {
    // Check if we have recordings instead of testResults
    if (!this.recordings || this.recordings.length === 0) {
    if (Object.keys(this.testResults).length === 0) {
      alert('No results to compare. Please load JSON or run testing first.');
      return;
      }
    }

    this.showView('replicasView');
    // Always show the recording picker if we have recordings
    const picker = document.getElementById('replicaRecordingPicker');
    if (picker && this.recordings && this.recordings.length > 0) {
      picker.style.display = 'block';
    }
    this.renderReplicaRecordingSelect();
    this.updateTranslationStatusLabel();
    this.updateTranslationControlsState();
    this.initReplicasComparison();
    
    // Update separated speakers players immediately
    const activeRecording = this.getActiveRecording();
    if (activeRecording) {
      console.log('🔄 Updating separated speakers players in showReplicasComparison...');
      this.updateSeparatedSpeakersPlayers(activeRecording);
      this.updateReplicaAudioPlayer(activeRecording.id);
    }
    
    // Auto-apply overlap fixes when opening replica view (only for non-Azure engines)
    // Use setTimeout to ensure DOM is ready
    setTimeout(() => {
      const recording = this.getActiveRecording();
      if (recording) {
        // Check if Azure engine was used
        const engine = recording.overlapMetadata?.steps?.step1?.engine || 
                      recording.results?.speechmatics?.engine ||
                      recording.results?.['overlap-corrected']?.engine;
        const isAzure = engine && (engine.toLowerCase().includes('azure') || engine.toLowerCase() === 'azure_realtime');
        
        // Update button visibility based on engine
        this.updateApplyOverlapFixesButtonVisibility(recording, isAzure);
        
        // Only auto-apply for non-Azure engines
        if (!isAzure) {
          this.autoApplyOverlapFixes(false);
        } else {
          console.log('🔵 Skipping auto-apply overlap fixes for Azure engine - manual application required');
        }
      } else {
        // Fallback: try to apply if no recording found (for backward compatibility)
        this.autoApplyOverlapFixes(false);
        // Hide button if no recording
        const button = document.getElementById('applyOverlapFixesBtn');
        if (button) {
          button.style.display = 'none';
        }
      }
    }, 100);
  },

  renderReplicaRecordingSelect() {
    const picker = document.getElementById('replicaRecordingPicker');
    const select = document.getElementById('replicaRecordingSelect');
    if (!picker || !select) return;

    if (!this.recordings || this.recordings.length === 0) {
      picker.style.display = 'none';
      select.innerHTML = '<option value="">— No Recordings —</option>';
      return;
    }

    // Always show picker if we have recordings
    picker.style.display = 'block';
    
    // Build options
    const options = this.recordings.map(rec => {
      const name = rec.name || rec.file?.name || `Recording ${rec.id}`;
      const selected = rec.id === this.activeRecordingId ? 'selected' : '';
      return `<option value="${rec.id}" ${selected}>${name}</option>`;
    }).join('');
    
    select.innerHTML = '<option value="">— Select Recording —</option>' + options;
    
    // If we have an active recording, select it
    if (this.activeRecordingId) {
      select.value = this.activeRecordingId;
    } else if (this.recordings.length > 0) {
      // Select first recording by default
      select.value = this.recordings[0].id;
      this.setActiveRecording(this.recordings[0].id);
    }
    
    console.log(`📋 Updated replica recording select: ${this.recordings.length} recordings`);
    const currentRecordingId = select.value || this.activeRecordingId;
    this.updateReplicaAudioPlayer(currentRecordingId);
  },

  handleReplicaRecordingChange(recordingId) {
    if (!recordingId || recordingId === this.activeRecordingId) return;
    this.setActiveRecording(recordingId, { skipResultsRender: true });
    this.renderReplicaRecordingSelect();
    this.updateTranslationStatusLabel();
    this.updateTranslationControlsState();
    this.initReplicasComparison();
    this.updateReplicaAudioPlayer(recordingId);
    
    // Update separated speakers players
    const recording = this.recordings.find(rec => rec.id === recordingId) || this.recordings[0];
    if (recording) {
      this.updateSeparatedSpeakersPlayers(recording);
    }
  },

  updateReplicaAudioPlayer(recordingId) {
    const wrapper = document.getElementById('replicaAudioPlayerWrapper');
    const audioEl = document.getElementById('replicaAudioPlayer');
    if (!wrapper || !audioEl) return;

    const recording = this.recordings.find(rec => rec.id === recordingId) || this.recordings[0];
    if (!recording) {
      wrapper.style.display = 'none';
      audioEl.removeAttribute('src');
      audioEl.load();
      // Hide separated speakers players
      const separatedPlayers = document.getElementById('separatedSpeakersPlayers');
      if (separatedPlayers) separatedPlayers.style.display = 'none';
      return;
    }

    let sourceUrl = recording.audioUrl || recording.source?.audioUrl || null;
    let createdObjectUrl = false;

    if (!sourceUrl && recording.file instanceof Blob) {
      if (this.replicaAudioObjectUrl) {
        URL.revokeObjectURL(this.replicaAudioObjectUrl);
        this.replicaAudioObjectUrl = null;
      }
      try {
        sourceUrl = URL.createObjectURL(recording.file);
        if (sourceUrl) {
          this.replicaAudioObjectUrl = sourceUrl;
          createdObjectUrl = true;
        }
      } catch (error) {
        console.error('Failed to create object URL from recording.file:', error);
      }
    }

    if (!sourceUrl && this.uploadedFile && recording.id === this.activeRecordingId) {
      if (this.replicaAudioObjectUrl) {
        URL.revokeObjectURL(this.replicaAudioObjectUrl);
      }
      try {
        // Validate that uploadedFile is a valid File or Blob
        if ((this.uploadedFile instanceof File) || (this.uploadedFile instanceof Blob)) {
          sourceUrl = URL.createObjectURL(this.uploadedFile);
          if (sourceUrl) {
            this.replicaAudioObjectUrl = sourceUrl;
            createdObjectUrl = true;
          }
        } else {
          console.error('Invalid uploadedFile type:', typeof this.uploadedFile);
        }
      } catch (error) {
        console.error('Failed to create object URL from uploadedFile:', error);
      }
    }

    // Check if we have separated speakers or voice tracks - if yes, show wrapper even without main audio
    const voiceTracks = this.getVoiceTracksFromRecording(recording);
    const hasVoiceTracks = voiceTracks && voiceTracks.length > 0;
    const separatedSpeakers = recording.overlapMetadata?.separation?.speakers || recording.separatedSpeakers || [];
    const hasSeparatedSpeakers = separatedSpeakers.length > 0;
    
    if (!sourceUrl) {
      // If we have separated speakers or voice tracks, show wrapper anyway
      if (hasSeparatedSpeakers || hasVoiceTracks) {
        wrapper.style.display = 'block';
        audioEl.removeAttribute('src');
        audioEl.load();
        console.log('✅ Showing replicaAudioPlayerWrapper for separated speakers even without main audio', {
          hasSeparatedSpeakers,
          hasVoiceTracks,
          separatedSpeakersCount: separatedSpeakers.length,
          voiceTracksCount: voiceTracks.length
        });
        // Don't return - continue to update separated speakers players
      } else {
        wrapper.style.display = 'none';
        audioEl.removeAttribute('src');
        audioEl.load();
        const separatedPlayers = document.getElementById('separatedSpeakersPlayers');
        if (separatedPlayers) separatedPlayers.style.display = 'none';
        if (this.replicaAudioObjectUrl && !createdObjectUrl) {
          URL.revokeObjectURL(this.replicaAudioObjectUrl);
          this.replicaAudioObjectUrl = null;
        }
        return;
      }
    } else {
      wrapper.style.display = 'block';
    }
    
    // Clean up object URL if needed
    if (this.replicaAudioObjectUrl && !createdObjectUrl && sourceUrl !== this.replicaAudioObjectUrl) {
      URL.revokeObjectURL(this.replicaAudioObjectUrl);
      this.replicaAudioObjectUrl = null;
    }
    if (audioEl.src !== sourceUrl) {
      audioEl.src = sourceUrl;
      audioEl.load();
    }
    
    // Update separated speakers players if overlap result exists
    this.updateSeparatedSpeakersPlayers(recording);
  },

  // Helper function to get voice tracks from multiple possible locations
  getVoiceTracksFromRecording(recording) {
    let tracks = recording?.overlapMetadata?.voiceTracks || [];
    if (tracks.length === 0 && recording?.voiceTracks && Array.isArray(recording.voiceTracks)) {
      console.log('⚠️ Voice tracks not found in overlapMetadata, trying top-level voiceTracks field');
      tracks = recording.voiceTracks;
    }
    return tracks;
  },

  updateSeparatedSpeakersPlayers(recording) {
    const separatedPlayers = document.getElementById('separatedSpeakersPlayers');
    const speaker0Container = document.getElementById('speaker0PlayerContainer');
    const speaker0Player = document.getElementById('speaker0Player');
    const speaker1Container = document.getElementById('speaker1PlayerContainer');
    const speaker1Player = document.getElementById('speaker1Player');
    const speaker0Transcript = document.getElementById('speaker0Transcript');
    const speaker1Transcript = document.getElementById('speaker1Transcript');
    const speaker0TranscriptMeta = document.getElementById('speaker0TranscriptMeta');
    const speaker1TranscriptMeta = document.getElementById('speaker1TranscriptMeta');
    const normalizedSegments = this.collectSegmentsForFix(recording) || [];
    
    if (!separatedPlayers || !speaker0Container || !speaker0Player || !speaker1Container || !speaker1Player) {
      return;
    }
    
    // Check if recording has overlap metadata with separated speakers
    const overlapMetadata = recording?.overlapMetadata;
    const separation = overlapMetadata?.separation;
    // Use helper function to get voice tracks from multiple possible locations
    const voiceTracks = this.getVoiceTracksFromRecording(recording);
    const speakers = separation?.speakers || [];
    
    // Log for debugging
    console.log('🔍 updateSeparatedSpeakersPlayers:', {
      hasOverlapMetadata: !!overlapMetadata,
      hasSeparation: !!separation,
      speakersCount: speakers.length,
      voiceTracksCount: voiceTracks.length,
      recordingId: recording.id
    });
    
    // If no speakers in separation, try to get from top-level separatedSpeakers
    let finalSpeakers = speakers;
    if (finalSpeakers.length === 0 && recording.separatedSpeakers && Array.isArray(recording.separatedSpeakers)) {
      console.log('⚠️ Speakers not found in overlapMetadata.separation, trying top-level separatedSpeakers field');
      finalSpeakers = recording.separatedSpeakers;
      // Also update separation object if it exists
      if (overlapMetadata && !overlapMetadata.separation) {
        overlapMetadata.separation = { speakers: finalSpeakers };
      }
    }
    
    if (finalSpeakers.length === 0) {
      separatedPlayers.style.display = 'none';
      console.log('⚠️ No separated speakers found, hiding separated players', {
        hasOverlapMetadata: !!overlapMetadata,
        hasSeparation: !!separation,
        hasTopLevelSeparatedSpeakers: !!recording.separatedSpeakers,
        topLevelSeparatedSpeakersCount: recording.separatedSpeakers?.length || 0
      });
      return;
    }
    
    separatedPlayers.style.display = 'block';
    console.log(`✅ Displaying ${finalSpeakers.length} separated speakers`, {
      speakers: finalSpeakers.map(s => ({ name: s.name, hasUrl: !!s.url, hasDownloadUrl: !!s.downloadUrl })),
      voiceTracks: voiceTracks.map(vt => ({ speaker: vt.speaker, hasAudioUrl: !!vt.audioUrl, hasTranscription: !!vt.transcription })),
      separatedPlayersId: separatedPlayers.id,
      separatedPlayersComputedDisplay: window.getComputedStyle(separatedPlayers).display,
      separatedPlayersParentDisplay: separatedPlayers.parentElement ? window.getComputedStyle(separatedPlayers.parentElement).display : 'N/A'
    });
    
    const findVoiceTrack = (speakerName) => {
      if (!speakerName || !voiceTracks.length) {
        console.log(`🔍 findVoiceTrack: no speakerName or no voiceTracks`, { speakerName, voiceTracksCount: voiceTracks.length });
        return null;
      }
      // Try exact match first
      let track = voiceTracks.find(vt => vt.speaker === speakerName);
      if (track) {
        console.log(`✅ findVoiceTrack: exact match for ${speakerName}`, { speaker: track.speaker });
        return track;
      }
      // Try normalized match (SPEAKER_00 vs SPEAKER_00)
      track = voiceTracks.find(vt => {
        const vtSpeaker = (vt.speaker || '').toUpperCase();
        const searchSpeaker = (speakerName || '').toUpperCase();
        return vtSpeaker === searchSpeaker;
      });
      if (track) {
        console.log(`✅ findVoiceTrack: normalized match for ${speakerName}`, { speaker: track.speaker });
        return track;
      }
      // Try partial match
      track = voiceTracks.find(vt => speakerName.includes(vt.speaker || '') || (vt.speaker || '').includes(speakerName));
      if (track) {
        console.log(`✅ findVoiceTrack: partial match for ${speakerName}`, { speaker: track.speaker });
        return track;
      }
      // Try number-based match (00, 01, etc.)
      const speakerNum = speakerName.match(/(\d+)/)?.[1];
      if (speakerNum) {
        track = voiceTracks.find(vt => {
          const vtNum = (vt.speaker || '').match(/(\d+)/)?.[1];
          return vtNum === speakerNum;
        });
        if (track) {
          console.log(`✅ findVoiceTrack: number-based match for ${speakerName}`, { speaker: track.speaker });
          return track;
        }
      }
      console.log(`❌ findVoiceTrack: no match found for ${speakerName}`, { availableSpeakers: voiceTracks.map(vt => vt.speaker) });
      return null;
    };
    
    const updateTranscriptField = (textarea, metaEl, trackData, speakerLabel) => {
      if (!textarea) {
        console.warn(`⚠️ Textarea not found for ${speakerLabel}`);
        return;
      }
      console.log(`🔍 Building transcript payload for ${speakerLabel}:`, {
        hasTrackData: !!trackData,
        trackDataKeys: trackData ? Object.keys(trackData) : [],
        hasTranscription: !!trackData?.transcription
      });
      const transcriptPayload = this.buildVoiceTrackTranscriptPayload(trackData, speakerLabel, normalizedSegments);
      console.log(`🔍 Transcript payload for ${speakerLabel}:`, {
        hasPayload: !!transcriptPayload,
        hasSegments: !!transcriptPayload?.segments,
        segmentsCount: transcriptPayload?.segments?.length || 0
      });
      if (transcriptPayload) {
        // Always show JSON with segments from Speechmatics
        const jsonString = JSON.stringify(transcriptPayload, null, 2);
        console.log(`✅ Setting transcript for ${speakerLabel}:`, {
          jsonLength: jsonString.length,
          hasTextarea: !!textarea,
          textareaId: textarea?.id,
          textareaDisplay: textarea ? window.getComputedStyle(textarea).display : 'N/A',
          textareaParentDisplay: textarea?.parentElement ? window.getComputedStyle(textarea.parentElement).display : 'N/A'
        });
        textarea.value = jsonString;
        textarea.placeholder = '';
        if (metaEl) {
          const parts = [];
          const jsonString = JSON.stringify(transcriptPayload, null, 2);
          parts.push(`JSON length: ${jsonString.length.toLocaleString()} chars`);
          parts.push(transcriptPayload.source ? `source: ${transcriptPayload.source}` : 'source: transcription');
          if (trackData?.roleAnalysis?.role) {
            const roleLabel = trackData.roleAnalysis.role === 'operator' ? '✅ оператор' : '🙋‍♂️ клієнт';
            const confidence = typeof trackData.roleAnalysis.confidence === 'number'
              ? ` (${Math.round(trackData.roleAnalysis.confidence * 100)}%)`
              : '';
            parts.push(`${roleLabel}${confidence}`);
          }
          if (transcriptPayload.segments) {
            parts.push(`segments: ${transcriptPayload.segments.length}`);
          }
          metaEl.textContent = parts.filter(Boolean).join(' • ');
        }
      } else {
        textarea.value = '';
        textarea.placeholder = 'Транскрипт недоступний';
        if (metaEl) {
          metaEl.textContent = speakerLabel ? `${speakerLabel}: транскрипт недоступний` : '';
        }
      }
    };
    
    // Update speaker 0 player
    const speaker0 = finalSpeakers.find(s => s.name === 'SPEAKER_00' || s.name?.includes('00'));
    if (speaker0) {
      console.log('🔍 Speaker 0 found:', {
        name: speaker0.name,
        hasUrl: !!speaker0.url,
        hasDownloadUrl: !!speaker0.downloadUrl,
        keys: Object.keys(speaker0)
      });
      speaker0Container.style.display = 'block';
      console.log('✅ Speaker 0 container display set to block', {
        containerId: speaker0Container.id,
        computedDisplay: window.getComputedStyle(speaker0Container).display,
        parentDisplay: speaker0Container.parentElement ? window.getComputedStyle(speaker0Container.parentElement).display : 'N/A'
      });
      // Try multiple URL sources: downloadUrl, url, audioUrl (from voice track)
      const speaker0Track = findVoiceTrack(speaker0.name);
      let speaker0Url = speaker0.downloadUrl || speaker0.url || speaker0Track?.audioUrl || speaker0Track?.url || speaker0Track?.downloadUrl;
      console.log('🔍 Speaker 0 URL:', speaker0Url, { 
        fromSpeaker: speaker0.downloadUrl || speaker0.url,
        fromTrack: speaker0Track?.audioUrl || speaker0Track?.url || speaker0Track?.downloadUrl
      });
      if (speaker0Url) {
        // Convert URL if needed (localtunnel to localhost)
        if (speaker0Url.includes('loca.lt') || speaker0Url.includes('ngrok') || speaker0Url.includes('tunnel')) {
          try {
            const url = new URL(speaker0Url);
            speaker0Url = `http://localhost:3000${url.pathname}`;
          } catch (e) {
            const pathMatch = speaker0Url.match(/\/uploads\/[^\/]+$/);
            if (pathMatch) {
              speaker0Url = `http://localhost:3000${pathMatch[0]}`;
            }
          }
        } else if (speaker0Url.startsWith('/uploads/') || speaker0Url.startsWith('/')) {
          speaker0Url = `http://localhost:3000${speaker0Url}`;
        }
        
        if (speaker0Player.src !== speaker0Url) {
          speaker0Player.src = speaker0Url;
          speaker0Player.load();
        }
      } else {
        speaker0Player.removeAttribute('src');
        speaker0Player.load();
      }
      // speaker0Track already found above
      console.log('🔍 Speaker 0 voice track:', {
        found: !!speaker0Track,
        speaker: speaker0Track?.speaker,
        hasTranscription: !!speaker0Track?.transcription,
        hasTranscriptText: !!speaker0Track?.transcriptText,
        keys: speaker0Track ? Object.keys(speaker0Track) : []
      });
      updateTranscriptField(speaker0Transcript, speaker0TranscriptMeta, speaker0Track, speaker0.name);
    } else {
      speaker0Container.style.display = 'none';
      if (speaker0Transcript) {
        speaker0Transcript.value = '';
      }
      if (speaker0TranscriptMeta) {
        speaker0TranscriptMeta.textContent = '';
      }
    }
    
    // Update speaker 1 player
    const speaker1 = finalSpeakers.find(s => s.name === 'SPEAKER_01' || s.name?.includes('01'));
    if (speaker1) {
      console.log('🔍 Speaker 1 found:', {
        name: speaker1.name,
        hasUrl: !!speaker1.url,
        hasDownloadUrl: !!speaker1.downloadUrl,
        keys: Object.keys(speaker1)
      });
      speaker1Container.style.display = 'block';
      console.log('✅ Speaker 1 container display set to block', {
        containerId: speaker1Container.id,
        computedDisplay: window.getComputedStyle(speaker1Container).display,
        parentDisplay: speaker1Container.parentElement ? window.getComputedStyle(speaker1Container.parentElement).display : 'N/A'
      });
      // Try multiple URL sources: downloadUrl, url, audioUrl (from voice track)
      const speaker1Track = findVoiceTrack(speaker1.name);
      let speaker1Url = speaker1.downloadUrl || speaker1.url || speaker1Track?.audioUrl || speaker1Track?.url || speaker1Track?.downloadUrl;
      console.log('🔍 Speaker 1 URL:', speaker1Url, { 
        fromSpeaker: speaker1.downloadUrl || speaker1.url,
        fromTrack: speaker1Track?.audioUrl || speaker1Track?.url || speaker1Track?.downloadUrl
      });
      if (speaker1Url) {
        // Convert URL if needed (localtunnel to localhost)
        if (speaker1Url.includes('loca.lt') || speaker1Url.includes('ngrok') || speaker1Url.includes('tunnel')) {
          try {
            const url = new URL(speaker1Url);
            speaker1Url = `http://localhost:3000${url.pathname}`;
          } catch (e) {
            const pathMatch = speaker1Url.match(/\/uploads\/[^\/]+$/);
            if (pathMatch) {
              speaker1Url = `http://localhost:3000${pathMatch[0]}`;
            }
          }
        } else if (speaker1Url.startsWith('/uploads/') || speaker1Url.startsWith('/')) {
          speaker1Url = `http://localhost:3000${speaker1Url}`;
        }
        
        if (speaker1Player.src !== speaker1Url) {
          speaker1Player.src = speaker1Url;
          speaker1Player.load();
        }
      } else {
        speaker1Player.removeAttribute('src');
        speaker1Player.load();
      }
      // speaker1Track already found above
      console.log('🔍 Speaker 1 voice track:', {
        found: !!speaker1Track,
        speaker: speaker1Track?.speaker,
        hasTranscription: !!speaker1Track?.transcription,
        hasTranscriptText: !!speaker1Track?.transcriptText,
        keys: speaker1Track ? Object.keys(speaker1Track) : []
      });
      updateTranscriptField(speaker1Transcript, speaker1TranscriptMeta, speaker1Track, speaker1.name);
    } else {
      speaker1Container.style.display = 'none';
      if (speaker1Transcript) {
        speaker1Transcript.value = '';
      }
      if (speaker1TranscriptMeta) {
        speaker1TranscriptMeta.textContent = '';
      }
    }
  },

  /**
   * Initialize replicas comparison
   */
  initReplicasComparison() {
    // Collect all unique speakers from active recording
    const allSpeakers = new Set();
    let segments = [];
    const activeRecording = this.getActiveRecording();
    
    if (activeRecording) {
      const textResult = this.getTextServiceResult(activeRecording);
      if (textResult && textResult.segments) {
        segments = textResult.segments;
        textResult.segments.forEach(seg => {
          if (seg.speaker) {
            allSpeakers.add(seg.speaker);
          }
        });
      }
    }
    
    // Fallback to testResults if no recordings
    if (allSpeakers.size === 0) {
      Object.values(this.testResults).forEach(result => {
        if (result.success && result.segments) {
          if (segments.length === 0) {
            segments = result.segments;
          }
          result.segments.forEach(seg => {
            allSpeakers.add(seg.speaker);
          });
        }
      });
    }

    // Sort speakers by their first segment timestamp (earliest first)
    const speakersArray = Array.from(allSpeakers).sort((speaker1, speaker2) => {
      const speaker1Segments = segments.filter(seg => seg.speaker === speaker1);
      const speaker2Segments = segments.filter(seg => seg.speaker === speaker2);
      
      if (speaker1Segments.length === 0 && speaker2Segments.length === 0) return 0;
      if (speaker1Segments.length === 0) return 1;
      if (speaker2Segments.length === 0) return -1;
      
      const speaker1FirstTime = Math.min(...speaker1Segments.map(seg => seg.start || 0));
      const speaker2FirstTime = Math.min(...speaker2Segments.map(seg => seg.start || 0));
      
      return speaker1FirstTime - speaker2FirstTime;
    });
    
    // Populate speaker selects
    const speaker1Select = document.getElementById('speaker1Select');
    const speaker2Select = document.getElementById('speaker2Select');
    
    speaker1Select.innerHTML = '<option value="">-- Select Speaker 1 --</option>';
    speaker2Select.innerHTML = '<option value="">-- Select Speaker 2 --</option>';

    speakersArray.forEach(speaker => {
      const option1 = document.createElement('option');
      option1.value = speaker;
      option1.textContent = speaker;
      speaker1Select.appendChild(option1);

      const option2 = document.createElement('option');
      option2.value = speaker;
      option2.textContent = speaker;
      speaker2Select.appendChild(option2);
    });

    // Auto-select first two speakers if available (first speaker = earliest timestamp)
    if (speakersArray.length >= 2) {
      speaker1Select.value = speakersArray[0]; // First speaker (earliest)
      speaker2Select.value = speakersArray[1]; // Second speaker
      this.updateReplicasComparison();
    }
  },

  /**
   * Update replicas comparison visualization
   */
  updateReplicasComparison() {
    const speaker1 = document.getElementById('speaker1Select').value;
    const speaker2 = document.getElementById('speaker2Select').value;

    if (!speaker1 || !speaker2) {
      document.getElementById('replicasComparisonContainer').innerHTML = 
        '<div class="alert alert-info">Select both speakers to start the comparison.</div>';
      return;
    }

    this.renderReplicasComparison(speaker1, speaker2);
  },

  /**
   * Render replicas comparison
   */
  renderReplicasComparison(speaker1, speaker2) {
    const container = document.getElementById('replicasComparisonContainer');
    const activeRecording = this.getActiveRecording();
    
    // Try to get segments from active recording first
    // CRITICAL FIX: Check if overlap-corrected was recently updated (has lastUpdated timestamp)
    // Priority order: recently-updated overlap-corrected > pause-based-merge > overlap-corrected > text-service
    let segments = null;
    let isOverlapCorrected = false;
    let isPauseBasedMerge = false;
    if (activeRecording) {
      const overlapCorrected = activeRecording.results?.['overlap-corrected'];
      const pauseBasedMerge = activeRecording.results?.['pause-based-merge'];
      
      // Check if overlap-corrected was recently updated (within last 5 minutes)
      const recentlyUpdated = overlapCorrected?.rawData?.lastUpdated || overlapCorrected?.rawData?.appliedAt;
      const isRecent = recentlyUpdated && (Date.now() - new Date(recentlyUpdated).getTime()) < 5 * 60 * 1000;
      
      // Priority: recently updated overlap-corrected > pause-based-merge > overlap-corrected > text-service
      if (isRecent && overlapCorrected?.segments) {
        segments = overlapCorrected.segments;
        isOverlapCorrected = true;
        console.log('📊 Using recently updated overlap-corrected segments');
      }
      // Check for pause-based-merge (if overlap-corrected not recently updated)
      else if (pauseBasedMerge && pauseBasedMerge.segments) {
        segments = pauseBasedMerge.segments;
        isPauseBasedMerge = true;
        console.log('📊 Using pause-based-merge segments');
      }
      // Check for overlap-corrected (if not recently updated)
      else if (overlapCorrected && overlapCorrected.segments) {
        segments = overlapCorrected.segments;
        isOverlapCorrected = true;
        console.log('📊 Using overlap-corrected segments');
      } 
      // Fallback to text-service
      else {
        const textResult = this.getTextServiceResult(activeRecording);
        if (textResult && textResult.segments) {
          segments = textResult.segments;
          console.log('📊 Using text-service segments');
        }
      }
    }
    
    // Fallback to testResults if no recording segments
    const successfulResults = Object.entries(this.testResults)
      .filter(([_, result]) => result.success && result.segments);

    if (!segments && successfulResults.length === 0) {
      container.innerHTML = '<div class="alert alert-warning">No successful results for comparison</div>';
      return;
    }

    // Render text-based comparison table
    let html = '<div class="replicas-comparison">';
    
    // Show info about pause-based merge if available (highest priority)
    if (isPauseBasedMerge && activeRecording?.results?.['pause-based-merge']) {
      const pauseMergeResult = activeRecording.results['pause-based-merge'];
      const rawData = pauseMergeResult.rawData || {};
      html += `
        <div class="alert alert-success" style="margin-bottom: var(--space-16);">
          <strong>✅ Pause-Based Replica Merge Applied</strong><br>
          This comparison shows results from pause-based merge algorithm with ${segments.length} segments 
          (${rawData.replicasCount || 'N/A'} replicas identified, ${rawData.additionalPhrasesCount || 0} additional phrases from voice tracks).
        </div>
      `;
    }
    // Show info about overlap correction if available
    else if (isOverlapCorrected && activeRecording?.overlapMetadata) {
      const metadata = activeRecording.overlapMetadata;
      const missingReplicas = metadata.diagnostics?.missingReplicas || [];
      if (missingReplicas.length > 0) {
        html += `
          <div class="alert alert-info" style="margin-bottom: var(--space-16);">
            <strong>✅ Overlap Correction Applied</strong><br>
            This comparison shows corrected results with ${missingReplicas.length} previously missing replica${missingReplicas.length === 1 ? '' : 's'} added from separated audio tracks.
          </div>
        `;
      }
    }
    
    if (segments) {
      const preparedSegments = this.prepareSegmentsForDisplay(segments);
      // Use segments from active recording
      const textAnalysis = activeRecording?.textAnalysis || result?.textAnalysis || null;
      
      // Діагностичне логування
      if (textAnalysis) {
        console.log('📊 renderTextComparisonTable: textAnalysis found', {
          hasBlue: !!(textAnalysis.Blue?.length),
          hasGreen: !!(textAnalysis.Green?.length),
          hasRed: !!(textAnalysis.Red?.length),
          blueCount: textAnalysis.Blue?.length || 0,
          greenCount: textAnalysis.Green?.length || 0,
          redCount: textAnalysis.Red?.length || 0
        });
      } else {
        console.warn('⚠️ renderTextComparisonTable: textAnalysis is null', {
          hasActiveRecording: !!activeRecording,
          hasResult: !!result,
          activeRecordingTextAnalysis: !!activeRecording?.textAnalysis,
          resultTextAnalysis: !!result?.textAnalysis
        });
      }
      
      html += this.renderTextComparisonTableFromSegments(speaker1, speaker2, preparedSegments, activeRecording, textAnalysis);
    } else {
      // Use testResults
      const normalizedResults = successfulResults.map(([serviceId, result]) => ([
        serviceId,
        {
          ...result,
          segments: this.prepareSegmentsForDisplay(result.segments)
        }
      ]));
      html += this.renderTextComparisonTable(speaker1, speaker2, normalizedResults);
    }
    
    html += '</div>';
    container.innerHTML = html;
  },
  
  /**
   * Render text comparison table from segments
   */
  renderTextComparisonTableFromSegments(speaker1, speaker2, segments, recording = null, textAnalysis = null) {
    const normalizedSegments = this.prepareSegmentsForDisplay(segments);
    // Filter segments by speakers
    const sortByTime = (a, b) => {
      const startA = parseFloat(a.start) || 0;
      const startB = parseFloat(b.start) || 0;
      if (startA !== startB) return startA - startB;
      const endA = parseFloat(a.end) || startA;
      const endB = parseFloat(b.end) || startB;
      return endA - endB;
    };
    
    const speaker1Segments = normalizedSegments
      .filter(seg => seg.speaker === speaker1)
      .sort(sortByTime);
    const speaker2Segments = normalizedSegments
      .filter(seg => seg.speaker === speaker2)
      .sort(sortByTime);
    
    // Get roles for speakers
    const speaker1Role = speaker1Segments[0]?.role;
    const speaker2Role = speaker2Segments[0]?.role;
    
    // Format speaker labels with roles
    const speaker1Label = speaker1Role 
      ? `${speaker1Role === 'operator' ? 'Agent' : 'Client'} (${speaker1})`
      : speaker1;
    const speaker2Label = speaker2Role 
      ? `${speaker2Role === 'operator' ? 'Agent' : 'Client'} (${speaker2})`
      : speaker2;
    
    // Build a map of missing replicas for quick lookup
    const missingReplicasMap = new Map();
    if (recording?.overlapMetadata?.diagnostics?.missingReplicas) {
      recording.overlapMetadata.diagnostics.missingReplicas.forEach(mr => {
        // Create a key based on speaker and normalized text
        const normalizedText = (mr.text || '').trim().toLowerCase().substring(0, 100);
        const key = `${mr.speaker}_${normalizedText}`;
        if (!missingReplicasMap.has(key)) {
          missingReplicasMap.set(key, mr);
        }
      });
    }
    
    // Helper to check if a segment was added from voice tracks
    const isAddedReplica = (seg) => {
      if (!recording?.overlapMetadata?.diagnostics?.missingReplicas || !seg) return false;
      const normalizedText = (seg.text || '').trim().toLowerCase().substring(0, 100);
      const key = `${seg.speaker}_${normalizedText}`;
      const missingReplica = missingReplicasMap.get(key);
      if (!missingReplica) return false;
      // Also check if time is approximately matching (within 2 seconds)
      const timeDiff = Math.abs((seg.start || 0) - (missingReplica.startTime || 0));
      return timeDiff <= 2.0;
    };
    
    // Create a map of segment index to overlaps (based on sorted segments)
    const sortedSegments = [...normalizedSegments].sort(sortByTime);
    const segmentIndexToOverlaps = {};
    if (this.detectedReplicaOverlaps) {
      Object.keys(this.detectedReplicaOverlaps).forEach(turnIndex => {
        const idx = parseInt(turnIndex);
        if (idx >= 0 && idx < sortedSegments.length) {
          const segment = sortedSegments[idx];
          // Find this segment in speaker segments
          const speaker1Idx = speaker1Segments.findIndex(s => 
            s.start === segment.start && s.end === segment.end && s.speaker === segment.speaker
          );
          const speaker2Idx = speaker2Segments.findIndex(s => 
            s.start === segment.start && s.end === segment.end && s.speaker === segment.speaker
          );
          
          if (speaker1Idx >= 0) {
            if (!segmentIndexToOverlaps[`speaker1_${speaker1Idx}`]) {
              segmentIndexToOverlaps[`speaker1_${speaker1Idx}`] = [];
            }
            segmentIndexToOverlaps[`speaker1_${speaker1Idx}`].push(...this.detectedReplicaOverlaps[turnIndex]);
          }
          if (speaker2Idx >= 0) {
            if (!segmentIndexToOverlaps[`speaker2_${speaker2Idx}`]) {
              segmentIndexToOverlaps[`speaker2_${speaker2Idx}`] = [];
            }
            segmentIndexToOverlaps[`speaker2_${speaker2Idx}`].push(...this.detectedReplicaOverlaps[turnIndex]);
          }
        }
      });
    }
    
    let html = `
      <div class="text-comparison-container">
        <div style="margin-bottom: var(--space-24);">
          <h3 style="margin-bottom: var(--space-16);">Replica Text Comparison</h3>
          <p style="color: var(--color-text-secondary); margin-bottom: var(--space-16);">
            Compare the replicas for speakers ${speaker1Label} and ${speaker2Label}
          </p>
        </div>
        <div class="text-comparison-table-wrapper">
          <table class="text-comparison-table">
            <thead>
              <tr>
                <th class="speaker-1-header">${speaker1Label}</th>
                <th class="speaker-2-header">${speaker2Label}</th>
              </tr>
            </thead>
            <tbody>
    `;
    
    const alignedRows = this.alignSpeakerSegmentsForTable(
      speaker1Segments,
      speaker2Segments,
      { maxPairGapSeconds: 1.0 }
    );

    alignedRows.forEach((row, rowIndex) => {
      const seg1 = row.speaker1Segment;
      const seg2 = row.speaker2Segment;
      
      html += '<tr>';
      
      // Speaker 1 cell
      if (seg1) {
        const overlapKeys = Array.isArray(row.speaker1Indices) && row.speaker1Indices.length > 0
          ? row.speaker1Indices
          : (row.speaker1Index != null ? [row.speaker1Index] : []);
        let overlaps1 = [];
        overlapKeys.forEach(idx => {
          const key = `speaker1_${idx}`;
          if (segmentIndexToOverlaps[key]) {
            overlaps1.push(...segmentIndexToOverlaps[key]);
          }
        });
        if (seg1?.overlap && overlaps1.length === 0) {
          const fallbackOverlap = this.createOverlapEntryFromSegment(seg1, speaker1, speaker2, seg1.overlapReason);
          if (fallbackOverlap) overlaps1.push(fallbackOverlap);
        }
        const overlapClasses1 = this.getOverlapClasses(overlaps1);
        const overlapIndicator1 = overlaps1.length > 0 ? this.renderOverlapIndicator(overlaps1) : '';
        const isAdded1 = isAddedReplica(seg1);
        const isLLMCorrected1 = seg1?.llmCorrected || recording?.results?.['overlap-corrected']?.rawData?.llmFixed;
        const addedBadge1 = isAdded1 ? '<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px;">+ Added</span>' : '';
        const llmBadge1 = isLLMCorrected1 ? '<span style="background: #6366f1; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px;" title="LLM Corrected">🤖 LLM</span>' : '';
        html += `
          <td class="replica-cell speaker-1-cell ${overlapClasses1}" ${isAdded1 ? 'style="border-left: 3px solid #10b981; background: rgba(16, 185, 129, 0.05);"' : isLLMCorrected1 ? 'style="border-left: 3px solid #6366f1; background: rgba(99, 102, 241, 0.05);"' : ''}>
            ${overlapIndicator1}
            <div class="replica-time" style="display: flex; align-items: center; justify-content: space-between;">
              <span>${this.formatTime(seg1.start)} - ${this.formatTime(seg1.end)}</span>
              <span style="display: flex; gap: 4px;">${addedBadge1}${llmBadge1}</span>
            </div>
            <div class="replica-text">${this.highlightTextWithColors(seg1.text || '', textAnalysis, seg1.start, seg1.end)}</div>
          </td>
        `;
      } else {
        html += '<td class="empty-cell">—</td>';
      }
      
      // Speaker 2 cell
      if (seg2) {
        const overlapKeys = Array.isArray(row.speaker2Indices) && row.speaker2Indices.length > 0
          ? row.speaker2Indices
          : (row.speaker2Index != null ? [row.speaker2Index] : []);
        let overlaps2 = [];
        overlapKeys.forEach(idx => {
          const key = `speaker2_${idx}`;
          if (segmentIndexToOverlaps[key]) {
            overlaps2.push(...segmentIndexToOverlaps[key]);
          }
        });
        if (seg2?.overlap && overlaps2.length === 0) {
          const fallbackOverlap = this.createOverlapEntryFromSegment(seg2, speaker2, speaker1, seg2.overlapReason);
          if (fallbackOverlap) overlaps2.push(fallbackOverlap);
        }
        const overlapClasses2 = this.getOverlapClasses(overlaps2);
        const overlapIndicator2 = overlaps2.length > 0 ? this.renderOverlapIndicator(overlaps2) : '';
        const isAdded2 = isAddedReplica(seg2);
        const isLLMCorrected2 = seg2?.llmCorrected || recording?.results?.['overlap-corrected']?.rawData?.llmFixed;
        const addedBadge2 = isAdded2 ? '<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px;">+ Added</span>' : '';
        const llmBadge2 = isLLMCorrected2 ? '<span style="background: #6366f1; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px;" title="LLM Corrected">🤖 LLM</span>' : '';
        html += `
          <td class="replica-cell speaker-2-cell ${overlapClasses2}" ${isAdded2 ? 'style="border-left: 3px solid #10b981; background: rgba(16, 185, 129, 0.05);"' : isLLMCorrected2 ? 'style="border-left: 3px solid #6366f1; background: rgba(99, 102, 241, 0.05);"' : ''}>
            ${overlapIndicator2}
            <div class="replica-time" style="display: flex; align-items: center; justify-content: space-between;">
              <span>${this.formatTime(seg2.start)} - ${this.formatTime(seg2.end)}</span>
              <span style="display: flex; gap: 4px;">${addedBadge2}${llmBadge2}</span>
            </div>
            <div class="replica-text">${this.highlightTextWithColors(seg2.text || '', textAnalysis, seg2.start, seg2.end)}</div>
          </td>
        `;
      } else {
        html += '<td class="empty-cell">—</td>';
      }
      
      html += '</tr>';
    });
    
    html += `
            </tbody>
          </table>
          ${textAnalysis ? `
          <div style="margin-top: 16px; padding: 12px; background: #f9fafb; border-radius: 8px; font-size: 12px;">
            <strong>Color Legend:</strong>
            <span style="color: #3b82f6; margin-left: 12px;">🔵 Blue</span> - Standard diarization (recognized by both general and speaker tracks)
            <span style="color: #10b981; margin-left: 12px;">🟢 Green</span> - Overlaps (our unique technology - simultaneous speech)
            <span style="color: #ef4444; margin-left: 12px;">🔴 Red</span> - Discrepancies (hallucinations - text not in general transcript)
          </div>
          ` : ''}
        </div>
      </div>
    `;
    
    return html;
  },

  /**
   * Виділяє слова в тексті кольорами на основі textAnalysis
   * Blue: повторювані фрази (звичайний діаризатор)
   * Green: overlaps (наша унікальна технологія)
   * Red: розбіжності (галюцинації)
   */
  highlightTextWithColors(text, textAnalysis, segmentStart = null, segmentEnd = null) {
    // Діагностичне логування (видалити після виправлення)
    if (!textAnalysis) {
      console.warn('🔴 highlightTextWithColors: textAnalysis is null/undefined', { text: text?.substring(0, 50) });
      return this.escapeHtml(text || '');
    }
    if (!text) {
      return '';
    }
    
    // Логування структури textAnalysis
    const hasBlue = !!(textAnalysis.Blue && Array.isArray(textAnalysis.Blue) && textAnalysis.Blue.length > 0);
    const hasGreen = !!(textAnalysis.Green && Array.isArray(textAnalysis.Green) && textAnalysis.Green.length > 0);
    const hasRed = !!(textAnalysis.Red && Array.isArray(textAnalysis.Red) && textAnalysis.Red.length > 0);
    
    if (!hasBlue && !hasGreen && !hasRed) {
      console.warn('🔴 highlightTextWithColors: textAnalysis has no data', {
        textAnalysis,
        text: text.substring(0, 50),
        segmentStart,
        segmentEnd
      });
      return this.escapeHtml(text);
    }

    // Нормалізуємо текст для порівняння (видаляємо пунктуацію, нижній регістр)
    const normalizeText = (str) => {
      if (!str || typeof str !== 'string') return '';
      return str.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
    };

    // Покращена функція для пошуку фрази в тексті з урахуванням пунктуації та пробілів
    const findPhraseInText = (text, phrase) => {
      if (!text || !phrase || typeof text !== 'string' || typeof phrase !== 'string') {
        return null;
      }
      
      phrase = phrase.trim();
      if (phrase.length < 1) return null;
      
      // Крок 1: Точний пошук (case-insensitive, без змін)
      const textLower = text.toLowerCase();
      const phraseLower = phrase.toLowerCase();
      let index = textLower.indexOf(phraseLower);
      
      if (index !== -1) {
        return { start: index, end: index + phrase.length };
      }
      
      // Крок 2: Пошук з ігноруванням пунктуації та множинних пробілів
      // Розбиваємо фразу на слова
      const phraseWords = phraseLower.split(/\s+/).filter(w => w.length > 0);
      if (phraseWords.length === 0) return null;
      
      // Якщо фраза - одне слово
      if (phraseWords.length === 1) {
        const word = phraseWords[0];
        // Шукаємо слово з word boundary
        const regex = new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
        const match = text.match(regex);
        if (match) {
          return { start: match.index, end: match.index + match[0].length };
        }
        // Якщо не знайшли, шукаємо без word boundary
        const simpleIndex = textLower.indexOf(word);
        if (simpleIndex !== -1) {
          return { start: simpleIndex, end: simpleIndex + word.length };
        }
        return null;
      }
      
      // Крок 3: Для фраз з кількох слів - шукаємо послідовність слів
      // Будуємо regex, який дозволяє множинні пробіли та пунктуацію між словами
      const escapeRegex = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const wordsPattern = phraseWords.map(w => escapeRegex(w)).join('[\\s\\p{P}]*'); // Дозволяємо пробіли та пунктуацію
      const regex = new RegExp(wordsPattern, 'iu'); // 'u' для Unicode, 'i' для case-insensitive
      const match = text.match(regex);
      
      if (match) {
        return { start: match.index, end: match.index + match[0].length };
      }
      
      // Крок 4: Альтернативний підхід - шукаємо слова з дозволом пунктуації між ними
      const flexiblePattern = phraseWords.map(w => escapeRegex(w)).join('[\\s\\p{P}]*\\s+');
      const flexibleRegex = new RegExp(flexiblePattern, 'iu');
      const flexibleMatch = text.match(flexibleRegex);
      
      if (flexibleMatch) {
        return { start: flexibleMatch.index, end: flexibleMatch.index + flexibleMatch[0].length };
      }
      
      // Крок 5: Остання спроба - пошук першого та останнього слова
      const firstWord = phraseWords[0];
      const lastWord = phraseWords[phraseWords.length - 1];
      
      const firstWordRegex = new RegExp(`\\b${escapeRegex(firstWord)}\\b`, 'i');
      const firstMatch = text.match(firstWordRegex);
      
      if (firstMatch) {
        const afterFirst = text.substring(firstMatch.index + firstMatch[0].length);
        const lastWordRegex = new RegExp(`\\b${escapeRegex(lastWord)}\\b`, 'i');
        const lastMatch = afterFirst.match(lastWordRegex);
        
        if (lastMatch) {
          const start = firstMatch.index;
          const end = firstMatch.index + firstMatch[0].length + lastMatch.index + lastMatch[0].length;
          return { start, end: Math.min(end, text.length) };
        }
      }
      
      return null;
    };

    // Збираємо всі фрази для виділення з позиціями
    const highlights = [];

    // Tolerance для перевірки часу (секунди) - дозволяє невеликі розбіжності
    const timeTolerance = 0.5;

    // Blue: повторювані фрази
    if (textAnalysis.Blue && Array.isArray(textAnalysis.Blue)) {
      textAnalysis.Blue.forEach(item => {
        if (segmentStart !== null && segmentEnd !== null) {
          const itemStart = parseFloat(item.start) || 0;
          const itemEnd = parseFloat(item.end) || itemStart;
          // Додаємо tolerance для більш гнучкої перевірки
          if (!(itemStart < segmentEnd + timeTolerance && itemEnd > segmentStart - timeTolerance)) return;
        }
        const pos = findPhraseInText(text, item.text);
        if (pos) {
          highlights.push({ ...pos, color: 'blue', type: 'repeated' });
        } else {
          // Діагностичне логування для незнайдених фраз
          console.debug('🔵 Blue phrase not found in text', {
            phrase: item.text?.substring(0, 50),
            textSample: text.substring(0, 100),
            segmentStart,
            segmentEnd
          });
        }
      });
    }

    // Green: overlaps
    if (textAnalysis.Green && Array.isArray(textAnalysis.Green)) {
      textAnalysis.Green.forEach(item => {
        if (segmentStart !== null && segmentEnd !== null) {
          const itemStart = parseFloat(item.start) || 0;
          const itemEnd = parseFloat(item.end) || itemStart;
          // Додаємо tolerance для більш гнучкої перевірки
          if (!(itemStart < segmentEnd + timeTolerance && itemEnd > segmentStart - timeTolerance)) return;
        }
        // Green містить "text1 | text2"
        const parts = item.text.split('|').map(p => p.trim()).filter(p => p.length > 0);
        parts.forEach(part => {
          const pos = findPhraseInText(text, part);
          if (pos) {
            highlights.push({ ...pos, color: 'green', type: 'overlap' });
          } else {
            // Діагностичне логування для незнайдених фраз
            console.debug('🟢 Green phrase not found in text', {
              phrase: part?.substring(0, 50),
              textSample: text.substring(0, 100),
              segmentStart,
              segmentEnd
            });
          }
        });
      });
    }

    // Red: розбіжності
    if (textAnalysis.Red && Array.isArray(textAnalysis.Red)) {
      textAnalysis.Red.forEach(item => {
        if (segmentStart !== null && segmentEnd !== null) {
          const itemStart = parseFloat(item.start) || 0;
          const itemEnd = parseFloat(item.end) || itemStart;
          // Додаємо tolerance для більш гнучкої перевірки
          if (!(itemStart < segmentEnd + timeTolerance && itemEnd > segmentStart - timeTolerance)) return;
        }
        const pos = findPhraseInText(text, item.text);
        if (pos) {
          highlights.push({ ...pos, color: 'red', type: 'discrepancy' });
        } else {
          // Діагностичне логування для незнайдених фраз
          console.debug('🔴 Red phrase not found in text', {
            phrase: item.text?.substring(0, 50),
            textSample: text.substring(0, 100),
            segmentStart,
            segmentEnd
          });
        }
      });
    }

    if (highlights.length === 0) {
      console.warn('🔴 highlightTextWithColors: No highlights found', {
        text: text.substring(0, 50),
        blueCount: textAnalysis.Blue?.length || 0,
        greenCount: textAnalysis.Green?.length || 0,
        redCount: textAnalysis.Red?.length || 0,
        segmentStart,
        segmentEnd,
        blueSamples: textAnalysis.Blue?.slice(0, 2).map(i => i?.text?.substring(0, 30)) || [],
        greenSamples: textAnalysis.Green?.slice(0, 2).map(i => i?.text?.substring(0, 30)) || [],
        redSamples: textAnalysis.Red?.slice(0, 2).map(i => i?.text?.substring(0, 30)) || []
      });
      return this.escapeHtml(text);
    }
    
    console.log('✅ highlightTextWithColors: Found highlights', {
      count: highlights.length,
      colors: highlights.map(h => h.color),
      text: text.substring(0, 50)
    });

    // Сортуємо за позицією
    highlights.sort((a, b) => a.start - b.start);

    // Об'єднуємо перекриваючі виділення (пріоритет: Red > Green > Blue)
    const merged = [];
    const priority = { red: 3, green: 2, blue: 1 };
    
    for (let i = 0; i < highlights.length; i++) {
      let current = { ...highlights[i] };
      for (let j = i + 1; j < highlights.length && highlights[j].start < current.end; j++) {
        current.end = Math.max(current.end, highlights[j].end);
        if (priority[highlights[j].color] > priority[current.color]) {
          current.color = highlights[j].color;
          current.type = highlights[j].type;
        }
        i = j;
      }
      merged.push(current);
    }

    // Будуємо HTML
    let result = '';
    let lastIndex = 0;

    merged.forEach(highlight => {
      if (highlight.start > lastIndex) {
        result += this.escapeHtml(text.substring(lastIndex, highlight.start));
      }
      
      const highlightedText = text.substring(highlight.start, Math.min(highlight.end, text.length));
      const bgColor = highlight.color === 'blue' ? '#dbeafe' : highlight.color === 'green' ? '#d1fae5' : '#fee2e2';
      const textColor = highlight.color === 'blue' ? '#1e40af' : highlight.color === 'green' ? '#065f46' : '#991b1b';
      result += `<span class="text-highlight-${highlight.color}" data-type="${highlight.type}" style="background-color: ${bgColor}; color: ${textColor}; padding: 2px 0; border-radius: 2px;">${this.escapeHtml(highlightedText)}</span>`;
      lastIndex = Math.min(highlight.end, text.length);
    });

    if (lastIndex < text.length) {
      result += this.escapeHtml(text.substring(lastIndex));
    }

    return result;
  },

  /**
   * Render text-based comparison table
   */
  renderTextComparisonTable(speaker1, speaker2, successfulResults) {
    let html = `
      <div class="text-comparison-container">
        <div style="margin-bottom: var(--space-24);">
          <h3 style="margin-bottom: var(--space-16);">Replica Text Comparison</h3>
          <p style="color: var(--color-text-secondary); margin-bottom: var(--space-16);">
            Compare how different diarization services split speaker replicas
          </p>
        </div>
    `;

    // Render table for each service
    successfulResults.forEach(([serviceId, result]) => {
      // IMPORTANT: Sort all segments by start time first to ensure correct chronological order
      const allSegmentsSorted = [...(result.segments || [])].sort((a, b) => {
        const startA = parseFloat(a.start) || 0;
        const startB = parseFloat(b.start) || 0;
        if (startA !== startB) return startA - startB;
        const endA = parseFloat(a.end) || startA;
        const endB = parseFloat(b.end) || startB;
        return endA - endB;
      });
      
      const speaker1Segments = allSegmentsSorted
        .filter(s => s.speaker === speaker1);
      
      const speaker2Segments = allSegmentsSorted
        .filter(s => s.speaker === speaker2);

      // Get roles for speakers
      const speaker1Role = speaker1Segments[0]?.role;
      const speaker2Role = speaker2Segments[0]?.role;
      
      // Format speaker labels with roles
      const speaker1Label = speaker1Role 
        ? `${speaker1Role === 'operator' ? 'Agent' : 'Client'} (${speaker1})`
        : speaker1;
      const speaker2Label = speaker2Role 
        ? `${speaker2Role === 'operator' ? 'Agent' : 'Client'} (${speaker2})`
        : speaker2;

      const alignedRows = this.alignSpeakerSegmentsForTable(
        speaker1Segments,
        speaker2Segments,
        { maxPairGapSeconds: 1.0 }
      );

      html += `
        <div class="service-text-comparison" style="margin-bottom: var(--space-32);">
          <div class="service-replicas-header" style="margin-bottom: var(--space-16);">
            <div class="service-replicas-title">${result.serviceName}</div>
            <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary);">
              ${speaker1Segments.length} replicas for ${speaker1Label} • ${speaker2Segments.length} replicas for ${speaker2Label}
            </div>
          </div>

          <div class="text-comparison-table-wrapper">
            <table class="text-comparison-table">
              <thead>
                <tr>
                  <th style="width: 80px;">№</th>
                  <th class="speaker-1-header">
                    <span class="speaker-label speaker-1">${speaker1Label}</span>
                    <div style="font-size: var(--font-size-xs); font-weight: normal; margin-top: var(--space-4);">
                      ${speaker1Segments.length} replicas
                    </div>
                  </th>
                  <th class="speaker-2-header">
                    <span class="speaker-label speaker-2">${speaker2Label}</span>
                    <div style="font-size: var(--font-size-xs); font-weight: normal; margin-top: var(--space-4);">
                      ${speaker2Segments.length} replicas
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody>
      `;

      // Create a map of segment index to overlaps for this service
      const sortedServiceSegments = [...(result.segments || [])].sort((a, b) => {
        const startA = parseFloat(a.start) || 0;
        const startB = parseFloat(b.start) || 0;
        if (startA !== startB) return startA - startB;
        const endA = parseFloat(a.end) || startA;
        const endB = parseFloat(b.end) || startB;
        return endA - endB;
      });
      const segmentIndexToOverlaps = {};
      if (this.detectedReplicaOverlaps) {
        Object.keys(this.detectedReplicaOverlaps).forEach(turnIndex => {
          const idx = parseInt(turnIndex);
          if (idx >= 0 && idx < sortedServiceSegments.length) {
            const segment = sortedServiceSegments[idx];
            // Find this segment in speaker segments
            const speaker1Idx = speaker1Segments.findIndex(s => 
              s.start === segment.start && s.end === segment.end && s.speaker === segment.speaker
            );
            const speaker2Idx = speaker2Segments.findIndex(s => 
              s.start === segment.start && s.end === segment.end && s.speaker === segment.speaker
            );
            
            if (speaker1Idx >= 0) {
              if (!segmentIndexToOverlaps[`speaker1_${speaker1Idx}`]) {
                segmentIndexToOverlaps[`speaker1_${speaker1Idx}`] = [];
              }
              segmentIndexToOverlaps[`speaker1_${speaker1Idx}`].push(...this.detectedReplicaOverlaps[turnIndex]);
            }
            if (speaker2Idx >= 0) {
              if (!segmentIndexToOverlaps[`speaker2_${speaker2Idx}`]) {
                segmentIndexToOverlaps[`speaker2_${speaker2Idx}`] = [];
              }
              segmentIndexToOverlaps[`speaker2_${speaker2Idx}`].push(...this.detectedReplicaOverlaps[turnIndex]);
            }
          }
        });
      }

      // Render rows with replicas
      alignedRows.forEach((row, rowIndex) => {
        const replica1 = row.speaker1Segment;
        const replica2 = row.speaker2Segment;

        html += '<tr>';
        html += `<td style="text-align: center; color: var(--color-text-secondary); font-weight: var(--font-weight-semibold);">${rowIndex + 1}</td>`;
        
        // Speaker 1 replica
        if (replica1) {
        const overlapKeys1 = Array.isArray(row.speaker1Indices) && row.speaker1Indices.length > 0
          ? row.speaker1Indices
          : (row.speaker1Index != null ? [row.speaker1Index] : []);
        let overlaps1 = [];
        overlapKeys1.forEach(idx => {
          const key = `speaker1_${idx}`;
          if (segmentIndexToOverlaps[key]) {
            overlaps1.push(...segmentIndexToOverlaps[key]);
          }
        });
          if (replica1?.overlap && overlaps1.length === 0) {
            const fallbackOverlap = this.createOverlapEntryFromSegment(replica1, speaker1, speaker2, replica1.overlapReason);
            if (fallbackOverlap) overlaps1.push(fallbackOverlap);
          }
          const overlapClasses1 = this.getOverlapClasses(overlaps1);
          const overlapIndicator1 = overlaps1.length > 0 ? this.renderOverlapIndicator(overlaps1) : '';
          html += `
            <td class="replica-cell speaker-1-cell ${overlapClasses1}">
              ${overlapIndicator1}
              <div class="replica-time">${this.formatTime(replica1.start)} - ${this.formatTime(replica1.end)}</div>
              <div class="replica-text">${replica1.text || '(no text)'}</div>
            </td>
          `;
        } else {
          html += '<td class="replica-cell empty-cell">—</td>';
        }

        // Speaker 2 replica
        if (replica2) {
        const overlapKeys2 = Array.isArray(row.speaker2Indices) && row.speaker2Indices.length > 0
          ? row.speaker2Indices
          : (row.speaker2Index != null ? [row.speaker2Index] : []);
        let overlaps2 = [];
        overlapKeys2.forEach(idx => {
          const key = `speaker2_${idx}`;
          if (segmentIndexToOverlaps[key]) {
            overlaps2.push(...segmentIndexToOverlaps[key]);
          }
        });
          if (replica2?.overlap && overlaps2.length === 0) {
            const fallbackOverlap = this.createOverlapEntryFromSegment(replica2, speaker2, speaker1, replica2.overlapReason);
            if (fallbackOverlap) overlaps2.push(fallbackOverlap);
          }
          const overlapClasses2 = this.getOverlapClasses(overlaps2);
          const overlapIndicator2 = overlaps2.length > 0 ? this.renderOverlapIndicator(overlaps2) : '';
          html += `
            <td class="replica-cell speaker-2-cell ${overlapClasses2}">
              ${overlapIndicator2}
              <div class="replica-time">${this.formatTime(replica2.start)} - ${this.formatTime(replica2.end)}</div>
              <div class="replica-text">${replica2.text || '(no text)'}</div>
            </td>
          `;
        } else {
          html += '<td class="replica-cell empty-cell">—</td>';
        }

        html += '</tr>';
      });

      html += `
              </tbody>
            </table>
          </div>
        </div>
      `;
    });

    html += `</div>`;
    return html;
  },

  /**
   * Get CSS classes for overlap based on confidence
   */
  getOverlapClasses(overlaps) {
    if (!overlaps || overlaps.length === 0) return '';
    
    // Get the highest confidence from all overlaps
    const maxConfidence = Math.max(...overlaps.map(o => o.confidence || 0));
    
    let classes = 'has-overlap';
    if (maxConfidence >= 0.8) {
      classes += ' high-confidence';
    } else if (maxConfidence >= 0.5) {
      classes += ' medium-confidence';
    } else {
      classes += ' low-confidence';
    }
    
    return classes;
  },

  /**
   * Render overlap indicator icon with tooltip
   */
  renderOverlapIndicator(overlaps) {
    if (!overlaps || overlaps.length === 0) return '';
    
    const maxConfidence = Math.max(...overlaps.map(o => o.confidence || 0));
    const patterns = overlaps.flatMap(o => o.detected_patterns || []).filter((v, i, a) => a.indexOf(v) === i);
    const evidence = overlaps.map(o => o.evidence).filter(Boolean).join('; ');
    
    const tooltipText = `Possible overlap detected (${(maxConfidence * 100).toFixed(0)}% confidence)\n` +
      `Patterns: ${patterns.join(', ')}\n` +
      (evidence ? `Evidence: ${evidence}` : '');
    
    return `
      <div class="overlap-indicator" title="${tooltipText.replace(/\n/g, ' ')}">
        ⚠️
        <div class="overlap-tooltip">${tooltipText.replace(/\n/g, '<br>')}</div>
      </div>
    `;
  },

  /**
   * Create fallback overlap entry when segment carries overlap flag but no structured overlaps detected
   */
  createOverlapEntryFromSegment(segment, primarySpeaker, secondarySpeaker, reason = null) {
    if (!segment || !segment.overlap) return null;
    const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
    const end = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end) || start;
    return {
      start,
      end,
      speakers: [primarySpeaker || segment.speaker, secondarySpeaker].filter(Boolean),
      source: 'segment',
      confidence: reason === 'contextual_abrupt' ? 0.5 : 0.6,
      type: 'overlap',
      evidence: reason === 'contextual_abrupt'
        ? 'Abrupt ending detected; likely interruption overlap'
        : 'Segment flagged as overlapping in diarization result',
      detected_patterns: [reason || segment.overlapReason || 'model_overlap_flag']
    };
  },

  /**
   * Render speaker comparison across all services (legacy - kept for reference)
   */
  renderSpeakerComparison(speakerLabel, servicesData, maxDuration, speakerClass) {
    const speakerColor = speakerClass === 'speaker-1' ? 'teal' : 'orange';
    
    let html = `
      <div class="speaker-comparison-section" style="margin-bottom: var(--space-32);">
        <div class="speaker-replicas-header ${speakerClass}" style="margin-bottom: var(--space-20);">
          <span class="speaker-label ${speakerClass}">${speakerLabel}</span>
          <span style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-left: var(--space-12);">
            Cross-service comparison
          </span>
        </div>
        
        <div class="unified-timeline-container" style="background: var(--color-surface); padding: var(--space-20); border-radius: var(--radius-lg); border: 2px solid var(--color-border);">
          <!-- Time scale -->
          <div class="timeline-scale" style="position: relative; height: 30px; margin-bottom: var(--space-8); border-bottom: 2px solid var(--color-border);">
    `;

    // Add time markers
    const timeMarkers = [];
    const step = Math.max(5, Math.floor(maxDuration / 15));
    for (let i = 0; i <= maxDuration; i += step) {
      timeMarkers.push(i);
    }
    if (timeMarkers[timeMarkers.length - 1] !== maxDuration) {
      timeMarkers.push(maxDuration);
    }

    timeMarkers.forEach(time => {
      const left = (time / maxDuration) * 100;
      html += `
        <div style="position: absolute; left: ${left}%; top: 0; height: 100%; border-left: 1px solid var(--color-border);">
          <div style="position: absolute; left: 2px; top: 4px; font-size: var(--font-size-xs); color: var(--color-text-secondary); font-weight: var(--font-weight-medium);">
            ${this.formatTime(time)}
          </div>
        </div>
      `;
    });

    html += `</div>`;

    // Render each service's timeline
    servicesData.forEach((serviceData, serviceIdx) => {
      const segments = serviceData.segments;
      const totalDuration = segments.reduce((sum, s) => sum + (s.end - s.start), 0);
      const segmentCount = segments.length;

      html += `
        <div class="service-timeline-row" style="margin-bottom: var(--space-16); position: relative; min-height: 60px;">
          <div style="display: flex; align-items: center; margin-bottom: var(--space-8);">
            <div style="font-weight: var(--font-weight-semibold); min-width: 150px; font-size: var(--font-size-base);">
              ${serviceData.serviceName}
            </div>
            <div style="font-size: var(--font-size-sm); color: var(--color-text-secondary); margin-left: var(--space-12);">
              ${segmentCount} replicas • ${this.formatTime(totalDuration)}
            </div>
          </div>
          
          <div class="service-timeline" style="position: relative; height: 50px; background: var(--color-bg-2); border-radius: var(--radius-base); border: 1px solid var(--color-border); overflow: visible;">
      `;

      segments.forEach((seg) => {
        const left = (seg.start / maxDuration) * 100;
        const width = ((seg.end - seg.start) / maxDuration) * 100;
        const segmentText = seg.text || '';
        const displayText = segmentText.length > 0 ? segmentText.substring(0, Math.floor(width / 2)) : '';
        
        const overlapStyle = seg.overlap ? 'border: 2px dashed var(--color-warning); box-shadow: 0 0 8px rgba(255, 193, 7, 0.5);' : '';
        html += `
          <div class="replica-segment ${speakerClass}" 
               style="left: ${left}%; width: ${width}%; height: 48px; top: 1px; min-width: ${Math.max(0.5, width)}%; ${overlapStyle}"
               title="${serviceData.serviceName} - ${seg.speaker}${seg.role ? ` (${seg.role === 'operator' ? 'Оператор' : 'Клієнт'})` : ''}${seg.overlap ? ' ⚠️ Overlap' : ''}: ${this.formatTime(seg.start)} - ${this.formatTime(seg.end)}${seg.text ? '\n' + seg.text : ''}">
            ${width > 3 ? `
              <div class="replica-segment-text" style="padding: var(--space-4) var(--space-8); font-size: ${width > 8 ? 'var(--font-size-sm)' : 'var(--font-size-xs)'}; font-weight: var(--font-weight-medium);">
                ${displayText}${segmentText.length > displayText.length ? '...' : ''}
              </div>
            ` : ''}
            ${width > 5 ? `
              <div style="position: absolute; bottom: 2px; left: 4px; font-size: 9px; color: var(--color-text-secondary); opacity: 0.8;">
                ${this.formatTime(seg.start)}
              </div>
            ` : ''}
          </div>
        `;
      });

      html += `</div></div>`;
    });

    html += `</div></div>`;
    return html;
  },

  /**
   * Render comparison table
   */
  renderComparisonTable(speaker1, speaker2, successfulResults) {
    const servicesComparison = [];
    successfulResults.forEach(([serviceId, result]) => {
      const speaker1Segments = result.segments.filter(s => s.speaker === speaker1);
      const speaker2Segments = result.segments.filter(s => s.speaker === speaker2);
      
      servicesComparison.push({
        name: result.serviceName,
        speaker1Count: speaker1Segments.length,
        speaker2Count: speaker2Segments.length,
        speaker1Duration: speaker1Segments.reduce((sum, s) => sum + (s.end - s.start), 0),
        speaker2Duration: speaker2Segments.reduce((sum, s) => sum + (s.end - s.start), 0)
      });
    });

    let html = `
      <div class="comparison-card" style="margin-top: var(--space-32);">
        <h3 style="margin-bottom: var(--space-16);">Comparison Table</h3>
        <table class="results-table">
          <thead>
            <tr>
              <th>Service</th>
              <th>${speaker1}</th>
              <th>${speaker2}</th>
              <th>Replica Difference</th>
            </tr>
          </thead>
          <tbody>
    `;

    servicesComparison.forEach(service => {
      const diff = Math.abs(service.speaker1Count - service.speaker2Count);
      html += `
        <tr>
          <td><strong>${service.name}</strong></td>
          <td>${service.speaker1Count} replicas<br><small style="color: var(--color-text-secondary);">${this.formatTime(service.speaker1Duration)}</small></td>
          <td>${service.speaker2Count} replicas<br><small style="color: var(--color-text-secondary);">${this.formatTime(service.speaker2Duration)}</small></td>
          <td><strong>${diff}</strong></td>
        </tr>
      `;
    });

    html += `
          </tbody>
        </table>
      </div>
    `;

    return html;
  },

  // Listen for audio files from Audio Generator
  setupAudioGeneratorListener() {
    // Listen for postMessage from Audio Generator
    window.addEventListener('message', (event) => {
      if (event.data && event.data.type === 'AUDIO_GENERATOR_FILE') {
        this.handleAudioGeneratorFile(event.data.file);
      }
    });
  },

  // Check localStorage for audio file from Audio Generator
  checkAudioGeneratorFile() {
    try {
      const ready = localStorage.getItem('audioGeneratorReady');
      if (ready === 'true') {
        const fileDataStr = localStorage.getItem('audioGeneratorFile');
        if (fileDataStr) {
          const fileData = JSON.parse(fileDataStr);
          this.handleAudioGeneratorFile(fileData);
          
          // Clean up
          localStorage.removeItem('audioGeneratorFile');
          localStorage.removeItem('audioGeneratorReady');
          
          // Switch to upload view if not already there
          if (window.location.hash !== '#upload') {
            this.showView('uploadView');
          }
        }
      }
    } catch (error) {
      console.error('Error checking audio generator file:', error);
    }
  },

  // Get file from IndexedDB with retry logic
  async getFileFromIndexedDB(retries = 2, delay = 200) {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const result = await new Promise((resolve, reject) => {
          const request = indexedDB.open('AudioGeneratorDB', 1);
          
          request.onerror = () => reject(request.error || new Error('IndexedDB open failed'));
          request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('files')) {
              db.createObjectStore('files');
            }
          };
          request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction(['files'], 'readonly');
            const store = transaction.objectStore('files');
            const getRequest = store.get('currentFile');
            
            transaction.oncomplete = () => db.close();
            transaction.onerror = () => {
              db.close();
              reject(transaction.error || new Error('IndexedDB transaction error'));
            };
            
            getRequest.onsuccess = () => {
              resolve(getRequest.result || null);
            };
            getRequest.onerror = () => reject(getRequest.error || new Error('IndexedDB read error'));
          };
        });
        
        if (result) {
          return result;
        }
      } catch (error) {
        if (attempt === retries) {
          throw error;
        }
      }
      
      if (attempt < retries) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw new Error('No file found in IndexedDB');
  },

  // Handle audio file from Audio Generator
  async handleAudioGeneratorFile(fileData) {
    try {
      let file;
      
      // Check if file is stored in IndexedDB
      if (fileData.storageType === 'indexeddb') {
        try {
          const storedData = await this.getFileFromIndexedDB();
          file = new File([storedData.blob], storedData.name, { type: storedData.type || 'audio/wav' });
          
          // Clean up IndexedDB after reading
          const request = indexedDB.open('AudioGeneratorDB', 1);
          request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');
            store.delete('currentFile');
            transaction.oncomplete = () => db.close();
            transaction.onerror = () => {
              console.warn('Failed to clean up IndexedDB file store');
              db.close();
            };
          };
          request.onerror = () => {
            console.warn('Failed to open IndexedDB for cleanup', request.error);
          };
        } catch (indexedDBError) {
          console.error('Error reading from IndexedDB:', indexedDBError);
          throw new Error('Could not retrieve file from storage');
        }
      } else {
        // Convert base64 back to File object (localStorage method)
        const base64Data = fileData.data.includes(',') 
          ? fileData.data.split(',')[1] 
          : fileData.data;
        
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: fileData.type || 'audio/wav' });
        file = new File([blob], fileData.name, { type: fileData.type || 'audio/wav' });
      }

      // Process the file
      this.processFile(file);
      if (this.pendingFlowIntent?.targetView === 'overlap') {
        this.showView('configView');
        this.switchDiarizationMethod('overlap');
        this.pendingFlowIntent = null;
      }
      
      // Show success message
      const statusDiv = document.createElement('div');
      statusDiv.className = 'status status-success';
      statusDiv.style.cssText = 'margin-top: var(--space-16); padding: var(--space-12); border-radius: var(--radius-base);';
      statusDiv.textContent = `✓ Audio file from Audio Generator loaded: ${fileData.name}`;
      const uploadView = document.getElementById('uploadView');
      if (uploadView) {
        uploadView.insertBefore(statusDiv, uploadView.firstChild);
        setTimeout(() => statusDiv.remove(), 5000);
      }
    } catch (error) {
      console.error('Error handling audio generator file:', error);
      alert('Error loading audio file from Audio Generator: ' + error.message);
    }
  },

  // Overlap Detector Functions
  audioFile: null,
  transcriptFile: null,
  transcriptText: '',
  detectedOverlaps: [],
  normalizeDetectedOverlaps(overlaps = []) {
    const cleaned = [];
    const seen = new Map();
    
    overlaps.forEach((overlap = {}) => {
      const range = this.getOverlapRange(overlap);
      if (!range) return;
      
      const { start, end } = range;
      if (start === null || end === null || end <= start) return;
      
      const normalized = {
        ...overlap,
        start,
        end,
        source: overlap.source || (overlap.fromAudio ? 'audio' : 'llm'),
        confidence: typeof overlap.confidence === 'number'
          ? overlap.confidence
          : (parseFloat(overlap.confidence) || 0.5),
        speakers: overlap.speakers || overlap.speakersInvolved || overlap.participants || null,
        type: overlap.type || overlap.category || 'overlap'
      };
      
      const key = `${Math.round(start * 100)}-${Math.round(end * 100)}-${normalized.source}-${normalized.type}`;
      if (seen.has(key)) {
        const idx = seen.get(key);
        const existing = cleaned[idx];
        if (normalized.confidence > existing.confidence) {
          cleaned[idx] = { ...existing, ...normalized };
        } else if (!existing.speakers && normalized.speakers) {
          cleaned[idx] = { ...existing, speakers: normalized.speakers };
        }
      } else {
        seen.set(key, cleaned.length);
        cleaned.push(normalized);
      }
    });
    
    return cleaned.sort((a, b) => a.start - b.start);
  },
  parseTimeToSeconds(value) {
    if (typeof value === 'number') return value;
    if (typeof value !== 'string') return null;
    
    const trimmed = value.trim();
    if (!trimmed) return null;
    
    const colonMatch = trimmed.match(/^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$/);
    if (colonMatch) {
      const minutes = parseInt(colonMatch[1], 10);
      const seconds = parseInt(colonMatch[2], 10);
      const millis = colonMatch[3] ? parseInt(colonMatch[3], 10) / Math.pow(10, colonMatch[3].length) : 0;
      return minutes * 60 + seconds + millis;
    }
    
    const numeric = parseFloat(trimmed);
    return Number.isFinite(numeric) ? numeric : null;
  },
  parseRangeString(rangeStr) {
    if (!rangeStr || typeof rangeStr !== 'string') return null;
    const regex = /(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s*[-–]\s*(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?/;
    const match = rangeStr.match(regex);
    if (!match) return null;
    
    const toSeconds = (minutes, seconds, fraction) => {
      const mins = parseInt(minutes, 10);
      const secs = parseInt(seconds, 10);
      const frac = fraction ? parseInt(fraction, 10) / Math.pow(10, fraction.length) : 0;
      return mins * 60 + secs + frac;
    };
    
    return {
      start: toSeconds(match[1], match[2], match[3]),
      end: toSeconds(match[4], match[5], match[6])
    };
  },
  getOverlapRange(overlap = {}) {
    const directStart = overlap.start ?? overlap.startTime ?? overlap.time_start ?? overlap.begin;
    const directEnd = overlap.end ?? overlap.endTime ?? overlap.time_end ?? overlap.finish;
    const rangeFromStrings = overlap.time_range || overlap.timeRange || overlap.range || overlap.time;
    
    let startSeconds = this.parseTimeToSeconds(directStart);
    let endSeconds = this.parseTimeToSeconds(directEnd);
    
    if ((startSeconds === null || endSeconds === null) && typeof rangeFromStrings === 'string') {
      const parsed = this.parseRangeString(rangeFromStrings);
      if (parsed) {
        startSeconds = startSeconds ?? parsed.start;
        endSeconds = endSeconds ?? parsed.end;
      }
    }
    
    if (startSeconds === null || endSeconds === null) return null;
    
    return { start: startSeconds, end: endSeconds };
  },
  rangesOverlap(aStart, aEnd, bStart, bEnd, tolerance = 0.05) {
    if ([aStart, aEnd, bStart, bEnd].some(v => typeof v !== 'number')) return false;
    return Math.max(aStart, bStart) < Math.min(aEnd, bEnd) - tolerance;
  },
  extractRangeFromLine(line) {
    if (typeof line !== 'string') return null;
    const bracketRegex = /\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s*[-–]\s*(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]/;
    const simpleRegex = /(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\s*[-–]\s*(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?/;
    
    const match = line.match(bracketRegex) || line.match(simpleRegex);
    if (!match) return null;
    
    const minutesToSeconds = (m, s, frac) => {
      const mins = parseInt(m, 10);
      const secs = parseInt(s, 10);
      const fraction = frac ? parseInt(frac, 10) / Math.pow(10, frac.length) : 0;
      return mins * 60 + secs + fraction;
    };
    
    return {
      start: minutesToSeconds(match[1], match[2], match[3]),
      end: minutesToSeconds(match[4], match[5], match[6])
    };
  },
  filterFalsePositiveOverlaps(overlaps, transcriptText, statusDiv = null) {
    if (!Array.isArray(overlaps) || overlaps.length === 0) {
      return overlaps;
    }

    const bounded = overlaps.filter(overlap => {
      if (!overlap) return false;
      const duration = overlap.end - overlap.start;
      return Number.isFinite(duration) && duration >= 0.08 && duration <= 45;
    });

    if (bounded.length === 0) {
      return [];
    }

    if (!transcriptText) {
      return bounded;
    }

    const coverage = this.calculateOverlapCoverage(bounded, transcriptText);
    if (coverage >= 0.85 && bounded.length >= 3) {
      console.warn(`Overlap coverage ${coverage} indicates likely false positive, discarding overlaps`);
      if (statusDiv) {
        statusDiv.className = 'status show warning';
        statusDiv.textContent = '⚠️ LLM позначив майже увесь діалог як перетин — ігноруємо результат. Спробуйте ще раз або використайте аудіо аналіз.';
      }
      return [];
    }

    return bounded;
  },
  calculateOverlapCoverage(overlaps, transcriptText) {
    if (!transcriptText) return 0;
    const lines = transcriptText.split(/\r?\n/).map(line => this.extractRangeFromLine(line)).filter(Boolean);
    if (lines.length === 0) return 0;

    let overlappedLines = 0;
    lines.forEach(range => {
      const hasOverlap = overlaps.some(overlap => this.rangesOverlap(range.start, range.end, overlap.start, overlap.end, 0.1));
      if (hasOverlap) overlappedLines++;
    });

    return overlappedLines / lines.length;
  },

  handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.add('drag-over');
  },

  handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('drag-over');
  },

  handleFileDrop(event, type) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('drag-over');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      if (type === 'audio') {
        this.handleAudioFile(files[0]);
      } else if (type === 'transcript') {
        this.handleTranscriptFile(files[0]);
      }
    }
  },

  handleAudioFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
      this.handleAudioFile(file);
    }
  },

  handleTranscriptFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
      this.handleTranscriptFile(file);
    }
  },

  handleAudioFile(file) {
    this.audioFile = file;
    const area = document.getElementById('audioUploadArea');
    const text = document.getElementById('audioFileText');
    const info = document.getElementById('audioFileInfo');
    
    area.classList.add('has-file');
    text.textContent = `✓ ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
    info.textContent = `Type: ${file.type || 'Unknown'} | Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  },

  async handleTranscriptFile(file) {
    this.transcriptFile = file;
    const area = document.getElementById('transcriptUploadArea');
    const text = document.getElementById('transcriptFileText');
    const info = document.getElementById('transcriptFileInfo');
    
    area.classList.add('has-file');
    text.textContent = `✓ ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
    
    // Read file content
    const reader = new FileReader();
    reader.onload = (e) => {
      this.transcriptText = e.target.result;
      document.getElementById('transcriptTextArea').value = this.transcriptText;
      info.textContent = `Loaded ${this.transcriptText.length} characters`;
    };
    reader.readAsText(file);
  },

  async detectOverlaps() {
    const button = document.getElementById('detectOverlapsBtn');
    const statusDiv = document.getElementById('overlapDetectionStatus');
    const overlapsListContainer = document.getElementById('overlapsListContainer');
    const overlapsList = document.getElementById('overlapsList');
    
    button.disabled = true;
    button.textContent = '⏳ Detecting...';
    statusDiv.className = 'status show processing';
    statusDiv.textContent = 'Starting overlap detection...';
    overlapsListContainer.style.display = 'none';
    this.detectedOverlaps = [];

    try {
      const overlaps = [];

      // Audio-based detection (Pyannote)
      if (this.audioFile) {
        statusDiv.textContent = 'Detecting overlaps in audio using Pyannote...';
        const formData = new FormData();
        formData.append('audio', this.audioFile);

        const audioResponse = await fetch('/api/overlap-audio', {
          method: 'POST',
          body: formData
        });

        if (!audioResponse.ok) {
          throw new Error(`Audio detection failed: ${audioResponse.statusText}`);
        }

        const audioData = await audioResponse.json();
        if (audioData.overlaps && Array.isArray(audioData.overlaps)) {
          overlaps.push(...audioData.overlaps.map(o => ({ ...o, source: 'audio' })));
        }
      }

      // LLM-based detection (Transcript)
      const transcriptText = document.getElementById('transcriptTextArea').value || this.transcriptText;
      if (transcriptText.trim().length > 0) {
        statusDiv.textContent = 'Detecting overlaps in transcript using LLM...';
        const llmResponse = await fetch('/api/overlap-llm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transcript: transcriptText })
        });

        if (!llmResponse.ok) {
          throw new Error(`LLM detection failed: ${llmResponse.statusText}`);
        }

        const llmData = await llmResponse.json();
        if (llmData.overlaps && Array.isArray(llmData.overlaps)) {
          overlaps.push(...llmData.overlaps.map(o => ({ ...o, source: 'llm' })));
        }
      }

      const normalizedOverlaps = this.normalizeDetectedOverlaps(overlaps);
      const filteredOverlaps = this.filterFalsePositiveOverlaps(normalizedOverlaps, transcriptText, statusDiv);

      if (filteredOverlaps.length === 0) {
        statusDiv.className = 'status show info';
        statusDiv.textContent = 'No overlaps detected.';
        button.disabled = false;
        button.textContent = '🔍 Detect Possible Overlaps';
        return;
      }

      this.detectedOverlaps = filteredOverlaps;
      this.renderOverlapsList();
      this.highlightOverlapsInTranscript(transcriptText);

      statusDiv.className = 'status show success';
      statusDiv.textContent = `✓ Detected ${filteredOverlaps.length} overlap(s)`;
      overlapsListContainer.style.display = 'block';

    } catch (error) {
      console.error('Overlap detection error:', error);
      statusDiv.className = 'status show error';
      statusDiv.textContent = `Error: ${error.message}`;
    } finally {
      button.disabled = false;
      button.textContent = '🔍 Detect Overlaps';
    }
  },

  renderOverlapsList() {
    const container = document.getElementById('overlapsList');
    if (!container) return;

    if (this.detectedOverlaps.length === 0) {
      container.innerHTML = '<p style="color: var(--color-text-secondary);">No overlaps detected.</p>';
      return;
    }

    const html = this.detectedOverlaps.map((overlap, idx) => {
      const confidenceClass = overlap.confidence >= 0.8 ? 'high-confidence' : 
                             overlap.confidence >= 0.5 ? 'medium-confidence' : 'low-confidence';
      const rangeLabel = `${this.formatTime(overlap.start)} – ${this.formatTime(overlap.end)}`;
      const speakerLabel = Array.isArray(overlap.speakers)
        ? overlap.speakers.join(', ')
        : (overlap.speakers || 'Unknown');
      const sourceLabel = overlap.source ? overlap.source.toUpperCase() : 'LLM';
      
      return `
        <div class="overlap-item ${confidenceClass}" onclick="app.highlightOverlapInTranscript(${idx})">
          <div class="overlap-header">
            <span class="overlap-source ${overlap.source || 'llm'}">${sourceLabel}</span>
            <span class="overlap-confidence" style="color: ${overlap.confidence >= 0.8 ? 'var(--color-error)' : overlap.confidence >= 0.5 ? 'var(--color-warning)' : 'var(--color-info)'};">
              ${(overlap.confidence * 100).toFixed(0)}% confidence
            </span>
          </div>
          <div class="overlap-time">
            ⏱ ${rangeLabel}
          </div>
          <div style="margin-top: var(--space-4); font-size: var(--font-size-sm); color: var(--color-text-secondary);">
            Type: ${overlap.type || 'overlap'} | Speakers: ${speakerLabel}
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = html;
  },

  highlightOverlapsInTranscript(transcriptText) {
    const container = document.getElementById('transcriptView');
    const containerDiv = document.getElementById('transcriptViewContainer');
    if (!container) return;
    
    if (!transcriptText) {
      container.innerHTML = '';
      containerDiv.style.display = 'none';
      return;
    }

    const lines = transcriptText.split(/\r?\n/);
    const html = lines.map((line, lineIdx) => {
      const range = this.extractRangeFromLine(line);
      if (!range) {
        return `<div class="transcript-line" data-line="${lineIdx}">${this.escapeHtml(line)}</div>`;
      }

      const overlapsForLine = this.detectedOverlaps.reduce((acc, overlap, idx) => {
        if (this.rangesOverlap(range.start, range.end, overlap.start, overlap.end, 0.1)) {
          acc.push({ overlap, idx });
        }
        return acc;
      }, []);

      if (overlapsForLine.length === 0) {
        return `<div class="transcript-line" data-line="${lineIdx}">${this.escapeHtml(line)}</div>`;
      }

      const chips = overlapsForLine.map(({ overlap, idx }) => {
        const sourceLabel = overlap.source ? overlap.source.toUpperCase() : 'LLM';
        const rangeLabel = `${this.formatTime(overlap.start)} – ${this.formatTime(overlap.end)}`;
        return `<span class="overlap-chip overlap-highlight" data-overlap-idx="${idx}" title="${sourceLabel} ${rangeLabel} (${Math.round(overlap.confidence * 100)}%)">
          ${sourceLabel} • ${rangeLabel}
        </span>`;
      }).join('');

      return `
        <div class="transcript-line overlap-line" data-line="${lineIdx}">
          <span class="line-text">${this.escapeHtml(line)}</span>
          <div class="overlap-chip-list">
            ${chips}
          </div>
        </div>
      `;
    }).join('');

    container.innerHTML = html;
    containerDiv.style.display = 'block';

    container.querySelectorAll('.overlap-highlight').forEach(el => {
      el.addEventListener('click', () => {
        const idx = parseInt(el.getAttribute('data-overlap-idx'), 10);
        if (!Number.isNaN(idx)) {
          this.highlightOverlapInTranscript(idx);
        }
      });
    });
  },

  async detectOverlapsFromReplicas() {
    const button = document.getElementById('detectOverlapsReplicaBtn');
    const statusDiv = document.getElementById('replicaTranslationStatus');
    
    if (!button) return;
    
    button.disabled = true;
    button.textContent = '⏳ Detecting...';
    statusDiv.className = 'status status-processing';
    statusDiv.textContent = 'Starting possible overlap detection...';

    try {
      // Get active recording and segments (same logic as renderReplicasComparison)
      const recording = this.getActiveRecording();
      let segments = null;
      
      // Try to get segments from active recording first
      if (recording) {
        const textResult = this.getTextServiceResult(recording);
        if (textResult && textResult.segments) {
          segments = textResult.segments;
        }
      }
      
      // Fallback to testResults if no recording segments
      if (!segments && this.testResults) {
        const successfulResults = Object.entries(this.testResults)
          .filter(([_, result]) => result.success && result.segments);
        
        if (successfulResults.length > 0) {
          // Use the first successful result's segments
          // Or combine all segments from all services
          segments = [];
          successfulResults.forEach(([_, result]) => {
            if (result.segments && Array.isArray(result.segments)) {
              segments.push(...result.segments);
            }
          });
          
          // Remove duplicates and sort by start time
          if (segments.length > 0) {
            segments = segments
              .filter((seg, idx, self) => 
                idx === self.findIndex(s => 
                  s.start === seg.start && 
                  s.end === seg.end && 
                  s.speaker === seg.speaker &&
                  s.text === seg.text
                )
              )
              .sort((a, b) => a.start - b.start);
          }
        }
      }

      if (!segments || segments.length === 0) {
        throw new Error('No transcript segments found. Please ensure you have processed a recording first.');
      }

      // Format transcript with turn indices
      const transcript = segments
        .map((seg, idx) => {
          return `Turn ${idx}: [${this.formatTime(seg.start)} - ${this.formatTime(seg.end)}] ${seg.speaker}: ${seg.text || ''}`;
        })
        .join('\n');

      if (!transcript.trim()) {
        throw new Error('Transcript is empty');
      }

      statusDiv.textContent = 'Detecting possible overlaps using pattern-based analysis...';

      // Call the pattern-based overlap detection API
      const response = await fetch('/api/overlap-llm-patterns', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript })
      });

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorData.details || errorMessage;
        } catch (e) {
          // If response is not JSON, use status text
          const text = await response.text().catch(() => '');
          if (text) errorMessage = text;
        }
        
        if (response.status === 404) {
          throw new Error(`API endpoint not found. Please restart the server to load the new endpoint. Error: ${errorMessage}`);
        }
        throw new Error(`Possible overlap detection failed: ${errorMessage}`);
      }

      const data = await response.json();
      const overlaps = data.overlaps || [];

      // Store overlaps mapped by turn index for rendering
      this.detectedReplicaOverlaps = {};
      overlaps.forEach(overlap => {
        const turnIndex = overlap.turn_index;
        if (!this.detectedReplicaOverlaps[turnIndex]) {
          this.detectedReplicaOverlaps[turnIndex] = [];
        }
        this.detectedReplicaOverlaps[turnIndex].push(overlap);
      });

      if (overlaps.length === 0) {
        statusDiv.className = 'status status-success';
        statusDiv.textContent = '✓ No possible overlaps detected';
        button.disabled = false;
        button.textContent = '🔍 Detect Possible Overlaps';
        // Refresh the comparison view to clear any previous overlap indicators
        this.updateReplicasComparison();
        return;
      }

      // Display results
      statusDiv.className = 'status status-success';
      statusDiv.textContent = `✓ Detected ${overlaps.length} possible overlap(s) - segments highlighted`;

      // Refresh the comparison view to show overlap indicators
      this.updateReplicasComparison();

    } catch (error) {
      console.error('Overlap detection error:', error);
      statusDiv.className = 'status status-error';
      statusDiv.textContent = `Error: ${error.message}`;
    } finally {
      button.disabled = false;
      button.textContent = '🔍 Detect Overlaps';
    }
  },

  // Helper function to collect voice track segments
  // Speechmatics determines speakers, then we map them to correct labels
  collectVoiceTrackSegments(voiceTracks, primarySegments = []) {
    if (!overlapMergeUtils || typeof overlapMergeUtils.collectVoiceTrackSegments !== 'function') {
      console.error('OverlapMergeUtils.collectVoiceTrackSegments is not available');
      return [];
    }
    return overlapMergeUtils.collectVoiceTrackSegments(voiceTracks, primarySegments, {
      logger: console
    });
  },

  async autoApplyOverlapFixes(silent = false) {
    // Auto-apply overlap fixes without showing modal
    const statusDiv = document.getElementById('replicaTranslationStatus');
    
    try {
      // Get active recording
      const recording = this.getActiveRecording();
      if (!recording) {
        if (!silent && statusDiv) {
          statusDiv.className = 'status status-error';
          statusDiv.textContent = 'No recording found. Please load a recording first.';
        }
        return false;
      }

      // Get primary diarization segments (Speechmatics result)
      const primarySegments = recording.results?.speechmatics?.segments || [];
      if (primarySegments.length === 0) {
        if (!silent && statusDiv) {
          statusDiv.className = 'status status-error';
          statusDiv.textContent = 'No primary diarization found. Please run diarization first.';
        }
        return false;
      }

      // Get voice tracks
      const voiceTracks = this.getVoiceTracksFromRecording(recording);
      if (voiceTracks.length === 0) {
        if (!silent && statusDiv) {
          statusDiv.className = 'status status-error';
          statusDiv.textContent = 'No voice tracks found. Please run overlap diarization first.';
        }
        return false;
      }

      // Check if already applied
      if (recording.results?.['overlap-corrected']?.rawData?.appliedAt) {
        if (!silent && statusDiv) {
          statusDiv.className = 'status status-success';
          statusDiv.innerHTML = '✨ <strong>Overlap Voice Recognition Fixed</strong> — Applied automatically';
        }
        return true;
      }

      // Show processing status
      if (!silent && statusDiv) {
        statusDiv.className = 'status status-info';
        statusDiv.innerHTML = '⏳ <strong>Applying Overlap Fixes...</strong> Processing voice tracks...';
      }

      // Process voice tracks - ensure they are diarized
      const processedVoiceTracks = await this.processVoiceTracksForDiarization(voiceTracks, recording.language || 'en');
      const voiceTrackTranscriptions = processedVoiceTracks.reduce((acc, track) => {
        if (track?.speaker && track?.transcription) {
          const sanitized = this.sanitizeTranscriptionForSpeaker(track.transcription, track.speaker);
          if (sanitized) {
            acc[track.speaker] = sanitized;
          }
        }
        return acc;
      }, {});
      
      // Get segments from voice tracks (already diarized)
      const voiceTrackSegments = this.collectVoiceTrackSegments(processedVoiceTracks, primarySegments);
      
      // Merge segments from voice tracks in correct format
      const mergedSegments = this.mergeVoiceTrackSegments(voiceTrackSegments, primarySegments, recording);
      
      // Apply merged segments directly
      await this.applyMergedSegments(mergedSegments, recording.id);
      
      // Show success status
      if (!silent && statusDiv) {
        statusDiv.className = 'status status-success';
        statusDiv.innerHTML = '✨ <strong>Overlap Voice Recognition Fixed</strong> — Applied automatically';
      }
      
      return true;
      
    } catch (error) {
      console.error('Error auto-applying overlap fixes:', error);
      if (!silent && statusDiv) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = `❌ Error: ${error.message}`;
      }
      return false;
    }
  },

  updateApplyOverlapFixesButtonVisibility(recording, isAzure) {
    const button = document.getElementById('applyOverlapFixesBtn');
    if (!button) return;
    
    // Hide button for Azure engines, show for Speechmatics and others
    if (isAzure) {
      button.style.display = 'none';
      console.log('🔵 Hiding Apply Overlap Fixes button for Azure engine');
    } else {
      button.style.display = '';
      console.log('✅ Showing Apply Overlap Fixes button for non-Azure engine');
    }
  },

  async applyOverlapFixes() {
    const button = document.getElementById('applyOverlapFixesBtn');
    const statusDiv = document.getElementById('replicaTranslationStatus');
    
    if (!button) return;

    try {
      // Get active recording
      const recording = this.getActiveRecording();
      if (!recording) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No recording found. Please load a recording first.';
        return;
      }

      // Get primary diarization segments (Speechmatics result)
      const primarySegments = recording.results?.speechmatics?.segments || [];
      if (primarySegments.length === 0) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No primary diarization found. Please run diarization first.';
        return;
      }

      // Get voice tracks using helper function
      const voiceTracks = this.getVoiceTracksFromRecording(recording);
      
      // Log for debugging
      console.log('🔍 Checking voice tracks:', {
        hasOverlapMetadata: !!recording.overlapMetadata,
        voiceTracksInOverlapMetadata: recording.overlapMetadata?.voiceTracks?.length || 0,
        voiceTracksTopLevel: recording.voiceTracks?.length || 0,
        finalVoiceTracksCount: voiceTracks.length,
        recordingId: recording.id
      });
      
      if (voiceTracks.length === 0) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No voice tracks found. Please run overlap diarization first.';
        console.error('❌ No voice tracks found in recording:', {
          recordingId: recording.id,
          hasOverlapMetadata: !!recording.overlapMetadata,
          overlapMetadataKeys: recording.overlapMetadata ? Object.keys(recording.overlapMetadata) : [],
          topLevelKeys: Object.keys(recording).filter(k => k.includes('voice') || k.includes('track'))
        });
        return;
      }
      
      console.log(`✅ Found ${voiceTracks.length} voice tracks for processing`);

      // Show modal with loading state
      this.showOverlapFixesModal(recording, primarySegments, voiceTracks);
      
    } catch (error) {
      console.error('Error opening overlap fixes modal:', error);
      statusDiv.className = 'status status-error';
      statusDiv.textContent = `❌ Error: ${error.message}`;
    }
  },

  async applyMarkdownFixes() {
    const button = document.getElementById('applyMarkdownFixesBtn');
    const statusDiv = document.getElementById('replicaTranslationStatus');
    
    if (!button) return;

    try {
      // Get active recording
      const recording = this.getActiveRecording();
      if (!recording) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No recording found. Please load a recording first.';
        return;
      }

      // Get primary diarization segments (Speechmatics result)
      const primarySegments = recording.results?.speechmatics?.segments || [];
      if (primarySegments.length === 0) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No primary diarization found. Please run diarization first.';
        return;
      }

      // Get voice tracks
      const voiceTracks = recording.overlapMetadata?.voiceTracks || [];
      if (voiceTracks.length === 0) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No voice tracks found. Please run overlap diarization first.';
        return;
      }

      // Get JSON transcripts from textarea fields (Agent and Client)
      const speaker0Transcript = document.getElementById('speaker0Transcript');
      const speaker1Transcript = document.getElementById('speaker1Transcript');
      
      if (!speaker0Transcript || !speaker1Transcript) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'Transcript fields not found. Please ensure separated speakers are displayed.';
        return;
      }
      
      let agentTranscriptJSON = null;
      let clientTranscriptJSON = null;
      
      try {
        const agentText = speaker0Transcript.value.trim();
        const clientText = speaker1Transcript.value.trim();
        
        if (!agentText || !clientText) {
          statusDiv.className = 'status status-error';
          statusDiv.textContent = 'Transcripts are empty. Please ensure voice tracks are transcribed.';
          return;
        }
        
        // Parse JSON transcripts
        agentTranscriptJSON = JSON.parse(agentText);
        clientTranscriptJSON = JSON.parse(clientText);
        
        console.log('📋 Agent transcript:', {
          hasSegments: !!agentTranscriptJSON.segments,
          segmentsCount: agentTranscriptJSON.segments?.length || 0
        });
        console.log('📋 Client transcript:', {
          hasSegments: !!clientTranscriptJSON.segments,
          segmentsCount: clientTranscriptJSON.segments?.length || 0
        });
        
      } catch (parseError) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = `Failed to parse transcripts as JSON: ${parseError.message}`;
        return;
      }

      // Show loading state
      button.disabled = true;
      statusDiv.className = 'status status-info';
      statusDiv.textContent = '🔄 Processing with LLM (Markdown format)...';

      // Get LLM mode from radio buttons (replicaLLMModel) or fallback to config
      const modeRadio = document.querySelector('input[name="replicaLLMModel"]:checked');
      const mode = modeRadio ? modeRadio.value : (this.config?.llmMode || 'smart');
      console.log('🔍 Using LLM mode for Markdown Fixes:', mode);

      // Send to backend for Markdown processing with JSON transcripts
      const response = await fetch('/api/apply-markdown-fixes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          agentTranscript: agentTranscriptJSON,
          clientTranscript: clientTranscriptJSON,
          mode: mode,
          recordingId: recording.id
        })
      });

      if (!response.ok) {
        // Try to parse error as JSON, but handle HTML error pages
        let errorMessage = 'Failed to apply markdown fixes';
        try {
          const errorText = await response.text();
          // Check if response is JSON
          if (errorText.trim().startsWith('{')) {
            const error = JSON.parse(errorText);
            errorMessage = error.error || error.message || errorMessage;
          } else {
            // HTML error page
            errorMessage = `Server error (${response.status}): ${response.statusText}`;
            console.error('Server returned HTML instead of JSON:', errorText.substring(0, 200));
          }
        } catch (parseError) {
          errorMessage = `Server error (${response.status}): ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      // Parse response as JSON
      const resultText = await response.text();
      let result;
      try {
        result = JSON.parse(resultText);
      } catch (parseError) {
        console.error('Failed to parse response as JSON:', parseError);
        console.error('Response text:', resultText.substring(0, 500));
        throw new Error('Server returned invalid JSON response');
      }
      
      // Render Markdown table
      this.renderMarkdownTable(result.markdown, statusDiv);
      
      button.disabled = false;
      
    } catch (error) {
      console.error('Error applying markdown fixes:', error);
      statusDiv.className = 'status status-error';
      statusDiv.textContent = `❌ Error: ${error.message}`;
      button.disabled = false;
    }
  },

  renderMarkdownTable(markdown, statusDiv) {
    if (!markdown) {
      statusDiv.className = 'status status-error';
      statusDiv.textContent = 'No markdown content received.';
      return;
    }

    console.log('📋 Rendering markdown table, length:', markdown.length);
    console.log('📋 First 500 chars:', markdown.substring(0, 500));

    // Create container for markdown table
    const markdownContainer = document.createElement('div');
    markdownContainer.id = 'markdownFixesContainer';
    markdownContainer.style.marginTop = 'var(--space-16)';
    markdownContainer.style.padding = 'var(--space-16)';
    markdownContainer.style.backgroundColor = '#000000';
    markdownContainer.style.borderRadius = 'var(--radius-md, 8px)';
    markdownContainer.style.maxHeight = '600px';
    markdownContainer.style.overflowY = 'auto';
    markdownContainer.style.color = '#ffffff';

    // Parse markdown table and render as HTML
    const lines = markdown.split('\n');
    console.log('📋 Total lines:', lines.length);
    
    let html = '<table class="table" style="width: 100%; border-collapse: collapse; margin: 0; color: #ffffff; background-color: #000000;">';
    let inTable = false;
    let headerProcessed = false;
    let hasTableRows = false;
    let rowCount = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim();
      
      // Check if this is a markdown table row
      if (trimmedLine.startsWith('|') && trimmedLine.endsWith('|')) {
        hasTableRows = true;
        
        if (!inTable) {
          inTable = true;
          html += '<thead><tr>';
        }
        
        // Split by | - markdown tables have | at start and end, so we get empty strings
        // Example: "| col1 | col2 |" -> ["", " col1 ", " col2 ", ""]
        const parts = trimmedLine.split('|');
        const cells = [];
        
        // Process parts: skip first empty, process middle, skip last empty
        for (let j = 0; j < parts.length; j++) {
          const part = parts[j];
          // First and last parts are usually empty (outside table boundaries)
          if (j === 0 && part.trim() === '') continue;
          if (j === parts.length - 1 && part.trim() === '') continue;
          cells.push(part.trim());
        }
        
        // Skip separator row (|---|---| or |:---:|)
        const isSeparator = cells.length > 0 && cells.every(c => {
          const trimmed = c.trim();
          return trimmed === '' || trimmed.match(/^:?-+:?$/) || trimmed.match(/^:?-+:?$/);
        });
        
        if (isSeparator) {
          if (!headerProcessed && inTable) {
            html += '</tr></thead><tbody>';
            headerProcessed = true;
          }
          continue;
        }
        
        // Valid data row
        rowCount++;
        html += '<tr>';
        cells.forEach(cell => {
          const tag = headerProcessed ? 'td' : 'th';
          const bgColor = headerProcessed ? '#000000' : '#1a1a1a';
          const cellText = this.escapeHtml(cell);
          html += `<${tag} style="padding: var(--space-8); border: 1px solid #333333; text-align: left; color: #ffffff; background-color: ${bgColor};">${cellText}</${tag}>`;
        });
        html += '</tr>';
        
        if (rowCount <= 3 || rowCount === rowCount) { // Log first 3 rows and last row
          console.log(`📋 Row ${rowCount}:`, cells);
        }
      } else if (inTable && trimmedLine === '' && headerProcessed) {
        // Empty line after table body - might indicate end, but continue in case there's more
        continue;
      } else if (inTable && !trimmedLine.startsWith('|') && headerProcessed) {
        // Non-table line after table started - might be end of table
        // But continue to see if there are more table rows
        continue;
      }
    }
    
    // Close table if still open
    if (inTable) {
      if (headerProcessed) {
        html += '</tbody>';
      } else {
        html += '</tr></thead>';
      }
    }
    
    html += '</table>';

    console.log('📋 Parsed table rows:', rowCount);
    console.log('📋 HTML length:', html.length);

    // If no table found, render as preformatted text
    if (!hasTableRows) {
      console.warn('⚠️ No table rows found, rendering as preformatted text');
      html = `<div style="white-space: pre-wrap; font-family: monospace; padding: var(--space-16); background: #000000; color: #ffffff; border-radius: var(--radius-md, 8px); border: 1px solid #333333;">${this.escapeHtml(markdown)}</div>`;
    }

    // Remove existing container if present
    const existing = document.getElementById('markdownFixesContainer');
    if (existing) {
      existing.remove();
    }

    // Set innerHTML after removing existing container
    markdownContainer.innerHTML = html;
    
    console.log('✅ Markdown table rendered');
    console.log('📋 HTML preview (first 1000 chars):', html.substring(0, 1000));
    console.log('📋 Container innerHTML length:', markdownContainer.innerHTML.length);
    console.log('📋 Container children count:', markdownContainer.children.length);
    
    // Verify table was created
    const table = markdownContainer.querySelector('table');
    if (table) {
      const rows = table.querySelectorAll('tr');
      console.log('📋 Table found with', rows.length, 'rows');
      if (rows.length === 0) {
        console.error('❌ Table has no rows!');
      }
    } else {
      console.error('❌ No table element found in container!');
      console.error('📋 Container HTML:', markdownContainer.innerHTML.substring(0, 500));
    }

    // Insert after status div's parent (or after status div if it has no parent)
    const parent = statusDiv.parentNode;
    if (parent) {
      parent.insertBefore(markdownContainer, statusDiv.nextSibling);
    } else {
      statusDiv.insertAdjacentElement('afterend', markdownContainer);
    }
    
    console.log('✅ Container inserted into DOM');
    console.log('📋 Final container HTML length:', markdownContainer.innerHTML.length);
    
    statusDiv.className = 'status status-success';
    statusDiv.textContent = `✅ Markdown fixes applied successfully (${rowCount} rows)`;
  },

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  },

  async showOverlapFixesModal(recording, primarySegments, voiceTracks) {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'flex';
    modal.id = 'overlapFixesModal';
    
    // Initial loading state
    modal.innerHTML = `
      <div class="modal-content" style="max-width: 95%; max-height: 95%;">
        <div class="modal-header">
          <h2>✅ Apply Overlap Fixes</h2>
          <button class="btn-close" onclick="document.getElementById('overlapFixesModal').remove()">✕</button>
        </div>
        <div class="modal-body">
          <div style="text-align: center; padding: var(--space-32);">
            <div class="spinner" style="margin: 0 auto;"></div>
            <p style="margin-top: var(--space-16);">Processing voice tracks and preparing data...</p>
          </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);

    try {
      // Process voice tracks - ensure they are diarized
      const processedVoiceTracks = await this.processVoiceTracksForDiarization(voiceTracks, recording.language || 'en');
      
      // Build map of voice track transcriptions by speaker
      const voiceTrackTranscriptions = processedVoiceTracks.reduce((acc, track) => {
        if (track?.speaker && track?.transcription) {
          const sanitized = this.sanitizeTranscriptionForSpeaker(track.transcription, track.speaker);
          if (sanitized) {
            acc[track.speaker] = sanitized;
          }
        }
        return acc;
      }, {});
      
      // Get segments from voice tracks (already diarized)
      // Pass primary segments for speaker mapping
      const voiceTrackSegments = this.collectVoiceTrackSegments(processedVoiceTracks, primarySegments);
      
      // Merge segments from voice tracks in correct format
      const mergedSegments = this.mergeVoiceTrackSegments(voiceTrackSegments, primarySegments, recording);
      
      // Get speaker segments from merged result (includes both primary and voice tracks)
      const speaker0Segments = mergedSegments.filter(s => s.speaker === 'SPEAKER_00');
      const speaker1Segments = mergedSegments.filter(s => s.speaker === 'SPEAKER_01');
      
      // Create JSON structures - use voice track transcription if available, otherwise fall back to main recording
      const speaker0RecordingSource = voiceTrackTranscriptions['SPEAKER_00'] || recording;
      const speaker1RecordingSource = voiceTrackTranscriptions['SPEAKER_01'] || recording;
      const speaker0JSON = this.createSpeakerJSON(speaker0RecordingSource, speaker0Segments, 'SPEAKER_00');
      const speaker1JSON = this.createSpeakerJSON(speaker1RecordingSource, speaker1Segments, 'SPEAKER_01');
      const mergedJSON = this.createMergedJSON(recording, mergedSegments);
      
      // Update modal with content
      this.updateOverlapFixesModal(modal, speaker0JSON, speaker1JSON, mergedJSON, recording, mergedSegments);
      
    } catch (error) {
      console.error('Error processing overlap fixes:', error);
      modal.querySelector('.modal-body').innerHTML = `
        <div class="alert alert-error">
          <strong>Error:</strong> ${error.message}
        </div>
      `;
    }
  },

  async processVoiceTracksForDiarization(voiceTracks, language) {
    const processedTracks = [];
    
    for (const track of voiceTracks) {
      if (track.error) {
        processedTracks.push(track);
        continue;
      }
      
      // Check if already diarized
      const existingSegments = track.transcription?.recordings?.[0]?.results?.speechmatics?.segments;
      if (existingSegments && existingSegments.length > 0) {
        // Already diarized, use existing
        console.log(`✅ Voice track ${track.speaker} already diarized (${existingSegments.length} segments)`);
        processedTracks.push(track);
        continue;
      }
      
      // Need to diarize - get audio file path or URL
      const audioPath = track.local_path;
      const audioUrl = track.url;
      
      if (!audioPath && !audioUrl) {
        console.warn('Voice track has no audio path or URL:', track);
        processedTracks.push(track);
        continue;
      }
      
      try {
        console.log(`🔄 Diarizing voice track ${track.speaker}...`);
        
        // Call backend API to diarize voice track
        // IMPORTANT: Don't use isSeparatedTrack=true, so Speechmatics can detect multiple speakers
        // (voice tracks may contain residual audio from other speakers)
        // We'll map detected speakers to correct labels afterwards
        const response = await fetch('/api/diarize-voice-track', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            local_path: audioPath,
            url: audioUrl,
            language: language,
            speakerCount: '2', // Allow detection of multiple speakers (main + residual)
            isSeparatedTrack: false // Let Speechmatics detect all speakers
          })
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
          throw new Error(errorData.error || `Diarization failed: ${response.statusText}`);
        }
        
        const diarizeResult = await response.json();
        
        // Update track with transcription
        const updatedTrack = {
          ...track,
          transcription: diarizeResult
        };
        
        processedTracks.push(updatedTrack);
        console.log(`✅ Voice track ${track.speaker} diarized successfully`);
        
      } catch (error) {
        console.error(`❌ Error diarizing voice track ${track.speaker}:`, error);
        // Continue with original track (may have partial transcription)
        processedTracks.push(track);
      }
    }
    
    return processedTracks;
  },

  // NOTE: This function is DEPRECATED and will be replaced by the proxy version from overlap_merge_utils.js
  // See bottom of file where mergeVoiceTrackSegments is reassigned to use overlap_merge_utils version
  // This local version is kept for backward compatibility but should not be used
  mergeVoiceTrackSegments(voiceTrackSegments, primarySegments, recording) {
    // CRITICAL: Voice track segments now have correct speaker labels (from track, not from Speechmatics)
    // Voice track segments have timestamps relative to the original audio
    // So we can directly merge them with primary segments
    
    console.log(`🔄 Merging segments: ${voiceTrackSegments.length} voice track + ${primarySegments.length} primary`);
    console.warn(`⚠️ WARNING: Using DEPRECATED local mergeVoiceTrackSegments. Should use overlap_merge_utils version.`);
    
    // CRITICAL: Validate all voice track segments have valid speaker labels
    const invalidVoiceTracks = voiceTrackSegments.filter(seg => {
      const spk = seg.speaker || '';
      return !spk || !spk.startsWith('SPEAKER_');
    });
    if (invalidVoiceTracks.length > 0) {
      console.error(`❌ Found ${invalidVoiceTracks.length} voice track segments with invalid speaker labels:`, invalidVoiceTracks.map(s => s.speaker));
      // Filter out invalid segments
      voiceTrackSegments = voiceTrackSegments.filter(seg => {
        const spk = seg.speaker || '';
        return spk && spk.startsWith('SPEAKER_');
      });
    }
    
    // Group voice track segments by speaker to verify correct labeling
    const voiceTracksBySpeaker = {};
    voiceTrackSegments.forEach(seg => {
      const spk = seg.speaker || 'UNKNOWN';
      if (!voiceTracksBySpeaker[spk]) voiceTracksBySpeaker[spk] = [];
      voiceTracksBySpeaker[spk].push(seg);
    });
    console.log(`📊 Voice track segments by speaker:`, Object.keys(voiceTracksBySpeaker).map(s => `${s}: ${voiceTracksBySpeaker[s].length}`).join(', '));
    
    // CRITICAL: Log sample segments to verify speaker attribution
    if (voiceTrackSegments.length > 0) {
      console.log(`📋 Sample voice track segments (first 3):`);
      voiceTrackSegments.slice(0, 3).forEach((seg, idx) => {
        console.log(`  [${idx}] Speaker: ${seg.speaker}, Text: "${(seg.text || '').substring(0, 50)}...", Time: ${seg.start}-${seg.end}`);
      });
    }
    
    // STEP 1: Remove duplicates within voice track segments first
    // This handles cases where:
    // 1. The same text appears in multiple voice tracks or is split multiple times
    // 2. Speechmatics transcribes the same audio fragment twice with different texts
    //    (e.g., "What do you have for me" vs "Yes For me" for the same time range)
    const deduplicatedVoiceTracks = [];
    
    // Sort by start time for easier comparison
    const sortedVoiceTracks = [...voiceTrackSegments].sort((a, b) => {
      const startDiff = (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0);
      if (startDiff !== 0) return startDiff;
      return (parseFloat(a.end) || 0) - (parseFloat(b.end) || 0);
    });
    
    for (let i = 0; i < sortedVoiceTracks.length; i++) {
      const current = sortedVoiceTracks[i];
      const currentStart = parseFloat(current.start) || 0;
      const currentEnd = parseFloat(current.end) || currentStart;
      const currentSpeaker = current.speaker || 'SPEAKER_00';
      const currentText = (current.text || '').trim().toLowerCase();
      const currentDuration = currentEnd - currentStart;
      
      let isDuplicate = false;
      
      // Check against already added segments (same speaker only)
      for (const existing of deduplicatedVoiceTracks) {
        if (existing.speaker !== currentSpeaker) continue;
        
        const existingStart = parseFloat(existing.start) || 0;
        const existingEnd = parseFloat(existing.end) || existingStart;
        const existingText = (existing.text || '').trim().toLowerCase();
        const existingDuration = existingEnd - existingStart;
        
        // Calculate temporal overlap
        const overlapStart = Math.max(currentStart, existingStart);
        const overlapEnd = Math.min(currentEnd, existingEnd);
        const overlapDuration = Math.max(0, overlapEnd - overlapStart);
        
        // Calculate overlap percentage for both segments
        const currentOverlapRatio = currentDuration > 0 ? overlapDuration / currentDuration : 0;
        const existingOverlapRatio = existingDuration > 0 ? overlapDuration / existingDuration : 0;
        
        // Check text similarity
        const textSimilarity = currentText.includes(existingText) || existingText.includes(currentText);
        
        // CRITICAL: Mark as duplicate if:
        // 1. High temporal overlap (>80% of either segment) - same audio fragment, different transcription
        // 2. Text similarity AND any temporal overlap - same text, overlapping time
        const highTemporalOverlap = currentOverlapRatio > 0.8 || existingOverlapRatio > 0.8;
        const textAndTimeOverlap = textSimilarity && overlapDuration > 0.1;
        
        if (highTemporalOverlap || textAndTimeOverlap) {
          // Keep the segment with longer/more complete text
          if (currentText.length > existingText.length) {
            // Replace existing with current
            const index = deduplicatedVoiceTracks.indexOf(existing);
            if (index >= 0) {
              deduplicatedVoiceTracks[index] = current;
            }
          }
          // Mark as duplicate and skip
          isDuplicate = true;
          break;
        }
      }
      
      if (!isDuplicate) {
        deduplicatedVoiceTracks.push(current);
      }
    }
    
    console.log(`✅ Deduplicated voice tracks: ${voiceTrackSegments.length} → ${deduplicatedVoiceTracks.length} (removed ${voiceTrackSegments.length - deduplicatedVoiceTracks.length} duplicates)`);
    
    // STEP 2: Start with deduplicated voice track segments
    const mergedSegments = [...deduplicatedVoiceTracks];
    
    // Helper function to validate short segments using contextual checks
    // Uses script-based validation (faster, more predictable than LLM)
    const validateShortSegment = (seg, contextSegments, voiceTracks) => {
      const start = parseFloat(seg.start) || 0;
      const end = parseFloat(seg.end) || start;
      const duration = end - start;
      const speaker = seg.speaker || 'SPEAKER_00';
      const text = (seg.text || '').trim().toLowerCase();
      
      // Only validate segments shorter than 0.8 seconds
      if (duration >= 0.8) return { valid: true, reason: 'duration_ok' };
      
      // Check 1: Significant voice track overlap (strongest signal - if voice track confirms it, it's valid)
      const hasSignificantVoiceTrackOverlap = voiceTracks.some(vtSeg => {
        const vtStart = parseFloat(vtSeg.start) || 0;
        const vtEnd = parseFloat(vtSeg.end) || vtStart;
        const vtSpeaker = vtSeg.speaker || 'SPEAKER_00';
        
        if (vtSpeaker !== speaker) return false;
        
        const overlapStart = Math.max(start, vtStart);
        const overlapEnd = Math.min(end, vtEnd);
        const overlapDuration = Math.max(0, overlapEnd - overlapStart);
        const minOverlap = Math.max(duration * 0.5, 0.2);
        return overlapDuration >= minOverlap;
      });
      
      if (hasSignificantVoiceTrackOverlap) {
        return { valid: true, reason: 'voice_track_overlap' };
      }
      
      // Check 2: Chronological validity - find previous segment from same speaker in context
      // CRITICAL: Check for overlapping segments from the same speaker (chronological violation)
      const overlappingSameSpeaker = contextSegments.find(s => {
        const sStart = parseFloat(s.start) || 0;
        const sEnd = parseFloat(s.end) || sStart;
        const sSpeaker = s.speaker || 'SPEAKER_00';
        
        if (sSpeaker !== speaker) return false;
        // Skip if it's the same segment (by comparing timestamps)
        if (Math.abs(sStart - start) < 0.01 && Math.abs(sEnd - end) < 0.01) return false;
        
        // Check for temporal overlap
        const overlapStart = Math.max(start, sStart);
        const overlapEnd = Math.min(end, sEnd);
        const overlapDuration = overlapEnd - overlapStart;
        
        // If there's significant overlap (>0.1s), it's a chronological violation
        return overlapDuration > 0.1;
      });
      
      if (overlappingSameSpeaker) {
        return { valid: false, reason: 'chronological_violation_overlap' };
      }
      
      // Also check if this segment starts before previous segment ends (even without overlap)
      const previousSameSpeaker = contextSegments
        .filter(s => {
          const sStart = parseFloat(s.start) || 0;
          const sSpeaker = s.speaker || 'SPEAKER_00';
          return sStart < start && sSpeaker === speaker;
        })
        .sort((a, b) => (parseFloat(b.start) || 0) - (parseFloat(a.start) || 0))[0];
      
      if (previousSameSpeaker) {
        const prevEnd = parseFloat(previousSameSpeaker.end) || parseFloat(previousSameSpeaker.start) || 0;
        // If this segment starts before previous ends (with small tolerance), it's likely an artifact
        if (start < prevEnd - 0.05) { // Reduced tolerance to 0.05s for stricter checking
          return { valid: false, reason: 'chronological_violation_start_before_end' };
        }
      }
      
      // Check 3: Context check - is there a nearby segment from same speaker?
      const nearbySameSpeaker = contextSegments.some(s => {
        const sStart = parseFloat(s.start) || 0;
        const sEnd = parseFloat(s.end) || sStart;
        const sSpeaker = s.speaker || 'SPEAKER_00';
        
        if (sSpeaker !== speaker) return false;
        if (Math.abs(sStart - start) < 0.01 && Math.abs(sEnd - end) < 0.01) return false; // Same segment
        
        // Check if within 3 seconds (reasonable dialogue context)
        const gap = Math.min(Math.abs(sStart - start), Math.abs(sEnd - end));
        return gap < 3.0;
      });
      
      // Check 4: Text type - common confirmation words are more likely to be valid
      const confirmationWords = ['right', 'yes', 'ok', 'okay', 'sure', 'works', 'fine', 'good', 'alright', 'yeah', 'yep', 'uh-huh', 'correct', 'exactly'];
      const isConfirmationWord = confirmationWords.some(word => {
        const wordLower = word.toLowerCase();
        return text === wordLower || text === wordLower + '.' || text.startsWith(wordLower + ' ') || text.endsWith(' ' + wordLower);
      });
      
      // If it's a confirmation word AND has nearby same-speaker context, it's likely valid
      if (isConfirmationWord && nearbySameSpeaker) {
        return { valid: true, reason: 'confirmation_word_with_context' };
      }
      
      // Check 5: If no nearby same-speaker context, it's suspicious (likely artifact)
      if (!nearbySameSpeaker) {
        return { valid: false, reason: 'no_context' };
      }
      
      // Check 6: Check if there's an overlapping segment from different speaker
      const hasOverlappingDifferentSpeaker = contextSegments.some(s => {
        const sStart = parseFloat(s.start) || 0;
        const sEnd = parseFloat(s.end) || sStart;
        const sSpeaker = s.speaker || 'SPEAKER_00';
        
        if (sSpeaker === speaker) return false;
        
        const overlapStart = Math.max(start, sStart);
        const overlapEnd = Math.min(end, sEnd);
        return overlapEnd > overlapStart + 0.1; // At least 100ms overlap
      });
      
      // If overlapping with different speaker and no voice track confirmation, likely artifact
      if (hasOverlappingDifferentSpeaker) {
        return { valid: false, reason: 'overlaps_different_speaker' };
      }
      
      // Default: if has context and passes other checks, allow it
      return { valid: true, reason: 'has_context' };
    };
    
    // STEP 3: Add primary segments that don't overlap significantly with voice tracks
    // Voice tracks take priority because they have complete text
    primarySegments.forEach(pSeg => {
      const pStart = parseFloat(pSeg.start) || 0;
      const pEnd = parseFloat(pSeg.end) || pStart;
      const pDuration = pEnd - pStart;
      const pSpeaker = pSeg.speaker || 'SPEAKER_00';
      const pText = (pSeg.text || '').trim().toLowerCase();
      
      // Validate short segments using contextual checks
      // Use primarySegments for context (not mergedSegments, which is still being built)
      if (pDuration < 0.8) {
        const validation = validateShortSegment(pSeg, primarySegments, deduplicatedVoiceTracks);
        if (!validation.valid) {
          console.log(`⚠️ Skipping short primary segment (${pDuration.toFixed(2)}s) "${pText.substring(0, 30)}..." - ${validation.reason}`);
          return;
        }
      }
      
      // CRITICAL: Check if this primary segment overlaps with voice tracks
      // 1. Check overlap with SAME speaker voice tracks (skip if overlaps - voice track has better text)
      // 2. Check overlap with DIFFERENT speaker voice tracks (skip if overlaps - prevents speaker mixing)
      const overlapsSameSpeaker = deduplicatedVoiceTracks.some(vtSeg => {
        const vtStart = parseFloat(vtSeg.start) || 0;
        const vtEnd = parseFloat(vtSeg.end) || vtStart;
        const vtSpeaker = vtSeg.speaker || 'SPEAKER_00';
        const vtText = (vtSeg.text || '').trim().toLowerCase();
        
        // Only consider overlap if same speaker
        if (vtSpeaker !== pSpeaker) return false;
        
        // Check temporal overlap
        const overlapDuration = Math.max(0, Math.min(pEnd, vtEnd) - Math.max(pStart, vtStart));
        
        // Also check text similarity - if texts are very similar, it's a duplicate
        const textSimilarity = pText.includes(vtText) || vtText.includes(pText);
        
        // Consider overlap if:
        // 1. Temporal overlap > 50% of primary segment duration OR
        // 2. Texts are similar and there's any temporal overlap
        return (overlapDuration > Math.max(0.2, pDuration * 0.5)) || (textSimilarity && overlapDuration > 0.1);
      });
      
      // CRITICAL: Check if this primary segment overlaps with DIFFERENT speaker voice tracks
      // If it does, skip it to prevent speaker mixing
      const overlapsDifferentSpeaker = deduplicatedVoiceTracks.some(vtSeg => {
        const vtStart = parseFloat(vtSeg.start) || 0;
        const vtEnd = parseFloat(vtSeg.end) || vtStart;
        const vtSpeaker = vtSeg.speaker || 'SPEAKER_00';
        
        // Only consider overlap if DIFFERENT speaker
        if (vtSpeaker === pSpeaker) return false;
        
        // Check temporal overlap
        const overlapDuration = Math.max(0, Math.min(pEnd, vtEnd) - Math.max(pStart, vtStart));
        
        // If there's significant overlap (>30% of primary segment), skip it
        // This prevents primary segments from being assigned to wrong speaker
        const overlapPercent = pDuration > 0 ? (overlapDuration / pDuration) * 100 : 0;
        if (overlapPercent > 30) {
          console.log(`⚠️ Skipping primary segment "${pText.substring(0, 30)}..." (${pSpeaker}) - overlaps ${overlapPercent.toFixed(1)}% with voice track from ${vtSpeaker}`);
          return true;
        }
        
        return false;
      });
      
      // Only add if:
      // 1. Doesn't overlap with same speaker voice track (voice track has better text)
      // 2. Doesn't overlap with different speaker voice track (prevents speaker mixing)
      if (!overlapsSameSpeaker && !overlapsDifferentSpeaker && (pSeg.text || '').trim().length > 0) {
        mergedSegments.push({
          ...pSeg,
          source: 'primary'
        });
      } else if (overlapsDifferentSpeaker) {
        console.log(`🔇 Skipping primary segment "${pText.substring(0, 30)}..." (${pSpeaker}) - overlaps with different speaker voice track`);
      }
    });
    
    // STEP 4: Sort by start time
    const sorted = mergedSegments.sort((a, b) => {
      const startDiff = (a.start || 0) - (b.start || 0);
      if (startDiff !== 0) return startDiff;
      return (a.end || 0) - (b.end || 0);
    });
    
    // STEP 4.1: First, merge consecutive segments from the same speaker that are split by short pauses
    // This fixes cases where one dialogue turn is split into multiple segments
    // Example: "Well wait wait wait for a second" (35.2-36.96) + "Uh can you please..." (37.84-42.88)
    // CRITICAL: Do this BEFORE chronological validation to preserve continuation segments
    // CRITICAL: NEVER merge segments from different speakers - this would corrupt speaker attribution
    const mergedConsecutive = [];
    const maxPauseForMerge = 2.0; // Merge if gap < 2 seconds
    
    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i];
      const currentStart = parseFloat(current.start) || 0;
      const currentEnd = parseFloat(current.end) || currentStart;
      const currentSpeaker = current.speaker || 'SPEAKER_00';
      const currentText = (current.text || '').trim();
      
      // CRITICAL: Verify speaker label is valid
      if (!currentSpeaker || !currentSpeaker.startsWith('SPEAKER_')) {
        console.warn(`⚠️ Skipping segment with invalid speaker label: ${currentSpeaker}`);
        continue;
      }
      
      // Look ahead to find consecutive segments from the same speaker
      let mergedSegment = { ...current };
      let j = i + 1;
      
      while (j < sorted.length) {
        const next = sorted[j];
        const nextStart = parseFloat(next.start) || 0;
        const nextEnd = parseFloat(next.end) || nextStart;
        const nextSpeaker = next.speaker || 'SPEAKER_00';
        const nextText = (next.text || '').trim();
        
        // CRITICAL: Verify next segment has valid speaker label
        if (!nextSpeaker || !nextSpeaker.startsWith('SPEAKER_')) {
          console.warn(`⚠️ Skipping segment with invalid speaker label: ${nextSpeaker}`);
          j++;
          continue;
        }
        
        // CRITICAL: Only merge if speakers match EXACTLY
        // This prevents segments from different speakers from being merged together
        if (nextSpeaker === currentSpeaker) {
          const gap = nextStart - currentEnd;
          const smallOverlap = currentEnd > nextStart && (currentEnd - nextStart) < 0.5;
          
          // Check if there are any segments from other speakers that significantly overlap with the gap/overlap region
          // Only prevent merging if there's a significant overlap (>0.5s) or if another speaker's segment
          // is chronologically between current and next (starts after current ends and ends before next starts)
          let hasInterveningSpeaker = false;
          const gapStart = currentEnd;
          const gapEnd = nextStart;
          const minInterveningOverlap = 0.5; // Only consider it intervening if overlap > 0.5s
          
          // Check segments that are chronologically between current and next
          for (let k = i + 1; k < j; k++) {
            const between = sorted[k];
            const betweenStart = parseFloat(between.start) || 0;
            const betweenEnd = parseFloat(between.end) || betweenStart;
            const betweenSpeaker = between.speaker || 'SPEAKER_00';
            
            // If there's a segment from a different speaker that is chronologically between current and next
            if (betweenSpeaker !== currentSpeaker) {
              // Check if this segment is fully between current and next (chronologically)
              if (betweenStart >= gapStart && betweenEnd <= gapEnd) {
                // This segment is chronologically between - don't merge
                hasInterveningSpeaker = true;
                break;
              }
              
              // Also check for significant overlap with the gap
              const overlapStart = Math.max(gapStart, betweenStart);
              const overlapEnd = Math.min(gapEnd, betweenEnd);
              const overlapDuration = overlapEnd - overlapStart;
              if (overlapDuration > minInterveningOverlap) {
                // Significant overlap with gap - don't merge
                hasInterveningSpeaker = true;
                break;
              }
            }
          }
          
      // For small overlaps, also check if there's significant overlap with the merged region
      if (!hasInterveningSpeaker && smallOverlap) {
        for (let k = 0; k < sorted.length; k++) {
          if (k === i || k === j) continue;
          
          const between = sorted[k];
              const betweenStart = parseFloat(between.start) || 0;
              const betweenEnd = parseFloat(between.end) || betweenStart;
              const betweenSpeaker = between.speaker || 'SPEAKER_00';
              
              if (betweenSpeaker !== currentSpeaker) {
                const mergedOverlapStart = Math.max(currentStart, betweenStart);
                const mergedOverlapEnd = Math.min(nextEnd, betweenEnd);
                const mergedOverlapDuration = mergedOverlapEnd - mergedOverlapStart;
                // Only prevent merge if overlap is significant (>0.5s)
                if (mergedOverlapDuration > minInterveningOverlap) {
                  hasInterveningSpeaker = true;
                  break;
                }
              }
            }
          }
          
          if (hasInterveningSpeaker) {
            // There's an intervening speaker segment - stop merging
            break;
          }
          
          if (gap >= 0 && gap < maxPauseForMerge || smallOverlap) {
            // CRITICAL: Double-check speakers match before merging
            if (mergedSegment.speaker !== nextSpeaker) {
              console.error(`❌ CRITICAL ERROR: Attempted to merge segments with different speakers: ${mergedSegment.speaker} vs ${nextSpeaker}`);
              break;
            }
            
            // Merge: extend end time and combine text
            mergedSegment.end = Math.max(mergedSegment.end || 0, nextEnd);
            mergedSegment.text = (mergedSegment.text + ' ' + nextText).trim();
            // Preserve original text if available
            if (mergedSegment.originalText && next.originalText) {
              mergedSegment.originalText = (mergedSegment.originalText + ' ' + next.originalText).trim();
            } else if (next.originalText) {
              mergedSegment.originalText = (mergedSegment.text + ' ' + next.originalText).trim();
            }
            // Merge words arrays if available
            if (mergedSegment.words && next.words) {
              mergedSegment.words = [...(mergedSegment.words || []), ...(next.words || [])];
            } else if (next.words) {
              mergedSegment.words = next.words;
            }
            // Ensure speaker label is preserved
            mergedSegment.speaker = currentSpeaker;
            j++; // Continue to next segment
          } else {
            // Gap too large or overlap too significant - stop merging
            break;
          }
        } else {
          // Different speaker - stop merging immediately
          // This is expected and correct - we should never merge segments from different speakers
          break;
        }
      }
      
      mergedConsecutive.push(mergedSegment);
      i = j - 1; // Skip merged segments
    }
    
    console.log(`✅ Merged consecutive segments: ${sorted.length} → ${mergedConsecutive.length}`);
    
    // STEP 4.2: Now remove segments that overlap with previous segments from the same speaker
    // This catches chronological violations AFTER merging continuations
    // Only remove if the overlap is significant AND the segment is not a continuation
    // CRITICAL: Only process segments with valid speaker labels and never mix speakers
    const chronologicallyValid = [];
    for (let i = 0; i < mergedConsecutive.length; i++) {
      const current = mergedConsecutive[i];
      const currentStart = parseFloat(current.start) || 0;
      const currentEnd = parseFloat(current.end) || currentStart;
      const currentSpeaker = current.speaker || 'SPEAKER_00';
      
      // CRITICAL: Verify speaker label is valid
      if (!currentSpeaker || !currentSpeaker.startsWith('SPEAKER_')) {
        console.warn(`⚠️ Skipping segment with invalid speaker label in chronological validation: ${currentSpeaker}`);
        continue;
      }
      
      // Check if this segment overlaps with any previously added segment from the same speaker
      let wasMerged = false;
      let shouldSkip = false;
      
      for (let k = 0; k < chronologicallyValid.length; k++) {
        const prev = chronologicallyValid[k];
        const prevStart = parseFloat(prev.start) || 0;
        const prevEnd = parseFloat(prev.end) || prevStart;
        const prevSpeaker = prev.speaker || 'SPEAKER_00';
        
        // CRITICAL: Only compare segments from the same speaker
        // If speakers don't match, skip this comparison (this is expected for different speakers)
        if (prevSpeaker !== currentSpeaker) continue;
        
        // CRITICAL: Double-check speakers match (defensive programming)
        if (prevSpeaker !== currentSpeaker) {
          console.error(`❌ CRITICAL ERROR: Speaker mismatch in chronological validation: ${prevSpeaker} vs ${currentSpeaker}`);
          continue;
        }
        
        // Check for temporal overlap
        const overlapStart = Math.max(currentStart, prevStart);
        const overlapEnd = Math.min(currentEnd, prevEnd);
        const overlapDuration = overlapEnd - overlapStart;
        
        // If overlap is significant (>0.1s), check if it's a continuation or a violation
        if (overlapDuration > 0.1) {
          // Check if current is a continuation (starts before prev ends but extends beyond)
          const isContinuation = currentStart < prevEnd && currentEnd > prevEnd;
          // Check if current is fully contained in previous (violation)
          const isFullyContained = currentStart >= prevStart && currentEnd <= prevEnd;
          
          // If it's a continuation, merge it instead of removing
          if (isContinuation && !isFullyContained) {
            // CRITICAL: Verify speakers match before merging
            if (prevSpeaker !== currentSpeaker) {
              console.error(`❌ CRITICAL ERROR: Attempted to merge continuation with different speakers: ${prevSpeaker} vs ${currentSpeaker}`);
              shouldSkip = true;
              break;
            }
            
            // Merge: extend previous segment
            prev.end = Math.max(prevEnd, currentEnd);
            prev.text = (prev.text + ' ' + (current.text || '')).trim();
            if (prev.words && current.words) {
              prev.words = [...(prev.words || []), ...(current.words || [])];
            } else if (current.words) {
              prev.words = current.words;
            }
            // Ensure speaker label is preserved
            prev.speaker = prevSpeaker;
            wasMerged = true;
            console.log(`✅ Merged continuation segment "${(current.text || '').substring(0, 30)}..." (${currentStart.toFixed(2)}-${currentEnd.toFixed(2)}) with previous (speaker: ${prevSpeaker})`);
            break;
          }
          
          // If fully contained or significant overlap without continuation, it's a violation
          if (isFullyContained || overlapDuration > 0.3) {
            shouldSkip = true;
            console.log(`⚠️ Skipping segment "${(current.text || '').substring(0, 30)}..." (${currentStart.toFixed(2)}-${currentEnd.toFixed(2)}) - overlaps with previous segment from same speaker`);
            break;
          }
        }
      }
      
      if (!wasMerged && !shouldSkip) {
        chronologicallyValid.push(current);
      }
    }
    
    console.log(`✅ Chronological validation: ${mergedConsecutive.length} → ${chronologicallyValid.length} segments`);
    
    // Use chronologically valid segments for further processing
    const sortedValid = chronologicallyValid;
    
    // STEP 5: Final deduplication pass - remove segments that are fully contained in others
    // CRITICAL: Only compare segments from the same speaker - never mix speakers
    const finalSegments = [];
    for (let i = 0; i < sortedValid.length; i++) {
      const current = sortedValid[i];
      const currentStart = parseFloat(current.start) || 0;
      const currentEnd = parseFloat(current.end) || currentStart;
      const currentText = (current.text || '').trim().toLowerCase();
      const currentSpeaker = current.speaker || 'SPEAKER_00';
      
      // CRITICAL: Verify speaker label is valid
      if (!currentSpeaker || !currentSpeaker.startsWith('SPEAKER_')) {
        console.warn(`⚠️ Skipping segment with invalid speaker label in final deduplication: ${currentSpeaker}`);
        continue;
      }
      
      let isFullyContained = false;
      
      // Check if this segment is fully contained in another segment of the same speaker
      for (let j = 0; j < sortedValid.length; j++) {
        if (i === j) continue;
        
        const other = sortedValid[j];
        const otherStart = parseFloat(other.start) || 0;
        const otherEnd = parseFloat(other.end) || otherStart;
        const otherText = (other.text || '').trim().toLowerCase();
        const otherSpeaker = other.speaker || 'SPEAKER_00';
        
        // CRITICAL: Only check same speaker - never compare segments from different speakers
        if (otherSpeaker !== currentSpeaker) continue;
        
        // CRITICAL: Double-check speakers match (defensive programming)
        if (otherSpeaker !== currentSpeaker) {
          console.error(`❌ CRITICAL ERROR: Speaker mismatch in final deduplication: ${otherSpeaker} vs ${currentSpeaker}`);
          continue;
        }
        
        // Check if current is fully contained in other (time-wise and text-wise)
        if (otherStart <= currentStart && otherEnd >= currentEnd) {
          // If other text contains current text, it's a duplicate
          if (otherText.includes(currentText) && otherText.length > currentText.length) {
            isFullyContained = true;
            break;
          }
        }
      }
      
      if (!isFullyContained) {
        finalSegments.push(current);
      }
    }
    
    console.log(`✅ Final deduplication: ${sorted.length} → ${finalSegments.length} segments`);
    
    // CRITICAL: Final validation - ensure all segments have valid speaker labels
    const invalidFinalSegments = finalSegments.filter(seg => {
      const spk = seg.speaker || '';
      return !spk || !spk.startsWith('SPEAKER_');
    });
    if (invalidFinalSegments.length > 0) {
      console.error(`❌ Found ${invalidFinalSegments.length} final segments with invalid speaker labels - removing them`);
      finalSegments = finalSegments.filter(seg => {
        const spk = seg.speaker || '';
        return spk && spk.startsWith('SPEAKER_');
      });
    }
    
    // Verify final speaker distribution
    const finalBySpeaker = {};
    finalSegments.forEach(seg => {
      const spk = seg.speaker || 'UNKNOWN';
      if (!finalBySpeaker[spk]) finalBySpeaker[spk] = [];
      finalBySpeaker[spk].push(seg);
    });
    console.log(`✅ Final merged segments by speaker:`, Object.keys(finalBySpeaker).map(s => `${s}: ${finalBySpeaker[s].length}`).join(', '));
    
    // CRITICAL: Log sample final segments to verify speaker attribution
    if (finalSegments.length > 0) {
      console.log(`📋 Sample final segments (first 5):`);
      finalSegments.slice(0, 5).forEach((seg, idx) => {
        console.log(`  [${idx}] Speaker: ${seg.speaker}, Text: "${(seg.text || '').substring(0, 50)}...", Time: ${seg.start}-${seg.end}, Source: ${seg.source || 'unknown'}`);
      });
    }
    
    return finalSegments;
  },

  createSpeakerJSON(recording, segments, speaker) {
    // Determine engine from recording
    const recordingData = recording?.recordings?.[0] || recording;
    const engine = recordingData?._engine || 
                   recordingData?.results?.speechmatics?.engine ||
                   recordingData?.results?.azure?.engine ||
                   'speechmatics';
    const isAzure = engine && (engine.includes('azure') || engine.includes('Azure'));
    
    // Get actual service name and raw data from recording
    const azureResult = recordingData?.results?.azure;
    const speechmaticsResult = recordingData?.results?.speechmatics;
    const serviceName = isAzure 
      ? (azureResult?.serviceName || 'Azure STT')
      : (speechmaticsResult?.serviceName || 'Speechmatics');
    
    // Filter segments for this speaker
    const speakerSegments = segments.filter(seg => seg.speaker === speaker);
    
    const baseRecording = {
      id: recordingData.id || recording.id,
      name: recordingData.name || recording.name || 'Recording',
      fileName: recordingData.fileName || recording.fileName || '',
      size: recordingData.size || recording.size || 0,
      duration: recordingData.duration || recording.duration || 0,
      language: recordingData.language || recording.language || 'en',
      speakerCount: "1",
      status: "completed",
      results: {}
    };
    
    if (isAzure) {
      // Azure native format
      const azureRawData = azureResult?.rawData || speechmaticsResult?.rawData;
      
      // Filter Azure segments for this speaker
      const azureSpeakerSegments = azureRawData?.segments?.filter(seg => {
        if (seg.type !== 'final' || !seg.text) return false;
        const speakerMap = azureRawData.speakerMap || {};
        const azureSpeakerId = seg.speakerId || 'Unknown';
        const mappedSpeaker = speakerMap[azureSpeakerId] || `SPEAKER_${azureSpeakerId}`;
        return mappedSpeaker === speaker;
      }) || [];
      
      baseRecording.results.azure = {
        success: true,
        serviceName: serviceName,
        engine: engine,
        segments: speakerSegments.map(seg => ({
          speaker: speaker,
          text: seg.text || '',
          start: seg.start || 0,
          end: seg.end || seg.start || 0,
          words: seg.words || [],
          confidence: seg.confidence || 0.9,
          pauses: seg.pauses || []
        })),
        rawData: {
          service: azureRawData?.service || 'azure_realtime',
          language: azureRawData?.language || recordingData.language || 'en-US',
          segments: azureSpeakerSegments,
          speakerMap: azureRawData?.speakerMap || {}
        }
      };
      
      // Also keep in speechmatics key for backward compatibility
      baseRecording.results.speechmatics = {
        success: true,
        serviceName: serviceName,
        engine: engine,
        segments: speakerSegments.map(seg => ({
          speaker: speaker,
          text: seg.text || '',
          start: seg.start || 0,
          end: seg.end || seg.start || 0,
          originalText: seg.originalText || seg.text || '',
          translations: seg.translations || {}
        })),
        rawData: azureRawData
      };
    } else {
      // Speechmatics format
      baseRecording.results.speechmatics = {
        success: true,
        serviceName: serviceName,
        engine: engine,
        segments: speakerSegments.map(seg => ({
          speaker: speaker,
          text: seg.text || '',
          start: seg.start || 0,
          end: seg.end || seg.start || 0,
          originalText: seg.originalText || seg.text || '',
          translations: seg.translations || {},
          words: seg.words || [],
          confidence: seg.confidence || 0.9,
          pauses: seg.pauses || []
        }))
      };
      
      // Include raw data if available
      if (speechmaticsResult?.rawData) {
        baseRecording.results.speechmatics.rawData = speechmaticsResult.rawData;
      }
    }
    
    return {
      version: "2.0",
      exportedAt: new Date().toISOString(),
      activeRecordingId: recordingData.id || recording.id,
      _engine: engine,
      recordings: [baseRecording]
    };
  },

  createMergedJSON(recording, mergedSegments) {
    // Determine engine from recording
    const recordingData = recording?.recordings?.[0] || recording;
    const engine = recordingData?._engine || 
                   recordingData?.results?.speechmatics?.engine ||
                   recordingData?.results?.azure?.engine ||
                   'speechmatics';
    const isAzure = engine && (engine.includes('azure') || engine.includes('Azure'));
    
    // Get actual service name and raw data from recording
    const azureResult = recordingData?.results?.azure;
    const speechmaticsResult = recordingData?.results?.speechmatics;
    const serviceName = isAzure 
      ? (azureResult?.serviceName || 'Azure STT')
      : (speechmaticsResult?.serviceName || 'Speechmatics');
    
    const baseRecording = {
      id: recordingData.id || recording.id,
      name: recordingData.name || recording.name || 'Recording',
      fileName: recordingData.fileName || recording.fileName || '',
      size: recordingData.size || recording.size || 0,
      duration: recordingData.duration || recording.duration || 0,
      language: recordingData.language || recording.language || 'en',
      speakerCount: new Set(mergedSegments.map(s => s.speaker)).size.toString(),
      status: "completed",
      translationState: recordingData.translationState || recording.translationState || { currentLanguage: 'original', lastError: null },
      aggregated: recordingData.aggregated || recording.aggregated || {},
      results: {}
    };
    
    if (isAzure) {
      // Azure native format
      const azureRawData = azureResult?.rawData || speechmaticsResult?.rawData;
      
      baseRecording.results.azure = {
        success: true,
        serviceName: serviceName,
        engine: engine,
        segments: mergedSegments.map(seg => ({
          speaker: seg.speaker || 'SPEAKER_00',
          text: seg.text || '',
          start: seg.start || 0,
          end: seg.end || seg.start || 0,
          words: seg.words || [],
          confidence: seg.confidence || 0.9,
          pauses: seg.pauses || [],
          role: seg.role || null,
          overlap: seg.overlap || false,
          source: seg.source || 'primary'
        })),
        rawData: azureRawData || {
          service: 'azure_realtime',
          language: recordingData.language || 'en-US',
          segments: [],
          speakerMap: {}
        }
      };
      
      // Also keep in speechmatics key for backward compatibility
      baseRecording.results.speechmatics = {
        success: true,
        serviceName: serviceName,
        engine: engine,
        segments: mergedSegments.map(seg => ({
          speaker: seg.speaker || 'SPEAKER_00',
          text: seg.text || '',
          start: seg.start || 0,
          end: seg.end || seg.start || 0,
          originalText: seg.originalText || seg.text || '',
          translations: seg.translations || {},
          role: seg.role || null,
          overlap: seg.overlap || false,
          source: seg.source || 'primary'
        })),
        rawData: azureRawData
      };
    } else {
      // Speechmatics format
      baseRecording.results.speechmatics = {
        success: true,
        serviceName: serviceName,
        engine: engine,
        segments: mergedSegments.map(seg => ({
          speaker: seg.speaker || 'SPEAKER_00',
          text: seg.text || '',
          start: seg.start || 0,
          end: seg.end || seg.start || 0,
          originalText: seg.originalText || seg.text || '',
          translations: seg.translations || {},
          words: seg.words || [],
          confidence: seg.confidence || 0.9,
          pauses: seg.pauses || []
        }))
      };
      
      // Include raw data if available
      if (speechmaticsResult?.rawData) {
        baseRecording.results.speechmatics.rawData = speechmaticsResult.rawData;
      }
    }
    
    // Add overlap-corrected results
    baseRecording.results["overlap-corrected"] = {
      success: true,
      serviceName: "Overlap Corrected (Voice Tracks)",
      processingTime: 0,
      speedFactor: 0,
      speakerCount: new Set(mergedSegments.map(s => s.speaker)).size.toString(),
      cost: "0.0000",
      segments: mergedSegments.map(seg => ({
        speaker: seg.speaker || 'SPEAKER_00',
        text: seg.text || '',
        start: seg.start || 0,
        end: seg.end || seg.start || 0,
        originalText: seg.originalText || seg.text || '',
        translations: seg.translations || {},
        role: seg.role || null,
        overlap: seg.overlap || false,
        source: seg.source || 'primary',
        words: seg.words || [],
        confidence: seg.confidence || 0.9,
        pauses: seg.pauses || []
      })),
      rawData: {
        source: "overlap-corrected-voice-tracks",
        mergedAt: new Date().toISOString(),
        engine: engine
      }
    };
    
    return {
      version: "2.0",
      exportedAt: new Date().toISOString(),
      activeRecordingId: recordingData.id || recording.id,
      _engine: engine,
      recordings: [baseRecording]
    };
  },

  updateOverlapFixesModal(modal, speaker0JSON, speaker1JSON, mergedJSON, recording, mergedSegments) {
    const json0Str = JSON.stringify(speaker0JSON, null, 2);
    const json1Str = JSON.stringify(speaker1JSON, null, 2);
    const mergedStr = JSON.stringify(mergedJSON, null, 2);
    
    // Get current LLM mode from radio buttons
    const modeRadio = document.querySelector('input[name="replicaLLMModel"]:checked');
    const currentMode = modeRadio ? modeRadio.value : 'smart';
    
    modal.querySelector('.modal-body').innerHTML = `
      <div style="display: flex; flex-direction: column; gap: var(--space-16);">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-16); margin-bottom: var(--space-16);">
          <div>
            <h3 style="margin-bottom: var(--space-8);">SPEAKER_00 Transcript</h3>
            <div style="position: relative;">
              <button class="btn btn-sm" onclick="navigator.clipboard.writeText(document.querySelector('#json0').innerText)" style="position: absolute; top: 8px; right: 8px; z-index: 10;">📋 Copy</button>
              <pre id="json0" style="max-height: 400px; overflow: auto; background: #111216; color: #c9f0ff; padding: var(--space-16); border-radius: var(--radius-base); font-size: 11px; border: 1px solid var(--color-border); white-space: pre-wrap; word-wrap: break-word;">${this.escapeHtml(json0Str)}</pre>
            </div>
          </div>
          <div>
            <h3 style="margin-bottom: var(--space-8);">SPEAKER_01 Transcript</h3>
            <div style="position: relative;">
              <button class="btn btn-sm" onclick="navigator.clipboard.writeText(document.querySelector('#json1').innerText)" style="position: absolute; top: 8px; right: 8px; z-index: 10;">📋 Copy</button>
              <pre id="json1" style="max-height: 400px; overflow: auto; background: #111216; color: #c9f0ff; padding: var(--space-16); border-radius: var(--radius-base); font-size: 11px; border: 1px solid var(--color-border); white-space: pre-wrap; word-wrap: break-word;">${this.escapeHtml(json1Str)}</pre>
            </div>
          </div>
        </div>
        <div>
          <h3 style="margin-bottom: var(--space-8);">Merged Transcript (Combined)</h3>
          <div style="position: relative;">
            <button class="btn btn-sm" onclick="navigator.clipboard.writeText(document.querySelector('#jsonMerged').innerText)" style="position: absolute; top: 8px; right: 8px; z-index: 10;">📋 Copy</button>
            <pre id="jsonMerged" style="max-height: 400px; overflow: auto; background: #111216; color: #c9f0ff; padding: var(--space-16); border-radius: var(--radius-base); font-size: 11px; border: 1px solid var(--color-border); white-space: pre-wrap; word-wrap: break-word;">${this.escapeHtml(mergedStr)}</pre>
          </div>
        </div>
        <div style="display: flex; justify-content: flex-end; gap: var(--space-12); margin-top: var(--space-16); padding-top: var(--space-16); border-top: 1px solid var(--color-border);">
          <button class="btn btn-secondary" onclick="const modal = document.getElementById('overlapFixesModal'); if (modal) { const recordingId = modal.querySelector('#applyMergedBtn')?.getAttribute('data-recording-id'); if (recordingId && app._overlapFixesModalData) { delete app._overlapFixesModalData[recordingId]; } modal.remove(); }">Cancel</button>
          <button class="btn btn-primary" id="applyMergedBtn" data-recording-id="${recording.id}" data-llm-mode="${currentMode}">✅ Apply</button>
        </div>
      </div>
    `;
    
    // Store segments in app object for later retrieval
    if (!this._overlapFixesModalData) {
      this._overlapFixesModalData = {};
    }
    this._overlapFixesModalData[recording.id] = mergedSegments;
    
    // Attach event listener to Apply button
    setTimeout(() => {
      const applyBtn = document.getElementById('applyMergedBtn');
      if (applyBtn) {
        applyBtn.addEventListener('click', () => {
          const recordingId = applyBtn.getAttribute('data-recording-id');
          const llmMode = applyBtn.getAttribute('data-llm-mode') || 'smart';
          const segments = this._overlapFixesModalData?.[recordingId];
          if (!segments) {
            console.error('Segments not found for recording:', recordingId);
            alert('Error: Segments data not found. Please try again.');
            return;
          }
          // Store LLM mode for use in applyMergedSegments if needed
          if (!this._overlapFixesModalData) {
            this._overlapFixesModalData = {};
          }
          if (!this._overlapFixesModalData[recordingId]) {
            this._overlapFixesModalData[recordingId] = {};
          }
          this._overlapFixesModalData[recordingId].llmMode = llmMode;
          this.applyMergedSegments(segments, recordingId);
        });
      } else {
        console.error('Apply button not found');
      }
    }, 100);
  },

  async applyMergedSegments(segments, recordingId) {
    const modal = document.getElementById('overlapFixesModal');
    if (modal) {
      modal.querySelector('.modal-body').innerHTML = `
        <div style="text-align: center; padding: var(--space-32);">
          <div class="spinner" style="margin: 0 auto;"></div>
          <p style="margin-top: var(--space-16);">Applying changes...</p>
        </div>
      `;
    }

    try {
      const recording = this.recordings.find(r => r.id === recordingId) || this.getActiveRecording();
      if (!recording) {
        throw new Error('Recording not found');
      }

      // Update recording with merged segments
      if (!recording.results) {
        recording.results = {};
      }

      // Create/update overlap-corrected result
      recording.results['overlap-corrected'] = {
        success: true,
        serviceName: 'Overlap Corrected (Voice Tracks)',
        processingTime: 0,
        speedFactor: 0,
        speakerCount: new Set(segments.map(s => s.speaker)).size.toString(),
        cost: '0.0000',
        segments: segments,
        rawData: {
          source: 'overlap-corrected-voice-tracks',
          appliedAt: new Date().toISOString()
        }
      };

      // Also update text-service
      recording.results['text-service'] = recording.results['overlap-corrected'];

      // Update servicesTested
      if (!recording.servicesTested) {
        recording.servicesTested = [];
      }
      if (!recording.servicesTested.includes('overlap-corrected')) {
        recording.servicesTested.push('overlap-corrected');
      }

      // Close modal
      if (modal) {
        modal.remove();
      }

      // Clean up stored data
      if (this._overlapFixesModalData && this._overlapFixesModalData[recordingId]) {
        delete this._overlapFixesModalData[recordingId];
      }

      // Update status
      const statusDiv = document.getElementById('replicaTranslationStatus');
      if (statusDiv) {
        statusDiv.className = 'status status-success';
        statusDiv.innerHTML = '✨ <strong>Overlap Voice Recognition Fixed</strong> — Applied successfully';
      }

      // Refresh view
      this.updateReplicasComparison();

    } catch (error) {
      console.error('Error applying merged segments:', error);
      if (modal) {
        modal.querySelector('.modal-body').innerHTML = `
          <div class="alert alert-error">
            <strong>Error:</strong> ${error.message}
          </div>
        `;
      }
    }
  },

  async applyPauseBasedMerge() {
    const recording = this.getActiveRecording();
    if (!recording || !recording.id) {
      alert('No recording found. Please load a recording first.');
      return;
    }

    // Get recording statistics
    const primarySegments = recording.results?.speechmatics?.segments || [];
    const overlapResult = recording.results?.['overlap-corrected'];
    const pauseMergeResult = recording.results?.['pause-based-merge'];
    const voiceTracks = recording.overlapMetadata?.voiceTracks || [];

    // Calculate statistics
    const primarySegmentsCount = primarySegments.length;
    const voiceTracksCount = voiceTracks.length;
    const hasOverlapResult = !!overlapResult;
    const hasPauseMergeResult = !!pauseMergeResult;
    const pauseMergeSegmentsCount = pauseMergeResult?.segments?.length || 0;
    const pauseMergeRawData = pauseMergeResult?.rawData || {};

    // Create modal popup with information
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'flex';
    modal.id = 'pauseMergeInfoModal';
    
    modal.innerHTML = `
      <div class="modal-content" style="max-width: 700px;">
        <div class="modal-header">
          <h2>🔄 Pause-Based Merge Information</h2>
          <button class="btn-close" onclick="document.getElementById('pauseMergeInfoModal').remove()">✕</button>
        </div>
        <div class="modal-body">
          <div style="margin-bottom: var(--space-16);">
            <h3 style="margin-top: 0;">Algorithm Overview</h3>
            <p>The pause-based merge algorithm identifies speaker replicas by detecting pauses in the audio and enhances the primary diarization with missing phrases from separated voice tracks.</p>
          </div>

          <div style="margin-bottom: var(--space-16);">
            <h3>Current Recording Statistics</h3>
            <ul style="line-height: 1.8;">
              <li><strong>Primary Segments:</strong> ${primarySegmentsCount}</li>
              <li><strong>Voice Tracks:</strong> ${voiceTracksCount}</li>
              <li><strong>Overlap-Corrected Result:</strong> ${hasOverlapResult ? '✅ Available' : '❌ Not available'}</li>
              <li><strong>Pause-Based Merge Result:</strong> ${hasPauseMergeResult ? '✅ Available' : '❌ Not available'}</li>
              ${hasPauseMergeResult ? `
                <li><strong>Merged Segments:</strong> ${pauseMergeSegmentsCount}</li>
                <li><strong>Replicas Identified:</strong> ${pauseMergeRawData.replicasCount || 'N/A'}</li>
                <li><strong>Additional Phrases:</strong> ${pauseMergeRawData.additionalPhrasesCount || 0}</li>
              ` : ''}
            </ul>
          </div>

          <div style="margin-bottom: var(--space-16);">
            <h3>How It Works</h3>
            <ol style="line-height: 1.8; padding-left: var(--space-24);">
              <li>Analyzes the primary recording to detect pauses between segments</li>
              <li>Identifies continuous speech segments (replicas) without pauses</li>
              <li>Matches replicas with corresponding segments in separated voice tracks</li>
              <li>Adds missing phrases from voice tracks that weren't in the primary transcript</li>
              <li>Creates a complete dialogue with all replicas properly transferred</li>
            </ol>
          </div>

          ${!hasPauseMergeResult ? `
            <div class="alert alert-info" style="margin-bottom: var(--space-16);">
              <strong>💡 Ready to Apply:</strong> Click "Apply Merge" below to run the pause-based merge algorithm on this recording.
            </div>
          ` : `
            <div class="alert alert-success" style="margin-bottom: var(--space-16);">
              <strong>✅ Already Applied:</strong> Pause-based merge has been applied to this recording. The result is available in the "pause-based-merge" service.
            </div>
          `}
        </div>
        <div class="modal-footer">
          <button class="btn btn-secondary" onclick="document.getElementById('pauseMergeInfoModal').remove()">Close</button>
          ${!hasPauseMergeResult ? `
            <button class="btn btn-primary" onclick="app.executePauseBasedMerge(); document.getElementById('pauseMergeInfoModal').remove();">
              🔄 Apply Merge
            </button>
          ` : `
            <button class="btn btn-primary" onclick="app.showResults(); document.getElementById('pauseMergeInfoModal').remove();">
              👁️ View Results
            </button>
          `}
        </div>
      </div>
    `;
    
    document.body.appendChild(modal);
  },

  async executePauseBasedMerge() {
    const button = document.getElementById('applyPauseBasedMergeBtn');
    const statusDiv = document.getElementById('replicaTranslationStatus');

    if (!button) return;

    button.disabled = true;
    button.textContent = '⏳ Applying...';
    statusDiv.className = 'status status-processing';
    statusDiv.textContent = 'Applying pause-based replica merge algorithm...';

    try {
      const recording = this.getActiveRecording();
      if (!recording || !recording.id) {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = 'No recording found. Please load a recording first.';
        return;
      }

      const response = await fetch('/api/apply-pause-based-merge', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          recordingId: recording.id,
          recording: recording // Send full recording data from frontend
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        const updatedRecording = result.recording;
        
        // Update the recording in app state
        if (recording) {
          // Find recording in array and update it
          const recordingIndex = this.recordings.findIndex(rec => rec.id === recording.id);
          if (recordingIndex !== -1) {
            // Merge results - ensure pause-based-merge result is properly added
            this.recordings[recordingIndex].results = {
              ...this.recordings[recordingIndex].results,
              ...updatedRecording.results
            };
            
            // Ensure servicesTested includes pause-based-merge
            if (!this.recordings[recordingIndex].servicesTested) {
              this.recordings[recordingIndex].servicesTested = [];
            }
            if (!this.recordings[recordingIndex].servicesTested.includes('pause-based-merge')) {
              this.recordings[recordingIndex].servicesTested.push('pause-based-merge');
            }
            
            // Update testResults to reflect changes
            this.testResults = this.recordings[recordingIndex].results;
            
            // Update active recording reference
            const activeRec = this.getActiveRecording();
            if (activeRec && activeRec.id === recording.id) {
              // Update the active recording object directly
              Object.assign(activeRec, {
                results: this.recordings[recordingIndex].results,
                servicesTested: this.recordings[recordingIndex].servicesTested
              });
            }
            
            // Check current view - if on replicas view, stay there; otherwise show results
            const replicasView = document.getElementById('replicasView');
            const isReplicasViewActive = replicasView && replicasView.classList.contains('active');
            
            if (isReplicasViewActive) {
              // Stay on replicas view and update comparison (shows pause-based-merge result)
              this.updateReplicasComparison();
            } else {
              // Show results table (includes pause-based-merge result)
              this.showResults();
              // Also update replicas comparison in case user switches to it
              this.updateReplicasComparison();
            }
          }
        }
        
        statusDiv.className = 'status status-success';
        statusDiv.textContent = `✅ Pause-based merge applied successfully. Result available in "pause-based-merge" service and Replicas Comparison view.`;
      } else {
        statusDiv.className = 'status status-error';
        statusDiv.textContent = `❌ Failed to apply pause-based merge: ${result.error || 'Unknown error'}`;
      }

    } catch (error) {
      console.error('Error applying pause-based merge:', error);
      statusDiv.className = 'status status-error';
      statusDiv.textContent = `❌ Error applying pause-based merge: ${error.message || 'Unknown error'}`;
    } finally {
      button.disabled = false;
      button.textContent = '🔄 Apply Pause-Based Merge';
    }
  },

  highlightOverlapInTranscript(idx) {
    const container = document.getElementById('transcriptView');
    if (!container) return;

    // Remove active class from all highlights
    container.querySelectorAll('.overlap-highlight').forEach(el => {
      el.classList.remove('active');
    });

    // Add active class to clicked overlap
    const highlights = container.querySelectorAll(`[data-overlap-idx="${idx}"]`);
    highlights.forEach(el => {
      el.classList.add('active');
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    // Also highlight in list
    const listItems = document.querySelectorAll('.overlap-item');
    listItems.forEach((item, i) => {
      item.style.background = i === idx ? 'var(--color-bg-2)' : 'var(--color-surface)';
    });
  },

  exportOverlaps() {
    const data = {
      timestamp: new Date().toISOString(),
      overlaps: this.detectedOverlaps,
      audioFile: this.audioFile ? this.audioFile.name : null,
      transcriptFile: this.transcriptFile ? this.transcriptFile.name : null
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `overlaps_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  clearOverlaps() {
    this.detectedOverlaps = [];
    this.audioFile = null;
    this.transcriptFile = null;
    this.transcriptText = '';
    
    document.getElementById('audioFileInput').value = '';
    document.getElementById('transcriptFileInput').value = '';
    document.getElementById('transcriptTextArea').value = '';
    document.getElementById('audioUploadArea').classList.remove('has-file');
    document.getElementById('transcriptUploadArea').classList.remove('has-file');
    document.getElementById('audioFileText').textContent = 'Drag & drop audio file here or click to browse';
    document.getElementById('transcriptFileText').textContent = 'Drag & drop transcript file here or click to browse';
    document.getElementById('audioFileInfo').textContent = '';
    document.getElementById('transcriptFileInfo').textContent = '';
    document.getElementById('overlapsListContainer').style.display = 'none';
    document.getElementById('transcriptViewContainer').style.display = 'none';
    document.getElementById('overlapDetectionStatus').className = 'status';
    document.getElementById('overlapDetectionStatus').textContent = '';
  }
};

// IMPORTANT: This replaces the local mergeVoiceTrackSegments function with the version from overlap_merge_utils.js
// The overlap_merge_utils version has better logging, validation, and conflict resolution
// Do not create a local version in app.js - use the imported one
if (overlapMergeUtils && typeof overlapMergeUtils.mergeVoiceTrackSegments === 'function') {
  app.mergeVoiceTrackSegments = function mergeVoiceTrackSegmentsProxy(voiceTrackSegments, primarySegments, recording) {
    console.log(`✅ Using overlap_merge_utils.mergeVoiceTrackSegments (with enhanced logging and validation)`);
    return overlapMergeUtils.mergeVoiceTrackSegments(voiceTrackSegments, primarySegments, recording, {
      logger: console,
      areTextsSimilar: this.areTextsSimilar.bind(this)
    });
  };
}

if (overlapMergeUtils && typeof overlapMergeUtils.areTextsSimilar === 'function') {
  app.areTextsSimilar = function areTextsSimilarProxy(text1, text2, threshold = 0.8) {
    return overlapMergeUtils.areTextsSimilar(text1, text2, threshold);
  };
}

// Initialize app on page load
window.addEventListener('DOMContentLoaded', () => {
  app.init();
});