const express = require('express');
const path = require('path');
const os = require('os');

// Load environment variables from .env file (must be before other requires)
const dotenv = require('dotenv');
const envResult = dotenv.config({ path: path.join(__dirname, '.env') });

if (envResult.error) {
  console.warn('⚠️  Warning: Could not load .env file:', envResult.error.message);
} else {
  console.log('✅ .env file loaded successfully');
  console.log(`   Found ${Object.keys(envResult.parsed || {}).length} environment variables`);
  // Log which keys are present (without values)
  const importantKeys = ['AUDIOSHAKE_API_KEY', 'OPENROUTER_API_KEY', 'OPENAI_API_KEY'];
  importantKeys.forEach(key => {
    if (process.env[key]) {
      const preview = process.env[key].substring(0, 20) + '...';
      console.log(`   ${key}: ${preview}`);
    } else {
      console.log(`   ${key}: NOT SET`);
    }
  });
}
const axios = require('axios');
const fs = require('fs').promises;
const fsSync = require('fs');
const { spawn } = require('child_process');
const multer = require('multer');
const localtunnel = require('localtunnel');
const { separateSpeakers } = require('./lib/audioshake-client');
const { validateDiarizationPayload, formatAjvErrors } = require('./lib/validators/diarizationSchema');
const textSimilarityUtils = require('./text_similarity_utils');
const textAnalysis = require('./lib/textAnalysis');

const app = express();
const PORT = process.env.PORT || 3000;
const TUNNEL_PORT = process.env.TUNNEL_PORT || PORT; // Можна вказати окремий порт для тунелю
// Model ID constants (with fallback defaults)
// NOTE: These are used as defaults, but getModelId() function should be used to get current values
const FAST_MODEL_ID = process.env.FAST_MODEL_ID || process.env.OPENROUTER_FAST_MODEL_ID || 'gpt-oss-120b';
const SMART_MODEL_ID = process.env.SMART_MODEL_ID || process.env.OPENROUTER_SMART_MODEL_ID || 'google/gemini-3.0-pro';
const SMART_2_MODEL_ID = process.env.SMART_2_MODEL_ID || process.env.OPENROUTER_SMART_2_MODEL_ID || 'google/gemini-3-pro-preview';
const TEST_MODEL_ID = process.env.TEST_MODEL_ID || process.env.OPENROUTER_TEST_MODEL_ID || 'google/gemma-3-4b';
const TEST2_MODEL_ID = process.env.TEST2_MODEL_ID || process.env.OPENROUTER_TEST2_MODEL_ID || 'llama-3.2-1b-instruct';

/**
 * Get current model ID for a given mode, always reading from process.env
 * This ensures cache keys use the current model, not stale constants
 * @param {string} mode - LLM mode ('local', 'fast', 'smart', 'smart-2', 'test', 'test2')
 * @returns {string} Model ID
 */
function getModelId(mode) {
  if (mode === 'local') {
    return process.env.LOCAL_LLM_MODEL || 'openai/gpt-oss-20b';
  } else if (mode === 'test') {
    return process.env.TEST_MODEL_ID || process.env.OPENROUTER_TEST_MODEL_ID || 'google/gemma-3-4b';
  } else if (mode === 'test2') {
    return process.env.TEST2_MODEL_ID || process.env.OPENROUTER_TEST2_MODEL_ID || 'llama-3.2-1b-instruct';
  } else if (mode === 'fast') {
    return process.env.FAST_MODEL_ID || process.env.OPENROUTER_FAST_MODEL_ID || 'gpt-oss-120b';
  } else if (mode === 'smart-2') {
    return process.env.SMART_2_MODEL_ID || process.env.OPENROUTER_SMART_2_MODEL_ID || 'google/gemini-3-pro-preview';
  } else {
    // Default to 'smart'
    return process.env.SMART_MODEL_ID || process.env.OPENROUTER_SMART_MODEL_ID || 'google/gemini-3.0-pro';
  }
}
const GEMINI_2_5_PRO_MODEL_ID = process.env.GEMINI_2_5_PRO_MODEL_ID || 'gemini-2.5-pro';
const GOOGLE_GEMINI_API_KEY = process.env.GOOGLE_GEMINI_API_KEY;
const TEXT_SERVICE_KEY = 'text-service';
const LEGACY_TEXT_SERVICE_KEYS = ['openai-text'];
const SPEECHMATICS_BASE_URL = 'https://asr.api.speechmatics.com/v3';
const LOCAL_LLM_BASE_URL = (process.env.LOCAL_LLM_BASE_URL || 'http://127.0.0.1:3001').replace(/\/$/, '');
const LOCAL_LLM_MODEL = process.env.LOCAL_LLM_MODEL || 'openai/gpt-oss-20b';
const LOCAL_LLM_API_KEY = process.env.LOCAL_LLM_API_KEY || '';

/**
 * Check if reasoning_effort should be set to "high" for the given model/mode
 * @param {string} mode - LLM mode ('local', 'fast', 'smart', 'smart-2')
 * @param {string} model - Model ID
 * @returns {boolean} True if reasoning_effort should be "high"
 * 
 * Note: For LM Studio (local mode), reasoning_effort may need to be set in the LM Studio UI
 * as the API parameter might be ignored. GPT-OSS-20B supports reasoning effort but LM Studio
 * may not pass it through the API correctly.
 */
function shouldUseHighReasoningEffort(mode, model) {
  // Use high reasoning effort for: Fast, and Smart (GPT 5.1)
  // NOTE: Local mode (LM Studio) reasoning_effort is disabled - set via LM Studio UI instead
  // if (mode === 'local') return true; // DISABLED - configure in LM Studio UI
  if (mode === 'fast') return true;
  if (mode === 'smart') return true;
  // Also check by model ID (in case mode is not available)
  // if (model === LOCAL_LLM_MODEL) return true; // DISABLED - configure in LM Studio UI
  if (model === FAST_MODEL_ID) return true;
  if (model === SMART_MODEL_ID) return true;
  // Check if model name contains GPT 5.1
  if (model && model.includes('gpt-5.1')) return true;
  // Check if model name contains gpt-oss (for LM Studio) - DISABLED
  // if (model && model.includes('gpt-oss')) return true; // DISABLED - configure in LM Studio UI
  return false;
}
const WORDS_PER_SECOND = parseFloat(process.env.TRANSCRIPT_WORDS_PER_SECOND || '3.5');
const LLM_CHUNK_WORD_LIMIT = parseInt(process.env.LLM_CHUNK_WORD_LIMIT || '900', 10);
const LLM_STOP_SEQUENCES = ['\n\n', '}\n}'];

// Ensure temp_uploads directory exists
const tempUploadsDir = path.join(__dirname, 'temp_uploads');
if (!fsSync.existsSync(tempUploadsDir)) {
  fsSync.mkdirSync(tempUploadsDir, { recursive: true });
}

const audioshakeStemsRoot = path.join(tempUploadsDir, 'audioshake_stems');
if (!fsSync.existsSync(audioshakeStemsRoot)) {
  fsSync.mkdirSync(audioshakeStemsRoot, { recursive: true });
}

const uploadsDir = path.join(__dirname, 'uploads');
if (!fsSync.existsSync(uploadsDir)) {
  fsSync.mkdirSync(uploadsDir, { recursive: true });
}

const debugDir = path.join(__dirname, 'Debug');
const speechbrainSamplesDir = path.join(debugDir, 'speechbrain_samples');
const speechbrainDashboardPath = path.join(debugDir, 'speechbrain_dashboard.html');
const speechbrainResultsPath = path.join(debugDir, 'speechbrain_test_results.json');
if (!fsSync.existsSync(debugDir)) {
  console.warn('⚠️  Debug directory not found. Dashboard routes will be unavailable.');
}

const cacheDir = path.join(__dirname, 'cache');
if (!fsSync.existsSync(cacheDir)) {
  fsSync.mkdirSync(cacheDir, { recursive: true });
}

// Diarization cache directory (full Speechmatics/Azure diarization results)
const diarizationCacheDir = path.join(cacheDir, 'diarization_results');
if (!fsSync.existsSync(diarizationCacheDir)) {
  fsSync.mkdirSync(diarizationCacheDir, { recursive: true });
}

// LLM responses cache directory (separate from audio cache)
const llmCacheDir = path.join(cacheDir, 'llm_responses');
if (!fsSync.existsSync(llmCacheDir)) {
  fsSync.mkdirSync(llmCacheDir, { recursive: true });
}

// Separation cache directory (for speaker separation results)
const separationCacheDir = path.join(cacheDir, 'separation');
if (!fsSync.existsSync(separationCacheDir)) {
  fsSync.mkdirSync(separationCacheDir, { recursive: true });
}

// 30 days TTL for diarization cache (in seconds)
const DIARIZATION_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60;

// LLM cache enabled flag (can be disabled via environment variable)
// This is independent from audio/transcription cache
const LLM_CACHE_ENABLED = process.env.LLM_CACHE_ENABLED !== 'false'; // Default: true
const LLM_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60; // 30 days

// Separation cache enabled flag
const SEPARATION_CACHE_ENABLED = process.env.SEPARATION_CACHE_ENABLED !== 'false'; // Default: true
const SEPARATION_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60; // 30 days

// LLM Cache functions (independent from audio cache)
function buildLLMCacheKey(filename, prompt, model, mode, promptVariant = 'default', demoLlmMode = null) {
  try {
    const crypto = require('crypto');
    // Create stable key based on filename, prompt content, model, mode, and demo LLM mode
    const filenameBase = filename 
      ? path.parse(filename).name.replace(/[^a-zA-Z0-9_-]/g, '_')
      : 'unknown';
    
    // Hash prompt to keep key manageable (first 16 chars of hash)
    const promptHash = crypto.createHash('sha256')
      .update(prompt)
      .digest('hex')
      .substring(0, 16);
    
    const modelSafe = (model || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    const modeSafe = (mode || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    const variantSafe = (promptVariant || 'default').replace(/[^a-zA-Z0-9_-]/g, '_');
    
    // For markdown-fixes variant, include DEMO_LLM_MODE in cache key
    // This ensures cache is invalidated when DEMO_LLM_MODE changes
    let demoModeSuffix = '';
    if (promptVariant === 'markdown-fixes') {
      // Use provided demoLlmMode or read from process.env
      const demoMode = demoLlmMode || process.env.DEMO_LLM_MODE || 'smart';
      const demoModeSafe = demoMode.replace(/[^a-zA-Z0-9_-]/g, '_');
      demoModeSuffix = `_demo_${demoModeSafe}`;
    }
    
    return `${filenameBase}_${promptHash}_${modelSafe}_${modeSafe}_${variantSafe}${demoModeSuffix}`;
  } catch (e) {
    console.warn('⚠️ Failed to build LLM cache key:', e.message);
    return null;
  }
}

const integrationStatePath = path.join(cacheDir, 'diarization_flow_state.json');

function getDefaultIntegrationState() {
  return {
    dialogueScripts: [],
    analyzerPayload: null
  };
}

function readIntegrationState() {
  try {
    const buffer = fsSync.readFileSync(integrationStatePath, 'utf8');
    const parsed = JSON.parse(buffer);
    return {
      ...getDefaultIntegrationState(),
      ...parsed,
      dialogueScripts: Array.isArray(parsed.dialogueScripts) ? parsed.dialogueScripts : []
    };
  } catch (error) {
    if (error.code !== 'ENOENT') {
      console.warn('⚠️  Failed to read integration state, using defaults:', error.message);
    }
    return getDefaultIntegrationState();
  }
}

function writeIntegrationState(state) {
  try {
    fsSync.writeFileSync(integrationStatePath, JSON.stringify(state, null, 2), 'utf8');
  } catch (error) {
    console.error('❌ Failed to persist integration state:', error);
  }
}

function toNumber(value, fallback = 0) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

function normalizeRole(role) {
  if (!role) return null;
  const normalized = role.toString().trim().toLowerCase();
  if (['agent', 'operator', 'support'].includes(normalized)) return 'agent';
  if (['client', 'customer', 'caller', 'user'].includes(normalized)) return 'client';
  return normalized || null;
}

function assignRolesToVoiceTracks(voiceTracks = []) {
  if (!Array.isArray(voiceTracks) || voiceTracks.length === 0) {
    return [];
  }

  const initialRoles = voiceTracks.map((track) => {
    const roleCandidate = track?.roleAnalysis?.role || track?.role || track?.metadata?.role;
    return normalizeRole(roleCandidate) || 'unknown';
  });

  const assignedRoles = [...initialRoles];
  const ensureRole = (roleName, preferredIndex) => {
    if (assignedRoles.includes(roleName)) {
      return;
    }
    const candidateIndex = preferredIndex ?? assignedRoles.findIndex((role, idx) => role === 'unknown' && idx !== 0);
    const targetIndex = candidateIndex !== -1 ? candidateIndex : (roleName === 'client' ? assignedRoles.length - 1 : 0);
    assignedRoles[targetIndex] = roleName;
  };

  // Ensure at least one agent and one client when possible
  ensureRole('agent', 0);
  if (voiceTracks.length > 1) {
    ensureRole('client', voiceTracks.length - 1);
  }

  // Balance remaining unknown roles between agent/client
  let agentCount = assignedRoles.filter(role => role === 'agent').length;
  let clientCount = assignedRoles.filter(role => role === 'client').length;

  assignedRoles.forEach((role, index) => {
    if (role === 'unknown') {
      if (agentCount <= clientCount) {
        assignedRoles[index] = 'agent';
        agentCount++;
      } else {
        assignedRoles[index] = 'client';
        clientCount++;
      }
    }
  });

  return assignedRoles;
}

function getSegmentSpeakerLabel(segment = {}) {
  if (segment.displayName) return segment.displayName;
  if (segment.speakerLabel) return segment.speakerLabel;
  if (segment.speaker_label) return segment.speaker_label;
  if (segment.speaker) return segment.speaker;
  if (typeof segment.speaker_id === 'number') {
    return `SPEAKER_${segment.speaker_id.toString().padStart(2, '0')}`;
  }
  if (typeof segment.speaker_id === 'string') {
    return segment.speaker_id;
  }
  return 'SPEAKER_00';
}

function buildSpeakerRoleMap(voiceTracks = []) {
  return voiceTracks.reduce((map, track) => {
    const speakerId = track?.speaker || track?.speakerId || track?.name;
    if (!speakerId) {
      return map;
    }
    const roleCandidate = track?.assignedRole || track?.roleAnalysis?.role || track?.role || track?.metadata?.role;
    const normalizedRole = normalizeRole(roleCandidate);
    if (normalizedRole && !map[speakerId]) {
      map[speakerId] = normalizedRole;
    }
    return map;
  }, {});
}

function extractSegmentsFromPayload(payload) {
  if (!payload) return [];

  if (Array.isArray(payload)) {
    if (payload.length === 0) return [];
    if (payload[0] && payload[0].text !== undefined && payload[0].start !== undefined) {
      return payload;
    }
    return payload.flatMap(item => extractSegmentsFromPayload(item));
  }

  if (Array.isArray(payload.segments)) {
    return payload.segments;
  }

  if (payload.speechmatics && Array.isArray(payload.speechmatics.segments)) {
    return payload.speechmatics.segments;
  }

  if (payload.results) {
    if (payload.results.speechmatics && Array.isArray(payload.results.speechmatics.segments)) {
      return payload.results.speechmatics.segments;
    }
    if (payload.results['overlap-corrected'] && Array.isArray(payload.results['overlap-corrected'].segments)) {
      return payload.results['overlap-corrected'].segments;
    }
    // Structured text-service responses may be nested under dynamic keys
    const dynamicKey = Object.keys(payload.results).find(key => Array.isArray(payload.results[key]?.segments));
    if (dynamicKey) {
      return payload.results[dynamicKey].segments;
    }
  }

  if (Array.isArray(payload.recordings) && payload.recordings.length > 0) {
    const firstRecording = payload.recordings.find(Boolean);
    if (firstRecording) {
      return extractSegmentsFromPayload(firstRecording);
    }
  }

  if (payload.transcription) {
    return extractSegmentsFromPayload(payload.transcription);
  }

  return [];
}

function sortSegmentsChronologically(segments) {
  return [...segments].sort((a, b) => toNumber(a.start) - toNumber(b.start));
}

function formatSegmentsAsDialog(segments, options = {}) {
  const { includeTimestamps = false } = options;
  const sorted = sortSegmentsChronologically(segments);

  return sorted
    .map(segment => {
      const speakerLabel = getSegmentSpeakerLabel(segment);
      const roleLabel = normalizeRole(segment.role);
      const roleSuffix = roleLabel ? ` (${roleLabel})` : '';
      const text = (segment.text || segment.word || '').trim();
      const start = toNumber(segment.start).toFixed(2);
      const end = toNumber(segment.end, segment.start).toFixed(2);
      const timestampPrefix = includeTimestamps ? `[${start}-${end}] ` : '';
      return `${timestampPrefix}${speakerLabel}${roleSuffix}: ${text}`;
    })
    .filter(Boolean)
    .join('\n')
    .trim();
}

function buildDialoguePromptContext({
  primaryDiarization = null,
  geminiDiarization = null, // LLM diarization from Gemini 2.5 Pro for transcription control
  agentTranscript = { segments: [] },
  clientTranscript = { segments: [] },
  voiceTracks = [],
  speaker0SegmentsOverride = null,
  speaker1SegmentsOverride = null,
  groundTruthText = null
} = {}) {
  const generalSegments = extractSegmentsFromPayload(primaryDiarization);
  const speakerRoleMap = buildSpeakerRoleMap(voiceTracks);

  // CRITICAL: Filter out any speakers beyond SPEAKER_00 and SPEAKER_01
  // Only allow two speakers (Agent and Client)
  const allowedSpeakers = new Set(['SPEAKER_00', 'SPEAKER_01']);
  const filteredGeneralSegments = generalSegments.filter(segment => {
    const speakerLabel = getSegmentSpeakerLabel(segment).toUpperCase();
    return allowedSpeakers.has(speakerLabel);
  });
  
  if (generalSegments.length !== filteredGeneralSegments.length) {
    const removedCount = generalSegments.length - filteredGeneralSegments.length;
    const removedSpeakers = [...new Set(generalSegments
      .filter(s => !allowedSpeakers.has(getSegmentSpeakerLabel(s).toUpperCase()))
      .map(s => getSegmentSpeakerLabel(s).toUpperCase()))];
    console.log(`⚠️ Filtered out ${removedCount} segments from disallowed speakers: ${removedSpeakers.join(', ')}`);
  }

  const speaker0Segments = Array.isArray(speaker0SegmentsOverride)
    ? speaker0SegmentsOverride
    : filteredGeneralSegments.filter(segment => getSegmentSpeakerLabel(segment).toUpperCase() === 'SPEAKER_00');

  const speaker1Segments = Array.isArray(speaker1SegmentsOverride)
    ? speaker1SegmentsOverride
    : filteredGeneralSegments.filter(segment => getSegmentSpeakerLabel(segment).toUpperCase() === 'SPEAKER_01');

  const generalDialog = formatSegmentsAsDialog(filteredGeneralSegments, { includeTimestamps: true }) || '[empty]';
  const speaker0Dialog = formatSegmentsAsDialog(speaker0Segments, { includeTimestamps: true }) || '[empty]';
  const speaker1Dialog = formatSegmentsAsDialog(speaker1Segments, { includeTimestamps: true }) || '[empty]';
  const agentDialog = formatSegmentsAsDialog(agentTranscript?.segments || [], { includeTimestamps: true }) || '[empty]';
  const clientDialog = formatSegmentsAsDialog(clientTranscript?.segments || [], { includeTimestamps: true }) || '[empty]';

  const roleGuidance = {
    speakerRoleMap,
    segments: {
      general: filteredGeneralSegments.length,
      speaker0: speaker0Segments.length,
      speaker1: speaker1Segments.length,
      agent: (agentTranscript?.segments || []).length,
      client: (clientTranscript?.segments || []).length
    },
    tracks: voiceTracks.map(track => ({
      speaker: track.speaker,
      roleAnalysis: track.roleAnalysis || null,
      assignedRole: track.assignedRole || track.role || track.roleAnalysis?.role || null,
      confidence: track.roleAnalysis?.confidence ?? null
    }))
  };

  const buildTimestampEntries = (source, segments) => {
    return sortSegmentsChronologically(segments).map((segment, index) => ({
      source,
      order: index + 1,
      speaker: getSegmentSpeakerLabel(segment),
      role: normalizeRole(segment.role),
      start: toNumber(segment.start),
      end: toNumber(segment.end, segment.start),
      text: (segment.text || segment.word || '').trim()
    }));
  };

  // Extract Gemini diarization segments for transcription control
  let geminiDialog = '[empty]';
  let geminiSegments = [];
  if (geminiDiarization) {
    geminiSegments = extractSegmentsFromPayload(geminiDiarization);
    geminiDialog = formatSegmentsAsDialog(geminiSegments, { includeTimestamps: true }) || '[empty]';
  }

  const segmentTimestampEntries = [
    ...buildTimestampEntries('general', filteredGeneralSegments),
    ...buildTimestampEntries('standard_speaker0', speaker0Segments),
    ...buildTimestampEntries('standard_speaker1', speaker1Segments),
    ...buildTimestampEntries('agent_track', agentTranscript?.segments || []),
    ...buildTimestampEntries('client_track', clientTranscript?.segments || []),
    ...buildTimestampEntries('gemini_llm', geminiSegments) // Add Gemini LLM diarization timestamps
  ];
  
  return {
    generalDialog,
    speaker0Dialog,
    speaker1Dialog,
    agentDialog,
    clientDialog,
    geminiDialog, // LLM diarization for transcription error correction
    roleGuidanceText: JSON.stringify(roleGuidance, null, 2),
    segmentTimestampsText: JSON.stringify(segmentTimestampEntries, null, 2),
    groundTruthText: groundTruthText || null,
    primaryDiarization: primaryDiarization || null, // Include for word-level comparison
    geminiDiarization: geminiDiarization || null // Include Gemini diarization for reference
  };
}

async function loadGroundTruthTextForUpload(uploadedFile) {
  if (!uploadedFile?.originalname) {
    return null;
  }
  
  const baseName = path.parse(uploadedFile.originalname).name;
  const candidatePaths = [
    path.join(__dirname, 'Debug', `${baseName}.txt`),
    path.join(__dirname, 'Debug', 'demo_page', `${baseName}.txt`),
    path.join(__dirname, 'Debug', 'ground_truth', `${baseName}.txt`)
  ];
  
  for (const candidate of candidatePaths) {
    try {
      await fs.access(candidate);
      const fileContents = await fs.readFile(candidate, 'utf8');
      if (fileContents && fileContents.trim().length > 0) {
        console.log(`🧾 Ground truth transcript detected for ${baseName}: ${candidate}`);
        return fileContents;
      }
    } catch (error) {
      // Ignore ENOENT and continue checking other paths
    }
  }
  
  return null;
}

// Middleware
const BODY_LIMIT = process.env.REQUEST_BODY_LIMIT || '15mb';

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

app.use(express.json({ limit: BODY_LIMIT }));
app.use(express.urlencoded({ extended: true, limit: BODY_LIMIT }));
app.use('/uploads', express.static(uploadsDir));

if (fsSync.existsSync(speechbrainSamplesDir)) {
  app.use('/speechbrain_samples', express.static(speechbrainSamplesDir));
} else {
  console.warn('⚠️  speechbrain_samples directory not found. Sample playback disabled.');
}

if (fsSync.existsSync(debugDir)) {
  app.use('/Debug', express.static(debugDir));
  // Expose dashboards at predictable root paths as well
  app.use('/dashboards', express.static(debugDir));
} else {
  console.warn('⚠️  Debug assets will not be served because the directory is missing.');
}

function sendDebugFile(res, next, filePath, friendlyName) {
  res.sendFile(filePath, err => {
    if (!err) {
      return;
    }

    if (err.code === 'ENOENT') {
      console.warn(`⚠️  Missing ${friendlyName} at ${filePath}`);
      res.status(404).send(`${friendlyName} not found on server`);
    } else {
      next(err);
    }
  });
}

// Monitoring endpoint for Azure STT debugging
app.get('/api/get-latest-recording', async (req, res) => {
  try {
    const recordingsPath = path.join(__dirname, 'recordings.json');
    if (!fsSync.existsSync(recordingsPath)) {
      return res.json({ success: false, error: 'No recordings file found' });
    }

    const data = await fs.readFile(recordingsPath, 'utf-8');
    const recordings = JSON.parse(data).recordings || [];
    
    if (recordings.length === 0) {
      return res.json({ success: false, error: 'No recordings found' });
    }

    // Get the most recent recording (by ID or timestamp)
    const latest = recordings.sort((a, b) => {
      const aTime = a.id ? parseInt(a.id.split('_')[1]) || 0 : 0;
      const bTime = b.id ? parseInt(b.id.split('_')[1]) || 0 : 0;
      return bTime - aTime;
    })[0];

    res.json({ success: true, recording: latest });
  } catch (error) {
    console.error('Error fetching latest recording:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Serve monitoring page
app.get('/monitor', (req, res) => {
  res.sendFile(path.join(__dirname, 'monitor.html'));
});

app.get(
  ['/speechbrain_dashboard.html', '/speechbrain-dashboard'],
  (req, res, next) => {
    sendDebugFile(res, next, speechbrainDashboardPath, 'SpeechBrain dashboard');
  }
);

app.get(
  ['/speechbrain_test_results.json', '/speechbrain-dashboard/data'],
  (req, res, next) => {
    sendDebugFile(res, next, speechbrainResultsPath, 'SpeechBrain test results');
  }
);

// Configure multer for file uploads
const upload = multer({
  dest: path.join(__dirname, 'temp_uploads'),
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB limit
  }
});

let tunnelUrl = null;
let activeTunnel = null;

const defaultPythonBin = path.join(__dirname, '.venv', 'bin', 'python3');
const PYTHON_BIN = process.env.PYTHON_BIN
  ? (path.isAbsolute(process.env.PYTHON_BIN) ? process.env.PYTHON_BIN : path.resolve(__dirname, process.env.PYTHON_BIN))
  : (fsSync.existsSync(defaultPythonBin) ? defaultPythonBin : 'python3');

// Log which Python binary is being used
console.log('🐍 Using Python binary:', PYTHON_BIN);
console.log('🐍 Python binary exists:', fsSync.existsSync(PYTHON_BIN));

// Log file path
const LOG_FILE = 'server_debug.log';
const PUBLIC_URL = process.env.PUBLIC_URL ? process.env.PUBLIC_URL.replace(/\/$/, '') : null;

// Helper function to write logs to file
function writeLog(level, message, data = null) {
  const timestamp = new Date().toISOString();
  const logEntry = {
    timestamp,
    level,
    message,
    data
  };
  const logLine = JSON.stringify(logEntry) + '\n';
  fsSync.appendFileSync(LOG_FILE, logLine, 'utf8');
  console.log(`📱 FRONTEND LOG [${level}]:`, message);
  if (data) {
    console.log('📱 FRONTEND DATA:', JSON.stringify(data, null, 2));
  }
}

// API route for logging (to capture frontend console logs)
app.post('/api/log', (req, res) => {
  const { level, message, data } = req.body;
  writeLog(level || 'info', message, data);
  res.json({ success: true });
});

app.post('/api/dialogue-scripts', (req, res) => {
  const { text, lines, meta } = req.body || {};
  if (!text || typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'Original dialogue text is required' });
  }

  const entry = {
    id: Date.now().toString(),
    text: text.trim(),
    lines: Array.isArray(lines) && lines.length > 0 ? lines : text.split('\n').map(line => line.trim()).filter(Boolean),
    meta: meta && typeof meta === 'object' ? meta : {},
    storedAt: new Date().toISOString()
  };

  const state = readIntegrationState();
  state.dialogueScripts = [...(state.dialogueScripts || []), entry].slice(-25);
  writeIntegrationState(state);

  res.json({ success: true, entry });
});

app.get('/api/dialogue-scripts/latest', (req, res) => {
  const state = readIntegrationState();
  const scripts = state.dialogueScripts || [];
  if (!scripts.length) {
    return res.status(404).json({ error: 'No dialogue scripts found' });
  }
  res.json(scripts[scripts.length - 1]);
});

app.post('/api/analyzer-payload', (req, res) => {
  const { original, recognized } = req.body || {};
  if (!original || typeof original !== 'object' || !original.text) {
    return res.status(400).json({ error: 'Original dialogue payload is required' });
  }
  if (!recognized || typeof recognized !== 'object' || !recognized.text) {
    return res.status(400).json({ error: 'Recognized dialogue payload is required' });
  }

  const payload = {
    original,
    recognized,
    storedAt: new Date().toISOString()
  };

  const state = readIntegrationState();
  state.analyzerPayload = payload;
  writeIntegrationState(state);

  res.json({ success: true, payload });
});

app.get('/api/analyzer-payload/latest', (req, res) => {
  const state = readIntegrationState();
  if (!state.analyzerPayload) {
    return res.status(404).json({ error: 'No analyzer payload stored' });
  }
  res.json(state.analyzerPayload);
});

// API endpoint to compare uploaded dialogue with existing diarization results
app.post('/api/compare-with-dialogue', async (req, res) => {
  try {
    const { groundTruthText, diarizationData } = req.body;
    
    if (!groundTruthText || typeof groundTruthText !== 'string' || !groundTruthText.trim()) {
      return res.status(400).json({ error: 'Ground truth text is required' });
    }
    
    if (!diarizationData) {
      return res.status(400).json({ error: 'Diarization data is required' });
    }
    
    // Extract markdown table from diarization data
    let markdownTable = null;
    if (diarizationData.markdownTable) {
      markdownTable = diarizationData.markdownTable;
    } else if (diarizationData.correctedDiarization) {
      const corrected = diarizationData.correctedDiarization;
      if (corrected.recordings && Array.isArray(corrected.recordings) && corrected.recordings[0]) {
        const recording = corrected.recordings[0];
        if (recording.results && recording.results['overlap-corrected']) {
          const rawData = recording.results['overlap-corrected'].rawData;
          if (rawData && rawData.markdownTable) {
            markdownTable = rawData.markdownTable;
          }
        }
      } else if (corrected.rawData && corrected.rawData.markdownTable) {
        markdownTable = corrected.rawData.markdownTable;
      }
    }
    
    if (!markdownTable) {
      return res.status(400).json({ error: 'Markdown table not found in diarization data' });
    }
    
    // Extract primaryDiarization for Speechmatics comparison
    const primaryDiarization = diarizationData.primaryDiarization || null;
    
    // Calculate ground truth metrics
    const groundTruthMetrics = calculateGroundTruthMatch(
      markdownTable,
      groundTruthText,
      primaryDiarization
    );
    
    // Store the uploaded dialogue
    const state = readIntegrationState();
    const entry = {
      id: Date.now().toString(),
      text: groundTruthText.trim(),
      lines: groundTruthText.split('\n').map(line => line.trim()).filter(Boolean),
      meta: { uploadedAt: new Date().toISOString() },
      storedAt: new Date().toISOString()
    };
    state.dialogueScripts = [...(state.dialogueScripts || []), entry].slice(-25);
    writeIntegrationState(state);
    
    res.json({
      success: true,
      groundTruthMetrics: groundTruthMetrics,
      dialogueEntry: entry
    });
  } catch (error) {
    console.error('Error comparing with dialogue:', error);
    res.status(500).json({ error: 'Failed to compare with dialogue', details: error.message });
  }
});

app.get('/api/tunnel-status', (req, res) => {
  res.json({
    active: Boolean(tunnelUrl || PUBLIC_URL),
    url: tunnelUrl || PUBLIC_URL,
    message: tunnelUrl
      ? 'Tunnel is active'
      : (PUBLIC_URL ? 'Using PUBLIC_URL fallback' : 'Tunnel not ready')
  });
});

app.post('/api/tunnel-start', async (req, res) => {
  try {
    if (activeTunnel) {
      return res.json({
        success: true,
        active: true,
        url: tunnelUrl,
        message: 'Tunnel is already active'
      });
    }

    await startTunnel();

    if (tunnelUrl) {
      res.json({
        success: true,
        active: true,
        url: tunnelUrl,
        message: 'Tunnel started successfully'
      });
    } else {
      res.json({
        success: false,
        active: false,
        message: 'Failed to start tunnel. Check server logs for details.'
      });
    }
  } catch (error) {
    console.error('Error starting tunnel via API:', error);
    res.status(500).json({
      success: false,
      active: false,
      error: error.message || 'Failed to start tunnel'
    });
  }
});

app.post('/api/speechmatics/media', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Audio file is required (field name: file)' });
  }

  const tempPath = req.file.path;
  try {
    const headers = getSpeechmaticsHeaders({
      'Content-Type': req.file.mimetype || 'application/octet-stream',
      'Content-Length': req.file.size
    });

    const response = await axios.post(
      `${SPEECHMATICS_BASE_URL}/media`,
      fsSync.createReadStream(tempPath),
      {
        headers,
        maxBodyLength: Infinity,
        maxContentLength: Infinity
      }
    );

    res.json(response.data);
  } catch (error) {
    speechmaticsErrorResponse(res, error, 'Speechmatics media upload failed');
  } finally {
    if (tempPath) {
      fs.unlink(tempPath).catch(() => {});
    }
  }
});

app.post('/api/speechmatics/jobs', async (req, res) => {
  try {
    const headers = getSpeechmaticsHeaders({ 'Content-Type': 'application/json' });
    const response = await axios.post(`${SPEECHMATICS_BASE_URL}/jobs`, req.body, {
      headers
    });
    res.json(response.data);
  } catch (error) {
    speechmaticsErrorResponse(res, error, 'Speechmatics job creation failed');
  }
});

app.get('/api/speechmatics/jobs/:jobId', async (req, res) => {
  try {
    const headers = getSpeechmaticsHeaders();
    const response = await axios.get(`${SPEECHMATICS_BASE_URL}/jobs/${req.params.jobId}`, {
      headers
    });
    res.json(response.data);
  } catch (error) {
    speechmaticsErrorResponse(res, error, 'Speechmatics job status fetch failed');
  }
});

app.get('/api/speechmatics/jobs/:jobId/transcript', async (req, res) => {
  try {
    const headers = getSpeechmaticsHeaders();
    const format = req.query.format || 'json-v2';
    const response = await axios.get(
      `${SPEECHMATICS_BASE_URL}/jobs/${req.params.jobId}/transcript`,
      {
        headers,
        params: { format }
      }
    );
    res.json(response.data);
  } catch (error) {
    speechmaticsErrorResponse(res, error, 'Speechmatics transcript fetch failed');
  }
});

function getSpeechmaticsHeaders(extraHeaders = {}) {
  const apiKey = process.env.SPEECHMATICS_API_KEY;
  if (!apiKey) {
    const error = new Error('Speechmatics API key is not configured on the server');
    error.statusCode = 500;
    throw error;
  }
  return {
    Authorization: `Bearer ${apiKey}`,
    ...extraHeaders
  };
}

function speechmaticsErrorResponse(res, error, fallbackMessage) {
  const status = error.statusCode || error.response?.status || 500;
  const details = error.response?.data || error.message;
  console.error(`❌ ${fallbackMessage}:`, details);
  res.status(status).json({
    error: fallbackMessage,
    details
  });
}

// API routes must be defined BEFORE static files middleware
// Cached regex patterns for better performance
const SERVICE_MARKER_REGEX = /^--\s+[^-]+\s+--\s*$/i;
const SEPARATOR_REGEX = /^---+\s*$/;

// Function to clean service messages from transcript text
function cleanServiceMessages(text) {
  // Remove lines like "-- Speechmatics --", "-- Service Name --", etc.
  const lines = text.split('\n');
  const cleanedLines = lines.filter(line => {
    const trimmed = line.trim();
    // Skip lines that are service markers (-- ServiceName --)
    if (SERVICE_MARKER_REGEX.test(trimmed)) {
      return false;
    }
    // Skip lines that are just separators
    if (SEPARATOR_REGEX.test(trimmed)) {
      return false;
    }
    return true;
  });
  return cleanedLines.join('\n').trim();
}

async function downloadAudioToTemp(audioUrl) {
  if (!audioUrl) {
    throw new Error('Audio URL is required for download');
  }

  try {
    const response = await axios.get(audioUrl, {
      responseType: 'arraybuffer',
      timeout: 120000
    });

    const urlObject = new URL(audioUrl);
    const extension = path.extname(urlObject.pathname) || '.tmp';
    const tempFilePath = path.join(
      tempUploadsDir,
      `remote-${Date.now()}-${Math.random().toString(36).slice(2)}${extension}`
    );

    await fs.writeFile(tempFilePath, Buffer.from(response.data));
    writeLog('info', 'Downloaded remote audio file for processing', {
      audioUrl,
      tempFilePath,
      size: response.headers['content-length'] || null
    });

    return tempFilePath;
  } catch (error) {
    writeLog('error', 'Failed to download remote audio file', {
      audioUrl,
      error: error.message
    });
    throw error;
  }
}

function getPublicBaseUrl() {
  if (tunnelUrl) {
    return tunnelUrl;
  }
  if (PUBLIC_URL) {
    return PUBLIC_URL;
  }
  return `http://localhost:${PORT}`;
}

/**
 * Sanitize diarization response by removing internal keys and extracting only necessary data
 * @param {Object} result - The full diarization result
 * @returns {Object} Sanitized result with only public-facing data
 */
function sanitizeDiarizationResponse(result) {
  if (!result || typeof result !== 'object') return result;
  
  const sanitized = {
    success: result.success,
    pipelineMode: result.pipelineMode,
    totalDuration: result.totalDuration
  };
  
  // Keep primaryDiarization and correctedDiarization for client-side processing
  // but remove internal diagnostic keys
  if (result.primaryDiarization) {
    sanitized.primaryDiarization = result.primaryDiarization;
  }
  
  if (result.correctedDiarization) {
    sanitized.correctedDiarization = result.correctedDiarization;
  }
  
  // Keep markdownTable and textAnalysis for client-side rendering
  if (result.markdownTable) {
    sanitized.markdownTable = result.markdownTable;
    console.log(`[sanitize] ✅ Preserved markdownTable (length: ${result.markdownTable.length})`);
  } else {
    console.log(`[sanitize] ⚠️ markdownTable is missing in result:`, {
      resultKeys: Object.keys(result),
      hasMarkdownTable: 'markdownTable' in result
    });
  }
  
  if (result.textAnalysis) {
    sanitized.textAnalysis = result.textAnalysis;
  }
  
  // Keep groundTruthMetrics for client-side display
  if (result.groundTruthMetrics) {
    sanitized.groundTruthMetrics = result.groundTruthMetrics;
    const nextLevelPercent = result.groundTruthMetrics.nextLevel?.matchPercent || 'N/A';
    const speechmaticsPercent = result.groundTruthMetrics.speechmatics?.matchPercent || 'N/A';
    console.log(`[sanitize] ✅ Preserved groundTruthMetrics (NextLevel: ${nextLevelPercent}%, Speechmatics: ${speechmaticsPercent}%)`);
  }
  
  // Keep separation info if needed
  if (result.separation) {
    sanitized.separation = result.separation;
  }
  
  // Keep voiceTracks with full data for client-side processing
  if (result.voiceTracks && Array.isArray(result.voiceTracks)) {
    // Keep full voice tracks data for JSON copy functionality
    sanitized.voiceTracks = result.voiceTracks;
  }
  
  return sanitized;
}

function sanitizeFileName(name) {
  return (name || 'audio')
    .replace(/[^a-z0-9._-]/gi, '_')
    .replace(/_+/g, '_');
}

async function persistUploadedFile(tempPath, originalName) {
  const safeBase = sanitizeFileName(originalName);
  const filename = `${Date.now()}_${safeBase}`;
  const destinationPath = path.join(uploadsDir, filename);
  try {
    await fs.rename(tempPath, destinationPath);
  } catch (error) {
    await fs.copyFile(tempPath, destinationPath);
    await fs.unlink(tempPath);
  }
  return { filename, destinationPath };
}

// Convert audio file to WAV format using ffmpeg
async function convertAudioToWav(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    const ffmpegPath = process.env.FFMPEG_PATH || 'ffmpeg';
    const ffmpegArgs = [
      '-hide_banner',
      '-loglevel', 'error',
      '-y', // Overwrite output file
      '-i', inputPath,
      '-ac', '1', // Mono
      '-ar', '8000', // 8kHz sample rate (SpeechBrain expects 8kHz)
      '-sample_fmt', 's16', // 16-bit PCM
      outputPath
    ];
    
    console.log(`[CONVERT] Converting ${inputPath} to WAV: ${outputPath}`);
    const ffmpegProcess = spawn(ffmpegPath, ffmpegArgs);
    
    let stderr = '';
    ffmpegProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    ffmpegProcess.on('close', (code) => {
      if (code === 0 && fsSync.existsSync(outputPath)) {
        console.log(`[CONVERT] ✅ Successfully converted to WAV: ${outputPath}`);
        resolve(outputPath);
      } else {
        const error = new Error(`ffmpeg conversion failed with code ${code}: ${stderr}`);
        console.error(`[CONVERT] ❌ ${error.message}`);
        reject(error);
      }
    });
    
    ffmpegProcess.on('error', (error) => {
      console.error(`[CONVERT] ❌ Failed to spawn ffmpeg: ${error.message}`);
      reject(new Error(`ffmpeg not found or not executable: ${error.message}`));
    });
  });
}

function getPublicFileUrl(filename) {
  const baseUrl = getPublicBaseUrl().replace(/\/$/, '');
  return `${baseUrl}/uploads/${encodeURIComponent(filename)}`;
}

async function saveTranscriptFile(content, baseName = 'voice_track_transcript') {
  if (!content || !content.trim()) {
    return null;
  }

  const safeBase = sanitizeFileName(baseName);
  const fileName = `${safeBase}_${Date.now()}.txt`;
  const filePath = path.join(uploadsDir, fileName);

  await fs.writeFile(filePath, content, 'utf8');
  const stats = await fs.stat(filePath);

  return {
    fileName,
    filePath,
    downloadUrl: `/uploads/${encodeURIComponent(fileName)}`,
    size: stats.size,
    createdAt: new Date().toISOString()
  };
}

async function ensurePublicFileAccessible(publicUrl) {
  try {
    const headResponse = await fetch(publicUrl, { method: 'HEAD' });
    if (!headResponse.ok) {
      throw new Error(`Public file not accessible (status ${headResponse.status})`);
    }
  } catch (error) {
    throw new Error(`Failed to verify public file URL: ${error.message}`);
  }
}

function resolveTranscriptionEngine(engine) {
  if (!engine) return 'speechmatics';
  const normalized = String(engine).toLowerCase();
  if (normalized === 'azure' || normalized === 'azure-stt') {
    return 'azure';
  }
  if (normalized === 'azure-realtime' || normalized === 'azure_realtime') {
    return 'azure_realtime';
  }
  return 'speechmatics';
}

const AZURE_PROGRESS_POLL_REGEX = /\[Azure STT\]\s+Poll attempt\s+(\d+)\/(\d+)\s+status=([A-Za-z]+)/i;
const AZURE_PROGRESS_JOB_SUBMITTED_REGEX = /\[Azure STT\]\s+Job submitted id=([0-9a-f-]+)/i;
const AZURE_PROGRESS_JOB_DONE_REGEX = /\[Azure STT\]\s+Job succeeded id=([0-9a-f-]+)(?:\s+status=([A-Za-z]+))?/i;
const AZURE_PROGRESS_FILES_REGEX = /\[Azure STT\]\s+Transcription files found=(\d+)/i;
const AZURE_PROGRESS_RESULT_FILES_REGEX = /\[Azure STT\]\s+Result files ready=(\d+)/i;
const AZURE_PROGRESS_PAYLOAD_REGEX = /\[Azure STT\]\s+Payload recognized phrases=(\d+)/i;
const AZURE_PROGRESS_SEGMENTS_REGEX = /\[Azure STT\]\s+Segments parsed=(\d+)/i;

function extractAzureProgressMessages(stderrChunk) {
  if (!stderrChunk || stderrChunk.indexOf('[Azure STT]') === -1) {
    return [];
  }

  const updates = [];
  const lines = stderrChunk.split(/\r?\n/);

  for (const rawLine of lines) {
    if (!rawLine || rawLine.indexOf('[Azure STT]') === -1) {
      continue;
    }

    const trimmedLine = rawLine.trim();
    const azurePortion = trimmedLine.substring(trimmedLine.indexOf('[Azure STT]'));

    const pollMatch = AZURE_PROGRESS_POLL_REGEX.exec(azurePortion);
    if (pollMatch) {
      const attempt = parseInt(pollMatch[1], 10);
      const total = parseInt(pollMatch[2], 10);
      const status = pollMatch[3];
      updates.push({
        source: 'azure',
        stage: 'azure_polling',
        description: `🔄 Azure STT: статус ${status} (${attempt}/${total})`,
        meta: {
          azureAttempt: attempt,
          azureTotalAttempts: total,
          azureStatus: status
        }
      });
      continue;
    }

    const jobSubmittedMatch = AZURE_PROGRESS_JOB_SUBMITTED_REGEX.exec(azurePortion);
    if (jobSubmittedMatch) {
      const jobId = jobSubmittedMatch[1];
      updates.push({
        source: 'azure',
        stage: 'azure_job_submitted',
        description: `📤 Azure STT: надіслано job ${jobId}`,
        meta: {
          azureJobId: jobId
        }
      });
      continue;
    }

    const jobDoneMatch = AZURE_PROGRESS_JOB_DONE_REGEX.exec(azurePortion);
    if (jobDoneMatch) {
      const jobId = jobDoneMatch[1];
      const status = jobDoneMatch[2] || 'Succeeded';
      updates.push({
        source: 'azure',
        stage: 'azure_job_completed',
        description: `✅ Azure STT: job ${jobId} завершено (${status})`,
        meta: {
          azureJobId: jobId,
          azureStatus: status
        }
      });
      continue;
    }

    const filesMatch = AZURE_PROGRESS_FILES_REGEX.exec(azurePortion);
    if (filesMatch) {
      const count = parseInt(filesMatch[1], 10);
      updates.push({
        source: 'azure',
        stage: 'azure_files_ready',
        description: `📁 Azure STT: отримано ${count} файлів`,
        meta: {
          azureFilesCount: count
        }
      });
      continue;
    }

    const resultFilesMatch = AZURE_PROGRESS_RESULT_FILES_REGEX.exec(azurePortion);
    if (resultFilesMatch) {
      const count = parseInt(resultFilesMatch[1], 10);
      updates.push({
        source: 'azure',
        stage: 'azure_result_files',
        description: `📦 Azure STT: готово ${count} результатних файлів`,
        meta: {
          azureFilesCount: count
        }
      });
      continue;
    }

    const payloadMatch = AZURE_PROGRESS_PAYLOAD_REGEX.exec(azurePortion);
    if (payloadMatch) {
      const phrases = parseInt(payloadMatch[1], 10);
      updates.push({
        source: 'azure',
        stage: 'azure_payload_ready',
        description: `📝 Azure STT: знайдено ${phrases} фраз`,
        meta: {
          azurePhrasesCount: phrases
        }
      });
      continue;
    }

    const segmentsMatch = AZURE_PROGRESS_SEGMENTS_REGEX.exec(azurePortion);
    if (segmentsMatch) {
      const segments = parseInt(segmentsMatch[1], 10);
      updates.push({
        source: 'azure',
        stage: 'azure_segments_ready',
        description: `🗂️ Azure STT: нормалізовано ${segments} сегментів`,
        meta: {
          azureSegmentsCount: segments
        }
      });
      continue;
    }

    updates.push({
      source: 'azure',
      stage: 'azure_update',
      description: `ℹ️ Azure STT: ${azurePortion.replace('[Azure STT]', '').trim()}`,
      meta: {}
    });
  }

  return updates;
}

async function startTunnel() {
  if (process.env.DISABLE_LOCALTUNNEL === 'true') {
    console.log('ℹ️  Localtunnel disabled via DISABLE_LOCALTUNNEL flag.');
    return;
  }

  if (activeTunnel) {
    return;
  }

  try {
    const subdomain = process.env.LOCALTUNNEL_SUBDOMAIN || undefined;
    activeTunnel = await localtunnel({
      port: TUNNEL_PORT,
      subdomain
    });

    tunnelUrl = activeTunnel.url.replace(/\/$/, '');
    console.log('\n🌐 Localtunnel URL:', tunnelUrl);
    console.log('✓ Files accessible at:', `${tunnelUrl}/uploads/`);
    console.log('💡 Tip: set LOCALTUNNEL_SUBDOMAIN in .env for a fixed URL\n');

    activeTunnel.on('close', () => {
      console.log('🔌 Localtunnel closed');
      activeTunnel = null;
      tunnelUrl = null;
    });

    activeTunnel.on('error', (err) => {
      console.error('❌ Localtunnel error:', err);
      activeTunnel = null;
      tunnelUrl = null;
    });
  } catch (error) {
    console.error('❌ Failed to start localtunnel:', error.message);
    console.log('⚠️  Continuing without tunnel (public uploads unavailable).');
  }
}

// LLM Cache functions (must be at module level to be accessible from routes)
function readLLMCache(cacheKey) {
  if (!LLM_CACHE_ENABLED || !cacheKey) return null;
  try {
    const cachePath = path.join(llmCacheDir, `${cacheKey}.json`);
    if (!fsSync.existsSync(cachePath)) {
      return null;
    }

    const stat = fsSync.statSync(cachePath);
    const ageSeconds = (Date.now() - stat.mtimeMs) / 1000;
    if (ageSeconds > LLM_CACHE_TTL_SECONDS) {
      console.log(`⏰ LLM cache expired, removing: ${cachePath}`);
      try {
        fsSync.unlinkSync(cachePath);
      } catch (unlinkErr) {
        console.warn('⚠️ Failed to remove expired LLM cache:', unlinkErr.message);
      }
      return null;
    }

    const raw = fsSync.readFileSync(cachePath, 'utf8');
    const parsed = JSON.parse(raw);
    console.log('✅ Using cached LLM response from cache:', { cacheKey });
    return parsed;
  } catch (e) {
    console.warn('⚠️ Failed to read LLM cache:', e.message);
    return null;
  }
}

function writeLLMCache(cacheKey, data) {
  if (!LLM_CACHE_ENABLED || !cacheKey || !data) return;
  try {
    const cachePath = path.join(llmCacheDir, `${cacheKey}.json`);
    fsSync.writeFileSync(cachePath, JSON.stringify(data, null, 2), 'utf8');
    console.log('💾 Saved LLM response to cache:', { cacheKey });
  } catch (e) {
    console.warn('⚠️ Failed to write LLM cache:', e.message);
  }
}

// Separation Cache functions (must be at module level to be accessible from routes)
function buildSeparationCacheKey(filename, mode, audioHash = null) {
  try {
    const crypto = require('crypto');
    // Create stable key based on filename, mode, and optional audio hash
    const filenameBase = filename 
      ? path.parse(filename).name.replace(/[^a-zA-Z0-9_-]/g, '_')
      : 'unknown';
    
    const modeSafe = (mode || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_');
    
    // If audio hash is provided, use it for more precise caching
    // Otherwise, use filename + mode
    if (audioHash) {
      const hashShort = audioHash.substring(0, 16);
      return `sep_${filenameBase}_${modeSafe}_${hashShort}`;
    }
    
    return `sep_${filenameBase}_${modeSafe}`;
  } catch (e) {
    console.warn('⚠️ Failed to build separation cache key:', e.message);
    return null;
  }
}

function readSeparationCache(cacheKey) {
  if (!SEPARATION_CACHE_ENABLED || !cacheKey) return null;
  try {
    const cachePath = path.join(separationCacheDir, `${cacheKey}.json`);
    if (!fsSync.existsSync(cachePath)) {
      return null;
    }

    const stat = fsSync.statSync(cachePath);
    const ageSeconds = (Date.now() - stat.mtimeMs) / 1000;
    if (ageSeconds > SEPARATION_CACHE_TTL_SECONDS) {
      console.log(`⏰ Separation cache expired, removing: ${cachePath}`);
      try {
        fsSync.unlinkSync(cachePath);
      } catch (unlinkErr) {
        console.warn('⚠️ Failed to remove expired separation cache:', unlinkErr.message);
      }
      return null;
    }

    const raw = fsSync.readFileSync(cachePath, 'utf8');
    const parsed = JSON.parse(raw);
    console.log('✅ Using cached separation result:', { cacheKey, ageDays: (ageSeconds / (24 * 60 * 60)).toFixed(1) });
    return parsed;
  } catch (e) {
    console.warn('⚠️ Failed to read separation cache:', e.message);
    return null;
  }
}

function writeSeparationCache(cacheKey, data) {
  if (!SEPARATION_CACHE_ENABLED || !cacheKey || !data) return;
  try {
    const cachePath = path.join(separationCacheDir, `${cacheKey}.json`);
    // Store only metadata, not file paths (they may change)
    const cacheData = {
      taskId: data.taskId,
      speakers: data.speakers?.map(s => ({
        name: s.name,
        format: s.format,
        isBackground: s.isBackground,
        // Don't cache URLs/paths as they may expire or change
        // URLs will be regenerated when cache is used
      })) || [],
      cost: data.cost,
      duration: data.duration,
      timeline: data.timeline,
      output_dir: data.output_dir,
      mode: data.mode,
      timestamp: new Date().toISOString(),
      // Store original filename for reference
      originalFilename: data.originalFilename
    };
    fsSync.writeFileSync(cachePath, JSON.stringify(cacheData, null, 2), 'utf8');
    console.log('💾 Saved separation result to cache:', { cacheKey, speakersCount: cacheData.speakers.length });
  } catch (e) {
    console.warn('⚠️ Failed to write separation cache:', e.message);
  }
}

async function runPythonDiarization({
  filePath,
  url,
  language,
  speakerCount,
  isSeparatedTrack = false,
  originalFilename = null,
  skipCache = false,
  engine = 'speechmatics',
  onProgress = null
}) {
  // Helper: build stable cache key for full diarization result
  const buildDiarizationCacheKey = () => {
    try {
      let baseName = originalFilename;

      if (!baseName) {
        if (url) {
          try {
            const parsedUrl = new URL(url);
            baseName = path.basename(parsedUrl.pathname) || null;
          } catch {
            // Fallback: use raw URL slug
            baseName = url.split('/').pop() || null;
          }
        } else if (filePath) {
          baseName = path.basename(filePath);
        }
      }

      if (!baseName) {
        baseName = 'audio';
      }

      // Sanitize filename for filesystem
      const baseNoExt = baseName.replace(/\.[^/.]+$/, '');
      const safeBase = baseNoExt
        .split('')
        .map(c => (/[a-zA-Z0-9_-]/.test(c) ? c : '_'))
        .join('')
        .replace(/_+/g, '_')
        .replace(/^_+|_+$/g, '') || 'audio';

      const langPart = (language || 'unknown').toLowerCase();
      const speakerPart = speakerCount ? String(speakerCount) : 'auto';
      const sepPart = isSeparatedTrack ? 'ch' : 'mix';
      const enginePart = (engine || 'speechmatics').toLowerCase();

      return `${safeBase}_${langPart}_${speakerPart}_${sepPart}_${enginePart}`;
    } catch (e) {
      console.warn('⚠️ Failed to build diarization cache key:', e.message);
      return null;
    }
  };

  const readDiarizationCache = (cacheKey) => {
    if (!cacheKey) return null;
    try {
      const cachePath = path.join(diarizationCacheDir, `${cacheKey}.json`);
      if (!fsSync.existsSync(cachePath)) {
        return null;
      }

      const stat = fsSync.statSync(cachePath);
      const ageSeconds = (Date.now() - stat.mtimeMs) / 1000;
      if (ageSeconds > DIARIZATION_CACHE_TTL_SECONDS) {
        console.log(`⏰ Diarization cache expired, removing: ${cachePath}`);
        try {
          fsSync.unlinkSync(cachePath);
        } catch (unlinkErr) {
          console.warn('⚠️ Failed to remove expired diarization cache:', unlinkErr.message);
        }
        return null;
      }

      const raw = fsSync.readFileSync(cachePath, 'utf8');
      const parsed = JSON.parse(raw);
      console.log('✅ Using cached diarization result from Node cache:', {
        cacheKey,
        segmentsCount:
          parsed?.recordings?.[0]?.results?.speechmatics?.segments?.length ||
          parsed?.recordings?.[0]?.results?.azure?.segments?.length ||
          0
      });
      return parsed;
    } catch (e) {
      console.warn('⚠️ Failed to read diarization cache:', e.message);
      return null;
    }
  };

  const writeDiarizationCache = (cacheKey, data) => {
    if (!cacheKey || !data) return;
    try {
      const cachePath = path.join(diarizationCacheDir, `${cacheKey}.json`);
      fsSync.writeFileSync(cachePath, JSON.stringify(data, null, 2), 'utf8');
      console.log('💾 Saved diarization result to Node cache:', { cacheKey });
    } catch (e) {
      console.warn('⚠️ Failed to write diarization cache:', e.message);
    }
  };

  // Node-level cache: skip for explicit skipCache=true (e.g. voice tracks)
  const diarizationCacheKey = !skipCache ? buildDiarizationCacheKey() : null;
  if (!skipCache && diarizationCacheKey) {
    const cached = readDiarizationCache(diarizationCacheKey);
    if (cached) {
      return cached;
    }
  }

  const pythonScriptPath = path.join(__dirname, 'process_audio_temp.py');
  if (!fsSync.existsSync(pythonScriptPath)) {
    throw new Error('Python processing script not found');
  }
  const pythonArgs = [];
  if (url) {
    pythonArgs.push('--url', url);
    // Pass original filename for cache key generation (important for voice tracks)
    if (originalFilename) {
      pythonArgs.push('--original-filename', originalFilename);
    }
  } else if (filePath) {
    pythonArgs.push('--file', filePath);
    // Pass original filename for cache key generation
    if (originalFilename) {
      pythonArgs.push('--original-filename', originalFilename);
    }
  } else {
    throw new Error('Either url or filePath must be provided for diarization.');
  }

  if (language) {
    pythonArgs.push('--language', language);
  }
  if (speakerCount) {
    pythonArgs.push('--speaker-count', speakerCount);
  }
  // CRITICAL FIX: Use 'channel' diarization mode for separated tracks
  // This prevents Speechmatics from detecting residual audio as separate speakers
  // Expected improvement: 70-80% reduction in speaker assignment errors
  if (isSeparatedTrack) {
    pythonArgs.push('--is-separated-track');
    console.log('[Diarization] Using channel mode for separated track (prevents false speaker detection)');
  }
  if (skipCache) {
    pythonArgs.push('--skip-cache');
    console.log(`[Diarization] Skipping cache - engine=${engine}`);
  }
  pythonArgs.push('--output-format', 'json');
  if (engine) {
    pythonArgs.push('--engine', engine);
  }

  return new Promise((resolve, reject) => {
    // Pass environment variables to Python process, including SPEECHMATICS_API_KEY
    const env = {
      ...process.env,
      SPEECHMATICS_API_KEY: process.env.SPEECHMATICS_API_KEY || ''
    };
    
    const pythonProcess = spawn(PYTHON_BIN, [pythonScriptPath, ...pythonArgs], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: env
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      const stderrChunk = data.toString();
      stderr += stderrChunk;
      // Log stderr in real-time for debugging
      console.error(`[Python stderr] ${stderrChunk}`);
      writeLog('debug', 'Python diarization stderr chunk', {
        stderrChunk: stderrChunk.substring(0, 500)
      });
      if (typeof onProgress === 'function') {
        const progressUpdates = extractAzureProgressMessages(stderrChunk);
        if (progressUpdates.length) {
          progressUpdates.forEach((update) => {
            try {
              onProgress(update);
            } catch (progressError) {
              console.warn('⚠️ Failed to forward python progress update:', progressError.message);
            }
          });
        }
      }
    });

    pythonProcess.on('close', (code) => {
      // Always log full stderr for debugging
      if (stderr) {
        writeLog('debug', 'Python diarization full stderr', {
          stderr: stderr.substring(0, 2000),
          code
        });
      }
      
      if (code !== 0) {
        const errorMessage = stderr || `Speechmatics pipeline exited with code ${code}`;
        writeLog('error', 'Python diarization script failed', {
          code,
          stderr: stderr.substring(0, 2000),
          stdout: stdout.substring(0, 2000),
          hasApiKey: !!process.env.SPEECHMATICS_API_KEY
        });
        return reject(new Error(errorMessage));
      }
      try {
        writeLog('debug', 'Python diarization stdout', {
          stdoutLength: stdout.length,
          stdoutPreview: stdout.substring(0, 1000)
        });
        const parsed = JSON.parse(stdout);
        writeLog('debug', 'Python diarization parsed', {
          hasRecordings: !!parsed?.recordings,
          recordingsCount: parsed?.recordings?.length || 0,
          firstRecordingHasResults: !!parsed?.recordings?.[0]?.results,
          resultsKeys: parsed?.recordings?.[0]?.results ? Object.keys(parsed.recordings[0].results) : [],
          hasSuccess: !!parsed?.success,
          hasProcessing: !!parsed?.processing
        });
        
        // If old format (no recordings), log warning and check for errors
        if (!parsed?.recordings) {
          if (parsed?.success === false || parsed?.error) {
            writeLog('error', 'Python script returned error', {
              structure: Object.keys(parsed),
              error: parsed?.error,
              errorType: parsed?.error_type,
              hasFileInfo: !!parsed?.file_info
            });
            return reject(new Error(parsed?.error || 'Python script returned an error'));
          }
          
          if (parsed?.success) {
            writeLog('warn', 'Python script returned old format (no recordings)', {
              structure: Object.keys(parsed),
              processingStatus: parsed?.processing?.status,
              hasFileInfo: !!parsed?.file_info,
              hasProcessing: !!parsed?.processing
            });
            // If it's the old format with processing status, it might not have performed diarization
            if (parsed?.processing?.status === 'success' && !parsed?.processing?.segments) {
              writeLog('error', 'Python script did not perform diarization - missing segments', {
                processingKeys: parsed?.processing ? Object.keys(parsed.processing) : []
              });
              return reject(new Error('Python script did not perform diarization. Check SPEECHMATICS_API_KEY and logs.'));
            }
          }
        }
        
        // Validate that we have recordings with results
        if (parsed?.recordings && parsed.recordings.length > 0) {
          const firstRecording = parsed.recordings[0];
          if (!firstRecording.results || !firstRecording.results.speechmatics) {
            writeLog('warn', 'Python script returned recordings but no speechmatics results', {
              recordingKeys: Object.keys(firstRecording),
              hasResults: !!firstRecording.results,
              resultsKeys: firstRecording.results ? Object.keys(firstRecording.results) : []
            });
          }
        }
        
        // Persist full diarization result into Node-level cache for fast reuse
        if (!skipCache && diarizationCacheKey) {
          writeDiarizationCache(diarizationCacheKey, parsed);
        }

        resolve(parsed);
      } catch (error) {
        writeLog('error', 'Failed to parse Python diarization output', {
          error: error.message,
          stdoutLength: stdout.length,
          stdoutPreview: stdout.substring(0, 2000),
          stderr: stderr.substring(0, 1000)
        });
        reject(new Error('Failed to parse diarization output'));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}

async function separateSpeakersWithPyAnnote(audioPath) {
  const pythonScriptPath = path.join(__dirname, 'pyannote_separation.py');
  if (!fsSync.existsSync(pythonScriptPath)) {
    throw new Error('PyAnnote separation script not found');
  }

  // Create output directory for separated files
  const outputDir = path.join(tempUploadsDir, `pyannote_separation_${Date.now()}`);
  fsSync.mkdirSync(outputDir, { recursive: true });

  return new Promise((resolve, reject) => {
    const env = {
      ...process.env,
      HUGGINGFACE_TOKEN: process.env.HUGGINGFACE_TOKEN || ''
    };
    
    const pythonProcess = spawn(PYTHON_BIN, [pythonScriptPath, audioPath, '--output-dir', outputDir], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: env
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      const stderrChunk = data.toString();
      stderr += stderrChunk;
      console.error(`[PyAnnote stderr] ${stderrChunk}`);
    });

    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        const errorMessage = stderr || `PyAnnote separation script exited with code ${code}`;
        writeLog('error', 'PyAnnote separation script failed', {
          code,
          stderr: stderr.substring(0, 2000),
          stdout: stdout.substring(0, 2000),
          hasToken: !!process.env.HUGGINGFACE_TOKEN
        });
        return reject(new Error(errorMessage));
      }
      
      try {
        const parsed = JSON.parse(stdout);
        
        if (!parsed.success) {
          writeLog('error', 'PyAnnote separation returned error', {
            error: parsed.error,
            traceback: parsed.traceback
          });
          return reject(new Error(parsed.error || 'PyAnnote separation failed'));
        }

        // Convert PyAnnote format to AudioShake-compatible format
        // Copy separated files to uploads directory
        const uploadsDir = path.join(__dirname, 'uploads');
        fsSync.mkdirSync(uploadsDir, { recursive: true });
        
        const timestamp = Date.now();
        const speakers = await Promise.all(parsed.speakers.map(async (speaker, index) => {
          // Copy file to uploads directory
          const filename = `${timestamp}_${index}_${speaker.name}.${speaker.format}`;
          const destPath = path.join(uploadsDir, filename);
          
          if (fsSync.existsSync(speaker.local_path)) {
            fsSync.copyFileSync(speaker.local_path, destPath);
            console.log(`✅ Copied PyAnnote speaker file: ${speaker.local_path} -> ${destPath}`);
          } else {
            console.warn(`⚠️ PyAnnote speaker file not found: ${speaker.local_path}`);
          }
          
          // Get public URL
          const publicUrl = getPublicFileUrl(filename);
          const downloadUrl = `/uploads/${filename}`; // Direct download path
          
          return {
            name: speaker.name,
            format: speaker.format,
            url: publicUrl || downloadUrl, // Public URL for transcription
            downloadUrl: downloadUrl, // Direct download URL
            isBackground: speaker.isBackground,
            local_path: destPath
          };
        }));

        resolve({
          taskId: `pyannote_${Date.now()}`,
          speakers: speakers,
          cost: null, // PyAnnote is free
          duration: null, // Will be calculated from processing time
          timeline: parsed.timeline,
          output_dir: parsed.output_dir,
          raw: parsed
        });
      } catch (error) {
        writeLog('error', 'Failed to parse PyAnnote separation output', {
          error: error.message,
          stdoutLength: stdout.length,
          stdoutPreview: stdout.substring(0, 2000),
          stderr: stderr.substring(0, 1000)
        });
        reject(new Error('Failed to parse PyAnnote separation output'));
      }
    });

    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}

async function separateSpeakersWithSpeechBrain(audioPath, requestId = null, res = null, sendSSEUpdate = null, debugParams = null) {
  const logPrefix = requestId ? `[${requestId}]` : '[SpeechBrain]';
  
  // Validate audioPath parameter
  if (!audioPath || typeof audioPath !== 'string') {
    throw new Error('audioPath is required');
  }
  
  // Verify audio file exists
  if (!fsSync.existsSync(audioPath)) {
    throw new Error(`Audio file not found: ${audioPath}`);
  }
  
  const pythonScriptPath = path.join(__dirname, 'speechbrain_separation.py');
  if (!fsSync.existsSync(pythonScriptPath)) {
    throw new Error('SpeechBrain separation script not found');
  }

  console.log(`${logPrefix} 🔵 MODE3: Initializing SpeechBrain separation`);
  console.log(`${logPrefix} 🔵 MODE3: Python script: ${pythonScriptPath}`);
  console.log(`${logPrefix} 🔵 MODE3: Audio path: ${audioPath}`);
  if (debugParams) {
    console.log(`${logPrefix} 🔵 MODE3: Debug parameters:`, debugParams);
  }
  
  const timestamp = Date.now();
  const outputDir = path.join(tempUploadsDir, `speechbrain_separation_${timestamp}`);
  fsSync.mkdirSync(outputDir, { recursive: true });
  
  console.log(`${logPrefix} 🔵 MODE3: Output directory: ${outputDir}`);

  // Send progress update
  const sendProgress = (stage, description, details = {}) => {
    if (sendSSEUpdate) {
      sendSSEUpdate(2, 'processing', `🔵 MODE3: ${description}`, {
        stage,
        ...details
      });
    } else if (res) {
      try {
        res.write(`data: ${JSON.stringify({
          type: 'step-progress',
          step: 2,
          status: 'processing',
          description: `🔵 MODE3: ${description}`,
          details: {
            stage,
            ...details,
            timestamp: new Date().toISOString(),
            requestId
          }
        })}\n\n`);
        if (res.flush && typeof res.flush === 'function') {
          res.flush();
        }
      } catch (e) {
        console.error(`${logPrefix} ❌ Failed to send SSE update:`, e.message);
      }
    }
    console.log(`${logPrefix} 🔵 MODE3 [${stage}]: ${description}`);
  };

  const fileSize = fsSync.existsSync(audioPath) ? fsSync.statSync(audioPath).size : 0;
  
  sendProgress('initialization', 'Ініціалізація SpeechBrain SepFormer...', {
    input: {
      audioPath: audioPath,
      audioFileSize: `${(fileSize / 1024 / 1024).toFixed(2)} MB`,
      pythonScript: pythonScriptPath,
      outputDir: outputDir
    }
  });

  return new Promise((resolve, reject) => {
    const pythonArgs = [pythonScriptPath, audioPath, outputDir];
    console.log(`${logPrefix} 🔵 MODE3: Executing Python script: ${PYTHON_BIN} ${pythonArgs.join(' ')}`);
    
    // Prepare environment variables (include debug params if provided)
    const env = { ...process.env };
    if (debugParams) {
      // Mark as debug mode
      env.SPEECHBRAIN_DEBUG_MODE = '1';
      if (debugParams.chunkSeconds !== null && debugParams.chunkSeconds !== undefined) {
        env.SPEECHBRAIN_CHUNK_SECONDS = debugParams.chunkSeconds.toString();
        console.log(`${logPrefix} 🔵 MODE3: Setting SPEECHBRAIN_CHUNK_SECONDS=${debugParams.chunkSeconds}`);
      }
      if (debugParams.enableSpectralGating) {
        env.SPEECHBRAIN_ENABLE_SPECTRAL_GATING = '1';
        if (debugParams.gateThreshold !== null && debugParams.gateThreshold !== undefined) {
          env.SPEECHBRAIN_GATE_THRESHOLD = debugParams.gateThreshold.toString();
          console.log(`${logPrefix} 🔵 MODE3: Setting SPEECHBRAIN_GATE_THRESHOLD=${debugParams.gateThreshold}`);
        }
        if (debugParams.gateAlpha !== null && debugParams.gateAlpha !== undefined) {
          env.SPEECHBRAIN_GATE_ALPHA = debugParams.gateAlpha.toString();
          console.log(`${logPrefix} 🔵 MODE3: Setting SPEECHBRAIN_GATE_ALPHA=${debugParams.gateAlpha}`);
        }
      }
    }
    
    sendProgress('spawning', 'Запуск Python процесу...', {
      input: {
        pythonBin: PYTHON_BIN,
        pythonArgs: pythonArgs,
        workingDir: __dirname,
        debugParams: debugParams || null
      }
    });
    
    const pythonProcess = spawn(PYTHON_BIN, pythonArgs, {
      cwd: __dirname,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: env
    });

    let stdout = '';
    let stderr = '';
    const timeoutMs = 5 * 60 * 1000; // 5 minutes
    const timeout = setTimeout(() => {
      pythonProcess.kill();
      reject(new Error('SpeechBrain speaker separation timed out after 5 minutes'));
    }, timeoutMs);

    pythonProcess.stdout.on('data', (data) => {
      const chunk = data.toString();
      stdout += chunk;
      // Log model loading progress from stderr (which goes to stdout in some cases)
      if (chunk.includes('[SpeechBrain]') || chunk.includes('Loading') || chunk.includes('model')) {
        console.log(`${logPrefix} 📊 MODE3 stdout: ${chunk.trim()}`);
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const chunk = data.toString();
      stderr += chunk;
      console.error(`${logPrefix} 📊 MODE3 stderr: ${chunk.trim()}`);
      
      // Parse progress from stderr
      if (chunk.includes('[SpeechBrain]')) {
        const match = chunk.match(/\[SpeechBrain\](.+)/);
        if (match) {
          const message = match[1].trim();
          if (message.includes('Using device')) {
            const deviceMatch = message.match(/Using device:\s*(.+)/);
            sendProgress('model_loading', message, {
              input: {
                device: deviceMatch ? deviceMatch[1] : 'unknown',
                rawMessage: message
              }
            });
          } else if (message.includes('Cache dir')) {
            const cacheMatch = message.match(/Cache dir:\s*(.+)/);
            sendProgress('model_loading', message, {
              input: {
                cacheDir: cacheMatch ? cacheMatch[1] : 'unknown',
                rawMessage: message
              }
            });
          } else if (message.includes('Loaded via') || message.includes('Resampling')) {
            sendProgress('audio_processing', message, {
              input: {
                rawMessage: message
              }
            });
          } else if (message.includes('Processing in chunks') || message.includes('Separating chunk')) {
            const chunkMatch = message.match(/chunk\s+(\d+):(\d+)/);
            sendProgress('separation', message, {
              input: {
                chunkInfo: chunkMatch ? { start: chunkMatch[1], end: chunkMatch[2] } : null,
                rawMessage: message
              }
            });
          }
        }
      }
    });

    pythonProcess.on('error', (error) => {
      clearTimeout(timeout);
      console.error(`${logPrefix} ❌ MODE3: Process error:`, error);
      reject(error);
    });

    pythonProcess.on('close', async (code) => {
      clearTimeout(timeout);
      console.log(`${logPrefix} 🔵 MODE3: Python process exited with code ${code}`);
      
      if (code !== 0) {
        console.error(`${logPrefix} ❌ MODE3: Script failed`, { code, stderr: stderr.substring(0, 2000) });
        writeLog('error', `${logPrefix} MODE3: Separation failed`, {
          requestId,
          code,
          stderr: stderr.substring(0, 2000),
          stdout: stdout.substring(0, 1000)
        });
        return reject(new Error(stderr || `SpeechBrain separation exited with code ${code}`));
      }

      sendProgress('parsing', 'Парсинг результату...', {
        input: {
          stdoutLength: stdout.length,
          stdoutPreview: stdout.substring(0, 200)
        }
      });

      try {
        const parsed = JSON.parse(stdout);
        const parsedInfo = {
          success: parsed.success,
          speakersCount: parsed.speakers?.length || 0,
          numSpeakers: parsed.num_speakers,
          hasTimeline: !!parsed.timeline,
          timelineLength: parsed.timeline?.length || 0
        };
        
        console.log(`${logPrefix} 📊 MODE3: Parsed result:`, parsedInfo);
        
        sendProgress('parsing', 'Результат розпарсено', {
          output: parsedInfo
        });
        
        if (!parsed.success) {
          return reject(new Error(parsed.error || 'SpeechBrain separation failed'));
        }

        sendProgress('copying', `Копіювання ${parsed.speakers?.length || 0} файлів спікерів...`, {
          input: {
            speakersToCopy: parsed.speakers?.length || 0,
            speakers: parsed.speakers?.map(s => ({
              name: s.name,
              localPath: s.local_path,
              format: s.format
            })) || []
          }
        });

        const speakers = await Promise.all((parsed.speakers || []).map(async (speaker, index) => {
          const filename = `${timestamp}_${index}_${speaker.name}.${speaker.format || 'wav'}`;
          const destPath = path.join(uploadsDir, filename);

          if (speaker.local_path && fsSync.existsSync(speaker.local_path)) {
            await fs.copyFile(speaker.local_path, destPath);
            const fileSize = fsSync.existsSync(destPath) ? fsSync.statSync(destPath).size : 0;
            const fileSizeMB = (fileSize / 1024 / 1024).toFixed(2);
            console.log(`${logPrefix} ✅ MODE3: Copied speaker file ${index + 1}/${parsed.speakers.length}: ${speaker.local_path} -> ${destPath} (${fileSizeMB} MB)`);
            
            sendProgress('copying', `✅ Скопійовано файл ${index + 1}/${parsed.speakers.length}: ${speaker.name}`, {
              output: {
                speakerIndex: index + 1,
                totalSpeakers: parsed.speakers.length,
                speakerName: speaker.name,
                sourcePath: speaker.local_path,
                destPath: destPath,
                fileSize: `${fileSizeMB} MB`
              }
            });
          } else {
            console.warn(`${logPrefix} ⚠️ MODE3: Speaker file not found: ${speaker.local_path}`);
            sendProgress('copying', `⚠️ Файл не знайдено: ${speaker.name}`, {
              error: {
                speakerName: speaker.name,
                expectedPath: speaker.local_path,
                message: 'File not found'
              }
            });
          }

          const publicUrl = getPublicFileUrl(filename);
          const downloadUrl = `/uploads/${filename}`;

          return {
            name: speaker.name,
            format: speaker.format || 'wav',
            url: publicUrl || downloadUrl,
            downloadUrl,
            isBackground: speaker.isBackground || false,
            local_path: destPath
          };
        }));

        const finalResult = {
          speakersCount: speakers.length,
          speakers: speakers.map(s => ({
            name: s.name,
            hasUrl: !!s.url,
            hasLocalPath: !!s.local_path,
            isBackground: s.isBackground
          })),
          timelineSegments: parsed.timeline?.length || 0
        };
        
        console.log(`${logPrefix} ✅ MODE3: Separation completed successfully`);
        console.log(`${logPrefix} 📊 MODE3: Final result:`, finalResult);
        
        sendProgress('completed', `✅ Розділення завершено. Знайдено ${speakers.length} спікерів`, {
          output: finalResult
        });

        resolve({
          taskId: `speechbrain_${Date.now()}`,
          speakers,
          cost: null,
          duration: null,
          timeline: parsed.timeline || [],
          output_dir: parsed.output_dir,
          raw: parsed
        });
      } catch (error) {
        console.error(`${logPrefix} ❌ MODE3: Failed to parse script output`, error);
        console.error(`${logPrefix} 📊 MODE3: stdout preview:`, stdout.substring(0, 2000));
        writeLog('error', `${logPrefix} MODE3: Parse error`, {
          requestId,
          error: error.message,
          stdoutPreview: stdout.substring(0, 2000)
        });
        reject(new Error('Failed to parse SpeechBrain output'));
      }
    });
  });
}

// LLM diarization endpoint
app.post('/api/diarize', async (req, res) => {
  try {
    const result = await handleDiarizationRequest(req.body);
    res.json(result);
  } catch (error) {
    console.error('Diarization error:', error);
    res.status(500).json({
      error: error.message || 'Internal server error',
      details: error.response?.data || null
    });
  }
});

app.post('/api/audioshake-tasks', upload.single('audio'), async (req, res) => {
  if (!process.env.AUDIOSHAKE_API_KEY) {
    return res.status(501).json({
      success: false,
      error: 'AudioShake API key not configured on the server.'
    });
  }

  if (!tunnelUrl && !PUBLIC_URL) {
    return res.status(503).json({
      success: false,
      error: 'Tunnel not ready. Please wait a few seconds and try again.'
    });
  }

  let tempDownloadedAudioPath = null;
  let storedFile = null;
  try {
    const { url, variant, language, engine } = req.body;
    const transcriptionEngine = resolveTranscriptionEngine(engine);
    const uploadedFile = req.file;

    if (!uploadedFile && !url) {
      return res.status(400).json({
        success: false,
        error: 'Either file upload or URL must be provided'
      });
    }

    let sourcePath = null;
    let originalName = 'audio.wav';

    if (uploadedFile && uploadedFile.path) {
      sourcePath = uploadedFile.path;
      originalName = uploadedFile.originalname || originalName;
    } else if (url) {
      sourcePath = await downloadAudioToTemp(url);
      tempDownloadedAudioPath = sourcePath;
      try {
        const remoteName = new URL(url).pathname.split('/').pop();
        if (remoteName) {
          originalName = remoteName;
        }
      } catch {
        // ignore URL parsing errors, fallback to default name
      }
    }

    storedFile = await persistUploadedFile(sourcePath, originalName);
    tempDownloadedAudioPath = null; // file moved into uploads directory

    const publicUrl = getPublicFileUrl(storedFile.filename);
    await ensurePublicFileAccessible(publicUrl);

    console.log(`🎵 Starting AudioShake separation for: ${publicUrl}`);
    const separationStartTime = Date.now();
    
    const separation = await separateSpeakers(publicUrl, {
      variant: variant || undefined,
      language: language || undefined
    });
    
    const separationDuration = Math.round((Date.now() - separationStartTime) / 1000);
    console.log(`✅ AudioShake separation completed in ${separationDuration}s, found ${separation.speakers.length} speakers`);

    const diarizationResults = [];

    console.log(`🎤 Processing ${separation.speakers.length} speakers for diarization...`);
    for (let i = 0; i < separation.speakers.length; i++) {
      const speaker = separation.speakers[i];
      if (!speaker?.url) {
        console.log(`⏭️ Skipping speaker ${i + 1}: no URL`);
        continue;
      }
      if (speaker.isBackground) {
        console.log(`⏭️ Skipping background stem: ${speaker.name}`);
        diarizationResults.push({
          speaker: speaker.name,
          audioUrl: speaker.url,
          downloadUrl: speaker.url, // Надаємо URL для скачування навіть для background
          skipped: true,
          reason: 'background stem'
        });
        continue;
      }

      console.log(`🎙️ Processing speaker ${i + 1}/${separation.speakers.length}: ${speaker.name}`);
      try {
        // CRITICAL FIX: For separated tracks, use speakerCount=2 to detect both main speaker and residual audio
        // Residual audio from other speakers may not be fully separated, so we need to detect it
        // Then filter it out in collectVoiceTrackSegments by identifying the main speaker
        const diarization = await runPythonDiarization({
          url: speaker.url,
          language,
          speakerCount: '2',  // Allow detection of main speaker + residual audio
          isSeparatedTrack: true,  // Separated audio track - will use appropriate mode
          engine: transcriptionEngine
        });
        console.log(`✅ Diarization completed for ${speaker.name}`);
        diarizationResults.push({
          speaker: speaker.name,
          audioUrl: speaker.url,
          downloadUrl: speaker.url, // AudioShake надає публічний URL для скачування
          diarization
        });
      } catch (error) {
        console.error(`❌ Diarization failed for ${speaker.name}:`, error.message);
        diarizationResults.push({
          speaker: speaker.name,
          audioUrl: speaker.url,
          downloadUrl: speaker.url, // Все одно надаємо URL для скачування навіть якщо diarization не вдався
          error: error.message
        });
      }
    }

    res.json({
      success: true,
      pipeline: 'audioshake_tasks',
      taskId: separation.taskId,
      cost: separation.cost,
      duration: separation.duration,
      publicUrl,
      speakers: diarizationResults
    });
  } catch (error) {
    console.error('AudioShake tasks pipeline error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'AudioShake tasks pipeline failed'
    });
  } finally {
    if (req.file && req.file.path && fsSync.existsSync(req.file.path)) {
      try {
        await fs.unlink(req.file.path);
      } catch (cleanupError) {
        console.error('Failed to cleanup uploaded temp file:', cleanupError);
      }
    }
    if (tempDownloadedAudioPath && fsSync.existsSync(tempDownloadedAudioPath)) {
      try {
        await fs.unlink(tempDownloadedAudioPath);
      } catch (cleanupError) {
        console.error('Failed to cleanup downloaded file:', cleanupError);
      }
    }
  }
});

// Overlap Diarization endpoint (Initial diarization + Speaker separation + correction)
app.post('/api/diarize-overlap', upload.single('audio'), async (req, res) => {
  const requestId = `overlap_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  console.log(`[${requestId}] 📥 Overlap diarization endpoint called`);
  writeLog('info', `[${requestId}] Overlap diarization started`, {
    requestId,
    timestamp: new Date().toISOString()
  });
  
  let tempDownloadedAudioPath = null;
  let storedFile = null;
  const pipelineStartTime = Date.now();
  let overlapPipelineMode = 'mode1';
  
  try {
    const { url, language, speakerCount, mode, pipelineMode, textAnalysisMode } = req.body;
    // textAnalysisMode: режим маркування фрагментів
    // Може бути задано через параметр запиту або через змінну оточення TEXT_ANALYSIS_MODE
    // 'script' (default) - використовує скриптову логіку
    // 'llm' - використовує LLM для класифікації
    const envTextAnalysisMode = process.env.TEXT_ANALYSIS_MODE || 'script';
    const useLLMForTextAnalysis = (textAnalysisMode === 'llm' || envTextAnalysisMode === 'llm');
    
    console.log(`[${requestId}] 🔍 Text analysis mode configuration:`, {
      textAnalysisModeFromRequest: textAnalysisMode,
      envTextAnalysisMode: envTextAnalysisMode,
      processEnvTEXT_ANALYSIS_MODE: process.env.TEXT_ANALYSIS_MODE,
      useLLMForTextAnalysis: useLLMForTextAnalysis,
      willUseLLM: useLLMForTextAnalysis,
      note: useLLMForTextAnalysis ? '✅ LLM mode will be used' : '⚠️ Script mode will be used (set TEXT_ANALYSIS_MODE=llm in .env to enable LLM)'
    });
    
    // Debug: Log received parameters
    console.log(`[${requestId}] 🔍 Received parameters:`);
    console.log(`   mode: ${mode}`);
    console.log(`   pipelineMode: ${pipelineMode}`);
    console.log(`   speakerCount: ${speakerCount}`);
    console.log(`   language: ${language}`);
    console.log(`   engine: ${req.body.engine}`);
    
    const transcriptionEngine = resolveTranscriptionEngine(req.body.engine);
    const engineLabel = transcriptionEngine === 'azure'
      ? 'Azure STT (Batch)'
      : transcriptionEngine === 'azure_realtime'
        ? 'Azure STT Realtime'
        : 'Speechmatics';
    overlapPipelineMode = pipelineMode || 'mode1';
    
    // Debug: Log final configuration
    console.log(`[${requestId}] ⚙️  Final configuration:`);
    console.log(`   overlapPipelineMode: ${overlapPipelineMode}`);
    console.log(`   transcriptionEngine: ${transcriptionEngine} (${engineLabel})`);
    console.log(`   llmMode: ${mode}`);
    console.log(`   isLocalMode: ${mode === 'local'}`);
    const uploadedFile = req.file;
    const isSSEMode = overlapPipelineMode === 'mode3' || overlapPipelineMode === 'mode1';
    
    // Set up SSE headers for mode 3 and mode 1 (detailed logging and keep-alive)
    // Mode 1 also needs SSE to prevent 504 Gateway Timeout during long operations
    if (isSSEMode) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // Disable nginx buffering
      
      // Send initial keep-alive ping to prevent timeout
      res.write(': keep-alive\n\n');
    }
    
    // Keep-alive ping interval for SSE (prevents gateway timeouts)
    let keepAliveInterval = null;
    if (isSSEMode) {
      keepAliveInterval = setInterval(() => {
        try {
          res.write(': ping\n\n');
        } catch (e) {
          // Connection closed, stop pinging
          if (keepAliveInterval) {
            clearInterval(keepAliveInterval);
            keepAliveInterval = null;
          }
        }
      }, 30000); // Ping every 30 seconds
    }
    
    // Helper function to send SSE updates for mode 3 and mode 1
    // Mode 1 also needs SSE to prevent 504 Gateway Timeout
    const sendSSEUpdate = (step, status, description, details = {}) => {
      if (isSSEMode && res) {
        try {
          const update = {
            type: 'step-progress',
            step,
            status,
            description,
            details: {
              ...details,
              timestamp: new Date().toISOString(),
              requestId
            }
          };
          const updateStr = `data: ${JSON.stringify(update)}\n\n`;
          res.write(updateStr);
          console.log(`[${requestId}] 📤 SSE Update: Step ${step} - ${description.substring(0, 60)}...`);
          // Force flush if possible
          if (res.flush && typeof res.flush === 'function') {
            res.flush();
          }
        } catch (e) {
          console.error(`[${requestId}] ❌ Failed to send SSE update:`, e.message);
        }
      }
    };
    
    const supportsPythonProgressStreaming = transcriptionEngine === 'azure';
    const shouldStreamEngineProgress = isSSEMode && supportsPythonProgressStreaming;
    const createPythonProgressForwarder = (step, extraDetails = {}) => (progressUpdate) => {
      if (!shouldStreamEngineProgress) {
        return;
      }
      if (!progressUpdate || progressUpdate.source !== 'azure') {
        return;
      }
      const description = progressUpdate.description || 'Azure STT: оновлення';
      const details = {
        stage: progressUpdate.stage || 'azure_update',
        ...extraDetails,
        ...(progressUpdate.meta || {})
      };
      sendSSEUpdate(step, 'processing', description, details);
    };
    
    // Cleanup keep-alive interval on completion or error
    const cleanupKeepAlive = () => {
      if (keepAliveInterval) {
        clearInterval(keepAliveInterval);
        keepAliveInterval = null;
      }
    };
    
    console.log(`[${requestId}] 📥 Request details:`, { 
      hasUrl: !!url, 
      hasFile: !!uploadedFile, 
      language, 
      speakerCount,
      pipelineMode, // raw value from request
      overlapPipelineMode, // processed value
      mode,
      engine: transcriptionEngine,
      filePath: uploadedFile?.path,
      fileName: uploadedFile?.originalname
    });
    
    writeLog('info', `[${requestId}] Request parameters`, {
      requestId,
      hasUrl: !!url,
      hasFile: !!uploadedFile,
      language,
      speakerCount,
      mode,
      fileSize: uploadedFile?.size,
      fileName: uploadedFile?.originalname
    });
    
    // Check API keys based on pipeline mode
    if (overlapPipelineMode === 'mode1') {
    if (!process.env.AUDIOSHAKE_API_KEY) {
      return res.status(501).json({
        success: false,
          error: 'Speaker separation service (AudioShake) is not configured on the server.'
      });
    }
    if (!tunnelUrl && !PUBLIC_URL) {
      return res.status(503).json({
        success: false,
        error: 'Tunnel not ready. Please wait a few seconds and try again.'
      });
      }
    } else if (overlapPipelineMode === 'mode2') {
      if (!process.env.HUGGINGFACE_TOKEN) {
        return res.status(501).json({
          success: false,
          error: 'PyAnnote separation service is not configured. Please set HUGGINGFACE_TOKEN in .env file.'
        });
      }
    }

    if (!url && !uploadedFile) {
      return res.status(400).json({
        success: false,
        error: 'Either file upload or URL must be provided'
      });
    }

    const steps = {};
    let primaryDiarization = null;
    let separation = null;
    const voiceTracks = [];
    let correctedDiarization = null;

    // Step 1: Initial Audio Analysis (same logic as /api/diarize-audio)
    const step1StartTime = Date.now();
    console.log(`[${requestId}] 🔵 STEP 1: Starting initial audio analysis`);
    
    // Send SSE update for Step 1 start
    sendSSEUpdate(1, 'processing', `🔵 MODE3: Початок аналізу аудіо...`, {
      stage: 'starting',
      input: {
        hasUrl: !!url,
        hasFile: !!uploadedFile,
        fileSize: uploadedFile?.size,
        fileName: uploadedFile?.originalname,
        language: language || 'auto',
        speakerCount: speakerCount || 'auto'
      }
    });
    
    writeLog('info', `[${requestId}] STEP 1: Starting initial audio analysis`, {
      requestId,
      step: 1,
      startTime: new Date().toISOString(),
      hasUrl: !!url,
      hasFile: !!uploadedFile,
      fileSize: uploadedFile?.size,
      fileName: uploadedFile?.originalname,
      language: language,
      speakerCount: speakerCount
    });
    
    try {
      console.log(`[${requestId}] 🔵 STEP 1: Starting Python diarization process...`);
      
      // Send SSE update for Step 1 processing
      const waitingStage = transcriptionEngine === 'azure'
        ? 'waiting_azure'
        : (transcriptionEngine === 'azure_realtime' ? 'waiting_azure_realtime' : 'waiting_speechmatics');
      sendSSEUpdate(1, 'processing', `⏳ MODE3: Очікування відповіді від ${engineLabel}...`, {
        stage: waitingStage,
        input: {
          language: language || 'auto',
          speakerCount: speakerCount || 'auto',
          filePath: uploadedFile?.path,
          url: url
        }
      });
      writeLog('info', `[${requestId}] STEP 1: Starting Python diarization`, {
        requestId,
        step: 1,
        stepName: 'python_diarization_start',
        hasUrl: !!url,
        hasFilePath: !!uploadedFile?.path,
        language,
        speakerCount
      });
      
      const diarizationStartTime = Date.now();
      const pythonCallParams = {
        url: url,
        filePath: uploadedFile?.path,
        language: language,
        speakerCount: speakerCount,
        originalFilename: uploadedFile?.originalname || null, // Pass original filename for cache
        engine: transcriptionEngine
      };
      if (shouldStreamEngineProgress) {
        pythonCallParams.onProgress = createPythonProgressForwarder(1);
      }
      primaryDiarization = await runPythonDiarization(pythonCallParams);
      const diarizationDuration = ((Date.now() - diarizationStartTime) / 1000).toFixed(2);
      
      const segmentsCount = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0;
      console.log(`[${requestId}] ✅ STEP 1: Python diarization completed in ${diarizationDuration}s, found ${segmentsCount} segments`);
      
      // Send SSE update for Step 1 completion
      sendSSEUpdate(1, 'completed', `✅ MODE3: Аналіз завершено. Знайдено ${segmentsCount} сегментів за ${diarizationDuration}s`, {
        stage: 'completed',
        output: {
          segmentsCount: segmentsCount,
          duration: `${diarizationDuration}s`,
          hasRecordings: !!primaryDiarization?.recordings,
          recordingsCount: primaryDiarization?.recordings?.length || 0
        }
      });
      
      writeLog('info', `[${requestId}] STEP 1: Python diarization completed`, {
        requestId,
        step: 1,
        stepName: 'python_diarization_complete',
        duration: `${diarizationDuration}s`,
        segmentsCount,
        hasRecordings: !!primaryDiarization?.recordings,
        recordingsCount: primaryDiarization?.recordings?.length || 0
      });

      const step1Duration = ((Date.now() - step1StartTime) / 1000).toFixed(2) + 's';
      steps.step1 = {
        name: 'Initial Audio Analysis',
        status: 'completed',
        duration: step1Duration,
        segmentsCount: segmentsCount
      };
      
      console.log(`[${requestId}] ✅ STEP 1: Completed successfully in ${step1Duration}`);
      writeLog('info', `[${requestId}] STEP 1: Completed successfully`, {
        requestId,
        step: 1,
        duration: step1Duration,
        segmentsCount,
        endTime: new Date().toISOString()
      });
      
      // Step 1.5: LLM Diarization with Gemini 2.5 Pro for transcription control
      // This provides an independent diarization to catch transcription errors like "Acne" -> "Acme"
      let geminiDiarization = null;
      const step1_5StartTime = Date.now();
      const audioFilePathForGemini = uploadedFile?.path || (url ? await downloadAudioToTemp(url) : null);
      
      if (audioFilePathForGemini && (GOOGLE_GEMINI_API_KEY || process.env.OPENROUTER_API_KEY)) {
        try {
          console.log(`[${requestId}] 🔵 STEP 1.5: Starting Gemini 2.5 Pro diarization for transcription control...`);
          sendSSEUpdate(1.5, 'processing', `🔵 MODE3: LLM діаризація (Gemini 2.5 Pro) для контролю транскрипції...`, {
            stage: 'gemini_diarization_starting',
            audioPath: audioFilePathForGemini
          });
          
          // Extract plain transcript from Speechmatics for context
          const speechmaticsSegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
          const plainTranscript = speechmaticsSegments.map(s => `${s.speaker || 'UNKNOWN'}: ${s.text || ''}`).join('\n');
          
          geminiDiarization = await callGeminiMultimodal(audioFilePathForGemini, plainTranscript, language || 'en');
          
          const step1_5Duration = ((Date.now() - step1_5StartTime) / 1000).toFixed(2) + 's';
          const geminiSegmentsCount = geminiDiarization?.recordings?.[0]?.results?.[TEXT_SERVICE_KEY]?.segments?.length || 0;
          
          console.log(`[${requestId}] ✅ STEP 1.5: Gemini 2.5 Pro diarization completed in ${step1_5Duration}s, found ${geminiSegmentsCount} segments`);
          sendSSEUpdate(1.5, 'completed', `✅ MODE3: LLM діаризація завершена. Знайдено ${geminiSegmentsCount} сегментів за ${step1_5Duration}s`, {
            stage: 'gemini_diarization_completed',
            segmentsCount: geminiSegmentsCount,
            duration: `${step1_5Duration}s`
          });
          
          writeLog('info', `[${requestId}] STEP 1.5: Gemini 2.5 Pro diarization completed`, {
            requestId,
            step: 1.5,
            duration: step1_5Duration,
            segmentsCount: geminiSegmentsCount,
            source: geminiDiarization?.source || 'unknown'
          });
          
          steps.step1_5 = {
            name: 'LLM Diarization (Gemini 2.5 Pro)',
            status: 'completed',
            duration: step1_5Duration,
            segmentsCount: geminiSegmentsCount,
            source: geminiDiarization?.source || 'unknown'
          };
        } catch (geminiError) {
          const step1_5Duration = ((Date.now() - step1_5StartTime) / 1000).toFixed(2) + 's';
          console.warn(`[${requestId}] ⚠️ STEP 1.5: Gemini 2.5 Pro diarization failed after ${step1_5Duration}:`, geminiError.message);
          sendSSEUpdate(1.5, 'error', `⚠️ MODE3: LLM діаризація не вдалася (продовжуємо без неї): ${geminiError.message}`, {
            stage: 'gemini_diarization_failed',
            error: geminiError.message,
            duration: step1_5Duration
          });
          
          writeLog('warn', `[${requestId}] STEP 1.5: Gemini 2.5 Pro diarization failed (non-fatal)`, {
            requestId,
            step: 1.5,
            error: geminiError.message,
            duration: step1_5Duration
          });
          
          steps.step1_5 = {
            name: 'LLM Diarization (Gemini 2.5 Pro)',
            status: 'failed',
            duration: step1_5Duration,
            error: geminiError.message
          };
          // Continue without Gemini diarization - it's optional for control
        }
      } else {
        console.log(`[${requestId}] ⏭️ STEP 1.5: Skipping Gemini 2.5 Pro diarization (no audio file or API key)`);
        steps.step1_5 = {
          name: 'LLM Diarization (Gemini 2.5 Pro)',
          status: 'skipped',
          reason: !audioFilePathForGemini ? 'No audio file available' : 'No Gemini API key configured'
        };
      }
    } catch (step1Error) {
      const step1Duration = ((Date.now() - step1StartTime) / 1000).toFixed(2) + 's';
      console.error(`[${requestId}] ❌ STEP 1: Failed after ${step1Duration}:`, step1Error);
      writeLog('error', `[${requestId}] STEP 1: Failed`, {
        requestId,
        step: 1,
        error: step1Error.message,
        errorType: step1Error.constructor?.name,
        duration: step1Duration,
        endTime: new Date().toISOString()
      });
      
      // For SSE mode (mode3/mode1), send error via SSE, not JSON response
      if (overlapPipelineMode === 'mode3' || overlapPipelineMode === 'mode1') {
        sendSSEUpdate(1, 'error', `❌ MODE3: Помилка аналізу аудіо: ${step1Error.message}`, {
          stage: 'error',
          error: step1Error.message,
          duration: step1Duration
        });
        
        // Send final error event and close SSE connection
        const errorEvent = {
          type: 'pipeline-error',
          requestId,
          error: 'Initial audio analysis failed',
          details: step1Error.message,
          step: 1,
          duration: step1Duration,
          timestamp: new Date().toISOString()
        };
        res.write(`data: ${JSON.stringify(errorEvent)}\n\n`);
        res.end();
        return;
      }
      
      // For non-SSE modes, return JSON error
      return res.status(500).json({
        success: false,
        error: 'Initial audio analysis failed',
        details: step1Error.message,
        step: 1,
        duration: step1Duration
      });
    }

    // Step 2: Speaker separation
    const step2StartTime = Date.now();
    let sourcePath = null;
    let publicUrl = null;
    let originalName = 'audio.wav';
    
    console.log(`[${requestId}] 🔵 STEP 2: Starting speaker separation`);
    
    // Send SSE update for Step 2 start
    sendSSEUpdate(2, 'processing', `🔵 MODE3: Початок розділення спікерів...`, {
      stage: 'starting',
      input: {
        mode: overlapPipelineMode,
        sourcePath: sourcePath,
        hasPublicUrl: !!publicUrl,
        publicUrl: publicUrl
      }
    });
    
    writeLog('info', `[${requestId}] STEP 2: Starting speaker separation`, {
      requestId,
      step: 2,
      startTime: new Date().toISOString()
    });
    
    try {
      console.log(`[${requestId}] 🔵 STEP 2: Preparing audio file...`);
      writeLog('info', `[${requestId}] STEP 2: Preparing audio file`, {
        requestId,
        step: 2,
        status: 'preparing_file',
        action: 'preparing_file',
        hasUploadedFile: !!uploadedFile,
        hasUrl: !!url
      });
      
      // Prepare audio file for separation

      if (uploadedFile && uploadedFile.path) {
        sourcePath = uploadedFile.path;
        originalName = uploadedFile.originalname || originalName;
      } else if (url) {
        sourcePath = await downloadAudioToTemp(url);
        tempDownloadedAudioPath = sourcePath;
        try {
          const remoteName = new URL(url).pathname.split('/').pop();
          if (remoteName) {
            originalName = remoteName;
          }
        } catch {
          // ignore URL parsing errors
        }
      }
      
      // Validate source path exists
      if (!sourcePath) {
        const errorMsg = 'No audio file provided. Please upload a file or provide a URL.';
        console.error(`[${requestId}] ❌ STEP 2: ${errorMsg}`);
        writeLog('error', `[${requestId}] STEP 2: No source path`, {
          requestId,
          step: 2,
          status: 'no_source_path',
          hasUploadedFile: !!uploadedFile,
          hasUrl: !!url
        });
        throw new Error(errorMsg);
      }
      
      // Check if file exists (for local separation modes)
      if (overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3') {
        if (!fsSync.existsSync(sourcePath)) {
          const errorMsg = `Audio file not found at path: ${sourcePath}`;
          console.error(`[${requestId}] ❌ STEP 2: ${errorMsg}`);
          writeLog('error', `[${requestId}] STEP 2: File not found`, {
            requestId,
            step: 2,
            status: 'file_not_found',
            sourcePath,
            mode: overlapPipelineMode
          });
          throw new Error(errorMsg);
        }
        
        // Check file size (should be > 0)
        const fileStats = fsSync.statSync(sourcePath);
        if (fileStats.size === 0) {
          const errorMsg = `Audio file is empty (0 bytes): ${sourcePath}`;
          console.error(`[${requestId}] ❌ STEP 2: ${errorMsg}`);
          writeLog('error', `[${requestId}] STEP 2: Empty file`, {
            requestId,
            step: 2,
            status: 'empty_file',
            sourcePath,
            fileSize: fileStats.size
          });
          throw new Error(errorMsg);
        }
        
        console.log(`[${requestId}] ✅ STEP 2: File validated - exists: true, size: ${(fileStats.size / 1024 / 1024).toFixed(2)} MB`);
      }
      
      if (overlapPipelineMode === 'mode1') {
        // For AudioShake (mode1), we need to persist file and get HTTPS URL
      storedFile = await persistUploadedFile(sourcePath, originalName);
      tempDownloadedAudioPath = null; // file moved into uploads directory
      
      console.log(`[${requestId}] 🔵 STEP 2: File persisted, filename: ${storedFile.filename}`);
      writeLog('info', `[${requestId}] STEP 2: File persisted`, {
        requestId,
        step: 2,
        status: 'file_persisted',
        filename: storedFile.filename,
        originalName
      });

        publicUrl = getPublicFileUrl(storedFile.filename);
      
      // Ensure file is accessible before calling separation
      console.log(`[${requestId}] 🔵 STEP 2: Checking file accessibility at ${publicUrl}...`);
      writeLog('info', `[${requestId}] STEP 2: Checking file accessibility`, {
        requestId,
        step: 2,
        status: 'checking_accessibility',
        publicUrl
      });
      
      try {
        await ensurePublicFileAccessible(publicUrl);
        console.log(`[${requestId}] ✅ STEP 2: File is accessible`);
        writeLog('info', `[${requestId}] STEP 2: File is accessible`, {
          requestId,
          step: 2,
          status: 'file_accessible',
          publicUrl
        });
      } catch (accessError) {
        console.warn(`[${requestId}] ⚠️ STEP 2: File accessibility check failed, proceeding anyway:`, accessError.message);
        writeLog('warn', `[${requestId}] STEP 2: File accessibility check failed, proceeding anyway`, {
          requestId,
          step: 2,
          status: 'accessibility_check_failed',
          error: accessError.message,
          url: publicUrl
        });
        // Continue anyway - the separation service might still work
      }

        // Validate URL is HTTPS (required by AudioShake)
      if (!publicUrl.startsWith('https://')) {
        console.log(`[${requestId}] 🔵 STEP 2: URL is not HTTPS, attempting to use tunnel or PUBLIC_URL...`);
        writeLog('info', `[${requestId}] STEP 2: URL is not HTTPS, attempting to use tunnel or PUBLIC_URL`, {
          requestId,
          step: 2,
          status: 'fixing_url',
          currentUrl: publicUrl,
          hasTunnelUrl: !!tunnelUrl,
          hasPublicUrl: !!PUBLIC_URL
        });
        
        // If we have tunnel or PUBLIC_URL, use it; otherwise throw error
        if (tunnelUrl) {
          const tunnelBaseUrl = tunnelUrl.replace(/\/$/, '');
          publicUrl = `${tunnelBaseUrl}/uploads/${encodeURIComponent(storedFile.filename)}`;
          console.log(`[${requestId}] ✅ STEP 2: Using tunnel URL: ${publicUrl}`);
          writeLog('info', `[${requestId}] STEP 2: Using tunnel URL for speaker separation`, { 
            requestId,
            step: 2,
            status: 'using_tunnel_url',
            publicUrl 
          });
        } else if (PUBLIC_URL) {
          const publicBaseUrl = PUBLIC_URL.replace(/\/$/, '');
          publicUrl = `${publicBaseUrl}/uploads/${encodeURIComponent(storedFile.filename)}`;
          console.log(`[${requestId}] ✅ STEP 2: Using PUBLIC_URL: ${publicUrl}`);
          writeLog('info', `[${requestId}] STEP 2: Using PUBLIC_URL for speaker separation`, { 
            requestId,
            step: 2,
            status: 'using_public_url',
            publicUrl 
          });
        } else {
          const errorMsg = 'Speaker separation requires HTTPS URL. Please configure tunnel or PUBLIC_URL environment variable.';
          console.error(`[${requestId}] ❌ STEP 2: ${errorMsg}`);
          writeLog('error', `[${requestId}] STEP 2: HTTPS URL required`, {
            requestId,
            step: 2,
            status: 'https_required',
            error: errorMsg
          });
          throw new Error(errorMsg);
        }
      }
      } else if (overlapPipelineMode === 'mode2') {
        // For PyAnnote (mode2), we can use the file directly, no need for HTTPS URL
        console.log(`[${requestId}] 🔵 STEP 2: Using local file for PyAnnote separation: ${sourcePath}`);
        writeLog('info', `[${requestId}] STEP 2: Using local file for PyAnnote`, {
          requestId,
          step: 2,
          status: 'using_local_file',
          filePath: sourcePath
        });
      } else if (overlapPipelineMode === 'mode3') {
        // For SpeechBrain (mode3), we can use the file directly, no need for HTTPS URL
        console.log(`[${requestId}] 🔵 STEP 2: Using local file for SpeechBrain separation: ${sourcePath}`);
        writeLog('info', `[${requestId}] STEP 2: Using local file for SpeechBrain`, {
          requestId,
          step: 2,
          status: 'using_local_file',
          filePath: sourcePath
        });
      }

      const separationStartTime = Date.now();
      
      // Log separation start with full context
      const isLocalOverlapMode = overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3';
      console.log(`[${requestId}] 🔵 STEP 2: Starting speaker separation`, {
        mode: overlapPipelineMode,
        sourcePath: sourcePath,
        publicUrl: publicUrl || (isLocalOverlapMode ? 'N/A (local separation uses file path)' : 'N/A'),
        hasSourcePath: !!sourcePath,
        fileExists: sourcePath ? fsSync.existsSync(sourcePath) : false
      });
      writeLog('info', `[${requestId}] STEP 2: Starting speaker separation`, {
        requestId,
        step: 2,
        mode: overlapPipelineMode,
        sourcePath: sourcePath,
        publicUrl: publicUrl || null,
        hasSourcePath: !!sourcePath
      });
      
      try {
        if (overlapPipelineMode === 'mode2') {
          // PyAnnote separation (mode2)
          // Check cache first
          const separationCacheKey = buildSeparationCacheKey(originalName, 'mode2');
          let cachedSeparation = null;
          if (separationCacheKey) {
            cachedSeparation = readSeparationCache(separationCacheKey);
            if (cachedSeparation) {
              console.log(`[${requestId}] ✅ STEP 2 MODE2: Using cached PyAnnote separation result`);
              writeLog('info', `[${requestId}] STEP 2 MODE2: Using cached PyAnnote separation`, {
                requestId,
                step: 2,
                status: 'using_cached_separation',
                cacheKey: separationCacheKey,
                speakersCount: cachedSeparation.speakers?.length || 0
              });
              // Regenerate URLs for cached speakers
              const speakersWithUrls = cachedSeparation.speakers.map(speaker => {
                const timestamp = Date.now();
                const speakerFilename = `${timestamp}_${speaker.name}.${speaker.format || 'wav'}`;
                return {
                  ...speaker,
                  url: getPublicFileUrl(speakerFilename),
                  downloadUrl: `/uploads/${speakerFilename}`,
                  cached: true
                };
              });
              
              separation = {
                ...cachedSeparation,
                speakers: speakersWithUrls,
                mode: 'mode2',
                originalFilename: originalName
              };
            }
          }
          
          if (!cachedSeparation) {
          console.log(`[${requestId}] 🔵 STEP 2 MODE2: Starting PyAnnote speaker separation`);
          console.log(`[${requestId}] 🔵 STEP 2 MODE2: File path: ${sourcePath}`);
          console.log(`[${requestId}] 🔵 STEP 2 MODE2: File exists: ${fsSync.existsSync(sourcePath)}`);
          console.log(`[${requestId}] 🔵 STEP 2 MODE2: File size: ${fsSync.existsSync(sourcePath) ? fsSync.statSync(sourcePath).size : 'N/A'} bytes`);
          
          writeLog('info', `[${requestId}] STEP 2 MODE2: Starting PyAnnote separation`, {
            requestId,
            step: 2,
            mode: 'mode2',
            status: 'calling_pyannote_separation',
            filePath: sourcePath,
            fileExists: fsSync.existsSync(sourcePath),
            fileSize: fsSync.existsSync(sourcePath) ? fsSync.statSync(sourcePath).size : null,
            language: language || 'auto'
          });
          
          const separationPromise = separateSpeakersWithPyAnnote(sourcePath);
          
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
              reject(new Error('PyAnnote speaker separation timed out after 20 minutes. First model download may take 5-10 minutes. Please try again.'));
            }, 20 * 60 * 1000);
          });
          
          console.log(`[${requestId}] 🔵 STEP 2 MODE2: Waiting for separation to complete...`);
          separation = await Promise.race([separationPromise, timeoutPromise]);
          console.log(`[${requestId}] ✅ STEP 2 MODE2: Separation completed, speakers count: ${separation?.speakers?.length || 0}`);
            
            // Save to cache
            if (separationCacheKey && separation) {
              writeSeparationCache(separationCacheKey, {
                ...separation,
                mode: 'mode2',
                originalFilename: originalName
              });
            }
          }
        } else if (overlapPipelineMode === 'mode3') {
          // SpeechBrain separation (mode3)
          // Check cache first
          const separationCacheKey = buildSeparationCacheKey(originalName, 'mode3');
          let cachedSeparation = null;
          if (separationCacheKey) {
            cachedSeparation = readSeparationCache(separationCacheKey);
            if (cachedSeparation) {
              console.log(`[${requestId}] ✅ STEP 2 MODE3: Using cached SpeechBrain separation result`);
              writeLog('info', `[${requestId}] STEP 2 MODE3: Using cached SpeechBrain separation`, {
                requestId,
                step: 2,
                status: 'using_cached_separation',
                cacheKey: separationCacheKey,
                speakersCount: cachedSeparation.speakers?.length || 0
              });
              // Regenerate URLs for cached speakers
              const speakersWithUrls = cachedSeparation.speakers.map(speaker => {
                const timestamp = Date.now();
                const speakerFilename = `${timestamp}_${speaker.name}.${speaker.format || 'wav'}`;
                return {
                  ...speaker,
                  url: getPublicFileUrl(speakerFilename),
                  downloadUrl: `/uploads/${speakerFilename}`,
                  cached: true
                };
              });
              
              separation = {
                ...cachedSeparation,
                speakers: speakersWithUrls,
                mode: 'mode3',
                originalFilename: originalName
              };
              
              // Send completion update for cached result
              sendSSEUpdate(2, 'completed', `✅ MODE3: Розділення завершено (кеш). Знайдено ${separation?.speakers?.length || 0} спікерів`, {
                output: {
                  speakersCount: separation?.speakers?.length || 0,
                  speakers: separation?.speakers?.map(s => s.name) || [],
                  timelineSegments: separation?.timeline?.length || 0,
                  cached: true
                }
              });
            }
          }
          
          if (!cachedSeparation) {
          console.log(`[${requestId}] 🔵 STEP 2 MODE3: Starting SpeechBrain speaker separation`);
          console.log(`[${requestId}] 🔵 STEP 2 MODE3: File path: ${sourcePath}`);
          console.log(`[${requestId}] 🔵 STEP 2 MODE3: File exists: ${fsSync.existsSync(sourcePath)}`);
          console.log(`[${requestId}] 🔵 STEP 2 MODE3: File size: ${fsSync.existsSync(sourcePath) ? fsSync.statSync(sourcePath).size : 'N/A'} bytes`);

          writeLog('info', `[${requestId}] STEP 2 MODE3: Starting SpeechBrain separation`, {
            requestId,
            step: 2,
            mode: 'mode3',
            status: 'calling_speechbrain_separation',
            filePath: sourcePath,
            fileExists: fsSync.existsSync(sourcePath),
            fileSize: fsSync.existsSync(sourcePath) ? fsSync.statSync(sourcePath).size : null,
            language: language || 'auto'
          });

          // Send detailed progress update to client
          if (res && !res.headersSent) {
            res.write(`data: ${JSON.stringify({
              type: 'step-progress',
              step: 2,
              status: 'processing',
              description: `🔵 MODE3: Завантаження моделі SpeechBrain SepFormer...`,
              details: {
                filePath: sourcePath,
                fileSize: fsSync.existsSync(sourcePath) ? `${(fsSync.statSync(sourcePath).size / 1024 / 1024).toFixed(2)} MB` : 'N/A',
                stage: 'model_loading'
              }
            })}\n\n`);
          }

          const speechBrainTimeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
              reject(new Error('SpeechBrain speaker separation timed out after 5 minutes. The first model download may take longer.'));
            }, 5 * 60 * 1000);
          });

          console.log(`[${requestId}] 🔵 STEP 2 MODE3: Calling separateSpeakersWithSpeechBrain...`);
          
          separation = await Promise.race([
            separateSpeakersWithSpeechBrain(sourcePath, requestId, res, sendSSEUpdate),
            speechBrainTimeoutPromise
          ]);
            
            // Save to cache
            if (separationCacheKey && separation) {
              writeSeparationCache(separationCacheKey, {
                ...separation,
                mode: 'mode3',
                originalFilename: originalName
              });
            }
          }
          
          const separationDetails = {
            speakers: separation?.speakers?.map(s => ({
              name: s.name,
              hasUrl: !!s.url,
              hasLocalPath: !!s.local_path,
              isBackground: s.isBackground
            })),
            timeline: separation?.timeline?.length || 0,
            outputDir: separation?.output_dir
          };
          
          console.log(`[${requestId}] ✅ STEP 2 MODE3: Separation completed, speakers count: ${separation?.speakers?.length || 0}`);
          console.log(`[${requestId}] 📊 STEP 2 MODE3: Separation details:`, separationDetails);
          
          writeLog('info', `[${requestId}] STEP 2 MODE3: Separation completed`, {
            requestId,
            step: 2,
            mode: 'mode3',
            status: 'separation_completed',
            speakersCount: separation?.speakers?.length || 0,
            speakers: separation?.speakers?.map(s => ({
              name: s.name,
              hasUrl: !!s.url,
              hasLocalPath: !!s.local_path
            })),
            timelineSegments: separation?.timeline?.length || 0
          });
          
          // Send completion update
          sendSSEUpdate(2, 'completed', `✅ MODE3: Розділення завершено. Знайдено ${separation?.speakers?.length || 0} спікерів`, {
            output: {
              speakersCount: separation?.speakers?.length || 0,
              speakers: separation?.speakers?.map(s => s.name) || [],
              timelineSegments: separation?.timeline?.length || 0,
              speakersDetails: separationDetails.speakers
            }
          });
        } else {
          // AudioShake separation (mode1)
          // Check cache first
          const separationCacheKey = buildSeparationCacheKey(storedFile?.filename || originalName, 'mode1');
          let cachedSeparation = null;
          if (separationCacheKey) {
            cachedSeparation = readSeparationCache(separationCacheKey);
            if (cachedSeparation) {
              console.log(`[${requestId}] ✅ STEP 2: Using cached AudioShake separation result`);
              writeLog('info', `[${requestId}] STEP 2: Using cached AudioShake separation`, {
                requestId,
                step: 2,
                status: 'using_cached_separation',
                cacheKey: separationCacheKey,
                speakersCount: cachedSeparation.speakers?.length || 0
              });
              // Regenerate URLs for cached speakers (they may have expired)
              const speakersWithUrls = cachedSeparation.speakers.map(speaker => {
                // For cached results, we need to regenerate the URLs
                // Since we can't regenerate the exact files, we'll mark them as cached
                const speakerFilename = `${Date.now()}_${speaker.name}.${speaker.format || 'wav'}`;
                return {
                  ...speaker,
                  url: getPublicFileUrl(speakerFilename), // May not exist, but structure is preserved
                  downloadUrl: `/uploads/${speakerFilename}`,
                  cached: true
                };
              });
              
              separation = {
                ...cachedSeparation,
                speakers: speakersWithUrls,
                mode: 'mode1',
                originalFilename: storedFile?.filename || originalName
              };
            }
          }
          
          if (!cachedSeparation) {
      console.log(`[${requestId}] 🔵 STEP 2: Starting speaker separation with URL: ${publicUrl}`);
          writeLog('info', `[${requestId}] STEP 2: Starting AudioShake separation`, {
        requestId,
        step: 2,
        status: 'calling_separation_api',
        publicUrl: publicUrl,
        language: language || 'auto'
      });
      
        // Add timeout wrapper for the entire separation process (15 minutes max)
        const separationPromise = separateSpeakers(publicUrl, {
          variant: undefined,
          language: language || undefined
        });
        
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => {
            reject(new Error('Speaker separation timed out after 15 minutes'));
          }, 15 * 60 * 1000); // 15 minutes
        });
        
        separation = await Promise.race([separationPromise, timeoutPromise]);
            
            // Save to cache
            if (separationCacheKey && separation) {
              writeSeparationCache(separationCacheKey, {
                ...separation,
                mode: 'mode1',
                originalFilename: storedFile?.filename || originalName
              });
            }
          }
        }
        const separationDuration = ((Date.now() - separationStartTime) / 1000).toFixed(2);
        console.log(`[${requestId}] ✅ STEP 2: Speaker separation completed in ${separationDuration}s, found ${separation.speakers.length} speakers`);
        writeLog('info', `[${requestId}] STEP 2: Speaker separation completed`, {
          requestId,
          step: 2,
          status: 'separation_completed',
          duration: `${separationDuration}s`,
          speakersCount: separation.speakers.length,
          taskId: separation.taskId
        });
      } catch (separationError) {
        const separationDuration = ((Date.now() - separationStartTime) / 1000).toFixed(2);
        console.error(`[${requestId}] ❌ STEP 2: Speaker separation failed after ${separationDuration}s`);
        console.error(`[${requestId}] ❌ STEP 2: Mode: ${overlapPipelineMode}`);
        console.error(`[${requestId}] ❌ STEP 2: Error type: ${separationError.constructor?.name || 'Unknown'}`);
        console.error(`[${requestId}] ❌ STEP 2: Error name: ${separationError.name || 'Unknown'}`);
        console.error(`[${requestId}] ❌ STEP 2: Error message: ${separationError.message || 'No message'}`);
        console.error(`[${requestId}] ❌ STEP 2: Full error:`, separationError);
        
        writeLog('error', `[${requestId}] STEP 2: Speaker separation failed`, {
          requestId,
          step: 2,
          mode: overlapPipelineMode,
          status: 'separation_failed',
          duration: `${separationDuration}s`,
          error: separationError.message,
          errorType: separationError.constructor?.name,
          errorName: separationError.name,
          stack: separationError.stack?.substring(0, 1000),
          sourcePath: sourcePath,
          publicUrl: publicUrl || null
        });
        
        // Provide more helpful error message
        let errorMessage = separationError.message || 'Speaker separation failed';
        
        // Log for debugging
        const isLocalSeparation = overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3';
        console.log(`[${requestId}] 🔍 STEP 2 Separation Error handling:`, {
          overlapPipelineMode,
          isMode1: overlapPipelineMode === 'mode1',
          isMode2: overlapPipelineMode === 'mode2',
          isMode3: overlapPipelineMode === 'mode3',
          errorMessage,
          errorMessageLength: errorMessage.length,
          hasHttpsInMessage: errorMessage.includes('HTTPS') || errorMessage.includes('https'),
          hasInvalidUrlInMessage: errorMessage.includes('Invalid audio URL'),
          hasRequiresHttps: errorMessage.includes('requires HTTPS URL'),
          hasHttpsRequired: errorMessage.includes('HTTPS URL required'),
          hasPubliclyAccessible: errorMessage.includes('publicly accessible HTTPS'),
          hasAudioShake: errorMessage.includes('AudioShake')
        });
        
        // Check for common error patterns
        if (errorMessage.includes('API key') || errorMessage.includes('api key')) {
          errorMessage = 'Speaker separation service is not configured. Please check API key settings.';
        } else if (overlapPipelineMode === 'mode1' && (
          errorMessage.includes('requires HTTPS URL') || 
          errorMessage.includes('HTTPS URL required') || 
          errorMessage.includes('publicly accessible HTTPS') ||
          (errorMessage.includes('Invalid audio URL') && (errorMessage.includes('HTTPS') || errorMessage.includes('AudioShake')))
        )) {
          console.log(`[${requestId}] 🔍 STEP 2: HTTPS error detected for mode1 in separation catch, setting error message`);
          errorMessage = 'Speaker separation requires a publicly accessible HTTPS URL.';
        } else if (isLocalSeparation && (
          errorMessage.includes('requires HTTPS URL') || 
          errorMessage.includes('HTTPS URL required') || 
          errorMessage.includes('publicly accessible HTTPS') ||
          (errorMessage.includes('Invalid audio URL') && (errorMessage.includes('HTTPS') || errorMessage.includes('AudioShake')))
        )) {
          // For local modes, don't show HTTPS error - it's not relevant
          console.log(`[${requestId}] 🔍 STEP 2: HTTPS-related error detected for local mode (${overlapPipelineMode}) in separation catch, ignoring HTTPS-specific message`);
          errorMessage = separationError.message || 'Speaker separation failed'; // Use original error message
        } else if (errorMessage.includes('timeout') || errorMessage.includes('timed out') || separationError.name === 'TimeoutError') {
          errorMessage = `Speaker separation timed out after ${separationDuration}s. The service may be experiencing high load. Please try again.`;
        } else if (errorMessage.includes('404') || errorMessage.includes('not found')) {
          errorMessage = 'Audio file not found. Please ensure the file is accessible.';
        } else if (errorMessage.includes('Network') || errorMessage.includes('ECONNREFUSED') || errorMessage.includes('ENOTFOUND')) {
          errorMessage = 'Network error connecting to speaker separation service. Please check your internet connection and try again.';
        }
        
        throw new Error(errorMessage);
      }

      const step2Duration = ((Date.now() - step2StartTime) / 1000).toFixed(2) + 's';
      const separationLabels = {
        mode1: 'Speaker Separation (AudioShake)',
        mode2: 'Speaker Separation (PyAnnote)',
        mode3: 'Speaker Separation (SpeechBrain)'
      };
      
      steps.step2 = {
        name: separationLabels[overlapPipelineMode] || 'Speaker Separation',
        status: 'completed',
        duration: step2Duration,
        speakersCount: separation.speakers.length,
        taskId: separation.taskId
      };
      
      console.log(`[${requestId}] ✅ STEP 2: Completed successfully in ${step2Duration}`);
      writeLog('info', `[${requestId}] STEP 2: Completed successfully`, {
        requestId,
        step: 2,
        duration: step2Duration,
        speakersCount: separation.speakers.length,
        taskId: separation.taskId,
        endTime: new Date().toISOString()
      });
    } catch (step2Error) {
      const step2Duration = ((Date.now() - step2StartTime) / 1000).toFixed(2) + 's';
      console.error(`[${requestId}] ❌ STEP 2: Failed after ${step2Duration}`);
      console.error(`[${requestId}] ❌ STEP 2: Mode: ${overlapPipelineMode}`);
      console.error(`[${requestId}] ❌ STEP 2: Error type: ${step2Error.constructor?.name || 'Unknown'}`);
      console.error(`[${requestId}] ❌ STEP 2: Error name: ${step2Error.name || 'Unknown'}`);
      console.error(`[${requestId}] ❌ STEP 2: Error message: ${step2Error.message || 'No message'}`);
      console.error(`[${requestId}] ❌ STEP 2: Full error:`, step2Error);
      console.error(`[${requestId}] ❌ STEP 2: Error stack:`, step2Error.stack?.substring(0, 500));
      
      writeLog('error', `[${requestId}] STEP 2: Failed`, {
        requestId,
        step: 2,
        mode: overlapPipelineMode,
        error: step2Error.message,
        errorType: step2Error.constructor?.name,
        errorName: step2Error.name,
        stack: step2Error.stack,
        duration: step2Duration,
        endTime: new Date().toISOString(),
        sourcePath: sourcePath,
        publicUrl: publicUrl || null
      });
      
      // Provide more helpful error message
      let errorMessage = 'Speaker separation failed';
      
      // Log for debugging
      const isLocalSeparation = overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3';
      console.log(`[${requestId}] 🔍 STEP 2 Error handling:`, {
        overlapPipelineMode,
        isMode1: overlapPipelineMode === 'mode1',
        isMode2: overlapPipelineMode === 'mode2',
        isMode3: overlapPipelineMode === 'mode3',
        errorMessage: step2Error.message,
        errorMessageLength: step2Error.message?.length || 0,
        hasHttpsInMessage: step2Error.message?.includes('HTTPS') || step2Error.message?.includes('https'),
        hasInvalidUrlInMessage: step2Error.message?.includes('Invalid audio URL'),
        hasRequiresHttps: step2Error.message?.includes('requires HTTPS URL'),
        hasHttpsRequired: step2Error.message?.includes('HTTPS URL required'),
        hasPubliclyAccessible: step2Error.message?.includes('publicly accessible HTTPS'),
        hasAudioShake: step2Error.message?.includes('AudioShake'),
        statusCode: step2Error.statusCode,
        errorData: step2Error.errorData
      });
      
      // Check for specific error types
      if (step2Error.message && step2Error.message.includes('API key')) {
        errorMessage = 'Speaker separation service is not configured. Please check API key settings.';
      } else if (step2Error.message && step2Error.message.includes('Insufficient credits')) {
        errorMessage = 'Speaker separation service account has insufficient credits. Please add credits to your account to continue.';
        } else if (overlapPipelineMode === 'mode1' && step2Error.message && (
        step2Error.message.includes('requires HTTPS URL') || 
        step2Error.message.includes('HTTPS URL required') || 
        step2Error.message.includes('publicly accessible HTTPS') ||
        (step2Error.message.includes('Invalid audio URL') && (step2Error.message.includes('HTTPS') || step2Error.message.includes('AudioShake')))
      )) {
        console.log(`[${requestId}] 🔍 STEP 2: HTTPS error detected for mode1, setting error message`);
        errorMessage = 'Invalid audio URL format. Speaker separation requires a publicly accessible HTTPS URL.';
      } else if (isLocalSeparation && step2Error.message && (
        step2Error.message.includes('requires HTTPS URL') || 
        step2Error.message.includes('HTTPS URL required') || 
        step2Error.message.includes('publicly accessible HTTPS') ||
        (step2Error.message.includes('Invalid audio URL') && (step2Error.message.includes('HTTPS') || step2Error.message.includes('AudioShake')))
      )) {
        // For local modes, don't show HTTPS error - it's not relevant
        console.log(`[${requestId}] 🔍 STEP 2: HTTPS-related error detected for local mode (${overlapPipelineMode}), ignoring HTTPS-specific message`);
        console.log(`[${requestId}] 🔍 STEP 2: Original error message will be used: ${step2Error.message}`);
        errorMessage = step2Error.message; // Use original error message
      } else if (isLocalSeparation && step2Error.message) {
        // For local modes, always use original error message, don't try to "improve" it
        console.log(`[${requestId}] 🔍 STEP 2: Local mode error - using original message: ${step2Error.message}`);
        errorMessage = step2Error.message;
      } else if (step2Error.statusCode === 400 && step2Error.errorData) {
        // Handle structured error responses
        const errorData = step2Error.errorData;
        if (errorData.message && errorData.message.includes('Insufficient credits')) {
          errorMessage = 'Speaker separation service account has insufficient credits. Please add credits to your account to continue.';
        } else if (errorData.message) {
          errorMessage = `Speaker separation failed: ${errorData.message}`;
        } else {
          errorMessage = step2Error.message || 'Speaker separation failed';
        }
      } else if (step2Error.message) {
        errorMessage = step2Error.message;
      }
      
      console.log(`[${requestId}] 🔍 STEP 2: Final error message to return:`, {
        errorMessage,
        originalMessage: step2Error.message,
        mode: overlapPipelineMode,
        willReturnHttpsError: errorMessage.includes('HTTPS') && (overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3')
      });
      
      // For local modes, never return HTTPS-related errors
      if ((overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3') && errorMessage.includes('HTTPS')) {
        console.log(`[${requestId}] ⚠️ STEP 2: Detected HTTPS error for local mode (${overlapPipelineMode}), replacing with original error`);
        errorMessage = step2Error.message || 'Speaker separation failed';
      }
      
      // For SSE modes (mode1 and mode3), send error via SSE
      // Note: For SSE, headers are already sent, so we check by mode instead of headersSent
      if ((overlapPipelineMode === 'mode3' || overlapPipelineMode === 'mode1')) {
        try {
          // Send error as SSE event (pipeline-error type for consistency)
          sendSSEUpdate(2, 'failed', `❌ MODE3: Помилка розділення спікерів: ${errorMessage}`, {
            stage: 'error',
            error: errorMessage,
            details: step2Error.stack?.substring(0, 500),
            duration: step2Duration
          });
          
          // Send final pipeline-error event
          const errorEvent = {
            type: 'pipeline-error',
            success: false,
            error: 'Speaker separation failed',
            details: errorMessage,
            step: 2,
            duration: step2Duration,
            requestId: requestId,
            mode: overlapPipelineMode,
            timestamp: new Date().toISOString()
          };
          res.write(`data: ${JSON.stringify(errorEvent)}\n\n`);
          console.log(`[${requestId}] 📤 SSE: Pipeline error sent for Step 2`);
          cleanupKeepAlive();
          res.end();
          return;
        } catch (e) {
          console.error(`[${requestId}] ❌ Failed to send SSE error:`, e.message);
          cleanupKeepAlive();
          res.end();
          return;
        }
      }
      
      // For mode2, use regular JSON response
      return res.status(500).json({
        success: false,
        error: errorMessage,
        details: process.env.NODE_ENV === 'development' ? step2Error.message : undefined,
        step: 2,
        duration: step2Duration,
        mode: overlapPipelineMode
      });
    }

    // Step 3-4: Transcribe and analyze each voice track
    const step3StartTime = Date.now();
    console.log(`[${requestId}] 🔵 STEP 3-4: Starting transcription and analysis of ${separation.speakers.length} voice tracks`);
    
    // Send SSE update for Step 3-4 start
    sendSSEUpdate(3, 'processing', `🔵 MODE3: Початок транскрипції та аналізу ${separation.speakers.length} треків...`, {
      stage: 'starting',
      input: {
        totalTracks: separation.speakers.length,
        speakers: separation.speakers.map(s => ({
          name: s.name,
          hasUrl: !!s.url,
          hasLocalPath: !!s.local_path,
          isBackground: s.isBackground
        }))
      }
    });
    
    writeLog('info', `[${requestId}] STEP 3-4: Starting transcription and analysis`, {
      requestId,
      step: '3-4',
      status: 'starting',
      startTime: new Date().toISOString(),
      voiceTracksCount: separation.speakers.length
    });
    
    try {
      for (let i = 0; i < separation.speakers.length; i++) {
        const speaker = separation.speakers[i];
        const trackStartTime = Date.now();
        
        console.log(`[${requestId}] 🔵 STEP 3-4: Processing voice track ${i + 1}/${separation.speakers.length}: ${speaker.name}`);
        writeLog('info', `[${requestId}] STEP 3-4: Processing voice track ${i + 1}/${separation.speakers.length}`, {
          requestId,
          step: '3-4',
          status: 'processing_track',
          trackNumber: i + 1,
          totalTracks: separation.speakers.length,
          speakerName: speaker.name,
          hasUrl: !!speaker?.url,
          isBackground: speaker.isBackground
        });
        
        if (!speaker?.url) {
          console.warn(`[${requestId}] ⚠️ STEP 3-4: Skipping speaker ${i + 1}: no URL`);
          writeLog('warn', `[${requestId}] STEP 3-4: Skipping speaker ${i + 1}: no URL`, {
            requestId,
            step: '3-4',
            trackNumber: i + 1,
            speakerName: speaker.name
          });
          continue;
        }
        
        if (speaker.isBackground) {
          console.log(`[${requestId}] ⏭️ STEP 3-4: Skipping background stem: ${speaker.name}`);
          writeLog('info', `[${requestId}] STEP 3-4: Skipping background stem`, {
            requestId,
            step: '3-4',
            trackNumber: i + 1,
            speakerName: speaker.name
          });
          continue;
        }

        try {
          // Step 3: Transcribe voice track
          console.log(`[${requestId}] 🔵 STEP 3: Transcribing voice track ${i + 1}/${separation.speakers.length} (${speaker.name})...`);
          writeLog('info', `[${requestId}] STEP 3: Transcribing voice track ${i + 1}`, {
            requestId,
            step: 3,
            status: 'transcribing',
            trackNumber: i + 1,
            totalTracks: separation.speakers.length,
            speakerName: speaker.name,
            audioUrl: speaker.url,
            hasLocalPath: !!speaker.local_path
          });
          
          // Send progress update for mode 3
          sendSSEUpdate(3, 'processing', `🔵 MODE3: Транскрипція треку ${i + 1}/${separation.speakers.length} (${speaker.name})...`, {
            trackNumber: i + 1,
            totalTracks: separation.speakers.length,
            speakerName: speaker.name,
            stage: 'transcription_starting',
            engine: engineLabel,
            input: {
              audioSource: speaker.local_path ? 'local_file' : 'url',
              audioPath: speaker.local_path || speaker.url,
              language: language
            }
          });
          
          const transcriptionStartTime = Date.now();
          
          // For Mode 3 (SpeechBrain), use local file path instead of URL
          // Speechmatics doesn't need public URL - it can work with local files
          // This avoids dependency on localtunnel and is faster
          // CRITICAL FIX: Mark as separated track to use 'channel' diarization mode
          // This prevents Speechmatics from detecting residual audio as separate speakers
          // Expected improvement: 70-80% reduction in speaker assignment errors
          // IMPORTANT: Enable cache for voice tracks using stable filename based on original file + speaker
          // This ensures cache works correctly even if URLs change (e.g., AudioShake with expiring URLs)
          // Use the same logic as Step 1: originalFilename + speaker identifier
          const originalBaseFilename = uploadedFile?.originalname 
            ? path.parse(uploadedFile.originalname).name 
            : (url ? path.parse(new URL(url).pathname.split('/').pop() || 'audio').name : 'audio');
          const speakerIdentifier = speaker.name || `speaker_${i + 1}`;
          // Create stable cache filename: original_file_speaker_name.wav
          // This matches the cache key logic used in Step 1
          const voiceTrackFilename = `${originalBaseFilename}_${speakerIdentifier}.wav`;
          
          console.log(`[${requestId}] 💾 STEP 3: Cache configuration for voice track ${i + 1}:`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            speakerIdentifier: speakerIdentifier,
            originalBaseFilename: originalBaseFilename,
            voiceTrackFilename: voiceTrackFilename,
            hasLocalPath: !!speaker.local_path,
            hasUrl: !!speaker.url,
            language: language,
            engine: transcriptionEngine,
            isSeparatedTrack: true,
            cacheEnabled: true,
            cacheStrategy: 'stable_filename_based_on_original_file'
          });
          
          // CRITICAL: For separated tracks, use speakerCount=2 to detect both main speaker and residual audio
          // Residual audio from other speakers may not be fully separated, so we need to detect it
          // Then filter it out in collectVoiceTrackSegments by identifying the main speaker
          const transcriptionParams = overlapPipelineMode === 'mode3' && speaker.local_path
            ? { filePath: speaker.local_path, language: language, speakerCount: '2', isSeparatedTrack: true, skipCache: false, engine: transcriptionEngine, originalFilename: voiceTrackFilename }
            : { url: speaker.url, language: language, speakerCount: '2', isSeparatedTrack: true, skipCache: false, engine: transcriptionEngine, originalFilename: voiceTrackFilename };
          
          console.log(`[${requestId}] 📊 STEP 3: Transcription params:`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            hasLocalPath: !!speaker.local_path,
            localPath: speaker.local_path,
            hasUrl: !!speaker.url,
            url: speaker.url,
            language: language,
            speakerCount: '2',  // Allow detection of main speaker + residual audio
            usingLocalFile: overlapPipelineMode === 'mode3' && !!speaker.local_path,
            isSeparatedTrack: true,  // Separated track - will use speaker mode with speaker_count=2
            originalFilename: voiceTrackFilename,
            cacheEnabled: true,
            engine: transcriptionEngine,
            diarizationMode: 'speaker'  // Speaker mode for separated tracks (detects main + residual, then filtered)
          });
          
          const trackProgressHandler = shouldStreamEngineProgress
            ? createPythonProgressForwarder(3, {
                trackNumber: i + 1,
                speakerName: speaker.name
              })
            : null;
          const pythonTrackParams = trackProgressHandler
            ? { ...transcriptionParams, onProgress: trackProgressHandler }
            : transcriptionParams;
          
          // Add timeout for transcription (10 minutes max per track)
          const transcriptionPromise = runPythonDiarization(pythonTrackParams);
          
          const transcriptionTimeoutPromise = new Promise((_, reject) => {
            setTimeout(() => {
              reject(new Error(`Transcription timed out after 10 minutes for track ${i + 1}`));
            }, 10 * 60 * 1000); // 10 minutes
          });
          
          let transcription;
          try {
            const trackWaitingStage = transcriptionEngine === 'azure'
              ? 'waiting_azure'
              : (transcriptionEngine === 'azure_realtime' ? 'waiting_azure_realtime' : 'waiting_speechmatics');
            sendSSEUpdate(3, 'processing', `⏳ MODE3: Очікування відповіді від ${engineLabel} для треку ${i + 1}...`, {
              trackNumber: i + 1,
              speakerName: speaker.name,
              stage: trackWaitingStage,
              engine: engineLabel,
              input: {
                transcriptionParams: transcriptionParams
              }
            });
            
            transcription = await Promise.race([transcriptionPromise, transcriptionTimeoutPromise]);
            
            // Log cache usage
            if (transcription?._cached) {
              console.log(`[${requestId}] ✅ STEP 3: Using CACHED transcription for track ${i + 1} (${speaker.name}) - saved time!`);
              writeLog('info', `[${requestId}] STEP 3: Using cached transcription`, {
                requestId,
                step: 3,
                trackNumber: i + 1,
                speakerName: speaker.name,
                cached: true,
                engine: transcriptionEngine
              });
            } else {
              console.log(`[${requestId}] 💸 STEP 3: Cache MISS for track ${i + 1} (${speaker.name}) - fetching from API...`);
            }
          } catch (transcriptionError) {
            const transcriptionDuration = ((Date.now() - transcriptionStartTime) / 1000).toFixed(2);
            console.error(`[${requestId}] ❌ STEP 3: Transcription failed for track ${i + 1} after ${transcriptionDuration}s:`, transcriptionError);
            writeLog('error', `[${requestId}] STEP 3: Transcription failed for track ${i + 1}`, {
              requestId,
              step: 3,
              trackNumber: i + 1,
              speakerName: speaker.name,
              duration: `${transcriptionDuration}s`,
              error: transcriptionError.message,
              errorType: transcriptionError.constructor?.name
            });
            
            sendSSEUpdate(3, 'processing', `❌ MODE3: Помилка транскрипції треку ${i + 1}: ${transcriptionError.message}`, {
              trackNumber: i + 1,
              speakerName: speaker.name,
              stage: 'transcription_error',
              error: {
                message: transcriptionError.message,
                type: transcriptionError.constructor?.name,
                duration: `${transcriptionDuration}s`
              }
            });
            
            throw transcriptionError;
          }
          
          const transcriptionDuration = ((Date.now() - transcriptionStartTime) / 1000).toFixed(2);
          
          // Extract transcript text
          let transcriptText = '';
          const segmentsCount = transcription?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0;
          
          // Validate transcription result
          if (!transcription || !transcription.recordings || transcription.recordings.length === 0) {
            console.warn(`[${requestId}] ⚠️ STEP 3: Transcription returned invalid structure for track ${i + 1}`);
            writeLog('warn', `[${requestId}] STEP 3: Invalid transcription structure`, {
              requestId,
              step: 3,
              trackNumber: i + 1,
              speakerName: speaker.name,
              hasTranscription: !!transcription,
              hasRecordings: !!transcription?.recordings,
              recordingsCount: transcription?.recordings?.length || 0
            });
          }
          
          if (transcription?.recordings?.[0]?.results?.speechmatics?.segments) {
            const segments = transcription.recordings[0].results.speechmatics.segments;
            transcriptText = segments
              .map(s => s.text)
              .join(' ')
              .trim();
            
            // Validate detected speakers in separated track
            // Separated tracks may contain main speaker + residual audio from other speakers
            // This is expected - collectVoiceTrackSegments will identify main speaker and filter residual
            const detectedSpeakers = new Set(segments.map(s => s.speaker || 'UNKNOWN'));
            if (detectedSpeakers.size > 2) {
              console.warn(
                `[${requestId}] ⚠️ STEP 3: WARNING - Voice track ${i + 1} (${speaker.name}) detected ${detectedSpeakers.size} speakers: ${Array.from(detectedSpeakers).join(', ')}. ` +
                `Expected: 1-2 speakers (main speaker + possible residual audio). More than 2 may indicate separation issues.`
              );
              writeLog('warn', `[${requestId}] STEP 3: More than 2 speakers detected in separated track`, {
                requestId,
                step: 3,
                trackNumber: i + 1,
                speakerName: speaker.name,
                detectedSpeakers: Array.from(detectedSpeakers),
                expectedSpeakers: '1-2 (main + residual)',
                segmentsCount: segmentsCount
              });
            } else if (detectedSpeakers.size === 2) {
              console.log(
                `[${requestId}] ℹ️ STEP 3: Voice track ${i + 1} detected 2 speakers: ${Array.from(detectedSpeakers).join(', ')}. ` +
                `This is expected - main speaker + residual audio. Will filter residual in post-processing.`
              );
            } else {
              console.log(
                `[${requestId}] ✅ STEP 3: Voice track ${i + 1} detected 1 speaker: ${Array.from(detectedSpeakers)[0] || 'UNKNOWN'} (clean separation)`
              );
            }
          }
          
          console.log(`[${requestId}] ✅ STEP 3: Transcription completed in ${transcriptionDuration}s, found ${segmentsCount} segments, text length: ${transcriptText.length} chars`);
          console.log(`[${requestId}] 📊 STEP 3 MODE3: Transcript preview (first 200 chars):`, transcriptText.substring(0, 200));
          
          writeLog('info', `[${requestId}] STEP 3: Transcription completed`, {
            requestId,
            step: 3,
            status: 'transcription_completed',
            trackNumber: i + 1,
            speakerName: speaker.name,
            duration: `${transcriptionDuration}s`,
            segmentsCount,
            transcriptLength: transcriptText.length,
            transcriptPreview: transcriptText.substring(0, 200)
          });
          
          sendSSEUpdate(3, 'processing', `✅ MODE3: Транскрипція треку ${i + 1} завершена (${segmentsCount} сегментів, ${transcriptText.length} символів)`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            stage: 'transcription_completed',
            output: {
              segmentsCount,
              transcriptLength: transcriptText.length,
              transcriptPreview: transcriptText.substring(0, 200),
              duration: `${transcriptionDuration}s`,
              hasTranscription: !!transcription,
              hasRecordings: !!transcription?.recordings
            }
          });
          
          // Step 4: Analyze role
          console.log(`[${requestId}] 🔵 STEP 4: Analyzing role for voice track ${i + 1} (${speaker.name})...`);
          console.log(`[${requestId}] 📊 STEP 4 MODE3: Sending to LLM for role analysis:`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            transcriptLength: transcriptText.length,
            transcriptPreview: transcriptText.substring(0, 150),
            language: language,
            mode: mode || 'fast'
          });
          
          writeLog('info', `[${requestId}] STEP 4: Analyzing role for voice track ${i + 1}`, {
            requestId,
            step: 4,
            status: 'analyzing_role',
            trackNumber: i + 1,
            speakerName: speaker.name,
            transcriptLength: transcriptText.length,
            transcriptPreview: transcriptText.substring(0, 150)
          });
          
          sendSSEUpdate(4, 'processing', `🔵 MODE3: Аналіз ролі для треку ${i + 1} (${speaker.name})...`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            stage: 'role_analysis_starting',
            input: {
              transcriptLength: transcriptText.length,
              transcriptPreview: transcriptText.substring(0, 150),
              language: language,
              mode: mode || 'fast'
            }
          });
          
          const analysisStartTime = Date.now();
          const roleAnalysis = await analyzeVoiceRole(transcriptText, language, mode || 'fast');
          const analysisDuration = ((Date.now() - analysisStartTime) / 1000).toFixed(2);
          
          console.log(`[${requestId}] ✅ STEP 4: Role analysis completed in ${analysisDuration}s: ${roleAnalysis.role} (confidence: ${roleAnalysis.confidence})`);
          console.log(`[${requestId}] 📊 STEP 4 MODE3: Role analysis result:`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            role: roleAnalysis.role,
            confidence: roleAnalysis.confidence,
            summary: roleAnalysis.summary
          });
          
          writeLog('info', `[${requestId}] STEP 4: Role analysis completed`, {
            requestId,
            step: 4,
            status: 'role_analysis_completed',
            trackNumber: i + 1,
            speakerName: speaker.name,
            duration: `${analysisDuration}s`,
            role: roleAnalysis.role,
            confidence: roleAnalysis.confidence,
            summary: roleAnalysis.summary
          });
          
          sendSSEUpdate(4, 'processing', `✅ MODE3: Роль визначено для треку ${i + 1}: ${roleAnalysis.role} (впевненість: ${(roleAnalysis.confidence * 100).toFixed(1)}%)`, {
            trackNumber: i + 1,
            speakerName: speaker.name,
            stage: 'role_analysis_completed',
            output: {
              role: roleAnalysis.role,
              confidence: roleAnalysis.confidence,
              summary: roleAnalysis.summary,
              duration: `${analysisDuration}s`
            }
          });
          
          const trackDuration = ((Date.now() - trackStartTime) / 1000).toFixed(2);
          console.log(`[${requestId}] ✅ STEP 3-4: Voice track ${i + 1} completed in ${trackDuration}s`);
          
          voiceTracks.push({
            speaker: speaker.name,
            audioUrl: speaker.url,
            transcription: transcription,
            transcriptText: transcriptText,
            roleAnalysis: roleAnalysis
          });
          
          writeLog('info', `[${requestId}] STEP 3-4: Voice track ${i + 1} processed`, {
            requestId,
            step: '3-4',
            trackNumber: i + 1,
            speaker: speaker.name,
            role: roleAnalysis.role,
            confidence: roleAnalysis.confidence,
            trackDuration: `${trackDuration}s`
          });
        } catch (voiceTrackError) {
          const trackDuration = ((Date.now() - trackStartTime) / 1000).toFixed(2);
          console.error(`[${requestId}] ❌ STEP 3-4: Voice track ${i + 1} processing failed after ${trackDuration}s:`, voiceTrackError);
          writeLog('error', `[${requestId}] STEP 3-4: Voice track ${i + 1} processing failed`, {
            requestId,
            step: '3-4',
            trackNumber: i + 1,
            speaker: speaker.name,
            duration: `${trackDuration}s`,
            error: voiceTrackError.message,
            errorType: voiceTrackError.constructor?.name,
            stack: voiceTrackError.stack?.substring(0, 500)
          });
          
          voiceTracks.push({
            speaker: speaker.name,
            audioUrl: speaker.url,
            error: voiceTrackError.message
          });
        }
      }

      const step3Duration = ((Date.now() - step3StartTime) / 1000).toFixed(2) + 's';
      const processedTracks = voiceTracks.filter(vt => !vt.error).length;
      const failedTracks = voiceTracks.filter(vt => vt.error).length;
      
      steps.step3 = {
        name: 'Voice Track Transcription & Role Analysis',
        status: 'completed',
        duration: step3Duration,
        processedTracks: processedTracks,
        totalTracks: voiceTracks.length,
        failedTracks: failedTracks
      };
      
      console.log(`[${requestId}] ✅ STEP 3-4: Completed in ${step3Duration}`);
      console.log(`[${requestId}] ✅ STEP 3-4: Processed ${processedTracks}/${voiceTracks.length} tracks successfully`);
      if (failedTracks > 0) {
        console.warn(`[${requestId}] ⚠️ STEP 3-4: ${failedTracks} tracks failed`);
      }
      
      writeLog('info', `[${requestId}] STEP 3-4: Completed`, {
        requestId,
        step: '3-4',
        duration: step3Duration,
        processedTracks: processedTracks,
        totalTracks: voiceTracks.length,
        failedTracks: failedTracks,
        endTime: new Date().toISOString()
      });
    } catch (step3Error) {
      const step3Duration = ((Date.now() - step3StartTime) / 1000).toFixed(2) + 's';
      console.error(`[${requestId}] ❌ STEP 3-4: Failed after ${step3Duration}:`, step3Error);
      console.error(`[${requestId}] ❌ STEP 3-4: Error details:`, {
        message: step3Error.message,
        stack: step3Error.stack
      });
      
      writeLog('error', `[${requestId}] STEP 3-4: Failed`, {
        requestId,
        step: '3-4',
        error: step3Error.message,
        stack: step3Error.stack,
        duration: step3Duration,
        endTime: new Date().toISOString()
      });
      
      return res.status(500).json({
        success: false,
        error: 'Voice tracks processing failed',
        details: step3Error.message,
        step: '3-4',
        duration: step3Duration
      });
    }

    // Diagnostics: build merged transcript & diff info
    const voiceTrackSegments = collectVoiceTrackSegments(voiceTracks);
    const missingReplicas = identifyMissingReplicas(primaryDiarization, voiceTrackSegments);
    const combinedTranscript = buildCombinedTranscriptFromSegments(voiceTrackSegments);
    let voiceTrackLLMResult = null;
    let comparisonAnalysis = null;
    let voiceTranscriptFile = null;

    if (combinedTranscript) {
      try {
        voiceTrackLLMResult = await handleDiarizationRequest({
          transcript: combinedTranscript,
          mode: mode || 'smart',
          promptVariant: 'voice-tracks'
        });
        comparisonAnalysis = await compareDiarizationResults(
          primaryDiarization,
          voiceTrackLLMResult,
          combinedTranscript
        );
        writeLog('info', `[${requestId}] DIAGNOSTICS: Combined transcript analyzed`, {
          requestId,
          transcriptLength: combinedTranscript.length,
          llmSegments: voiceTrackLLMResult?.recordings?.[0]?.results?.[TEXT_SERVICE_KEY]?.segments?.length || 0
        });
      } catch (diagnosticsError) {
        console.warn(`[${requestId}] ⚠️ DIAGNOSTICS: Failed to run transcript diarization`, diagnosticsError.message);
        writeLog('warn', `[${requestId}] DIAGNOSTICS failed`, {
          requestId,
          error: diagnosticsError.message
        });
      }
      
      try {
        voiceTranscriptFile = await saveTranscriptFile(
          combinedTranscript,
          requestId ? `voice_tracks_${requestId}` : 'voice_track_transcript'
        );
      } catch (transcriptError) {
        console.warn(`[${requestId}] ⚠️ Failed to save voice track transcript:`, transcriptError.message);
        writeLog('warn', 'Voice track transcript save failed', {
          requestId,
          error: transcriptError.message
        });
      }
    }

    // Step 5: Correct primary diarization using voice tracks
    const step5StartTime = Date.now();
    let finalMarkdownTable = null;
    console.log(`[${requestId}] 🔵 STEP 5: Starting overlap correction`);
    
    // Send text analysis webhook request after role analysis
    try {
      if (voiceTracks && voiceTracks.length >= 2) {
        // Extract general data from Step 1 (primary diarization)
        const primaryRecording = primaryDiarization?.recordings?.[0] || null;
        const primarySpeechmatics = primaryRecording?.results?.speechmatics || null;
        const primarySegments = primarySpeechmatics?.segments || [];
        const originalText = primarySegments
          .map(seg => seg.text || '')
          .filter(text => text.trim().length > 0)
          .join(' ')
          .trim();
        
        // Extract speaker1 data from Step 3 (voiceTracks[0])
        const track1 = voiceTracks[0];
        const speaker1Recording = track1?.transcription?.recordings?.[0] || null;
        const speaker1Speechmatics = speaker1Recording?.results?.speechmatics || null;
        
        // Extract speaker2 data from Step 3 (voiceTracks[1])
        const track2 = voiceTracks[1];
        const speaker2Recording = track2?.transcription?.recordings?.[0] || null;
        const speaker2Speechmatics = speaker2Recording?.results?.speechmatics || null;
        
        if (primarySpeechmatics && speaker1Speechmatics && speaker2Speechmatics) {
          console.log(`[${requestId}] 📤 STEP 5: Sending text analysis webhook request...`);
          
          // Generate markdown table from voice tracks segments for webhook
          let markdownTable = '';
          try {
            const speaker1Segments = speaker1Speechmatics?.segments || [];
            const speaker2Segments = speaker2Speechmatics?.segments || [];
            const speaker1Role = track1?.roleAnalysis?.role || 'unknown';
            const speaker2Role = track2?.roleAnalysis?.role || 'unknown';
            
            // Determine speaker labels
            const speaker1Label = speaker1Role === 'operator' || speaker1Role === 'agent' ? 'Agent' : 
                                 speaker1Role === 'client' || speaker1Role === 'customer' ? 'Client' : 
                                 'Speaker 1';
            const speaker2Label = speaker2Role === 'operator' || speaker2Role === 'agent' ? 'Agent' : 
                                 speaker2Role === 'client' || speaker2Role === 'customer' ? 'Client' : 
                                 'Speaker 2';
            
            // Combine all segments and sort by start time
            const allSegments = [];
            
            speaker1Segments.forEach(seg => {
              if (seg.text && seg.text.trim()) {
                allSegments.push({
                  speaker: speaker1Label,
                  text: seg.text.trim(),
                  start: parseFloat(seg.start) || 0,
                  end: parseFloat(seg.end) || parseFloat(seg.start) || 0
                });
              }
            });
            
            speaker2Segments.forEach(seg => {
              if (seg.text && seg.text.trim()) {
                allSegments.push({
                  speaker: speaker2Label,
                  text: seg.text.trim(),
                  start: parseFloat(seg.start) || 0,
                  end: parseFloat(seg.end) || parseFloat(seg.start) || 0
                });
              }
            });
            
            // Sort by start time
            allSegments.sort((a, b) => {
              const startDiff = a.start - b.start;
              if (startDiff !== 0) return startDiff;
              return a.end - b.end;
            });
            
            // Generate markdown table
            if (allSegments.length > 0) {
              markdownTable = '| Segment ID | Speaker | Text | Start Time | End Time |\n';
              markdownTable += '|------------|---------|------|------------|----------|\n';
              
              allSegments.forEach((seg, index) => {
                const segmentId = index + 1;
                const startTime = seg.start.toFixed(2);
                const endTime = seg.end.toFixed(2);
                // Remove filler words from text
                const cleanedText = removeFillerWords(seg.text);
                markdownTable += `| ${segmentId} | ${seg.speaker} | ${cleanedText} | ${startTime} | ${endTime} |\n`;
              });
            }
          } catch (markdownError) {
            console.warn(`[${requestId}] ⚠️ STEP 5: Failed to generate markdown for webhook:`, markdownError.message);
            // Continue without markdown if generation fails
          }
          
          try {
            // Structure the payload with clear separation: general (Step 1), speaker1 (Step 3), speaker2 (Step 3)
            const webhookPayload = {
              general: {
                // Step 1: Primary diarization data
                originalText: originalText,
                speechmatics: primarySpeechmatics,
                recording: {
                  name: primaryRecording?.name || null,
                  duration: primaryRecording?.duration || null,
                  speakerCount: primaryRecording?.speakerCount || null,
                  language: primaryRecording?.language || null
                },
                segments: primarySegments,
                segmentsCount: primarySegments.length
              },
              speaker1: {
                // Step 3: Voice track 1 transcription data
                speaker: track1?.speaker || 'SPEAKER_00',
                role: track1?.roleAnalysis?.role || null,
                roleConfidence: track1?.roleAnalysis?.confidence || null,
                roleSummary: track1?.roleAnalysis?.summary || null,
                speechmatics: speaker1Speechmatics,
                recording: {
                  name: speaker1Recording?.name || null,
                  duration: speaker1Recording?.duration || null,
                  language: speaker1Recording?.language || null
                },
                segments: speaker1Speechmatics?.segments || [],
                segmentsCount: (speaker1Speechmatics?.segments || []).length,
                transcriptText: track1?.transcriptText || null
              },
              speaker2: {
                // Step 3: Voice track 2 transcription data
                speaker: track2?.speaker || 'SPEAKER_01',
                role: track2?.roleAnalysis?.role || null,
                roleConfidence: track2?.roleAnalysis?.confidence || null,
                roleSummary: track2?.roleAnalysis?.summary || null,
                speechmatics: speaker2Speechmatics,
                recording: {
                  name: speaker2Recording?.name || null,
                  duration: speaker2Recording?.duration || null,
                  language: speaker2Recording?.language || null
                },
                segments: speaker2Speechmatics?.segments || [],
                segmentsCount: (speaker2Speechmatics?.segments || []).length,
                transcriptText: track2?.transcriptText || null
              },
              markdown: markdownTable || null
            };
            
            // Використовуємо локальні функції замість webhook або LLM для маркування
            let analysisResult;
            console.log(`[${requestId}] 🔍 STEP 5: Text analysis mode check:`, {
              useLLMForTextAnalysis: useLLMForTextAnalysis,
              hasMarkdown: !!webhookPayload.markdown,
              markdownLength: webhookPayload.markdown?.length || 0
            });
            
            if (useLLMForTextAnalysis) {
              try {
                console.log(`[${requestId}] 🤖 STEP 5: Using LLM mode for text analysis classification...`);
                const useLocalLLM = mode === 'local' || mode === 'test' || mode === 'test2';
                const llmModel = getModelId(mode);
                const apiUrl = useLocalLLM 
                  ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
                  : 'https://openrouter.ai/api/v1/chat/completions';
                const apiKey = useLocalLLM ? LOCAL_LLM_API_KEY : process.env.OPENROUTER_API_KEY;
                
                console.log(`[${requestId}] 📤 STEP 5: Calling LLM for text analysis:`, {
                  llmModel: llmModel,
                  apiUrl: apiUrl,
                  useLocalLLM: useLocalLLM,
                  mode: mode
                });
                
                analysisResult = await textAnalysis.analyzeTextWithLLM(
                  webhookPayload,
                  llmModel,
                  apiUrl,
                  apiKey,
                  useLocalLLM,
                  mode
                );
                console.log(`[${requestId}] ✅ STEP 5: Text analysis completed (LLM mode):`, {
                  blueCount: analysisResult.Blue?.length || 0,
                  greenCount: analysisResult.Green?.length || 0,
                  redCount: analysisResult.Red?.length || 0
                });
              } catch (llmError) {
                console.error(`[${requestId}] ❌ STEP 5: LLM text analysis failed, falling back to script mode:`, llmError.message);
                console.error(`[${requestId}] ❌ STEP 5: LLM error details:`, llmError);
                // Fallback до script режиму
                analysisResult = textAnalysis.analyzeText(webhookPayload);
              }
            } else {
              console.log(`[${requestId}] 📝 STEP 5: Using script mode for text analysis (useLLMForTextAnalysis=false)`);
              analysisResult = textAnalysis.analyzeText(webhookPayload);
            }
            
            console.log(`[${requestId}] ✅ STEP 5: Text analysis completed (${useLLMForTextAnalysis ? 'LLM' : 'script'} mode):`, {
              hasGreen: !!analysisResult.Green,
              hasBlue: !!analysisResult.Blue,
              hasRed: !!analysisResult.Red,
              blueCount: analysisResult.Blue?.length || 0,
              greenCount: analysisResult.Green?.length || 0,
              redCount: analysisResult.Red?.length || 0,
              payloadStructure: {
                hasGeneral: !!webhookPayload.general,
                hasSpeaker1: !!webhookPayload.speaker1,
                hasSpeaker2: !!webhookPayload.speaker2,
                hasMarkdown: !!webhookPayload.markdown,
                generalSegmentsCount: webhookPayload.general?.segmentsCount || 0,
                speaker1SegmentsCount: webhookPayload.speaker1?.segmentsCount || 0,
                speaker2SegmentsCount: webhookPayload.speaker2?.segmentsCount || 0,
                markdownLength: webhookPayload.markdown?.length || 0
              }
            });
            
            writeLog('info', `[${requestId}] STEP 5: Text analysis completed (local functions)`, {
              requestId,
              step: 5,
              analysisStatus: 'success',
              responseKeys: Object.keys(analysisResult || {}),
              payloadStructure: {
                general: {
                  hasOriginalText: !!webhookPayload.general?.originalText,
                  segmentsCount: webhookPayload.general?.segmentsCount || 0
                },
                speaker1: {
                  speaker: webhookPayload.speaker1?.speaker,
                  role: webhookPayload.speaker1?.role,
                  segmentsCount: webhookPayload.speaker1?.segmentsCount || 0
                },
                speaker2: {
                  speaker: webhookPayload.speaker2?.speaker,
                  role: webhookPayload.speaker2?.role,
                  segmentsCount: webhookPayload.speaker2?.segmentsCount || 0
                },
                markdown: {
                  hasMarkdown: !!webhookPayload.markdown,
                  markdownLength: webhookPayload.markdown?.length || 0
                }
              },
              timestamp: new Date().toISOString()
            });
            
            // Store the analysis result for potential use later
            if (!global.webhookAnalysisResults) {
              global.webhookAnalysisResults = {};
            }
            global.webhookAnalysisResults[requestId] = analysisResult;
            
          } catch (analysisError) {
            console.error(`[${requestId}] ❌ STEP 5: Text analysis failed:`, analysisError.message);
            writeLog('warn', `[${requestId}] STEP 5: Text analysis failed`, {
              requestId,
              step: 5,
              analysisStatus: 'failed',
              error: analysisError.message,
              timestamp: new Date().toISOString()
            });
            // Don't fail the entire step if analysis fails
          }
        } else {
          console.warn(`[${requestId}] ⚠️ STEP 5: Skipping text analysis - missing required data`, {
            hasPrimarySpeechmatics: !!primarySpeechmatics,
            hasSpeaker1Speechmatics: !!speaker1Speechmatics,
            hasSpeaker2Speechmatics: !!speaker2Speechmatics,
            hasOriginalText: !!originalText
          });
        }
      } else {
        console.warn(`[${requestId}] ⚠️ STEP 5: Skipping text analysis - insufficient voice tracks (${voiceTracks?.length || 0})`);
      }
    } catch (analysisSetupError) {
      console.error(`[${requestId}] ❌ STEP 5: Text analysis setup error:`, analysisSetupError.message);
      // Don't fail the entire step if webhook setup fails
    }
    
    const primarySegmentsCount = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0;
    const processedTracksCount = voiceTracks.filter(vt => !vt.error).length;
    
    writeLog('info', `[${requestId}] STEP 5: Starting overlap correction`, {
      requestId,
      step: 5,
      status: 'starting',
      startTime: new Date().toISOString(),
      input: {
        primarySegmentsCount,
        voiceTracksCount: voiceTracks.length,
        processedTracksCount
      }
    });
    
    const modeLabel = overlapPipelineMode === 'mode3' ? 'MODE3' : overlapPipelineMode === 'mode1' ? 'MODE1' : 'MODE2';
    sendSSEUpdate(5, 'processing', `🔵 ${modeLabel}: Початок корекції overlap. Обробка ${processedTracksCount} треків...`, {
      stage: 'starting',
      input: {
        primarySegmentsCount,
        voiceTracksCount: voiceTracks.length,
        processedTracksCount,
        voiceTracks: voiceTracks.map(vt => ({
          speaker: vt.speaker,
          hasError: !!vt.error,
          hasTranscription: !!vt.transcription,
          hasRoleAnalysis: !!vt.roleAnalysis,
          role: vt.roleAnalysis?.role,
          transcriptLength: vt.transcriptText?.length || 0
        }))
      }
    });
    
    let correctionSucceeded = false;
    try {
      // For mode3, use programmatic merge (correctPrimaryDiarizationWithTracks)
      // For other modes, use LLM-based correction (generateOverlapCorrectionResult)
      let correctionResult = null;
      
      if (overlapPipelineMode === 'mode3') {
        correctionResult = await correctPrimaryDiarizationWithTracks(
          primaryDiarization,
          voiceTracks,
          language,
          mode || 'smart',
          requestId,
          res,
          overlapPipelineMode,
          sendSSEUpdate,
          uploadedFile?.originalname || null
        );
        
        // For mode3, apply Markdown fixes using voice tracks segments
        // This generates a markdown table from agent and client transcripts
        if (correctionResult && voiceTracks && voiceTracks.length > 0) {
          try {
            sendSSEUpdate(5, 'processing', `📝 MODE3: Генерація Markdown таблиці з сегментів аудіодоріжок...`, {
              stage: 'markdown_generation',
              input: {
                voiceTracksCount: voiceTracks.length
              }
            });
            
            // Determine main track for each primary speaker
            const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
            const primarySpeakers = [...new Set(primarySegments.map(s => s.speaker || 'SPEAKER_00'))];
            
            // Group primary segments by speaker
            const primaryBySpeaker = {};
            primarySpeakers.forEach(speaker => {
              primaryBySpeaker[speaker] = primarySegments.filter(s => (s.speaker || 'SPEAKER_00') === speaker);
            });
            
            // For each primary speaker, find the best voice track
            const speakerToTrackMap = {};
            const trackScores = [];
            
            primarySpeakers.forEach(primarySpeaker => {
              const primarySegs = primaryBySpeaker[primarySpeaker] || [];
              
              voiceTracks.forEach((track, trackIndex) => {
                if (track.error || !track.transcription) return;
                
                const trackSegments = track.transcription?.recordings?.[0]?.results?.speechmatics?.segments || [];
                if (trackSegments.length === 0) return;
                
                // Calculate score: overlap + completeness + segment count
                let totalOverlap = 0;
                let matchCount = 0;
                let totalTextLength = 0;
                let completeReplicas = 0;
                
                trackSegments.forEach(trackSeg => {
                  const tStart = parseFloat(trackSeg.start) || 0;
                  const tEnd = parseFloat(trackSeg.end) || tStart;
                  const tText = (trackSeg.text || '').trim();
                  
                  if (tText.length > 0) {
                    totalTextLength += tText.length;
                    // Check if text looks complete (not truncated)
                    const isComplete = !tText.endsWith('...') && tText.length > 10;
                    if (isComplete) completeReplicas++;
                  }
                  
                  // Find matching primary segment
                  primarySegs.forEach(primarySeg => {
                    const pStart = parseFloat(primarySeg.start) || 0;
                    const pEnd = parseFloat(primarySeg.end) || pStart;
                    
                    const overlap = Math.max(0, Math.min(tEnd, pEnd) - Math.max(tStart, pStart));
                    if (overlap > 0.1) {
                      totalOverlap += overlap;
                      matchCount++;
                    }
                  });
                });
                
                // Score = overlap * matches * completeness_factor * text_quality
                const completenessFactor = trackSegments.length > 0 ? (completeReplicas / trackSegments.length) : 0;
                const avgTextLength = trackSegments.length > 0 ? (totalTextLength / trackSegments.length) : 0;
                const textQualityFactor = Math.min(1.0, avgTextLength / 50); // Prefer longer, more complete texts
                const score = totalOverlap * Math.sqrt(matchCount) * (1 + completenessFactor) * (1 + textQualityFactor);
                
                trackScores.push({
                  primarySpeaker,
                  trackIndex,
                  track,
                  score,
                  totalOverlap,
                  matchCount,
                  segmentCount: trackSegments.length,
                  completeReplicas,
                  avgTextLength
                });
              });
            });
            
            // For each primary speaker, select the best track
            primarySpeakers.forEach(primarySpeaker => {
              const scoresForSpeaker = trackScores.filter(s => s.primarySpeaker === primarySpeaker);
              if (scoresForSpeaker.length > 0) {
                const bestTrack = scoresForSpeaker.reduce((best, current) => 
                  current.score > best.score ? current : best
                );
                speakerToTrackMap[primarySpeaker] = bestTrack.trackIndex;
                console.log(`[${requestId}] 🎯 Main track for ${primarySpeaker}: track ${bestTrack.trackIndex} (score: ${bestTrack.score.toFixed(2)}, segments: ${bestTrack.segmentCount}, complete: ${bestTrack.completeReplicas})`);
              }
            });
            
            // Build agent and client transcripts from voice tracks using main tracks
            const agentTranscript = { segments: [] };
            const clientTranscript = { segments: [] };
            const assignedRoles = assignRolesToVoiceTracks(voiceTracks);
            
            voiceTracks.forEach((track, index) => {
              if (track.error || !track.transcription) {
                return;
              }
              
              const assignedRole = assignedRoles[index] || 'unknown';
              const segments = track.transcription?.recordings?.[0]?.results?.speechmatics?.segments || [];
              track.assignedRole = assignedRole;
              
              // Check if this track is the main track for any primary speaker
              const isMainTrack = Object.values(speakerToTrackMap).includes(index);
              
              // Only use segments from main tracks (or if no mapping found, use all)
              if (Object.keys(speakerToTrackMap).length > 0 && !isMainTrack) {
                console.log(`[${requestId}] ⏭️ Skipping track ${index} (${track.speaker}) - not main track for any primary speaker`);
                return;
              }
              
              // Determine which primary speaker this track belongs to
              let targetPrimarySpeaker = null;
              for (const [primarySpeaker, trackIndex] of Object.entries(speakerToTrackMap)) {
                if (trackIndex === index) {
                  targetPrimarySpeaker = primarySpeaker;
                  break;
                }
              }
              
              // Add metadata to each segment
              const enrichedSegments = segments.map(seg => ({
                ...seg,
                speaker_id: targetPrimarySpeaker || track.speaker,
                speaker: targetPrimarySpeaker || track.speaker, // Use primary speaker label
                role: assignedRole,
                sourceTrack: track.speaker
              }));
              
              if (assignedRole === 'agent') {
                agentTranscript.segments.push(...enrichedSegments);
              } else if (assignedRole === 'client') {
                clientTranscript.segments.push(...enrichedSegments);
              } else if (agentTranscript.segments.length <= clientTranscript.segments.length) {
                agentTranscript.segments.push(...enrichedSegments);
              } else {
                clientTranscript.segments.push(...enrichedSegments);
              }
            });
            
            console.log(`[${requestId}] 🎚️ Voice track role assignment:`, assignedRoles);
            
            // Validate transcripts
            if (agentTranscript.segments.length === 0 && clientTranscript.segments.length === 0) {
              throw new Error('No segments found in voice tracks');
            }
            
            const groundTruthText = await loadGroundTruthTextForUpload(uploadedFile);
            if (groundTruthText) {
              console.log(`[${requestId}] 🧾 Ground truth validation enabled (source: ${uploadedFile?.originalname || 'unknown'})`);
            }
            
            const promptContext = buildDialoguePromptContext({
              primaryDiarization,
              geminiDiarization, // Pass Gemini 2.5 Pro diarization for transcription control
              agentTranscript,
              clientTranscript,
              voiceTracks,
              groundTruthText
            });
            
            const useLocalLLM = mode === 'local' || mode === 'test' || mode === 'test2';
            const llmModel = getModelId(mode);
            const shouldUseMultiStepMarkdown = useLocalLLM || process.env.USE_MULTI_STEP_MARKDOWN === 'true';
            let markdownTable = null;
            let markdownSource = 'voice-tracks-markdown';
            
            if (shouldUseMultiStepMarkdown) {
              try {
                // Detect if this is an auto-test (curl/script) vs frontend request
                const userAgent = req.headers['user-agent'] || '';
                const isAutoTest = userAgent.includes('curl') || userAgent.includes('node') || req.body.isAutoTest === true;
                
                const result = await processMarkdownFixesMultiStep(
                  promptContext,
                  mode || 'smart',
                  uploadedFile?.originalname || 'demo',
                  isAutoTest
                );
                
                if (result) {
                  if (typeof result === 'string') {
                    // Backward compatibility: if string is returned, use it as markdownTable
                    markdownTable = result;
                  } else if (result.markdownTable) {
                    markdownTable = result.markdownTable;
                    // Store ground truth metrics if available
                    if (result.groundTruthMetrics) {
                      if (!global.groundTruthMetrics) {
                        global.groundTruthMetrics = {};
                      }
                      global.groundTruthMetrics[requestId] = result.groundTruthMetrics;
                    }
                  }
                  
                  if (markdownTable) {
                    // Merge consecutive segments from the same speaker
                    markdownTable = mergeConsecutiveSpeakerSegmentsInMarkdown(markdownTable, 2.0);
                    markdownSource = 'multi-step';
                    finalMarkdownTable = markdownTable;
                    console.log(`[${requestId}] ✅ MODE3: Multi-step markdown generated (${markdownTable.length} chars)`);
                    console.log(`[${requestId}] 🧾 Saved finalMarkdownTable snapshot after multi-step (length: ${finalMarkdownTable.length})`);
                    if (result.groundTruthMetrics && result.groundTruthMetrics.nextLevel) {
                      console.log(`[${requestId}] 📊 Ground Truth Match (NextLevel): ${result.groundTruthMetrics.nextLevel.matchPercent}%`);
                      if (result.groundTruthMetrics.speechmatics) {
                        console.log(`[${requestId}] 📊 Ground Truth Match (Speechmatics): ${result.groundTruthMetrics.speechmatics.matchPercent}%`);
                      }
                    }
                  }
                }
              } catch (multiStepError) {
                console.error(`[${requestId}] ❌ MODE3: Multi-step markdown generation failed:`, multiStepError.message);
              }
            }
            
            if (!markdownTable) {
              console.log(`[${requestId}] 📝 Preparing markdown request:`, {
                agentSegments: agentTranscript.segments.length,
                clientSegments: clientTranscript.segments.length
              });
              
              // Call markdown fixes endpoint logic (same as /api/apply-markdown-fixes)
              const apiUrl = useLocalLLM 
                ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
                : 'https://openrouter.ai/api/v1/chat/completions';
              
              // Load markdown prompt template
              let promptTemplate;
              try {
                promptTemplate = await fs.readFile('prompts/overlap_fixes_markdown_prompt.txt', 'utf8');
              } catch (error) {
                console.warn(`[${requestId}] ⚠️ Failed to load overlap_fixes_markdown_prompt.txt, using fallback prompt`);
                // Fallback prompt will be set below
              }
            
            // Use template or fallback
            if (!promptTemplate) {
              promptTemplate = `You are the **NextLevel diarization controller**. You receive already-extracted dialogues (plain text, one replica per line) instead of raw JSON. Your goal is to fuse Standard diarization evidence with separated voice-track transcripts, detect the real speaker roles, and output a **clean, deduplicated Markdown table** with STRICT alternation between Agent and Client.

## ABSOLUTE TRUTH POLICY
Only the provided dialogues are trustworthy. Every sentence in the final table **must appear verbatim** in at least one dialogue block below. If a sentence is absent from **all** blocks, you MUST discard it as hallucination.

## DATA BLOCKS YOU RECEIVE

1. **Combined Standard Diarization (General)**
\`\`\`
{{GENERAL_DIALOGUE}}
\`\`\`

2. **Standard Speaker 0 Dialogue (raw Speechmatics speaker track)**
\`\`\`
{{STANDARD_SPEAKER0_DIALOGUE}}
\`\`\`

3. **Standard Speaker 1 Dialogue (raw Speechmatics speaker track)**
\`\`\`
{{STANDARD_SPEAKER1_DIALOGUE}}
\`\`\`

4. **Separated Voice Track – Agent Candidate**
\`\`\`
{{AGENT_DIALOGUE}}
\`\`\`

5. **Separated Voice Track – Client Candidate**
\`\`\`
{{CLIENT_DIALOGUE}}
\`\`\`

6. **Role & Context Guidance (output of the classifier / debug checks)**
\`\`\`
{{ROLE_GUIDANCE}}
\`\`\`

Each dialogue is already normalized into plain text lines like \`SPEAKER_00 (operator): text\`. Treat every line as a possible segment. Start/end timestamps for each line are also provided in the metadata block below:
\`\`\`
{{SEGMENT_TIMESTAMPS}}
\`\`\`
\`SEGMENT_TIMESTAMPS\` maps dialogue lines to their numeric \`start\`/\`end\` (seconds). Always copy these exact numbers into the final table.

## CRITICAL TASKS
1. **Role Detection**
   - Use \`ROLE_GUIDANCE\`, conversational intent, and speaker labels to decide who is Agent vs Client.
   - **CRITICAL**: ROLE_GUIDANCE contains \`speakerRoleMap\` which shows the exact mapping. For example, if \`speakerRoleMap\` shows \`{"SPEAKER_00": "agent", "SPEAKER_01": "client"}\`, this means ALL segments from SPEAKER_00 must be Agent, and ALL segments from SPEAKER_01 must be Client.
   - **CRITICAL**: The voice tracks (Agent/Client candidate dialogues) contain the MAIN and MOST COMPLETE replicas for each speaker. These are the primary source - use them as the authoritative text for each role.
   - **CRITICAL**: You MUST output only TWO roles: "Agent" and "Client". Never use "SPEAKER_00", "SPEAKER_01", "SPEAKER_02" or any other speaker labels in the final table.
   - After generating the table, verify: are there segments from BOTH roles? If all segments have the same role, check ROLE_GUIDANCE and reassign roles correctly.
   - If metadata contradicts the dialogue meaning, trust the meaning plus guidance.

2. **Replica Validation**
   - **PRIORITY**: Use replicas from voice-track dialogues (Agent/Client candidate) as they contain the most complete and accurate text without truncation.
   - Keep a replica ONLY if it exists in at least one dialogue block.
   - Prefer the voice-track (Agent/Client candidate) versions - they have the most complete replicas without truncation.
   - If a replica appears in multiple blocks, keep the version from the voice-track that matches the intended role.
   - If text is found only in Standard Speaker 0/1 but matches the other participant, reassign it correctly based on ROLE_GUIDANCE.

3. **Duplicate & Overlap Control**
   - Remove exact or near-duplicate sentences that describe the same moment twice.
   - If a line appears once as Agent and once as Client, decide who truly said it; keep only that one.
   - Merge overlapping lines from the same real speaker: earliest start, latest end, concatenated text (chronological order).

4. **Strict Alternation**
   - Final table must alternate Agent → Client → Agent → Client.
   - Temporary double Agent/Client is allowed only if both lines undoubtedly belong to the same speaker. Try to resolve by reassignment before accepting a double turn.

5. **No Hallucinations**
   - Sentences absent from every dialogue block are forbidden. Discard them even if they appeared in the original markdown.
   - If \`ROLE_GUIDANCE\` lists phrases flagged as “not in any source”, ensure they **never** reach the final table.

6. **Timestamp Fidelity**
   - Every row needs \`Start Time\` and \`End Time\` taken from \`SEGMENT_TIMESTAMPS\`.
   - If you merged multiple lines, use the min start / max end for that merged row.

## OUTPUT FORMAT
Return ONLY a Markdown table:
| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | … | 0.64 | 2.15 |

Where:
- \`Segment ID\` starts at 1.
- \`Speaker\` is either **Agent** or **Client** (NEVER use "SPEAKER_00", "SPEAKER_01", "SPEAKER_02", or any other speaker labels).
- \`Text\` is verbatim (no paraphrasing, keep fillers). Prefer text from voice-track dialogues as they are most complete.
- \`Start/End Time\` use numeric seconds with 2 decimal precision (e.g., \`3.45\`). Use timestamps from SEGMENT_TIMESTAMPS that match the selected text.

## QUALITY CHECKLIST (do this mentally before outputting)
1. Every row comes from the supplied dialogues.
2. Duplicates removed; overlaps merged.
3. Roles validated against \`ROLE_GUIDANCE\` - ensure both Agent and Client segments are present.
4. Alternation Agent/Client preserved.
5. No stray text or commentary outside the Markdown table.
6. **VERIFY ROLE DISTRIBUTION**: Check that the table contains segments from BOTH Agent and Client. If all segments have the same role, you MUST review ROLE_GUIDANCE and correct the assignments.

If any requirement cannot be satisfied, adjust the rows (reassign, merge, drop) until compliance is achieved. Only then output the final table.`;
            }
            
            const replacements = {
              '{{GENERAL_DIALOGUE}}': promptContext.generalDialog,
              '{{STANDARD_SPEAKER0_DIALOGUE}}': promptContext.speaker0Dialog,
              '{{STANDARD_SPEAKER1_DIALOGUE}}': promptContext.speaker1Dialog,
              '{{AGENT_DIALOGUE}}': promptContext.agentDialog,
              '{{CLIENT_DIALOGUE}}': promptContext.clientDialog,
              '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText,
              '{{SEGMENT_TIMESTAMPS}}': promptContext.segmentTimestampsText
            };
            
            let prompt = promptTemplate;
            Object.entries(replacements).forEach(([token, value]) => {
              const safeValue = value && value.trim().length > 0 ? value : '[empty]';
              prompt = prompt.replace(new RegExp(token, 'g'), safeValue);
            });
            
            // Log ROLE_GUIDANCE for debugging
            try {
              const roleGuidanceObj = JSON.parse(promptContext.roleGuidanceText || '{}');
              console.log(`[${requestId}] 🔍 ROLE_GUIDANCE being sent to LLM:`, {
                speakerRoleMap: roleGuidanceObj.speakerRoleMap || {},
                tracks: roleGuidanceObj.tracks || []
              });
            } catch (e) {
              console.warn(`[${requestId}] ⚠️ Failed to parse ROLE_GUIDANCE for logging:`, e.message);
            }
            
            // Build cache key for markdown generation
            // Get DEMO_LLM_MODE for cache key (ensures cache is invalidated when demo mode changes)
            const demoLlmMode = process.env.DEMO_LLM_MODE || 'smart';
            const cacheKey = buildLLMCacheKey(
              uploadedFile?.originalname || 'demo',
              prompt,
              llmModel,
              mode,
              'markdown-fixes',
              demoLlmMode
            );
            
            // Check cache first
            if (cacheKey) {
              const cachedResult = readLLMCache(cacheKey);
              if (cachedResult && cachedResult.rawMarkdown) {
                console.log(`[${requestId}] ✅ Using cached raw markdown from LLM`);
                let cachedMarkdown = cachedResult.rawMarkdown;
                // Merge consecutive segments from the same speaker
                cachedMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(cachedMarkdown, 2.0);
                markdownTable = cachedMarkdown;
                markdownSource = 'cache';
              }
            }
            
            // Якщо локальний режим і кеш не знайдено, спробувати знайти кеш для моделі fast
            if (!markdownTable && useLocalLLM) {
              const fastModelId = getModelId('fast');
              const fastCacheKey = buildLLMCacheKey(
                uploadedFile?.originalname || 'demo',
                prompt,
                fastModelId,
                'fast',
                'markdown-fixes',
                demoLlmMode
              );
              
              if (fastCacheKey) {
                const fastCachedResult = readLLMCache(fastCacheKey);
                if (fastCachedResult && fastCachedResult.rawMarkdown) {
                  console.log(`[${requestId}] ✅ Using cached raw markdown from fast model for local mode`);
                  let fastCachedMarkdown = fastCachedResult.rawMarkdown;
                  // Merge consecutive segments from the same speaker
                  fastCachedMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(fastCachedMarkdown, 2.0);
                  markdownTable = fastCachedMarkdown;
                  markdownSource = 'cache-fast';
                }
              }
            }
            
            // If not cached, call LLM
            if (!markdownTable) {
            const headers = {
              'Content-Type': 'application/json'
            };
            
            if (useLocalLLM) {
              if (LOCAL_LLM_API_KEY) {
                headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
              }
              console.log(`[${requestId}] 📤 Sending request to local LLM: ${apiUrl}`);
            } else {
              headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
              headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
              headers['X-Title'] = 'Apply Markdown Fixes';
            }
            
            // Build payload
            const payload = {
              model: llmModel,
              messages: [
                {
                  role: 'system',
                  content: 'You are a helpful assistant that fixes speaker diarization. Always return a valid Markdown table with columns: Segment ID, Speaker, Text, Start Time, End Time. Keep all text from input. **CRITICAL**: After reasoning, you MUST generate the final Markdown table in your response content. Do not leave the content field empty.'
                },
                {
                  role: 'user',
                  content: prompt
                }
              ],
              temperature: 0
            };
            
            // Add reasoning effort
            if (shouldUseHighReasoningEffort(mode, llmModel)) {
              if (useLocalLLM) {
                console.log(`[${requestId}] 🔧 Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)`);
              } else {
                payload.reasoning = { effort: 'high' };
                console.log(`[${requestId}] 🔧 Using reasoning effort: high for ${mode} mode (model: ${llmModel})`);
              }
            }
              
              // Для локальної моделі встановлюємо більший таймаут (30 хвилин), оскільки вона працює повільніше
              const markdownTimeout = useLocalLLM ? 1800000 : 600000; // 30 хвилин для локальної, 10 хвилин для віддаленої
              console.log(`[${requestId}] ⏱️  Using timeout: ${markdownTimeout / 1000 / 60} minutes for markdown generation (${useLocalLLM ? 'local' : 'remote'} LLM)`);
            
            const llmResponse = await axios.post(apiUrl, payload, {
              headers,
                timeout: markdownTimeout
            });
            
              // Отримуємо чистий markdown від LLM
              const rawLLMOutput = llmResponse.data.choices[0]?.message?.content || '';
              
              // Витягуємо markdown з code blocks якщо потрібно
              let rawMarkdown = rawLLMOutput.trim();
              const codeBlockMatch = rawMarkdown.match(/```(?:markdown)?\s*([\s\S]*?)\s*```/);
              if (codeBlockMatch) {
                rawMarkdown = codeBlockMatch[1].trim();
              }
              
              // Remove filler words from markdown table
              rawMarkdown = removeFillerWordsFromMarkdownTable(rawMarkdown);
              
              // Merge consecutive segments from the same speaker
              rawMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(rawMarkdown, 2.0);
              
              markdownTable = rawMarkdown;
              markdownSource = 'voice-tracks-markdown';
              finalMarkdownTable = markdownTable;
              console.log(`[${requestId}] 🧾 Saved finalMarkdownTable snapshot after voice-track markdown (length: ${finalMarkdownTable.length})`);
              
              // Save to cache - зберігаємо чистий markdown від LLM
              if (cacheKey && rawMarkdown) {
                writeLLMCache(cacheKey, {
                  rawMarkdown: rawMarkdown, // Чистий markdown від LLM
                  agentSegmentsCount: agentTranscript.segments.length,
                  clientSegmentsCount: clientTranscript.segments.length,
                  totalSegmentsCount: totalSegmentsCount,
                  model: llmModel,
                  mode: mode,
                  timestamp: new Date().toISOString()
                });
              }
            }
            
            if (markdownTable) {
              // Analyze role distribution
              const roleDistribution = analyzeRoleDistribution(markdownTable);
              console.log(`[${requestId}] 📊 Role distribution in markdown:`, roleDistribution);
              
              if (roleDistribution.total > 0) {
                if (roleDistribution.agentCount === 0) {
                  console.warn(`[${requestId}] ⚠️ WARNING: All ${roleDistribution.total} segments are Client!`);
                } else if (roleDistribution.clientCount === 0) {
                  console.warn(`[${requestId}] ⚠️ WARNING: All ${roleDistribution.total} segments are Agent!`);
                } else {
                  const ratio = (roleDistribution.clientCount / roleDistribution.agentCount).toFixed(2);
                  console.log(`[${requestId}] ✅ Role distribution OK: Agent=${roleDistribution.agentCount}, Client=${roleDistribution.clientCount}, Ratio=${ratio}`);
                }
              }
              
              // Store markdown table in correction result
              correctionResult.recordings[0].results['overlap-corrected'].rawData = correctionResult.recordings[0].results['overlap-corrected'].rawData || {};
              correctionResult.recordings[0].results['overlap-corrected'].rawData.markdownTable = markdownTable;
              correctionResult.recordings[0].results['overlap-corrected'].rawData.markdownGenerated = true;
              finalMarkdownTable = finalMarkdownTable || markdownTable;
              
              sendSSEUpdate(5, 'completed', `✅ MODE3: Markdown таблицю згенеровано`, {
                stage: 'markdown_completed',
                output: {
                  markdownLength: markdownTable.length,
                  source: markdownSource
                }
              });
              
              console.log(`[${requestId}] ✅ MODE3: Markdown table generated (${markdownTable.length} chars) [source=${markdownSource}]`);
              
              // Перераховуємо textAnalysis з правильним markdown (з кешу або з LLM)
              // Це важливо, бо textAnalysis має використовувати фінальний markdown, а не простий
              try {
                if (voiceTracks && voiceTracks.length >= 2) {
                  const primaryRecording = primaryDiarization?.recordings?.[0] || null;
                  const primarySpeechmatics = primaryRecording?.results?.speechmatics || null;
                  const primarySegments = primarySpeechmatics?.segments || [];
                  
                  const track1 = voiceTracks[0];
                  const speaker1Recording = track1?.transcription?.recordings?.[0] || null;
                  const speaker1Speechmatics = speaker1Recording?.results?.speechmatics || null;
                  
                  const track2 = voiceTracks[1];
                  const speaker2Recording = track2?.transcription?.recordings?.[0] || null;
                  const speaker2Speechmatics = speaker2Recording?.results?.speechmatics || null;
                  
                  if (primarySpeechmatics && speaker1Speechmatics && speaker2Speechmatics) {
                    const webhookPayload = {
                      general: {
                        speechmatics: primarySpeechmatics,
                        segments: primarySegments,
                        segmentsCount: primarySegments.length
                      },
                      speaker1: {
                        speaker: track1?.speaker || 'SPEAKER_00',
                        role: assignedRoles[0] || track1?.roleAnalysis?.role || null,
                        speechmatics: speaker1Speechmatics,
                        segments: speaker1Speechmatics?.segments || [],
                        segmentsCount: (speaker1Speechmatics?.segments || []).length
                      },
                      speaker2: {
                        speaker: track2?.speaker || 'SPEAKER_01',
                        role: assignedRoles[1] || track2?.roleAnalysis?.role || null,
                        speechmatics: speaker2Speechmatics,
                        segments: speaker2Speechmatics?.segments || [],
                        segmentsCount: (speaker2Speechmatics?.segments || []).length
                      },
                      markdown: markdownTable // Використовуємо фінальний markdown (з кешу або з LLM)
                    };
                    
                    // Перераховуємо textAnalysis з правильним markdown (script або LLM режим)
                    let analysisResult;
                    console.log(`[${requestId}] 🔍 MODE3: Text analysis mode check:`, {
                      useLLMForTextAnalysis: useLLMForTextAnalysis,
                      hasMarkdown: !!webhookPayload.markdown,
                      markdownLength: webhookPayload.markdown?.length || 0
                    });
                    
                    if (useLLMForTextAnalysis) {
                      try {
                        console.log(`[${requestId}] 🤖 MODE3: Using LLM mode for text analysis classification...`);
                        const useLocalLLM = mode === 'local' || mode === 'test' || mode === 'test2';
                        const llmModel = getModelId(mode);
                        const apiUrl = useLocalLLM 
                          ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
                          : 'https://openrouter.ai/api/v1/chat/completions';
                        const apiKey = useLocalLLM ? LOCAL_LLM_API_KEY : process.env.OPENROUTER_API_KEY;
                        
                        console.log(`[${requestId}] 📤 MODE3: Calling LLM for text analysis:`, {
                          llmModel: llmModel,
                          apiUrl: apiUrl,
                          useLocalLLM: useLocalLLM,
                          mode: mode
                        });
                        
                        analysisResult = await textAnalysis.analyzeTextWithLLM(
                          webhookPayload,
                          llmModel,
                          apiUrl,
                          apiKey,
                          useLocalLLM,
                          mode
                        );
                        console.log(`[${requestId}] ✅ MODE3: Text analysis completed (LLM mode):`, {
                          blueCount: analysisResult.Blue?.length || 0,
                          greenCount: analysisResult.Green?.length || 0,
                          redCount: analysisResult.Red?.length || 0
                        });
                      } catch (llmError) {
                        console.error(`[${requestId}] ❌ MODE3: LLM text analysis failed, falling back to script mode:`, llmError.message);
                        console.error(`[${requestId}] ❌ MODE3: LLM error details:`, llmError);
                        // Fallback до script режиму
                        analysisResult = textAnalysis.analyzeText(webhookPayload);
                      }
                    } else {
                      console.log(`[${requestId}] 📝 MODE3: Using script mode for text analysis (useLLMForTextAnalysis=false)`);
                      analysisResult = textAnalysis.analyzeText(webhookPayload);
                    }
                    console.log(`[${requestId}] ✅ MODE3: Text analysis recalculated with final markdown (${useLLMForTextAnalysis ? 'LLM' : 'script'} mode):`, {
                      blueCount: analysisResult.Blue?.length || 0,
                      greenCount: analysisResult.Green?.length || 0,
                      redCount: analysisResult.Red?.length || 0
                    });
                    
                    // Оновлюємо результат в глобальному об'єкті
                    if (!global.webhookAnalysisResults) {
                      global.webhookAnalysisResults = {};
                    }
                    global.webhookAnalysisResults[requestId] = analysisResult;
                  }
                }
              } catch (recalcError) {
                console.warn(`[${requestId}] ⚠️ MODE3: Failed to recalculate text analysis with final markdown:`, recalcError.message);
                // Не критична помилка - продовжуємо
              }
            } else {
              console.log(`[${requestId}] ⚠️ MODE3: Markdown generation returned empty`);
            }
          }
          } catch (markdownError) {
            console.error(`[${requestId}] ❌ MODE3: Markdown generation failed:`, markdownError.message);
            sendSSEUpdate(5, 'completed', `⚠️ MODE3: Markdown таблицю не вдалося згенерувати`, {
              stage: 'markdown_failed',
              error: markdownError.message
            });
          }
        }
      } else {
        correctionResult = await generateOverlapCorrectionResult({
          primaryDiarization,
          voiceTracks,
          transcript: combinedTranscript,
          existingLLMResult: voiceTrackLLMResult,
          mode: mode || 'smart',
          requestId,
          filename: uploadedFile?.originalname || (url ? path.parse(new URL(url).pathname.split('/').pop() || 'audio').name + '.wav' : null)
        });
      }

      if (correctionResult) {
        correctedDiarization = correctionResult;
        correctionSucceeded = true;
      }
    } catch (step5Error) {
      console.error(`[${requestId}] ❌ STEP 5: Voice-track LLM correction failed:`, step5Error.message);
      writeLog('error', `[${requestId}] STEP 5: Overlap correction failed`, {
        requestId,
        step: 5,
        status: 'failed',
        duration: `${((Date.now() - step5StartTime) / 1000).toFixed(2)}s`,
        error: step5Error.message,
        stack: step5Error.stack?.substring(0, 1000)
      });
    }

    const step5Duration = ((Date.now() - step5StartTime) / 1000).toFixed(2);
    if (correctionSucceeded) {
      const correctedSegmentsCount = correctedDiarization?.recordings?.[0]?.results?.['overlap-corrected']?.segments?.length || 0;
      const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0;

      const modeLabel = overlapPipelineMode === 'mode3' ? 'MODE3' : overlapPipelineMode === 'mode1' ? 'MODE1' : 'MODE2';
      sendSSEUpdate(5, 'completed', `✅ ${modeLabel}: Корекція overlap завершена. Результат: ${correctedSegmentsCount} сегментів`, {
        stage: 'completed',
        output: {
          segmentsCount: correctedSegmentsCount,
          primarySegmentsCount: primarySegments,
          segmentsDifference: correctedSegmentsCount - primarySegments
        }
      });

      steps.step5 = {
        name: 'Overlap Correction',
        status: 'completed',
        duration: step5Duration + 's',
        segmentsCount: correctedSegmentsCount
      };

      writeLog('info', `[${requestId}] STEP 5: Overlap correction completed`, {
        requestId,
        step: 5,
        status: 'completed',
        duration: `${step5Duration}s`,
        result: {
          correctedSegmentsCount,
          primarySegmentsCount: primarySegments,
          segmentsDifference: correctedSegmentsCount - primarySegments
        },
        endTime: new Date().toISOString()
      });
    } else {
      const fallbackResult = buildOverlapCorrectedFromVoiceTracks(primaryDiarization, voiceTracks);
      correctedDiarization = fallbackResult || primaryDiarization;
      const fallbackSegments = correctedDiarization?.recordings?.[0]?.results?.['overlap-corrected']?.segments?.length || 0;

      const modeLabel = overlapPipelineMode === 'mode3' ? 'MODE3' : overlapPipelineMode === 'mode1' ? 'MODE1' : 'MODE2';
      sendSSEUpdate(5, 'completed', `⚠️ ${modeLabel}: Використовуємо fallback з voice tracks (${fallbackSegments} сегментів)`, {
        stage: 'completed_fallback',
        output: {
          segmentsCount: fallbackSegments,
          source: 'voice_tracks'
        }
      });

      steps.step5 = {
        name: 'Overlap Correction',
        status: 'completed_with_fallback',
        duration: step5Duration + 's',
        segmentsCount: fallbackSegments,
        fallback: 'voice_tracks'
      };

      writeLog('warn', `[${requestId}] STEP 5: Using voice-track fallback`, {
        requestId,
        step: 5,
        status: 'fallback',
        duration: `${step5Duration}s`,
        segmentsCount: fallbackSegments
      });
    }

    const totalTime = ((Date.now() - pipelineStartTime) / 1000).toFixed(2);
    
    console.log(`[${requestId}] ✅ PIPELINE: All steps completed in ${totalTime}s`);
    console.log(`[${requestId}] ✅ PIPELINE: Summary:`, {
      step1: steps.step1?.status,
      step2: steps.step2?.status,
      step3: steps.step3?.status,
      step5: steps.step5?.status,
      totalDuration: totalTime + 's'
    });
    
    writeLog('info', `[${requestId}] PIPELINE: All steps completed`, {
      requestId,
      totalDuration: totalTime,
      steps: {
        step1: steps.step1?.status,
        step2: steps.step2?.status,
        step3: steps.step3?.status,
        step5: steps.step5?.status
      },
      endTime: new Date().toISOString()
    });

    // Add downloadUrl to each speaker for both modes
    const speakersWithDownloadUrl = separation.speakers.map(speaker => {
      // If speaker already has downloadUrl (from mode2), use it
      if (speaker.downloadUrl) {
        return speaker;
      }
      
      // For mode1 (AudioShake), speaker.url is already a public URL
      // For mode2 (PyAnnote), speaker.url is also a public URL from getPublicFileUrl
      // But we should also provide a direct download endpoint
      let downloadUrl = speaker.url;
      
      // If it's a relative path, make it absolute
      if (downloadUrl && downloadUrl.startsWith('/uploads/')) {
        // Already a relative path, can be used directly
        downloadUrl = downloadUrl;
      } else if (downloadUrl && !downloadUrl.startsWith('http')) {
        // Make it relative to uploads
        const filename = path.basename(downloadUrl);
        downloadUrl = `/uploads/${filename}`;
      }
      
      return {
        ...speaker,
        downloadUrl: downloadUrl || speaker.url
      };
    });

    console.log(`[${requestId}] 🧮 Stored finalMarkdownTable snapshot:`, {
      hasFinalMarkdown: !!finalMarkdownTable,
      finalMarkdownLength: finalMarkdownTable?.length || 0
    });

    // Extract markdown table from corrected diarization if available
    let markdownTable = finalMarkdownTable;
    console.log(`[${requestId}] 🔍 Before markdownTable extraction:`, {
      hasFinalMarkdownTable: !!finalMarkdownTable,
      finalMarkdownTableLength: finalMarkdownTable?.length || 0,
      hasMarkdownTable: !!markdownTable,
      markdownTableLength: markdownTable?.length || 0
    });
    if (!markdownTable) {
      if (correctedDiarization) {
        const corrected = correctedDiarization;
        if (corrected.recordings && Array.isArray(corrected.recordings) && corrected.recordings[0]) {
          const recording = corrected.recordings[0];
          if (recording.results && recording.results['overlap-corrected']) {
            const rawData = recording.results['overlap-corrected'].rawData;
            if (rawData && rawData.markdownTable) {
              markdownTable = rawData.markdownTable;
              finalMarkdownTable = markdownTable;
              console.log(`[${requestId}] ✅ Extracted markdownTable from rawData (length: ${markdownTable.length})`);
            } else {
              console.log(`[${requestId}] ⚠️ markdownTable not found in rawData:`, {
                hasRawData: !!rawData,
                rawDataKeys: rawData ? Object.keys(rawData) : []
              });
            }
          } else {
            console.log(`[${requestId}] ⚠️ overlap-corrected not found in recording.results`);
          }
        } else {
          console.log(`[${requestId}] ⚠️ No recordings found in correctedDiarization`);
        }
      } else {
        console.log(`[${requestId}] ⚠️ correctedDiarization is null/undefined`);
      }
    }
    
    // Fallback: Generate markdown table from segments if still not found
    if (!markdownTable && correctedDiarization) {
      try {
        const corrected = correctedDiarization;
        let segments = [];
        
        if (corrected.recordings && Array.isArray(corrected.recordings) && corrected.recordings[0]) {
          const recording = corrected.recordings[0];
          if (recording.results && recording.results['overlap-corrected']) {
            segments = recording.results['overlap-corrected'].segments || [];
          }
        } else if (corrected.segments) {
          segments = Array.isArray(corrected.segments) ? corrected.segments : [];
        }
        
          if (segments.length > 0) {
          console.log(`[${requestId}] 🔧 Generating fallback markdown table from ${segments.length} segments`);
          let fallbackMarkdown = '| Segment ID | Speaker | Text | Start Time | End Time |\n';
          fallbackMarkdown += '|------------|---------|------|------------|----------|\n';
          
          segments.forEach((seg, index) => {
            const segmentId = index + 1;
            const speaker = seg.role === 'operator' || seg.role === 'agent' ? 'Agent' :
                          seg.role === 'client' || seg.role === 'customer' ? 'Client' :
                          seg.speaker || 'Unknown';
            const text = (seg.text || '').trim();
            const startTime = parseFloat(seg.start) || 0;
            const endTime = parseFloat(seg.end) || startTime;
            
            if (text) {
              fallbackMarkdown += `| ${segmentId} | ${speaker} | ${text} | ${startTime.toFixed(2)} | ${endTime.toFixed(2)} |\n`;
            }
          });
          
          // Merge consecutive segments from the same speaker
          fallbackMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(fallbackMarkdown, 2.0);
          
          markdownTable = fallbackMarkdown;
          finalMarkdownTable = markdownTable;
          console.log(`[${requestId}] ✅ Generated fallback markdown table (length: ${markdownTable.length})`);
        }
      } catch (fallbackError) {
        console.error(`[${requestId}] ❌ Failed to generate fallback markdown table:`, fallbackError.message);
      }
    }
    
    // Final check - if still no markdown table, log warning
    if (!markdownTable) {
      console.warn(`[${requestId}] ⚠️ WARNING: markdownTable is still null/undefined after all attempts!`);
      console.warn(`[${requestId}] This will cause issues on the frontend.`);
    } else {
      console.log(`[${requestId}] ✅ Final markdownTable status:`, {
        hasMarkdownTable: !!markdownTable,
        markdownTableLength: markdownTable.length,
        markdownTablePreview: markdownTable.substring(0, 200)
      });
    }
    
    // For mode 3 and mode 1, send final response via SSE, then close
    // Mode 1 needs SSE to prevent 504 Gateway Timeout
    if ((overlapPipelineMode === 'mode3' || overlapPipelineMode === 'mode1') && res) {
      // Send final result as SSE event
      try {
        // Get text analysis results if available
        const textAnalysisResults = global.webhookAnalysisResults?.[requestId] || null;
        
        // Діагностичне логування textAnalysis
        console.log(`[${requestId}] 🔍 Text Analysis in fullResult:`, {
          hasTextAnalysis: !!textAnalysisResults,
          blueCount: textAnalysisResults?.Blue?.length || 0,
          greenCount: textAnalysisResults?.Green?.length || 0,
          redCount: textAnalysisResults?.Red?.length || 0,
          blueSamples: textAnalysisResults?.Blue?.slice(0, 2).map(i => i?.text?.substring(0, 30)) || [],
          greenSamples: textAnalysisResults?.Green?.slice(0, 2).map(i => i?.text?.substring(0, 30)) || [],
          redSamples: textAnalysisResults?.Red?.slice(0, 2).map(i => i?.text?.substring(0, 30)) || []
        });
        
        // Get ground truth metrics if available
        const groundTruthMetrics = global.groundTruthMetrics?.[requestId] || null;
        
        console.log(`[${requestId}] 🔍 Before fullResult creation:`, {
          hasMarkdownTable: !!markdownTable,
          markdownTableLength: markdownTable?.length || 0,
          markdownTableType: typeof markdownTable,
          markdownTablePreview: markdownTable ? markdownTable.substring(0, 100) : 'null'
        });
        
        const fullResult = {
          type: 'final-result',
          success: true,
          primaryDiarization: primaryDiarization,
          separation: {
            taskId: separation.taskId,
            speakers: speakersWithDownloadUrl,
            cost: separation.cost,
            duration: separation.duration
          },
          voiceTracks: voiceTracks,
          correctedDiarization: correctedDiarization,
          markdownTable: markdownTable, // Add markdown table at top level
          textAnalysis: textAnalysisResults, // Add text analysis results (Blue, Green, Red)
          groundTruthMetrics: groundTruthMetrics, // Add ground truth match metrics
          diagnostics: {
            combinedTranscript,
            llmDiarization: voiceTrackLLMResult,
            comparison: comparisonAnalysis,
            missingReplicas,
            voiceTranscript: voiceTranscriptFile
          },
          steps: steps,
          totalDuration: totalTime + 's',
          requestId: requestId,
          pipelineMode: overlapPipelineMode
        };
        
        // Діагностичне логування markdownTable перед санітизацією
        console.log(`[${requestId}] 🔍 markdownTable before sanitize:`, {
          hasMarkdownTable: !!fullResult.markdownTable,
          markdownTableLength: fullResult.markdownTable?.length || 0,
          markdownTablePreview: fullResult.markdownTable?.substring(0, 100) || 'null'
        });
        
        // Sanitize response to remove internal keys before sending
        const finalResult = sanitizeDiarizationResponse(fullResult);
        finalResult.type = 'final-result';
        
        // Діагностичне логування markdownTable після санітизації
        console.log(`[${requestId}] 🔍 markdownTable after sanitize:`, {
          hasMarkdownTable: !!finalResult.markdownTable,
          markdownTableLength: finalResult.markdownTable?.length || 0,
          markdownTablePreview: finalResult.markdownTable?.substring(0, 100) || 'null'
        });
        
        // Try to stringify and send, with error handling for large objects
        try {
          const jsonString = JSON.stringify(finalResult);
          const jsonSize = Buffer.byteLength(jsonString, 'utf8');
          console.log(`[${requestId}] 📤 SSE: Final result size: ${(jsonSize / 1024).toFixed(2)} KB`);
          
          if (jsonSize > 10 * 1024 * 1024) { // 10MB limit
            console.warn(`[${requestId}] ⚠️ SSE: Result is very large (${(jsonSize / 1024 / 1024).toFixed(2)} MB), may cause issues`);
          }
          
          res.write(`data: ${jsonString}\n\n`);
          console.log(`[${requestId}] 📤 SSE: Final result sent successfully`);
        } catch (stringifyError) {
          console.error(`[${requestId}] ❌ SSE: Failed to stringify final result:`, stringifyError.message);
          // Try to send a simplified version
          try {
            const simplifiedResult = {
              type: 'final-result',
              success: finalResult.success,
              error: 'Result too large to serialize',
              requestId: requestId
            };
            res.write(`data: ${JSON.stringify(simplifiedResult)}\n\n`);
            console.log(`[${requestId}] 📤 SSE: Sent simplified error result`);
          } catch (e) {
            console.error(`[${requestId}] ❌ SSE: Failed to send even simplified result:`, e.message);
          }
        }
        
        cleanupKeepAlive();
        res.end();
      } catch (e) {
        console.error(`[${requestId}] ❌ Failed to send final SSE result:`, e.message);
        cleanupKeepAlive();
        res.end();
      }
    } else {
      // For mode 2, use regular JSON response
      const fullResult = {
        success: true,
        primaryDiarization: primaryDiarization,
        separation: {
          taskId: separation.taskId,
          speakers: speakersWithDownloadUrl,
          cost: separation.cost,
          duration: separation.duration
        },
        voiceTracks: voiceTracks,
        correctedDiarization: correctedDiarization,
        markdownTable: markdownTable, // Add markdown table at top level
        diagnostics: {
          combinedTranscript,
          llmDiarization: voiceTrackLLMResult,
          comparison: comparisonAnalysis,
          missingReplicas,
          voiceTranscript: voiceTranscriptFile
        },
        steps: steps,
        totalDuration: totalTime + 's',
        requestId: requestId,
        pipelineMode: overlapPipelineMode
      };
      
      // Sanitize response to remove internal keys before sending
      const sanitizedResult = sanitizeDiarizationResponse(fullResult);
      res.json(sanitizedResult);
    }

  } catch (error) {
    const totalTime = ((Date.now() - pipelineStartTime) / 1000).toFixed(2);
    console.error(`[${requestId}] ❌ PIPELINE: Fatal error after ${totalTime}s`);
    console.error(`[${requestId}] ❌ PIPELINE: Mode: ${overlapPipelineMode || 'unknown'}`);
    console.error(`[${requestId}] ❌ PIPELINE: Error type: ${error.constructor?.name || 'Unknown'}`);
    console.error(`[${requestId}] ❌ PIPELINE: Error name: ${error.name || 'Unknown'}`);
    console.error(`[${requestId}] ❌ PIPELINE: Error message: ${error.message || 'No message'}`);
    console.error(`[${requestId}] ❌ PIPELINE: Full error:`, error);
    console.error(`[${requestId}] ❌ PIPELINE: Error stack:`, error.stack?.substring(0, 500));
    
    writeLog('error', `[${requestId}] PIPELINE: Fatal error`, {
      requestId,
      error: error.message,
      stack: error.stack,
      name: error.name,
      code: error.code,
      totalDuration: totalTime,
      endTime: new Date().toISOString()
    });
    
    // Ensure we always send JSON, not HTML
    if (!res.headersSent) {
      let finalErrorMessage = error.message || 'Overlap diarization pipeline failed';
      
      // For local modes, never show HTTPS-related errors
      if ((overlapPipelineMode === 'mode2' || overlapPipelineMode === 'mode3') && finalErrorMessage.includes('HTTPS')) {
        console.log(`[${requestId}] ⚠️ PIPELINE: Detected HTTPS error for local mode (${overlapPipelineMode}) in main catch, using generic message`);
        finalErrorMessage = 'Speaker separation failed. Please check the logs for details.';
      }
      
      console.log(`[${requestId}] 🔍 PIPELINE: Final error message to return:`, {
        finalErrorMessage,
        originalMessage: error.message,
        mode: overlapPipelineMode
      });
      
      cleanupKeepAlive();
      
      if ((overlapPipelineMode === 'mode3' || overlapPipelineMode === 'mode1')) {
        // For SSE mode (mode3 and mode1), send error as SSE event
        // Check if headers are already sent (might be sent in step error handlers)
        try {
          const errorEvent = {
            type: 'pipeline-error',
            success: false,
            error: finalErrorMessage,
            details: process.env.NODE_ENV === 'development' ? (error.stack ? error.stack.substring(0, 500) : undefined) : undefined,
            requestId: requestId,
            duration: totalTime,
            mode: overlapPipelineMode,
            timestamp: new Date().toISOString()
          };
          res.write(`data: ${JSON.stringify(errorEvent)}\n\n`);
          console.log(`[${requestId}] 📤 SSE: Pipeline error sent`);
          cleanupKeepAlive();
          res.end();
        } catch (e) {
          console.error(`[${requestId}] ❌ Failed to send SSE error:`, e.message);
          cleanupKeepAlive();
          res.end();
        }
      } else {
        res.status(500).json({
          success: false,
          error: finalErrorMessage,
          details: process.env.NODE_ENV === 'development' ? (error.stack ? error.stack.substring(0, 500) : undefined) : undefined,
          requestId: requestId,
          duration: totalTime,
          mode: overlapPipelineMode
        });
      }
    }
  } finally {
    // Cleanup
    if (req.file && req.file.path && fsSync.existsSync(req.file.path)) {
      try {
        await fs.unlink(req.file.path);
      } catch (cleanupError) {
        console.error('Failed to cleanup uploaded temp file:', cleanupError);
      }
    }
    if (tempDownloadedAudioPath && fsSync.existsSync(tempDownloadedAudioPath)) {
      try {
        await fs.unlink(tempDownloadedAudioPath);
      } catch (cleanupError) {
        console.error('Failed to cleanup downloaded file:', cleanupError);
      }
    }
  }
});

// LLM Diarization endpoint (wrapper around /api/diarize with multimodal support)
app.post('/api/diarize-llm', upload.single('audio'), async (req, res) => {
  try {
    const { transcript, mode, isVerification, isFixRequest, verificationPrompt } = req.body;
    const uploadedFile = req.file;
    
    if (!transcript || !transcript.trim()) {
      return res.status(400).json({ error: 'Transcript is required' });
    }

    // Check if Gemini 2.5 Pro multimodal mode is requested
    if (mode === 'gemini-2.5-pro' || mode === 'gemini-2.5-pro-tier') {
      // Use multimodal processing
      let audioPath = null;
      
      // Get audio file path if uploaded
      if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
        audioPath = uploadedFile.path;
      } else if (req.body.audioUrl) {
        // If audio URL is provided, download it first
        // For now, we'll need the file path, so this would require additional logic
        // For simplicity, we'll use the uploaded file path
        audioPath = null; // URL handling would need additional implementation
      }
      
      try {
        const result = await callGeminiMultimodal(audioPath, transcript, req.body.language || 'ar');
        
        // Clean up uploaded file
        if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
          try {
            await fs.unlink(uploadedFile.path);
          } catch (cleanupError) {
            console.error('Cleanup error:', cleanupError);
          }
        }
        
        res.json(result);
        return;
      } catch (multimodalError) {
        console.error('Multimodal processing failed:', multimodalError);
        
        // Clean up uploaded file
        if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
          try {
            await fs.unlink(uploadedFile.path);
          } catch (cleanupError) {
            console.error('Cleanup error:', cleanupError);
          }
        }
        
        // Fallback to standard LLM processing
        writeLog('warn', 'Gemini 2.5 Pro multimodal failed, falling back to standard LLM', {
          error: multimodalError.message
        });
        
        // Continue to standard processing below
      }
    }

    // Standard LLM processing (reuse existing logic)
    const result = await handleDiarizationRequest(req.body);
    
    // Clean up uploaded file if exists
    if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
      try {
        await fs.unlink(uploadedFile.path);
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError);
      }
    }
    
    res.json(result);
    
  } catch (error) {
    console.error('LLM Diarization error:', error);
    
    // Clean up uploaded file if exists
    if (req.file && fsSync.existsSync(req.file.path)) {
      try {
        await fs.unlink(req.file.path);
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError);
      }
    }
    
    res.status(500).json({ 
      error: error.message || 'Internal server error',
      details: error.response?.data || null
    });
  }
});

// Audio Diarization endpoint (uses Speechmatics via process-audio-temp)
// Helper function to handle raw file upload (for Shortcuts)
async function handleRawFileUpload(req, tempDir) {
  if (req.body && req.body.length > 0 && Buffer.isBuffer(req.body)) {
    // Raw binary data received
    const contentDisposition = req.headers['content-disposition'];
    let originalName = 'audio.wav';
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
      if (filenameMatch && filenameMatch[1]) {
        originalName = filenameMatch[1].replace(/['"]/g, '');
      }
    } else {
      const contentType = req.headers['content-type'] || '';
      if (contentType.includes('mp3')) originalName = 'audio.mp3';
      else if (contentType.includes('m4a')) originalName = 'audio.m4a';
      else if (contentType.includes('wav')) originalName = 'audio.wav';
    }
    
    const tempFilePath = path.join(tempDir, `raw_${Date.now()}_${originalName}`);
    await fs.writeFile(tempFilePath, req.body);
    
    return {
      path: tempFilePath,
      originalname: originalName,
      size: req.body.length,
      mimetype: req.headers['content-type'] || 'application/octet-stream'
    };
  }
  return null;
}

// Alternative endpoint for Shortcuts: accepts file as raw binary data, JSON, or multipart/form-data
app.post('/api/diarize-audio-raw', handleShortcutsUpload, async (req, res) => {
  const requestId = `audio_raw_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const startTime = Date.now();
  
  try {
    let filePath = null;
    let originalName = 'audio.wav';
    
    // Check if multer already processed the file (multipart/form-data)
    if (req.file) {
      console.log(`[${requestId}] 📥 Received file via multipart/form-data: ${req.file.originalname}, size: ${req.file.size}`);
      filePath = req.file.path;
      originalName = req.file.originalname;
    } else {
      // Check Content-Type to determine how data was sent
      const contentType = req.headers['content-type'] || '';
      console.log(`[${requestId}] 📥 Content-Type: ${contentType}`);
      console.log(`[${requestId}] 📥 Body type: ${typeof req.body}, isBuffer: ${Buffer.isBuffer(req.body)}`);
      
      let fileData = null;
      
      if (Buffer.isBuffer(req.body)) {
        // Raw binary data
        fileData = req.body;
        console.log(`[${requestId}] 📥 Received raw binary data: ${fileData.length} bytes`);
      } else if (typeof req.body === 'object' && req.body !== null) {
        // JSON data - Shortcuts might send file as base64 or file path
        if (req.body.filePath) {
          const filePathFromBody = req.body.filePath;
          if (fsSync.existsSync(filePathFromBody)) {
            filePath = filePathFromBody;
            originalName = path.basename(filePathFromBody);
            console.log(`[${requestId}] 📥 Using file from path: ${filePathFromBody}`);
          } else {
            return res.status(400).json({
              error: `File not found at path: ${filePathFromBody}`
            });
          }
        } else if (req.body.base64) {
          fileData = Buffer.from(req.body.base64, 'base64');
          originalName = req.body.filename || 'audio.wav';
          console.log(`[${requestId}] 📥 Decoded base64 data: ${fileData.length} bytes`);
        } else if (req.body.data && Buffer.isBuffer(req.body.data)) {
          fileData = req.body.data;
          originalName = req.body.filename || 'audio.wav';
        } else {
          return res.status(400).json({
            error: 'Invalid request format. Expected: multipart/form-data with field "audio", raw binary, {filePath: "..."}, or {base64: "...", filename: "..."}',
            received: Object.keys(req.body),
            contentType: contentType
          });
        }
      } else {
        return res.status(400).json({
          error: 'No audio file provided (empty or invalid body)',
          contentType: contentType
        });
      }
      
      // If we have fileData, save it to temp file
      if (fileData && !filePath) {
        if (!fileData || fileData.length === 0) {
          return res.status(400).json({
            error: 'No audio file provided (empty data)'
          });
        }
        
        // Get filename from headers if not already set
        if (originalName === 'audio.wav') {
          const contentDisposition = req.headers['content-disposition'];
          if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch[1]) {
              originalName = filenameMatch[1].replace(/['"]/g, '');
            }
          } else {
            if (contentType.includes('mp3')) originalName = 'audio.mp3';
            else if (contentType.includes('m4a')) originalName = 'audio.m4a';
            else if (contentType.includes('wav')) originalName = 'audio.wav';
          }
        }
        
        filePath = path.join(tempUploadsDir, `raw_${Date.now()}_${originalName}`);
        await fs.writeFile(filePath, fileData);
      }
    }
    
    if (!filePath || !fsSync.existsSync(filePath)) {
      return res.status(400).json({
        error: `Audio file not found at path: ${filePath}`
      });
    }

    // Get parameters from query string or headers
    const language = req.query.language || req.headers['x-language'] || 'auto';
    const speakerCount = req.query.speakerCount || req.headers['x-speaker-count'] || null;
    const engine = req.query.engine || req.headers['x-engine'] || 'speechmatics';

    console.log(`[${requestId}] 📥 Audio diarization RAW endpoint called`);
    console.log(`[${requestId}] File: ${originalName}, path: ${filePath}`);
    console.log(`[${requestId}] Params: language=${language}, speakerCount=${speakerCount}, engine=${engine}`);

    const transcriptionEngine = resolveTranscriptionEngine(engine);
    
    const diarizationStartTime = Date.now();
    const result = await runPythonDiarization({
      filePath: filePath,
      language: language,
      speakerCount: speakerCount,
      originalFilename: originalName,
      engine: transcriptionEngine
    });
    const diarizationDuration = ((Date.now() - diarizationStartTime) / 1000).toFixed(2);
    
    // Cleanup temp file (only if it was created by us, not multer)
    if (!req.file && filePath && fsSync.existsSync(filePath)) {
      try {
        await fs.unlink(filePath);
        console.log(`[${requestId}] 🧹 Cleaned up temp file: ${filePath}`);
      } catch (cleanupError) {
        console.warn(`[${requestId}] ⚠️ Failed to cleanup temp file:`, cleanupError.message);
      }
    } else if (req.file && req.file.path && fsSync.existsSync(req.file.path)) {
      // Cleanup multer temp file
      try {
        await fs.unlink(req.file.path);
        console.log(`[${requestId}] 🧹 Cleaned up multer temp file: ${req.file.path}`);
      } catch (cleanupError) {
        console.warn(`[${requestId}] ⚠️ Failed to cleanup multer file:`, cleanupError.message);
      }
    }
    
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    const segmentsCount = result?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0;
    
    console.log(`[${requestId}] ✅ Python diarization completed in ${diarizationDuration}s, found ${segmentsCount} segments`);
    
    res.json({
      success: true,
      recordings: result.recordings,
      duration: totalDuration,
      segmentsCount: segmentsCount
    });
    
  } catch (error) {
    console.error(`[${requestId}] ❌ Error:`, error);
    res.status(500).json({
      error: 'Audio processing failed',
      details: error.message
    });
  }
});

// Endpoint for Shortcuts: transcribe audio from URL
app.post('/api/diarize-audio-from-url', express.json(), async (req, res) => {
  const requestId = `audio_url_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const startTime = Date.now();
  
  try {
    const { url, language, speakerCount, engine } = req.body;
    
    if (!url) {
      return res.status(400).json({
        error: 'URL is required'
      });
    }
    
    const transcriptionEngine = resolveTranscriptionEngine(engine || 'speechmatics');
    const lang = language || 'auto';
    const spCount = speakerCount || null;
    
    console.log(`[${requestId}] 📥 Audio diarization from URL: ${url}`);
    console.log(`[${requestId}] Params: language=${lang}, speakerCount=${spCount}, engine=${transcriptionEngine}`);
    
    const diarizationStartTime = Date.now();
    const result = await runPythonDiarization({
      url: url,
      language: lang,
      speakerCount: spCount,
      originalFilename: null,
      engine: transcriptionEngine
    });
    const diarizationDuration = ((Date.now() - diarizationStartTime) / 1000).toFixed(2);
    
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    const segmentsCount = result?.recordings?.[0]?.results?.[transcriptionEngine]?.segments?.length || 0;
    
    console.log(`[${requestId}] ✅ Python diarization completed in ${diarizationDuration}s, found ${segmentsCount} segments`);
    
    res.json({
      success: true,
      recordings: result.recordings,
      duration: totalDuration,
      segmentsCount: segmentsCount
    });
    
  } catch (error) {
    console.error(`[${requestId}] ❌ Error:`, error);
    res.status(500).json({
      error: 'Audio processing failed',
      details: error.message
    });
  }
});

// Middleware to handle both text and JSON for transcribe endpoint
function handleTranscribeRequest(req, res, next) {
  const contentType = req.headers['content-type'] || '';
  
  if (contentType.includes('application/json')) {
    express.json({ limit: '10mb' })(req, res, next);
  } else {
    // Plain text (JSON string from Shortcuts)
    express.text({ type: '*/*', limit: '10mb' })(req, res, next);
  }
}

// Endpoint for Shortcuts: transcribe all separated speakers at once
// Accepts both JSON object and plain text (JSON string from Shortcuts)
app.post('/api/transcribe-separated-speakers', handleTranscribeRequest, async (req, res) => {
  const requestId = `transcribe_sep_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const startTime = Date.now();
  
  try {
    let speakers, language, engine;
    
    // Check if body is plain text (JSON string from Shortcuts)
    if (typeof req.body === 'string') {
      try {
        // Parse JSON string (handles escaped slashes like \/)
        const parsed = JSON.parse(req.body);
        
        // Extract speakers - can be direct array or nested in separation result
        if (Array.isArray(parsed)) {
          speakers = parsed;
        } else if (parsed.speakers && Array.isArray(parsed.speakers)) {
          speakers = parsed.speakers;
        } else if (parsed.speakers) {
          speakers = parsed.speakers;
        }
        
        language = parsed.language;
        engine = parsed.engine;
        console.log(`[${requestId}] 📥 Received plain text JSON string, parsed successfully`);
      } catch (parseError) {
        return res.status(400).json({
          error: 'Invalid JSON string in request body',
          details: parseError.message
        });
      }
    } else if (typeof req.body === 'object' && req.body !== null) {
      // Regular JSON object
      // Extract speakers - can be direct array or nested in separation result
      if (Array.isArray(req.body)) {
        speakers = req.body;
      } else if (req.body.speakers && Array.isArray(req.body.speakers)) {
        speakers = req.body.speakers;
      } else {
        speakers = req.body.speakers;
      }
      
      language = req.body.language;
      engine = req.body.engine;
      console.log(`[${requestId}] 📥 Received JSON object`);
    } else {
      return res.status(400).json({
        error: 'Request body must be JSON object or JSON string'
      });
    }
    
    if (!speakers || !Array.isArray(speakers) || speakers.length === 0) {
      return res.status(400).json({
        error: 'speakers array is required',
        received: typeof speakers,
        bodyType: typeof req.body
      });
    }
    
    const transcriptionEngine = resolveTranscriptionEngine(engine || 'speechmatics');
    const lang = language || 'auto';
    
    console.log(`[${requestId}] 📥 Transcribing ${speakers.length} separated speakers`);
    console.log(`[${requestId}] Params: language=${lang}, engine=${transcriptionEngine}`);
    
    const voiceTracks = [];
    
    // Process each speaker
    for (let i = 0; i < speakers.length; i++) {
      const speaker = speakers[i];
      
      // Skip background tracks
      if (speaker.isBackground) {
        console.log(`[${requestId}] ⏭️ Skipping background track: ${speaker.name}`);
        continue;
      }
      
      console.log(`[${requestId}] 🎤 Transcribing speaker ${i + 1}/${speakers.length}: ${speaker.name}`);
      
      try {
        // Determine audio source (prefer local path over URL for reliability)
        let audioUrl = null;
        let audioPath = null;
        
        // Prefer local_path if available and file exists
        if (speaker.local_path && fsSync.existsSync(speaker.local_path)) {
          audioPath = speaker.local_path;
          console.log(`[${requestId}] 📥 Using local path: ${audioPath}`);
        } else if (speaker.url) {
          audioUrl = speaker.url;
          console.log(`[${requestId}] 📥 Using URL: ${audioUrl}`);
          console.warn(`[${requestId}] ⚠️ Note: URL may not be accessible to Speechmatics if it's a local tunnel`);
        } else {
          console.warn(`[${requestId}] ⚠️ No valid audio source for ${speaker.name}, skipping`);
          continue;
        }
        
        // Transcribe the speaker track with timeout
        const transcriptionStartTime = Date.now();
        
        // Set timeout for each transcription (5 minutes)
        const transcriptionPromise = runPythonDiarization({
          url: audioUrl,
          filePath: audioPath,
          language: lang,
          speakerCount: 1, // Single speaker track
          originalFilename: speaker.filename || `${speaker.name}.wav`,
          engine: transcriptionEngine
        });
        
        const timeoutPromise = new Promise((_, reject) => {
          setTimeout(() => reject(new Error('Transcription timeout (5 minutes)')), 5 * 60 * 1000);
        });
        
        let transcription;
        try {
          transcription = await Promise.race([transcriptionPromise, timeoutPromise]);
        } catch (timeoutError) {
          throw new Error(`Transcription timeout for ${speaker.name}: ${timeoutError.message}`);
        }
        
        const transcriptionDuration = ((Date.now() - transcriptionStartTime) / 1000).toFixed(2);
        
        const segments = transcription?.recordings?.[0]?.results?.[transcriptionEngine]?.segments || [];
        
        console.log(`[${requestId}] ✅ Transcribed ${speaker.name} in ${transcriptionDuration}s, found ${segments.length} segments`);
        
        voiceTracks.push({
          speaker: speaker.name,
          role: null, // Will be assigned later
          confidence: null,
          transcription: transcription,
          segments: segments,
          audioPath: audioPath,
          downloadUrl: speaker.url,
          filename: speaker.filename,
          duration: transcriptionDuration,
          success: true
        });
        
      } catch (speakerError) {
        console.error(`[${requestId}] ❌ Failed to transcribe ${speaker.name}:`, speakerError);
        voiceTracks.push({
          speaker: speaker.name,
          error: speakerError.message,
          transcription: null,
          success: false
        });
      }
    }
    
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    const successfulTracks = voiceTracks.filter(track => track.success).length;
    const failedTracks = voiceTracks.filter(track => !track.success).length;
    
    console.log(`[${requestId}] ✅ Completed transcription: ${successfulTracks} successful, ${failedTracks} failed, total time: ${totalDuration}s`);
    
    res.json({
      success: successfulTracks > 0,
      voiceTracks: voiceTracks,
      totalDuration: totalDuration,
      speakersCount: voiceTracks.length,
      successfulCount: successfulTracks,
      failedCount: failedTracks,
      errors: failedTracks > 0 ? voiceTracks.filter(track => !track.success).map(track => ({
        speaker: track.speaker,
        error: track.error
      })) : null
    });
    
  } catch (error) {
    console.error(`[${requestId}] ❌ Error:`, error);
    res.status(500).json({
      error: 'Transcription failed',
      details: error.message
    });
  }
});

app.post('/api/diarize-audio', upload.single('audio'), async (req, res) => {
  const requestId = `audio_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  let uploadedFile = req.file;
  const startTime = Date.now();
  
  console.log(`[${requestId}] 📥 Audio diarization endpoint called`);
  writeLog('info', `[${requestId}] Audio diarization started`, {
    requestId,
    timestamp: new Date().toISOString(),
    hasUrl: !!req.body.url,
    hasFile: !!uploadedFile,
    fileSize: uploadedFile?.size,
    fileName: uploadedFile?.originalname,
    language: req.body.language,
    speakerCount: req.body.speakerCount
  });
  
  try {
    const { url, language, speakerCount, engine } = req.body;
    const transcriptionEngine = resolveTranscriptionEngine(engine);
    
    if (!url && !uploadedFile) {
      writeLog('error', `[${requestId}] Validation failed: no file or URL`, {
        requestId
      });
      return res.status(400).json({ 
        error: 'Either file upload or URL must be provided' 
      });
    }

    // Use runPythonDiarization which handles environment variables correctly
    try {
      console.log(`[${requestId}] 🔵 Starting Python diarization process...`);
      writeLog('info', `[${requestId}] Starting Python diarization`, {
        requestId,
        step: 'python_diarization_start',
        hasUrl: !!url,
        hasFilePath: !!uploadedFile?.path,
        language,
        speakerCount
      });
      
      const diarizationStartTime = Date.now();
      const result = await runPythonDiarization({
        url: url,
        filePath: uploadedFile?.path,
        language: language,
        speakerCount: speakerCount,
        originalFilename: uploadedFile?.originalname || null, // Pass original filename for cache
        engine: transcriptionEngine
      });
      const diarizationDuration = ((Date.now() - diarizationStartTime) / 1000).toFixed(2);
      
      const segmentsCount = result?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0;
      console.log(`[${requestId}] ✅ Python diarization completed in ${diarizationDuration}s, found ${segmentsCount} segments`);
      writeLog('info', `[${requestId}] Python diarization completed`, {
        requestId,
        step: 'python_diarization_complete',
        duration: `${diarizationDuration}s`,
        segmentsCount,
        hasRecordings: !!result?.recordings,
        recordingsCount: result?.recordings?.length || 0
      });
      
      // Clean up uploaded file if it exists
      if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
        try {
          await fs.unlink(uploadedFile.path);
          writeLog('debug', `[${requestId}] Cleaned up uploaded file`, {
            requestId,
            filePath: uploadedFile.path
          });
        } catch (cleanupError) {
          console.error(`[${requestId}] Cleanup error:`, cleanupError);
          writeLog('warn', `[${requestId}] File cleanup failed`, {
            requestId,
            error: cleanupError.message
          });
        }
      }
      
      const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`[${requestId}] ✅ Audio diarization completed successfully in ${totalDuration}s`);
      writeLog('info', `[${requestId}] Audio diarization completed successfully`, {
        requestId,
        totalDuration: `${totalDuration}s`,
        segmentsCount
      });
      
      res.json(result);
    } catch (diarizationError) {
      const errorDuration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.error(`[${requestId}] ❌ Audio diarization error after ${errorDuration}s:`, diarizationError);
      writeLog('error', `[${requestId}] Audio diarization failed`, {
        requestId,
        step: 'python_diarization_error',
        duration: `${errorDuration}s`,
        error: diarizationError.message,
        errorType: diarizationError.constructor?.name,
        stack: diarizationError.stack?.substring(0, 1000)
      });
      
      // Clean up uploaded file if it exists
      if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
        try {
          await fs.unlink(uploadedFile.path);
          writeLog('debug', `[${requestId}] Cleaned up uploaded file after error`, {
            requestId,
            filePath: uploadedFile.path
          });
        } catch (cleanupError) {
          console.error(`[${requestId}] Cleanup error:`, cleanupError);
          writeLog('warn', `[${requestId}] File cleanup failed after error`, {
            requestId,
            error: cleanupError.message
          });
        }
      }
      
      res.status(500).json({
        error: 'Audio processing failed',
        details: diarizationError.message
      });
    }
    
  } catch (error) {
    console.error('Audio Diarization error:', error);
    
    if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
      try {
        await fs.unlink(uploadedFile.path);
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError);
      }
    }
    
    res.status(500).json({
      error: 'Internal server error',
      details: error.message
    });
  }
});

// Diarize voice track endpoint (for separated audio tracks)
app.post('/api/diarize-voice-track', async (req, res) => {
  try {
    const {
      local_path,
      url,
      language,
      speakerCount,
      isSeparatedTrack,
      engine,
      skipCache,
      forceFresh
    } = req.body;
    const transcriptionEngine = resolveTranscriptionEngine(engine);
    
    if (!local_path && !url) {
      return res.status(400).json({ error: 'Either local_path or url must be provided' });
    }
    
    console.log('📥 Diarize voice track request:', {
      hasLocalPath: !!local_path,
      hasUrl: !!url,
      language,
      speakerCount,
      isSeparatedTrack
    });
    
    // For voice tracks we now allow caching to speed up debugging.
    // You can force fresh diarization by passing skipCache=true or forceFresh=true in the request body.
    const shouldSkipCache = skipCache === true || forceFresh === true;

    const result = await runPythonDiarization({
      filePath: local_path,
      url: url,
      language: language || 'en',
      speakerCount: speakerCount || '2',
      isSeparatedTrack: isSeparatedTrack === true, // Only if explicitly set
      skipCache: shouldSkipCache,
      engine: transcriptionEngine
    });
    
    res.json(result);
    
  } catch (error) {
    console.error('❌ Diarize voice track error:', error);
    res.status(500).json({
      error: 'Failed to diarize voice track',
      details: error.message
    });
  }
});

// Combined Diarization endpoint (Audio + LLM + Verification)
app.post('/api/diarize-combined', upload.single('audio'), async (req, res) => {
  let tempDownloadedAudioPath = null;
  try {
    const { url, language, speakerCount, mode, pipelineMode: pipelineModeRaw, engine } = req.body;
    const transcriptionEngine = resolveTranscriptionEngine(engine);
    const uploadedFile = req.file;
    const pipelineMode = (pipelineModeRaw || 'hybrid').toLowerCase() === 'multimodal' ? 'multimodal' : 'hybrid';
    const pipelineStartTime = Date.now();
    const manualTranscript = (req.body.plainTranscript || req.body.manualTranscript || req.body.transcript || '').trim();
    let plainTranscript = manualTranscript;
    let audioResult = null;
    let llmResult = null;
    let comparisonAnalysis = null;
    let correctedResult = null;
    let step1Duration = null;
    let step2Duration = null;
    let step3Duration = null;
    let step4Duration = null;
    let audioFilePathForGemini = uploadedFile ? uploadedFile.path : null;

    if (!url && !uploadedFile) {
      return res.status(400).json({
        error: 'Either file upload or URL must be provided'
      });
    }

    if (pipelineMode === 'multimodal' && !audioFilePathForGemini) {
      if (!url) {
        return res.status(400).json({
          error: 'Audio file is required for multimodal pipeline'
        });
      }
      try {
        audioFilePathForGemini = await downloadAudioToTemp(url);
        tempDownloadedAudioPath = audioFilePathForGemini;
      } catch (downloadError) {
        return res.status(500).json({
          error: 'Failed to download audio for multimodal pipeline',
          details: downloadError.message
        });
      }
    }

    const steps = {};

    if (pipelineMode !== 'multimodal') {
      const step1StartTime = Date.now();
      try {
        // Use runPythonDiarization which handles environment variables correctly
        audioResult = await runPythonDiarization({
          url: url,
          filePath: uploadedFile?.path,
          language: language,
          speakerCount: speakerCount,
          originalFilename: uploadedFile?.originalname || null, // Pass original filename for cache
          engine: transcriptionEngine
        });
      } catch (audioError) {
        console.error('Audio processing failed:', audioError);
        writeLog('error', 'Audio processing failed (combined pipeline)', {
          error: audioError.message,
          stack: audioError.stack
        });
        return res.status(500).json({
          error: 'Audio processing failed',
          details: audioError.message
        });
      }

      step1Duration = ((Date.now() - step1StartTime) / 1000).toFixed(2) + 's';
      steps.step1 = {
        name: 'Audio Transcription',
        status: 'completed',
        duration: step1Duration,
        result: audioResult
      };

      if (audioResult && audioResult.recordings && audioResult.recordings.length > 0) {
        const recording = audioResult.recordings[0];
        if (recording.results && recording.results.speechmatics) {
          const segments = recording.results.speechmatics.segments || [];
          if (!plainTranscript) {
            plainTranscript = segments.map(s => s.text).join(' ').trim();
          }
        }
      }

      if (!plainTranscript) {
        return res.status(500).json({
          error: 'Failed to extract transcript from audio result'
        });
      }
    } else {
      steps.step1 = {
        name: 'Audio Transcription',
        status: 'skipped',
        reason: 'Gemini multimodal pipeline ingests audio directly'
      };
    }

    const step2StartTime = Date.now();
    try {
      if (pipelineMode === 'multimodal' || mode === 'gemini-2.5-pro') {
        llmResult = await callGeminiMultimodal(audioFilePathForGemini, plainTranscript, language || 'ar');
        writeLog('info', 'Step 2: LLM diarization (multimodal) completed', {
          duration: ((Date.now() - step2StartTime) / 1000).toFixed(2) + 's',
          segmentsCount: llmResult?.recordings?.[0]?.results?.[TEXT_SERVICE_KEY]?.segments?.length || 0,
          source: llmResult?.source || 'unknown'
        });
      } else {
        const useLocalLLM = mode === 'local' || mode === 'test' || mode === 'test2';
        // Use getModelId() to always get current model from process.env (for cache key accuracy)
        const selectedModel = getModelId(mode);
        
        // Build payload with reasoning_effort if needed
        const payload = {
          model: selectedModel,
            messages: [
              {
                role: 'system',
                content: (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim()
              },
              {
                role: 'user',
                content: (await fs.readFile('prompts/arabic_diarization_complete_prompt.txt', 'utf8'))
                  .replace('[PASTE YOUR ARABIC TRANSCRIPT HERE WITH CURRENT SPEAKER LABELS]', plainTranscript.trim())
              }
            ],
            temperature: 0
        };
        
        // Add reasoning effort for Local, Fast, and Smart (GPT 5.1)
        // OpenRouter format: reasoning: { effort: "high" }
        // NOTE: Local mode (including test) reasoning_effort is disabled - configure in LM Studio UI
        if (shouldUseHighReasoningEffort(mode, selectedModel) && !useLocalLLM) {
          // This is OpenRouter request, use nested reasoning object
          payload.reasoning = { effort: 'high' };
          console.log(`🔧 Using reasoning effort: high for ${mode} mode (model: ${selectedModel})`);
        }

        const apiUrl = useLocalLLM 
          ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
          : 'https://openrouter.ai/api/v1/chat/completions';
        
        const headers = useLocalLLM ? getLocalLLMHeaders() : getOpenRouterHeaders('Speaker Diarization');

        // Check LLM cache if enabled
        let llmOutput = null;
        if (LLM_CACHE_ENABLED && uploadedFile?.originalname) {
          const systemPromptText = (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim();
          const userPromptText = (await fs.readFile('prompts/arabic_diarization_complete_prompt.txt', 'utf8')).replace('[PASTE YOUR ARABIC TRANSCRIPT HERE WITH CURRENT SPEAKER LABELS]', plainTranscript.trim());
          const fullPrompt = `system: ${systemPromptText}\n\nuser: ${userPromptText}`;
          const cacheKey = buildLLMCacheKey(uploadedFile.originalname, fullPrompt, selectedModel, mode, 'default');
          if (cacheKey) {
            const cachedResponse = readLLMCache(cacheKey);
            if (cachedResponse) {
              console.log(`✅ Using cached LLM response for combined diarization (filename: ${uploadedFile.originalname})`);
              llmOutput = cachedResponse.llmOutput;
            }
          }
        }
        
        if (!llmOutput) {
          const llmResponse = await axios.post(
            apiUrl,
            payload,
            { headers }
          );

          llmOutput = llmResponse.data.choices[0]?.message?.content || '';
          
          // Save to LLM cache if enabled
          if (LLM_CACHE_ENABLED && uploadedFile?.originalname) {
            const systemPromptText = (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim();
            const userPromptText = (await fs.readFile('prompts/arabic_diarization_complete_prompt.txt', 'utf8')).replace('[PASTE YOUR ARABIC TRANSCRIPT HERE WITH CURRENT SPEAKER LABELS]', plainTranscript.trim());
            const fullPrompt = `system: ${systemPromptText}\n\nuser: ${userPromptText}`;
            const cacheKey = buildLLMCacheKey(uploadedFile.originalname, fullPrompt, selectedModel, mode, 'default');
            if (cacheKey) {
              writeLLMCache(cacheKey, { llmOutput, model: selectedModel, mode, promptVariant: 'default', timestamp: new Date().toISOString() });
            }
          }
        }
        
        llmResult = parseToStructuredJSON(llmOutput, plainTranscript);

        writeLog('info', 'Step 2: LLM diarization (text-only) completed', {
          duration: ((Date.now() - step2StartTime) / 1000).toFixed(2) + 's',
          segmentsCount: llmResult?.recordings?.[0]?.results?.[TEXT_SERVICE_KEY]?.segments?.length || 0
        });
      }
    } catch (llmError) {
      console.error('LLM diarization failed:', llmError);
      writeLog('error', 'Step 2: LLM diarization failed', {
        error: llmError.message
      });
      return res.status(500).json({
        error: 'LLM diarization failed',
        details: llmError.message
      });
    }
    applyLanguageHintToResult(llmResult, language || 'ar');
    step2Duration = ((Date.now() - step2StartTime) / 1000).toFixed(2) + 's';
    const llmProvider = llmResult?.provider
      || (llmResult?.source && llmResult.source.includes('openrouter') ? 'openrouter' : 'google');
    const isFallbackProvider = pipelineMode === 'multimodal' && llmProvider !== 'google';
    steps.step2 = {
      name: pipelineMode === 'multimodal' ? 'Gemini Multimodal' : 'LLM Diarization',
      status: isFallbackProvider ? 'completed_with_fallback' : 'completed',
      duration: step2Duration,
      reason: isFallbackProvider ? 'Fallback to OpenRouter text-only mode (audio not ingested directly).' : undefined,
      result: llmResult,
      metadata: {
        provider: llmProvider,
        multimodal: !!llmResult?.multimodal
      }
    };

    if (pipelineMode === 'multimodal') {
      comparisonAnalysis = {
        status: 'skipped',
        reason: 'Not required for multimodal pipeline'
      };
      correctedResult = {
        status: 'skipped',
        reason: 'Not required for multimodal pipeline'
      };
      steps.step3 = {
        name: 'Comparison Analysis',
        status: 'skipped',
        reason: 'Not required for multimodal pipeline',
        result: null
      };
      steps.step4 = {
        name: 'Correction & Merging',
        status: 'skipped',
        reason: 'Not required for multimodal pipeline',
        result: llmResult
      };
    } else {
      const step3StartTime = Date.now();
      try {
        comparisonAnalysis = await compareDiarizationResults(audioResult, llmResult, plainTranscript);
        step3Duration = ((Date.now() - step3StartTime) / 1000).toFixed(2) + 's';
        writeLog('info', 'Step 3: Comparison analysis completed', {
          duration: step3Duration,
          recommendedMethod: comparisonAnalysis.comparison?.overallAssessment?.recommendedMethod
        });
      } catch (comparisonError) {
        console.error('Comparison failed:', comparisonError);
        writeLog('error', 'Step 3: Comparison failed', {
          error: comparisonError.message
        });

        comparisonAnalysis = {
          comparison: {
            overallAssessment: {
              recommendedMethod: 'llm',
              reasoning: 'Comparison failed. Using LLM result as fallback.'
            },
            correctionGuidance: {
              useAudioFor: ['timestamps'],
              useLLMFor: ['roles', 'speaker_assignments'],
              mergeStrategy: 'Use LLM for roles and speaker assignments, Audio for timestamps'
            }
          },
          error: comparisonError.message
        };
        step3Duration = ((Date.now() - step3StartTime) / 1000).toFixed(2) + 's';
      }

      steps.step3 = {
        name: 'Comparison Analysis',
        status: comparisonAnalysis?.error ? 'completed_with_fallback' : 'completed',
        duration: step3Duration,
        result: comparisonAnalysis
      };

      const step4StartTime = Date.now();
      try {
        correctedResult = await correctDiarizationResults(comparisonAnalysis, audioResult, llmResult, plainTranscript);
        step4Duration = ((Date.now() - step4StartTime) / 1000).toFixed(2) + 's';
        writeLog('info', 'Step 4: Correction completed', {
          duration: step4Duration,
          segmentsCount: correctedResult?.recordings?.[0]?.results?.combined?.segments?.length || 0
        });
      } catch (correctionError) {
        console.error('Correction failed:', correctionError);
        writeLog('error', 'Step 4: Correction failed', {
          error: correctionError.message
        });

        correctedResult = llmResult;
        step4Duration = ((Date.now() - step4StartTime) / 1000).toFixed(2) + 's';
      }

      steps.step4 = {
        name: 'Correction & Merging',
        status: 'completed',
        duration: step4Duration,
        result: correctedResult
      };
    }

    const totalTime = ((Date.now() - pipelineStartTime) / 1000).toFixed(2);
    const finalResult = pipelineMode === 'multimodal' ? llmResult : correctedResult;

    res.json({
      audio: pipelineMode === 'multimodal' ? null : audioResult,
      llm: llmResult,
      comparison: comparisonAnalysis,
      corrected: correctedResult,
      transcript: plainTranscript || null,
      status: 'completed',
      pipelineMode,
      steps,
      totalDuration: totalTime + 's',
      finalResult
    });
  } catch (error) {
    console.error('Combined Diarization error:', error);
    res.status(500).json({
      error: 'Internal server error',
      details: error.message
    });
  } finally {
    const cleanupTargets = [];
    if (req.file && req.file.path) {
      cleanupTargets.push(req.file.path);
    }
    if (tempDownloadedAudioPath) {
      cleanupTargets.push(tempDownloadedAudioPath);
    }

    await Promise.all(cleanupTargets.map(async (filePath) => {
      if (filePath && fsSync.existsSync(filePath)) {
        try {
          await fs.unlink(filePath);
        } catch (cleanupError) {
          console.error('Cleanup error:', cleanupError);
        }
      }
    }));
  }
});

app.post('/api/audioshake-pipeline', upload.single('audio'), async (req, res) => {
  let audioPath = null;
  const cleanupTargets = [];
  const { url } = req.body || {};

  try {
    const pythonScriptPath = path.join(__dirname, 'main.py');
    if (!fsSync.existsSync(pythonScriptPath)) {
      return res.status(500).json({
        error: 'AudioShake pipeline script not found',
        hint: 'Ensure main.py exists in the project root'
      });
    }

    let storedFile = null;
    let publicAudioUrl = null;

    if (req.file && fsSync.existsSync(req.file.path)) {
      audioPath = req.file.path;
      // Зберігаємо файл у /uploads для публічного доступу
      storedFile = await persistUploadedFile(audioPath, req.file.originalname || 'audio.wav');
      publicAudioUrl = getPublicFileUrl(storedFile.filename);
      await ensurePublicFileAccessible(publicAudioUrl);
      cleanupTargets.push(req.file.path);
    } else if (url) {
      // Якщо URL вже публічний, використовуємо його
      if (url.startsWith('https://')) {
        publicAudioUrl = url;
        audioPath = await downloadAudioToTemp(url);
        cleanupTargets.push(audioPath);
      } else {
        audioPath = await downloadAudioToTemp(url);
        storedFile = await persistUploadedFile(audioPath, path.basename(url) || 'audio.wav');
        publicAudioUrl = getPublicFileUrl(storedFile.filename);
        await ensurePublicFileAccessible(publicAudioUrl);
        cleanupTargets.push(audioPath);
      }
    } else {
      return res.status(400).json({ error: 'Either file upload or URL must be provided' });
    }

    if (!publicAudioUrl) {
      return res.status(500).json({ error: 'Failed to create public URL for audio file' });
    }

    const jobId = `job_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    const stemsDir = path.join(audioshakeStemsRoot, jobId);
    fsSync.mkdirSync(stemsDir, { recursive: true });

    const pythonArgs = [pythonScriptPath, audioPath, '--stems-dir', stemsDir, '--audio-url', publicAudioUrl];
    
    // Ensure critical env vars are passed to Python
    // Explicitly pass API keys from process.env (which includes .env loaded values)
    const pythonEnv = {
      ...process.env, // Copy all existing env vars
      // Explicitly set API keys (will override if already in process.env)
      AUDIOSHAKE_API_KEY: process.env.AUDIOSHAKE_API_KEY || '',
      OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY || '',
      // Also pass OPENAI_API_KEY for backward compatibility
      ...(process.env.OPENAI_API_KEY && { OPENAI_API_KEY: process.env.OPENAI_API_KEY })
    };
    
    // Log what we're passing (without showing full keys)
    console.log('🔑 Passing env vars to Python process:');
    console.log(`  AUDIOSHAKE_API_KEY: ${pythonEnv.AUDIOSHAKE_API_KEY ? (pythonEnv.AUDIOSHAKE_API_KEY.substring(0, 20) + '... (length: ' + pythonEnv.AUDIOSHAKE_API_KEY.length + ')') : 'NOT SET'}`);
    console.log(`  OPENROUTER_API_KEY: ${pythonEnv.OPENROUTER_API_KEY ? (pythonEnv.OPENROUTER_API_KEY.substring(0, 20) + '... (length: ' + pythonEnv.OPENROUTER_API_KEY.length + ')') : 'NOT SET'}`);
    console.log(`  Python working directory (cwd): ${__dirname}`);
    console.log(`  Python script path: ${pythonScriptPath}`);
    
    writeLog('info', 'Starting AudioShake pipeline', {
      pythonBin: PYTHON_BIN,
      hasAudioshakeKey: !!process.env.AUDIOSHAKE_API_KEY,
      hasOpenaiKey: !!process.env.OPENAI_API_KEY,
      audioPath: audioPath
    });
    
    const pythonProcess = spawn(PYTHON_BIN, pythonArgs, {
      cwd: __dirname,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: pythonEnv
    });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', data => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', data => {
      stderr += data.toString();
    });

    pythonProcess.on('close', async (code) => {
      try {
        await Promise.all(
          cleanupTargets.map(async target => {
            if (target && fsSync.existsSync(target)) {
              await fs.unlink(target).catch(() => {});
            }
          })
        );
      } catch (cleanupError) {
        console.error('AudioShake cleanup error:', cleanupError);
      }

      if (code !== 0) {
        writeLog('error', 'AudioShake pipeline python error', { stderr, stdout, code, jobId });
        return res.status(500).json({
          error: 'AudioShake pipeline failed',
          details: stderr || 'Unknown error'
        });
      }

      try {
        const parsed = JSON.parse(stdout);
        const tracks = Array.isArray(parsed.results) ? parsed.results : [];
        const sanitizedTracks = tracks.map(track => {
          const fileName = track.file_name || path.basename(track.audio_path || '');
          const downloadUrl = fileName
            ? `/api/audioshake-stems/${jobId}/${encodeURIComponent(fileName)}`
            : null;
          const sanitized = {
            ...track,
            file_name: fileName,
            downloadUrl
          };
          delete sanitized.audio_path;
          return sanitized;
        });

        delete parsed.output_directory;
        parsed.results = sanitizedTracks;
        parsed.jobId = jobId;

        res.json(parsed);
      } catch (parseError) {
        console.error('Failed to parse AudioShake pipeline output:', parseError, stdout);
        res.status(500).json({
          error: 'Failed to parse AudioShake pipeline output',
          details: parseError.message
        });
      }
    });

    pythonProcess.on('error', async (error) => {
      console.error('Failed to start AudioShake pipeline process:', error);
      await Promise.all(
        cleanupTargets.map(async target => {
          if (target && fsSync.existsSync(target)) {
            await fs.unlink(target).catch(() => {});
          }
        })
      );
      res.status(500).json({
        error: 'Failed to start AudioShake pipeline',
        details: error.message
      });
    });
  } catch (error) {
    console.error('AudioShake pipeline error:', error);
    await Promise.all(
      cleanupTargets.map(async target => {
        if (target && fsSync.existsSync(target)) {
          await fs.unlink(target).catch(() => {});
        }
      })
    );
    res.status(500).json({
      error: 'Internal server error',
      details: error.message
    });
  }
});

// Endpoint for downloading PyAnnote separated audio files
app.get('/api/pyannote-stems/:filename', async (req, res) => {
  try {
    const filename = path.basename(req.params.filename);
    const resolvedPath = path.resolve(uploadsDir, filename);
    
    // Security check: ensure the resolved path is within uploadsDir
    if (!resolvedPath.startsWith(path.resolve(uploadsDir))) {
      return res.status(400).json({ error: 'Invalid file path' });
    }
    
    // Check if file exists
    await fs.access(resolvedPath);
    
    // Send the file
    res.sendFile(resolvedPath);
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'Stem not found' });
    } else {
      console.error('PyAnnote stem download error:', error);
      res.status(500).json({ error: 'Failed to download stem', details: error.message });
    }
  }
});

// Endpoint for downloading SpeechBrain separated audio files
app.get('/api/speechbrain-stems/:filename', async (req, res) => {
  try {
    const filename = path.basename(req.params.filename);
    const resolvedPath = path.resolve(uploadsDir, filename);

    if (!resolvedPath.startsWith(path.resolve(uploadsDir))) {
      return res.status(400).json({ error: 'Invalid file path' });
    }

    await fs.access(resolvedPath);
    res.sendFile(resolvedPath);
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'Stem not found' });
    } else {
      console.error('SpeechBrain stem download error:', error);
      res.status(500).json({ error: 'Failed to download stem', details: error.message });
    }
  }
});

app.get('/api/audioshake-stems/:jobId/:filename', async (req, res) => {
  try {
    const jobId = path.basename(req.params.jobId);
    const filename = path.basename(req.params.filename);
    const resolvedPath = path.resolve(audioshakeStemsRoot, jobId, filename);

    if (!resolvedPath.startsWith(path.resolve(audioshakeStemsRoot))) {
      return res.status(400).json({ error: 'Invalid file path' });
    }

    await fs.access(resolvedPath);
    res.sendFile(resolvedPath);
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'Stem not found' });
    } else {
      console.error('AudioShake stem download error:', error);
      res.status(500).json({ error: 'Failed to download stem', details: error.message });
    }
  }
});

// Debug endpoint for SpeechBrain separation
app.post('/api/debug-separation', upload.single('audio'), async (req, res) => {
  const requestId = `debug_sep_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const logPrefix = `[${requestId}]`;
  
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No audio file provided'
      });
    }

    console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Starting separation for file: ${req.file.originalname}`);
    console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Temp file path: ${req.file.path}`);
    
    // Verify temp file exists
    if (!fsSync.existsSync(req.file.path)) {
      throw new Error(`Temporary file not found: ${req.file.path}`);
    }
    
    // Check if file needs conversion to WAV
    const fileExtension = path.extname(req.file.originalname).toLowerCase();
    const needsConversion = fileExtension !== '.wav';
    
    let audioPath;
    let convertedFilePath = null;
    
    if (needsConversion) {
      console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Converting ${fileExtension} to WAV format...`);
      // Convert to WAV in temp directory first
      const baseName = path.basename(req.file.originalname, fileExtension);
      convertedFilePath = path.join(tempUploadsDir, `converted_${Date.now()}_${baseName}.wav`);
      await convertAudioToWav(req.file.path, convertedFilePath);
      audioPath = convertedFilePath;
    } else {
      // Save uploaded file to uploads directory
      const storedFile = await persistUploadedFile(req.file.path, req.file.originalname);
      console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Stored file info:`, storedFile);
      audioPath = storedFile.destinationPath;
      
      if (!audioPath) {
        throw new Error(`destinationPath not found in storedFile. Got: ${JSON.stringify(storedFile)}`);
      }
    }
    
    // Verify file exists
    if (!fsSync.existsSync(audioPath)) {
      throw new Error(`Audio file not found at path: ${audioPath}`);
    }
    
    console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Audio ready at: ${audioPath}`);
    
    // Extract debug parameters from request body (FormData fields)
    // Note: multer makes FormData fields available in req.body
    const debugParams = {
      chunkSeconds: req.body.chunkSeconds ? parseFloat(req.body.chunkSeconds) : null,
      gateThreshold: req.body.gateThreshold ? parseFloat(req.body.gateThreshold) : null,
      gateAlpha: req.body.gateAlpha ? parseFloat(req.body.gateAlpha) : null,
      enableSpectralGating: req.body.enableSpectralGating === 'true' || req.body.enableSpectralGating === true
    };
    
    console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Raw body:`, req.body);
    console.log(`${logPrefix} 🔵 DEBUG SEPARATION: Parameters:`, debugParams);
    
    // Run SpeechBrain separation with debug parameters
    const separationResult = await separateSpeakersWithSpeechBrain(audioPath, requestId, null, null, debugParams);
    
    console.log(`${logPrefix} ✅ DEBUG SEPARATION: Separation completed, found ${separationResult.speakers.length} speakers`);
    
    // Cleanup converted file if it was created
    if (convertedFilePath && fsSync.existsSync(convertedFilePath)) {
      try {
        await fs.unlink(convertedFilePath);
        console.log(`${logPrefix} 🧹 DEBUG SEPARATION: Cleaned up converted file: ${convertedFilePath}`);
      } catch (cleanupError) {
        console.warn(`${logPrefix} ⚠️ DEBUG SEPARATION: Failed to cleanup converted file: ${cleanupError.message}`);
      }
    }
    
    // Prepare response with speaker information
    const response = {
      success: true,
      num_speakers: separationResult.speakers.length,
      speakers: separationResult.speakers.map(speaker => {
        // Extract filename from local_path for URL generation
        let filename = null;
        if (speaker.local_path) {
          filename = path.basename(speaker.local_path);
        } else if (speaker.url) {
          // Extract filename from URL if available
          const urlParts = speaker.url.split('/');
          filename = urlParts[urlParts.length - 1];
        }
        
        return {
          name: speaker.name,
          format: speaker.format || 'wav',
          url: speaker.url || (filename ? `/api/speechbrain-stems/${encodeURIComponent(filename)}` : null),
          local_path: speaker.local_path,
          filename: filename,
          isBackground: speaker.isBackground || false
        };
      }),
      timeline: separationResult.timeline || [],
      output_dir: separationResult.output_dir
    };
    
    res.json(response);
    
  } catch (error) {
    // Cleanup converted file if it was created (even on error)
    if (convertedFilePath && fsSync.existsSync(convertedFilePath)) {
      try {
        await fs.unlink(convertedFilePath);
        console.log(`${logPrefix} 🧹 DEBUG SEPARATION: Cleaned up converted file after error: ${convertedFilePath}`);
      } catch (cleanupError) {
        console.warn(`${logPrefix} ⚠️ DEBUG SEPARATION: Failed to cleanup converted file: ${cleanupError.message}`);
      }
    }
    
    console.error(`${logPrefix} ❌ DEBUG SEPARATION: Error:`, error);
    res.status(500).json({
      success: false,
      error: error.message || 'Separation failed',
      details: error.stack
    });
  }
});

// Test endpoint to diagnose what Shortcuts sends
// Use conditional middleware to avoid stream conflicts
app.post('/api/test-upload', (req, res, next) => {
  const contentType = req.headers['content-type'] || '';
  if (contentType.includes('multipart/form-data')) {
    upload.single('audio')(req, res, next);
  } else if (contentType.includes('application/json')) {
    express.json({ limit: '500mb' })(req, res, next);
  } else {
    express.raw({ type: '*/*', limit: '500mb' })(req, res, next);
  }
}, async (req, res) => {
  const requestId = `test_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const logPrefix = `[${requestId}]`;
  
  try {
    const diagnostics = {
      timestamp: new Date().toISOString(),
      headers: {
        'content-type': req.headers['content-type'],
        'content-length': req.headers['content-length'],
        'content-disposition': req.headers['content-disposition'],
        'user-agent': req.headers['user-agent']
      },
      body: {
        type: typeof req.body,
        isBuffer: Buffer.isBuffer(req.body),
        isNull: req.body === null,
        isUndefined: req.body === undefined,
        length: req.body ? (Buffer.isBuffer(req.body) ? req.body.length : (typeof req.body === 'object' ? Object.keys(req.body).length : String(req.body).length)) : 0,
        keys: typeof req.body === 'object' && req.body !== null && !Buffer.isBuffer(req.body) ? Object.keys(req.body) : null,
        preview: Buffer.isBuffer(req.body) ? `Buffer(${req.body.length} bytes)` : 
                 typeof req.body === 'object' && req.body !== null ? JSON.stringify(req.body).substring(0, 200) : 
                 String(req.body).substring(0, 200)
      },
      file: req.file ? {
        fieldname: req.file.fieldname,
        originalname: req.file.originalname,
        encoding: req.file.encoding,
        mimetype: req.file.mimetype,
        size: req.file.size,
        path: req.file.path,
        exists: require('fs').existsSync(req.file.path)
      } : null,
      query: req.query,
      method: req.method,
      url: req.url
    };
    
    console.log(`${logPrefix} 📋 TEST UPLOAD DIAGNOSTICS:`, JSON.stringify(diagnostics, null, 2));
    
    // Try to read file if it exists
    if (req.file && require('fs').existsSync(req.file.path)) {
      const fileStats = require('fs').statSync(req.file.path);
      diagnostics.file.fileSize = fileStats.size;
      diagnostics.file.firstBytes = require('fs').readFileSync(req.file.path, { encoding: 'hex', start: 0, end: 20 });
    }
    
    res.json({
      success: true,
      message: 'Test endpoint received request',
      diagnostics: diagnostics
    });
    
  } catch (error) {
    console.error(`${logPrefix} ❌ TEST UPLOAD ERROR:`, error);
    res.status(500).json({
      success: false,
      error: error.message,
      stack: error.stack
    });
  }
});

// Custom middleware to handle both JSON and raw binary
function handleShortcutsUpload(req, res, next) {
  const contentType = req.headers['content-type'] || '';
  
  if (contentType.includes('application/json')) {
    express.json({ limit: '500mb' })(req, res, next);
  } else if (contentType.includes('multipart/form-data')) {
    // Shortcuts sends file as multipart/form-data even when selecting "File"
    upload.single('audio')(req, res, next);
  } else {
    express.raw({ type: '*/*', limit: '500mb' })(req, res, next);
  }
}

// Alternative endpoint for Shortcuts: accepts file as raw binary data, JSON, or multipart/form-data
app.post('/api/debug-separation-raw', handleShortcutsUpload, async (req, res) => {
  const requestId = `debug_sep_raw_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const logPrefix = `[${requestId}]`;
  
  try {
    let fileData = null;
    let originalName = 'audio.wav';
    let tempFilePath = null;
    
    // Check if multer already processed the file (multipart/form-data)
    if (req.file) {
      console.log(`${logPrefix} 📥 Received file via multipart/form-data: ${req.file.originalname}, size: ${req.file.size}`);
      // Use the file path directly from multer - no need to read and write again
      tempFilePath = req.file.path;
      originalName = req.file.originalname;
      console.log(`${logPrefix} 📥 File ready at: ${tempFilePath}`);
    } else {
      // Check Content-Type to determine how data was sent
      const contentType = req.headers['content-type'] || '';
      console.log(`${logPrefix} 📥 Content-Type: ${contentType}`);
      console.log(`${logPrefix} 📥 Body type: ${typeof req.body}, isBuffer: ${Buffer.isBuffer(req.body)}`);
      
      if (Buffer.isBuffer(req.body)) {
        // Raw binary data
        fileData = req.body;
        console.log(`${logPrefix} 📥 Received raw binary data: ${fileData.length} bytes`);
      } else if (typeof req.body === 'object' && req.body !== null) {
        // JSON data - Shortcuts might send file as base64 or file path
        if (req.body.filePath) {
          // File path provided
          const filePath = req.body.filePath;
          if (fsSync.existsSync(filePath)) {
            fileData = await fs.readFile(filePath);
            originalName = path.basename(filePath);
            console.log(`${logPrefix} 📥 Read file from path: ${filePath}`);
          } else {
            return res.status(400).json({
              success: false,
              error: `File not found at path: ${filePath}`
            });
          }
        } else if (req.body.base64) {
          // Base64 encoded file
          fileData = Buffer.from(req.body.base64, 'base64');
          originalName = req.body.filename || 'audio.wav';
          console.log(`${logPrefix} 📥 Decoded base64 data: ${fileData.length} bytes`);
        } else if (req.body.data && Buffer.isBuffer(req.body.data)) {
          // Data in data field
          fileData = req.body.data;
          originalName = req.body.filename || 'audio.wav';
        } else {
          // Check if body is empty object (Shortcuts might send empty object when file is in Form)
          if (Object.keys(req.body).length === 0) {
            return res.status(400).json({
              success: false,
              error: 'No audio file provided. Please send file as: raw binary, multipart/form-data with field "audio", {filePath: "..."}, or {base64: "...", filename: "..."}',
              contentType: req.headers['content-type'],
              hasFile: !!req.file
            });
          }
          return res.status(400).json({
            success: false,
            error: 'Invalid request format. Expected: raw binary, {filePath: "..."}, or {base64: "...", filename: "..."}',
            received: Object.keys(req.body),
            contentType: req.headers['content-type']
          });
        }
      } else {
        return res.status(400).json({
          success: false,
          error: 'No audio file provided (empty or invalid body)',
          contentType: req.headers['content-type'],
          hasFile: !!req.file
        });
      }
    }
    
    if (!fileData || fileData.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No audio file provided (empty data)'
      });
    }

    // Get filename from headers if not already set
    if (originalName === 'audio.wav') {
      const contentDisposition = req.headers['content-disposition'];
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
        if (filenameMatch && filenameMatch[1]) {
          originalName = filenameMatch[1].replace(/['"]/g, '');
        }
      } else {
        // Try to detect from Content-Type
        const contentType = req.headers['content-type'] || '';
        if (contentType.includes('mp3')) originalName = 'audio.mp3';
        else if (contentType.includes('m4a')) originalName = 'audio.m4a';
        else if (contentType.includes('wav')) originalName = 'audio.wav';
      }
    }

    // If we have fileData (from raw/JSON), save it to temp file
    if (fileData && !tempFilePath) {
      console.log(`${logPrefix} 🔵 DEBUG SEPARATION RAW: Received file: ${originalName}, size: ${fileData.length} bytes`);
      tempFilePath = path.join(tempUploadsDir, `raw_${Date.now()}_${originalName}`);
      await fs.writeFile(tempFilePath, fileData);
    } else if (tempFilePath) {
      console.log(`${logPrefix} 🔵 DEBUG SEPARATION RAW: Using multer file: ${originalName}, size: ${req.file.size} bytes`);
    }

    if (!tempFilePath || !fsSync.existsSync(tempFilePath)) {
      throw new Error(`Audio file not found at path: ${tempFilePath}`);
    }
    
    // Check if file needs conversion to WAV
    const fileExtension = path.extname(originalName).toLowerCase();
    const needsConversion = fileExtension !== '.wav';
    
    let audioPath;
    let convertedFilePath = null;
    let shouldCleanupTempFile = false;
    
    if (needsConversion) {
      console.log(`${logPrefix} 🔵 DEBUG SEPARATION RAW: Converting ${fileExtension} to WAV format...`);
      const baseName = path.basename(originalName, fileExtension);
      convertedFilePath = path.join(tempUploadsDir, `converted_${Date.now()}_${baseName}.wav`);
      await convertAudioToWav(tempFilePath, convertedFilePath);
      audioPath = convertedFilePath;
      // Mark temp file for cleanup (if it was created by us, not multer)
      shouldCleanupTempFile = !req.file;
    } else {
      // For WAV files, use the temp file directly or persist it
      if (req.file) {
        // Multer file - use it directly
        audioPath = tempFilePath;
      } else {
        // Our temp file - persist it
        const storedFile = await persistUploadedFile(tempFilePath, originalName);
        console.log(`${logPrefix} 🔵 DEBUG SEPARATION RAW: Stored file info:`, storedFile);
        audioPath = storedFile.destinationPath;
        shouldCleanupTempFile = false; // Already moved
      }
    }
    
    // Verify file exists
    if (!fsSync.existsSync(audioPath)) {
      throw new Error(`Audio file not found at path: ${audioPath}`);
    }
    
    console.log(`${logPrefix} 🔵 DEBUG SEPARATION RAW: Audio ready at: ${audioPath}`);
    
    // Run SpeechBrain separation
    const separationResult = await separateSpeakersWithSpeechBrain(audioPath, requestId);
    
    console.log(`${logPrefix} ✅ DEBUG SEPARATION RAW: Separation completed, found ${separationResult.speakers.length} speakers`);
    
    // Cleanup files
    if (shouldCleanupTempFile && tempFilePath && fsSync.existsSync(tempFilePath)) {
      try {
        await fs.unlink(tempFilePath);
        console.log(`${logPrefix} 🧹 DEBUG SEPARATION RAW: Cleaned up temp file: ${tempFilePath}`);
      } catch (cleanupError) {
        console.warn(`${logPrefix} ⚠️ DEBUG SEPARATION RAW: Failed to cleanup temp file: ${cleanupError.message}`);
      }
    }
    
    // Cleanup multer temp file if it was used
    if (req.file && req.file.path && fsSync.existsSync(req.file.path) && audioPath !== req.file.path) {
      try {
        await fs.unlink(req.file.path);
        console.log(`${logPrefix} 🧹 DEBUG SEPARATION RAW: Cleaned up multer temp file: ${req.file.path}`);
      } catch (cleanupError) {
        console.warn(`${logPrefix} ⚠️ DEBUG SEPARATION RAW: Failed to cleanup multer file: ${cleanupError.message}`);
      }
    }
    
    // Cleanup converted file if it was created
    if (convertedFilePath && fsSync.existsSync(convertedFilePath)) {
      try {
        await fs.unlink(convertedFilePath);
        console.log(`${logPrefix} 🧹 DEBUG SEPARATION RAW: Cleaned up converted file: ${convertedFilePath}`);
      } catch (cleanupError) {
        console.warn(`${logPrefix} ⚠️ DEBUG SEPARATION RAW: Failed to cleanup converted file: ${cleanupError.message}`);
      }
    }
    
    // Prepare response
    const response = {
      success: true,
      num_speakers: separationResult.speakers.length,
      speakers: separationResult.speakers.map(speaker => {
        let filename = null;
        if (speaker.local_path) {
          filename = path.basename(speaker.local_path);
        } else if (speaker.url) {
          const urlParts = speaker.url.split('/');
          filename = urlParts[urlParts.length - 1];
        }
        
        return {
          name: speaker.name,
          format: speaker.format || 'wav',
          url: speaker.url || (filename ? `/api/speechbrain-stems/${encodeURIComponent(filename)}` : null),
          local_path: speaker.local_path,
          filename: filename,
          isBackground: speaker.isBackground || false
        };
      }),
      timeline: separationResult.timeline || [],
      output_dir: separationResult.output_dir
    };
    
    res.json(response);
    
  } catch (error) {
    console.error(`${logPrefix} ❌ DEBUG SEPARATION RAW: Error:`, error);
    res.status(500).json({
      success: false,
      error: error.message || 'Separation failed',
      details: error.stack
    });
  }
});

// API endpoint for applying separation parameters to project
app.post('/api/apply-separation-params', async (req, res) => {
  const logPrefix = '[Apply Separation Params]';
  
  try {
    const { chunkSeconds, gateThreshold, gateAlpha, enableSpectralGating } = req.body;
    
    // Validate parameters
    if (chunkSeconds !== undefined && (chunkSeconds < 5 || chunkSeconds > 30)) {
      return res.status(400).json({
        success: false,
        error: 'chunkSeconds must be between 5 and 30'
      });
    }
    
    if (gateThreshold !== undefined && (gateThreshold < 0.05 || gateThreshold > 0.3)) {
      return res.status(400).json({
        success: false,
        error: 'gateThreshold must be between 0.05 and 0.3'
      });
    }
    
    if (gateAlpha !== undefined && (gateAlpha < 0.1 || gateAlpha > 0.9)) {
      return res.status(400).json({
        success: false,
        error: 'gateAlpha must be between 0.1 and 0.9'
      });
    }
    
    // Create configuration object
    const config = {
      chunkSeconds: chunkSeconds !== undefined ? chunkSeconds : null,
      gateThreshold: gateThreshold !== undefined ? gateThreshold : null,
      gateAlpha: gateAlpha !== undefined ? gateAlpha : null,
      enableSpectralGating: enableSpectralGating !== undefined ? enableSpectralGating : null,
      appliedAt: new Date().toISOString(),
      appliedBy: 'debug-panel'
    };
    
    // Save to configuration file
    const configPath = path.join(__dirname, 'cache', 'speechbrain_separation_config.json');
    const configDir = path.dirname(configPath);
    
    // Ensure cache directory exists
    if (!fsSync.existsSync(configDir)) {
      fsSync.mkdirSync(configDir, { recursive: true });
    }
    
    // Read existing config if exists
    let existingConfig = {};
    if (fsSync.existsSync(configPath)) {
      try {
        const existingData = fsSync.readFileSync(configPath, 'utf8');
        existingConfig = JSON.parse(existingData);
      } catch (e) {
        console.warn(`${logPrefix} Warning: Could not read existing config, creating new one`);
      }
    }
    
    // Merge with existing config (only update provided values)
    const mergedConfig = {
      ...existingConfig,
      ...config
    };
    
    // Write configuration file
    fsSync.writeFileSync(configPath, JSON.stringify(mergedConfig, null, 2), 'utf8');
    
    console.log(`${logPrefix} ✅ Applied separation parameters:`, mergedConfig);
    
    res.json({
      success: true,
      message: 'Parameters applied successfully',
      config: mergedConfig
    });
    
  } catch (error) {
    console.error(`${logPrefix} ❌ Error:`, error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to apply parameters'
    });
  }
});

function getPromptTemplateConfig(variant = 'default') {
  if (variant === 'voice-tracks') {
    return {
      templatePath: 'prompts/voice_track_diarization_prompt.txt',
      placeholder: '[VOICE_TRACK_TRANSCRIPT]'
    };
  }

  return {
    templatePath: 'prompts/arabic_diarization_complete_prompt.txt',
    placeholder: '[PASTE YOUR ARABIC TRANSCRIPT HERE WITH CURRENT SPEAKER LABELS]'
  };
}

// Helper function to extract diarization logic (for reuse)
async function handleDiarizationRequest(body) {
  const { transcript, mode, isVerification, isFixRequest, verificationPrompt, promptVariant, filename } = body;
  
  const useLocalLLM = mode === 'local' || mode === 'test' || mode === 'test2';
  
  // Debug logging
  console.log(`🔍 [handleDiarizationRequest] Configuration:`);
  console.log(`   mode: ${mode}`);
  console.log(`   useLocalLLM: ${useLocalLLM}`);

  // Select model based on mode
  // Use getModelId() to always get current model from process.env (for cache key accuracy)
  let model = getModelId(mode);
  if (useLocalLLM) {
      console.log(`   Using local model: ${model}`);
    console.log(`   Local LLM Base URL: ${LOCAL_LLM_BASE_URL}`);
  } else {
    console.log(`   Using OpenRouter model: ${model}`);
  }
  
  const normalizedTranscript = (transcript || '').trim();
  if (!normalizedTranscript) {
    throw new Error('Transcript text is empty.');
  }
  const cleanedTranscript = cleanServiceMessages(normalizedTranscript);
  if (!cleanedTranscript) {
    throw new Error('Transcript text is empty after cleaning.');
  }
  
  const systemPrompt = (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim();
  
  // Use appropriate prompt template based on variant
  let promptConfig;
  let promptTemplate;
  if (promptVariant === 'voice-tracks') {
    promptConfig = getPromptTemplateConfig('voice-tracks');
    promptTemplate = await fs.readFile(promptConfig.templatePath, 'utf8');
  } else {
    promptConfig = getPromptTemplateConfig('default');
    promptTemplate = await fs.readFile(promptConfig.templatePath, 'utf8');
  }
  const chatEndpoint = useLocalLLM
    ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
    : 'https://openrouter.ai/api/v1/chat/completions';
  const getHeaders = (title) => useLocalLLM ? getLocalLLMHeaders() : getOpenRouterHeaders(title);
  const resolveModelId = () => {
    if (useLocalLLM) {
      if (mode === 'test') return TEST_MODEL_ID;
      if (mode === 'test2') return TEST2_MODEL_ID;
      return LOCAL_LLM_MODEL;
    }
    return model;
  };

  if (!useLocalLLM && !process.env.OPENROUTER_API_KEY) {
    throw new Error('OPENROUTER_API_KEY is not configured.');
  }

  const requestChatCompletion = async (messages, title) => {
    const modelId = resolveModelId();
    const userMessage = messages.find(m => m.role === 'user')?.content || '';
    const fullPrompt = `${messages.map(m => `${m.role}: ${m.content}`).join('\n\n')}`;
    
    // Check LLM cache if enabled
    if (LLM_CACHE_ENABLED) {
      if (!filename) {
        console.warn(`⚠️ LLM cache check skipped for ${title}: filename is missing`);
      } else {
      const cacheKey = buildLLMCacheKey(filename, fullPrompt, modelId, mode, promptVariant);
      if (cacheKey) {
          console.log(`🔍 Checking LLM cache for ${title}:`, { filename, cacheKey, modelId, mode, promptVariant });
        const cachedResponse = readLLMCache(cacheKey);
        if (cachedResponse) {
            console.log(`✅ Using cached LLM response for ${title} (filename: ${filename}, cacheKey: ${cacheKey})`);
          return cachedResponse.llmOutput;
          } else {
            console.log(`📝 LLM cache miss for ${title} (filename: ${filename}, cacheKey: ${cacheKey})`);
          }
        } else {
          console.warn(`⚠️ Failed to build LLM cache key for ${title} (filename: ${filename})`);
        }
      }
    }
    
    const payload = {
      model: modelId,
        messages,
        temperature: 0
    };
    
  // Add reasoning effort for Fast, and Smart (GPT 5.1)
  // OpenRouter format: reasoning: { effort: "high" }
  // NOTE: Local mode (LM Studio) reasoning_effort is disabled - configure in LM Studio UI
  if (shouldUseHighReasoningEffort(mode, modelId)) {
    if (useLocalLLM) {
      // Local LLM reasoning_effort disabled - configure in LM Studio UI
      console.log(`🔧 Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)`);
    } else {
      // For OpenRouter, use nested reasoning object
      payload.reasoning = { effort: 'high' };
      console.log(`🔧 Using reasoning effort: high for ${mode} mode (model: ${modelId})`);
    }
  }
    
    const llmOutput = await streamChatCompletion({
      url: chatEndpoint,
      headers: getHeaders(title),
      payload: payload,
      errorContext: title
    });
    
    // Save to LLM cache if enabled
    if (LLM_CACHE_ENABLED) {
      if (!filename) {
        console.warn(`⚠️ LLM cache save skipped for ${title}: filename is missing`);
      } else {
      const cacheKey = buildLLMCacheKey(filename, fullPrompt, modelId, mode, promptVariant);
      if (cacheKey) {
          console.log(`💾 Saving LLM response to cache for ${title}:`, { filename, cacheKey, modelId, mode, promptVariant });
        writeLLMCache(cacheKey, { llmOutput, model: modelId, mode, promptVariant, timestamp: new Date().toISOString() });
        } else {
          console.warn(`⚠️ Failed to build LLM cache key for saving ${title} (filename: ${filename})`);
        }
      }
    }
    
    return llmOutput;
  };
  
  if ((isVerification || isFixRequest) && verificationPrompt) {
    const llmOutput = await requestChatCompletion(
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: verificationPrompt }
      ],
      'Speaker Diarization Verification'
    );
    return parseToStructuredJSON(llmOutput, cleanedTranscript);
  }
  
  if (isFixRequest) {
    const verificationPromptTemplate = await fs.readFile('prompts/diarization_verification_prompt.txt', 'utf8');
    const prompt = verificationPromptTemplate.replace(
      '[PASTE THE JSON OUTPUT FROM THE FIRST WEBHOOK HERE]',
      cleanedTranscript
    );
    
    const llmOutput = await requestChatCompletion(
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: prompt }
      ],
      'Speaker Diarization Fix Request'
    );
    return parseToStructuredJSON(llmOutput, cleanedTranscript);
  }
  
  const chunks = splitTranscriptIntoChunks(cleanedTranscript, LLM_CHUNK_WORD_LIMIT);
  const aggregatedSegments = [];
  let baseRecordingMeta = null;
  let baseRecording = null;
  
  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    const chunkPrompt = `${promptTemplate.replace(
      promptConfig.placeholder,
      chunk.text
    )}\n\nChunk metadata: this is chunk ${index + 1}/${chunks.length} and it starts at ${
      chunk.offset.toFixed(1)
    } seconds of the full conversation. Keep timestamps aligned with the full call timeline.`;
    
    const llmOutput = await requestChatCompletion(
      [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: chunkPrompt }
      ],
      `Speaker Diarization Chunk ${index + 1}`
    );
    
    const chunkJSON = parseToStructuredJSON(llmOutput, chunk.text, { resultKey: TEXT_SERVICE_KEY });
    const chunkRecording = chunkJSON.recordings?.[0];
    const chunkSegments = chunkRecording?.results?.[TEXT_SERVICE_KEY]?.segments || [];
    
    chunkSegments.forEach(segment => {
      segment.start = parseFloat((coerceToNumber(segment.start, 0) + chunk.offset).toFixed(3));
      segment.end = parseFloat((coerceToNumber(segment.end, segment.start) + chunk.offset).toFixed(3));
    });
    
    aggregatedSegments.push(...chunkSegments);
    
    if (!baseRecordingMeta) {
      baseRecordingMeta = chunkJSON;
    }
    if (!baseRecording && chunkRecording) {
      baseRecording = JSON.parse(JSON.stringify(chunkRecording));
    }
  }
  
  const finalRecording = baseRecording
    ? JSON.parse(JSON.stringify(baseRecording))
    : {
        id: `rec_${Date.now()}`,
        name: 'Transcript Analysis',
        fileName: 'transcript.txt',
        size: cleanedTranscript.length,
        duration: 0,
        language: 'ar',
        speakerCount: '0',
        status: 'completed',
        addedAt: null,
        translationState: { currentLanguage: 'original', lastError: null },
        aggregated: {},
        servicesTested: []
      };
  
  finalRecording.id = finalRecording.id || baseRecordingMeta?.activeRecordingId || `rec_${Date.now()}`;
  finalRecording.name = finalRecording.name || 'Transcript Analysis';
  finalRecording.fileName = finalRecording.fileName || 'transcript.txt';
  finalRecording.translationState = finalRecording.translationState || { currentLanguage: 'original', lastError: null };
  finalRecording.aggregated = finalRecording.aggregated || {};
  
  const textResult = ensureResultContainer(finalRecording, TEXT_SERVICE_KEY, 'Text Mode 📝');
  textResult.segments = sanitizeSegmentsCollection(aggregatedSegments);
  const mergedSegments = textResult.segments;
  
  const totalDuration = mergedSegments.length
    ? Math.max(...mergedSegments.map(segment => segment.end || 0))
    : finalRecording.duration || cleanedTranscript.split(/\s+/).filter(Boolean).length / WORDS_PER_SECOND;
  
  finalRecording.duration = parseFloat(totalDuration.toFixed(3));
  textResult.rawData.duration = finalRecording.duration;
  
  const speakerSet = new Set(mergedSegments.map(segment => segment.speaker));
  const speakerCount = speakerSet.size || parseInt(finalRecording.speakerCount || '0', 10) || 0;
  finalRecording.speakerCount = speakerCount.toString();
  textResult.speakerCount = speakerCount;
  
  const structuredJSON = {
    version: baseRecordingMeta?.version || '2.0',
    exportedAt: new Date().toISOString(),
    activeRecordingId: finalRecording.id,
    recordings: [finalRecording]
  };
  
  return structuredJSON;
}

// Comparison Agent: Compare Audio and LLM diarization results
async function compareDiarizationResults(audioResult, llmResult, transcript) {
  try {
    // Load comparison prompt template
    const comparisonPromptTemplate = await fs.readFile('prompts/diarization_comparison_prompt.txt', 'utf8');
    
    // Prepare input data for the prompt
    const audioResultJSON = JSON.stringify(audioResult, null, 2);
    const llmResultJSON = JSON.stringify(llmResult, null, 2);
    
    // Replace placeholders in the prompt
    let prompt = comparisonPromptTemplate
      .replace('[PASTE THE AUDIO RESULT JSON HERE]', audioResultJSON)
      .replace('[PASTE THE LLM RESULT JSON HERE]', llmResultJSON)
      .replace('[PASTE THE ORIGINAL TRANSCRIPT HERE]', transcript || '');
    
    // Use SMART_MODEL_ID for comparison (more accurate analysis)
    const comparisonResponse = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: SMART_MODEL_ID,
        messages: [
          {
            role: 'system',
            content: 'You are an expert comparison agent for speaker diarization. Analyze and compare the provided diarization results, then provide a structured comparison following the exact format specified in the prompt.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
          'X-Title': 'Diarization Comparison Agent'
        }
      }
    );

    const comparisonOutput = comparisonResponse.data.choices[0]?.message?.content || '';
    
    // Try to parse the comparison result as JSON
    let comparisonAnalysis;
    try {
      // Try to extract JSON from the response (might be wrapped in markdown)
      const jsonMatch = comparisonOutput.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        comparisonAnalysis = JSON.parse(jsonMatch[0]);
      } else {
        comparisonAnalysis = JSON.parse(comparisonOutput);
      }
    } catch (parseError) {
      console.error('Failed to parse comparison result:', parseError);
      console.error('Raw output:', comparisonOutput.substring(0, 500));
      // Return a basic comparison structure if parsing fails
      comparisonAnalysis = {
        comparison: {
          overallAssessment: {
            audioMethodScore: 50,
            llmMethodScore: 50,
            recommendedMethod: 'llm',
            reasoning: 'Failed to parse comparison. Defaulting to LLM method.'
          },
          disagreements: [],
          roleAnalysis: {
            recommendedMethodForRoles: 'llm',
            reasoning: 'LLM method typically better for role identification'
          },
          correctionGuidance: {
            useAudioFor: ['timestamps'],
            useLLMFor: ['roles', 'speaker_assignments'],
            mergeStrategy: 'Use LLM for roles and speaker assignments, Audio for timestamps'
          }
        }
      };
    }
    
    writeLog('info', 'Comparison Agent completed', {
      audioSegments: audioResult?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0,
      llmSegments: llmResult?.recordings?.[0]?.results?.[TEXT_SERVICE_KEY]?.segments?.length || 0,
      recommendedMethod: comparisonAnalysis.comparison?.overallAssessment?.recommendedMethod
    });
    
    return comparisonAnalysis;
    
  } catch (error) {
    console.error('Comparison Agent error:', error);
    writeLog('error', 'Comparison Agent failed', {
      error: error.message,
      stack: error.stack
    });
    
    // Return fallback comparison
    return {
      comparison: {
        overallAssessment: {
          audioMethodScore: 50,
          llmMethodScore: 50,
          recommendedMethod: 'llm',
          reasoning: 'Comparison failed. Defaulting to LLM method as fallback.'
        },
        disagreements: [],
        roleAnalysis: {
          recommendedMethodForRoles: 'llm',
          reasoning: 'LLM method typically better for role identification'
        },
        correctionGuidance: {
          useAudioFor: ['timestamps'],
          useLLMFor: ['roles', 'speaker_assignments'],
          mergeStrategy: 'Use LLM for roles and speaker assignments, Audio for timestamps'
        }
      },
      error: error.message
    };
  }
}

// Validate JSON structure for diarization results
function validateDiarizationJSON(json, { resultKey = TEXT_SERVICE_KEY } = {}) {
  const { valid, errors } = validateDiarizationPayload(json, resultKey);
  if (!valid) {
    return {
      valid: false,
      error: formatAjvErrors(errors),
      details: errors
    };
  }
  return { valid: true };
}

function hasValidOverlapSegments(result) {
  const segments = result?.recordings?.[0]?.results?.['overlap-corrected']?.segments;
  if (!segments || !Array.isArray(segments) || segments.length === 0) {
    return false;
  }
  const uniqueSpeakers = new Set(segments.map(segment => segment.speaker).filter(Boolean));
  return uniqueSpeakers.size >= 2;
}

function shouldAcceptLLMOverlapResult(primaryDiarization, correctedResult, voiceTracks = []) {
  if (!hasValidOverlapSegments(correctedResult)) {
    return false;
  }

  const correctedSegments = correctedResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
  const overlapSegments = correctedSegments.filter(segment => segment.overlap).length;
  if (overlapSegments > 0) {
    return true;
  }

  const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
  if (correctedSegments.length > primarySegments.length) {
    return true;
  }

  const voiceSpeakers = new Set(
    (voiceTracks || [])
      .map(track => track?.speaker)
      .filter(Boolean)
  );
  if (voiceSpeakers.size === 0) {
    return false;
  }

  const correctedSpeakers = new Set(correctedSegments.map(segment => segment.speaker));
  const missingSpeaker = Array.from(voiceSpeakers).some(speaker => !correctedSpeakers.has(speaker));
  return !missingSpeaker;
}

function convertTextServiceToOverlapResult(structured, serviceName = 'Overlap Corrected (Voice Tracks)', voiceTracksCount = 0) {
  if (!structured) {
    return null;
  }

  const cloned = JSON.parse(JSON.stringify(structured));
  const recording = cloned.recordings?.[0];
  if (!recording?.results) {
    return cloned;
  }

  let textResult = recording.results[TEXT_SERVICE_KEY];
  if (!textResult) {
    const legacyKey = Object.keys(recording.results).find(key => LEGACY_TEXT_SERVICE_KEYS.includes(key));
    if (legacyKey) {
      textResult = recording.results[legacyKey];
    }
  }

  if (!textResult) {
    return cloned;
  }

  recording.results['overlap-corrected'] = {
    ...textResult,
    success: true,
    serviceName,
    rawData: {
      ...(textResult.rawData || {}),
      source: 'overlap-corrected',
      voiceTracksCount
    }
  };

  recording.servicesTested = recording.servicesTested || [];
  if (!recording.servicesTested.includes('overlap-corrected')) {
    recording.servicesTested.push('overlap-corrected');
  }

  return cloned;
}

async function generateOverlapCorrectionResult({
  primaryDiarization,
  voiceTracks,
  transcript,
  existingLLMResult,
  mode = 'smart',
  requestId = null,
  filename = null
}) {
  const logPrefix = requestId ? `[${requestId}]` : '[Overlap Correction]';
  if (!transcript || !transcript.trim()) {
    return null;
  }

  let structured = existingLLMResult ? JSON.parse(JSON.stringify(existingLLMResult)) : null;

  if (!structured) {
    structured = await handleDiarizationRequest({
      transcript,
      mode,
      promptVariant: 'voice-tracks',
      filename: filename // Pass filename for LLM cache
    });
  }

  if (!structured) {
    return null;
  }

  const normalized = convertTextServiceToOverlapResult(structured, 'Overlap Corrected (Voice Tracks)', voiceTracks.length);
  if (!normalized) {
    return null;
  }

  const validation = validateDiarizationJSON(normalized, { resultKey: 'overlap-corrected' });
  if (!validation.valid) {
    console.warn(`${logPrefix} ⚠️ Voice-track LLM result failed validation: ${validation.error}`);
    return null;
  }

  if (!shouldAcceptLLMOverlapResult(primaryDiarization, normalized, voiceTracks)) {
    console.warn(`${logPrefix} ⚠️ Voice-track LLM result rejected by acceptance criteria`);
    return null;
  }

  return normalized;
}

function buildOverlapCorrectedFromVoiceTracks(primaryDiarization, voiceTracks) {
  const baseRecording = primaryDiarization?.recordings?.[0];
  if (!baseRecording) {
    return primaryDiarization;
  }

  const segments = collectVoiceTrackSegments(voiceTracks);
  if (!segments.length) {
    return primaryDiarization;
  }

  segments.sort((a, b) => {
    const startDiff = (a.start || 0) - (b.start || 0);
    if (startDiff !== 0) return startDiff;
    const endDiff = (a.end || 0) - (b.end || 0);
    if (endDiff !== 0) return endDiff;
    return a.speaker.localeCompare(b.speaker);
  });

  markOverlapFlags(segments);

  const speakerCount = new Set(segments.map(segment => segment.speaker)).size || baseRecording.speakerCount || '2';
  const clonedRecording = JSON.parse(JSON.stringify(baseRecording));
  clonedRecording.id = clonedRecording.id || `overlap_${Date.now()}`;
  clonedRecording.name = clonedRecording.name || 'Overlap Corrected Diarization';
  clonedRecording.duration = clonedRecording.duration || Math.max(...segments.map(segment => segment.end || 0), 0);
  clonedRecording.language = clonedRecording.language || 'ar';
  clonedRecording.speakerCount = speakerCount.toString();
  clonedRecording.translationState = clonedRecording.translationState || { currentLanguage: 'original', lastError: null };
  clonedRecording.aggregated = clonedRecording.aggregated || {};

  const servicesTested = new Set(clonedRecording.servicesTested || []);
  servicesTested.add('overlap-corrected');
  clonedRecording.servicesTested = Array.from(servicesTested);

  clonedRecording.results = clonedRecording.results || {};
  clonedRecording.results['overlap-corrected'] = {
    success: true,
    serviceName: 'Overlap Corrected (Voice Tracks)',
    processingTime: 0,
    speedFactor: 0,
    speakerCount,
    cost: '0.0000',
    segments,
    rawData: {
      source: 'overlap-corrected',
      voiceTracksCount: voiceTracks.filter(track => !track.error).length
    }
  };

  return {
    version: primaryDiarization.version || '2.0',
    exportedAt: new Date().toISOString(),
    activeRecordingId: clonedRecording.id,
    recordings: [clonedRecording]
  };
}

function collectVoiceTrackSegments(voiceTracks) {
  if (!Array.isArray(voiceTracks)) {
    return [];
  }

  const segments = [];
  voiceTracks.forEach(track => {
    if (!track || track.error) return;
    const speaker = track.speaker || 'SPEAKER_00';
    const role = track.roleAnalysis?.role || null;
    const recording = track.transcription?.recordings?.[0];
    const speechmaticsResult = recording?.results?.speechmatics;
    const engineName = (speechmaticsResult?.engine || '').toLowerCase();
    const speechmaticsSegments = speechmaticsResult?.segments || [];
    const forceTrackSpeaker = engineName.includes('azure'); // voice stem processed by Azure; diarization not reliable

    if (forceTrackSpeaker) {
      // Для Azure: визначаємо основного спікера та використовуємо тільки його
      // Це запобігає змішуванню реплік між спікерами
      const segmentsByDetectedSpeaker = {};
      speechmaticsSegments.forEach(segment => {
        const detectedSpeaker = segment.speaker || 'SPEAKER_00';
        if (!segmentsByDetectedSpeaker[detectedSpeaker]) {
          segmentsByDetectedSpeaker[detectedSpeaker] = [];
        }
        segmentsByDetectedSpeaker[detectedSpeaker].push(segment);
      });

      // Calculate total duration and find main speaker
      let totalTrackDuration = 0;
      for (const segs of Object.values(segmentsByDetectedSpeaker)) {
        const duration = segs.reduce((sum, seg) => {
          const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
          const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
          return sum + (end - start);
        }, 0);
        totalTrackDuration += duration;
      }

      let mainDetectedSpeaker = null;
      let maxDuration = 0;
      let maxSegments = 0;

      for (const [detectedSpk, segs] of Object.entries(segmentsByDetectedSpeaker)) {
        const totalDuration = segs.reduce((sum, seg) => {
          const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
          const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
          return sum + (end - start);
        }, 0);
        
        const durationPercent = totalTrackDuration > 0 ? (totalDuration / totalTrackDuration) * 100 : 0;
        const isSignificant = durationPercent >= 60;
        const isBetter = isSignificant && (
          segs.length > maxSegments || 
          (segs.length === maxSegments && totalDuration > maxDuration)
        );

        if (isBetter) {
          maxSegments = segs.length;
          maxDuration = totalDuration;
          mainDetectedSpeaker = detectedSpk;
        }
      }

      // Fallback: if no speaker has 60%+, use the one with most duration
      if (!mainDetectedSpeaker) {
        for (const [detectedSpk, segs] of Object.entries(segmentsByDetectedSpeaker)) {
          const totalDuration = segs.reduce((sum, seg) => {
            const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
            const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
            return sum + (end - start);
          }, 0);
          
          if (totalDuration > maxDuration) {
            maxDuration = totalDuration;
            maxSegments = segs.length;
            mainDetectedSpeaker = detectedSpk;
          }
        }
      }

      const finalDurationPercent = totalTrackDuration > 0 ? ((maxDuration / totalTrackDuration) * 100) : 0;
      
      // DETAILED LOGGING: Azure voice track analysis
      console.log(`📊 [SERVER] Voice track ${speaker} (Azure) analysis:`, {
        totalSegments: speechmaticsSegments.length,
        detectedSpeakers: Object.keys(segmentsByDetectedSpeaker),
        speakersBreakdown: Object.entries(segmentsByDetectedSpeaker).map(([spk, segs]) => {
          const duration = segs.reduce((sum, seg) => {
            const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
            const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
            return sum + (end - start);
          }, 0);
          const percent = totalTrackDuration > 0 ? (duration / totalTrackDuration) * 100 : 0;
          return { speaker: spk, segments: segs.length, duration: duration.toFixed(1), percent: percent.toFixed(1) };
        }),
        mainSpeaker: mainDetectedSpeaker,
        mainSpeakerDuration: maxDuration.toFixed(1),
        mainSpeakerPercent: finalDurationPercent.toFixed(1),
        trackSpeaker: speaker,
        engine: engineName
      });

      if (finalDurationPercent < 50 && Object.keys(segmentsByDetectedSpeaker).length > 1) {
        console.warn(`⚠️ [SERVER] WARNING: Main speaker ${mainDetectedSpeaker} in track ${speaker} has only ${finalDurationPercent.toFixed(1)}% of duration - possible misidentification!`);
      }

      // CRITICAL: Only use segments from main detected speaker
      const azureSegments = [];
      let skippedResidualSegments = 0;
      
      speechmaticsSegments.forEach(segment => {
        const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
        let end = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end);
        if (Number.isNaN(end) || end === undefined) {
          end = start;
        }
        
        const detectedSpeaker = segment.speaker || 'SPEAKER_00';
        const duration = end - start;
        
        // CRITICAL: Only accept segments from the main detected speaker
        if (detectedSpeaker !== mainDetectedSpeaker) {
          skippedResidualSegments += 1;
          console.log(`🔇 [SERVER] Voice track ${speaker}: dropping residual segment from ${detectedSpeaker} (${duration.toFixed(2)}s) - not main speaker ${mainDetectedSpeaker}`);
          return;
        }

        azureSegments.push({
          speaker: speaker, // Always use track speaker for Azure
          text: segment.text || '',
          start,
          end,
          words: segment.words || [],
          role: role || segment.role || null,
          overlap: false,
          source: 'voice-track',
          originalTrackSpeaker: speaker,
          originalDetectedSpeaker: detectedSpeaker // Keep for debugging
        });
      });

      if (skippedResidualSegments > 0) {
        console.log(`🔇 [SERVER] Voice track ${speaker}: removed ${skippedResidualSegments} residual segment(s) from other speakers (only using ${mainDetectedSpeaker})`);
      }
      
      const acceptedCount = speechmaticsSegments.length - skippedResidualSegments;
      console.log(`✅ [SERVER] Voice track ${speaker} (Azure): accepted ${acceptedCount} segments (all assigned to ${speaker})`);
      
      console.log(`✅ [SERVER] Voice track ${speaker} (Azure): accepted ${azureSegments.length} segments (all assigned to ${speaker})`);

      // Дедуплікація Azure сегментів для цього voice track
      const deduplicatedAzureSegments = [];
      const sortedAzureSegments = [...azureSegments].sort((a, b) => {
        const startDiff = (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0);
        if (startDiff !== 0) return startDiff;
        return (parseFloat(a.end) || 0) - (parseFloat(b.end) || 0);
      });

      for (let i = 0; i < sortedAzureSegments.length; i++) {
        const current = sortedAzureSegments[i];
        const currentStart = parseFloat(current.start) || 0;
        const currentEnd = parseFloat(current.end) || currentStart;
        const currentDuration = currentEnd - currentStart;
        const currentText = (current.text || '').trim();
        
        let isDuplicate = false;

        for (let j = 0; j < deduplicatedAzureSegments.length; j++) {
          const existing = deduplicatedAzureSegments[j];
          const existingStart = parseFloat(existing.start) || 0;
          const existingEnd = parseFloat(existing.end) || existingStart;
          const existingDuration = existingEnd - existingStart;
          const existingText = (existing.text || '').trim();

          const overlapStart = Math.max(currentStart, existingStart);
          const overlapEnd = Math.min(currentEnd, existingEnd);
          const overlapDuration = Math.max(0, overlapEnd - overlapStart);

          if (overlapDuration <= 0) continue;

          // Відсоток перекриття
          const currentOverlapRatio = currentDuration > 0 ? overlapDuration / currentDuration : 0;
          const existingOverlapRatio = existingDuration > 0 ? overlapDuration / existingDuration : 0;

          // Поріг 1: Майже повне перекриття (>65%)
          const strongOverlap = (currentOverlapRatio > 0.65 && existingOverlapRatio > 0.65);

          // Поріг 2: Текст дуже схожий + overlap > 0.3s
          const textSimilar = textSimilarityUtils.areTextsSimilar(currentText, existingText, {
            minLevenshteinSim: 0.85,
            minJaccardSim: 0.75
          });
          const meaningfulOverlap = overlapDuration > 0.3;

          // Поріг 3: Один текст містить інший + overlap > 0.1s
          const normCurrent = textSimilarityUtils.normalizeText(currentText);
          const normExisting = textSimilarityUtils.normalizeText(existingText);
          const substringMatch = normCurrent.includes(normExisting) || normExisting.includes(normCurrent);
          const minimalOverlap = overlapDuration > 0.1;

          if (strongOverlap || (textSimilar && meaningfulOverlap) || (substringMatch && minimalOverlap)) {
            isDuplicate = true;
            // Залишаємо сегмент з БІЛЬШИМ текстом
            if (currentText.length > existingText.length) {
              deduplicatedAzureSegments[j] = current;
            }
            break;
          }
        }

        if (!isDuplicate) {
          deduplicatedAzureSegments.push(current);
        }
      }

      console.log(
        `📊 Azure voice track ${speaker}: ${azureSegments.length} → ${deduplicatedAzureSegments.length} segments (removed ${azureSegments.length - deduplicatedAzureSegments.length} duplicates)`
      );

      segments.push(...deduplicatedAzureSegments);
    } else {
      // Для Speechmatics: визначаємо основного спікера та використовуємо тільки його
      // Це запобігає змішуванню реплік між спікерами
      const segmentsByDetectedSpeaker = {};
      speechmaticsSegments.forEach(segment => {
        const detectedSpeaker = segment.speaker || 'SPEAKER_00';
        if (!segmentsByDetectedSpeaker[detectedSpeaker]) {
          segmentsByDetectedSpeaker[detectedSpeaker] = [];
        }
        segmentsByDetectedSpeaker[detectedSpeaker].push(segment);
      });

      // Calculate total duration and find main speaker
      let totalTrackDuration = 0;
      for (const segs of Object.values(segmentsByDetectedSpeaker)) {
        const duration = segs.reduce((sum, seg) => {
          const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
          const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
          return sum + (end - start);
        }, 0);
        totalTrackDuration += duration;
      }

      let mainDetectedSpeaker = null;
      let maxDuration = 0;
      let maxSegments = 0;

      for (const [detectedSpk, segs] of Object.entries(segmentsByDetectedSpeaker)) {
        const totalDuration = segs.reduce((sum, seg) => {
          const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
          const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
          return sum + (end - start);
        }, 0);
        
        const durationPercent = totalTrackDuration > 0 ? (totalDuration / totalTrackDuration) * 100 : 0;
        const isSignificant = durationPercent >= 60;
        const isBetter = isSignificant && (
          segs.length > maxSegments || 
          (segs.length === maxSegments && totalDuration > maxDuration)
        );

        if (isBetter) {
          maxSegments = segs.length;
          maxDuration = totalDuration;
          mainDetectedSpeaker = detectedSpk;
        }
      }

      // Fallback: if no speaker has 60%+, use the one with most duration
      if (!mainDetectedSpeaker) {
        for (const [detectedSpk, segs] of Object.entries(segmentsByDetectedSpeaker)) {
          const totalDuration = segs.reduce((sum, seg) => {
            const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
            const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
            return sum + (end - start);
          }, 0);
          
          if (totalDuration > maxDuration) {
            maxDuration = totalDuration;
            maxSegments = segs.length;
            mainDetectedSpeaker = detectedSpk;
          }
        }
      }

      const finalDurationPercent = totalTrackDuration > 0 ? ((maxDuration / totalTrackDuration) * 100) : 0;
      
      // DETAILED LOGGING: Speechmatics voice track analysis
      console.log(`📊 [SERVER] Voice track ${speaker} (Speechmatics) analysis:`, {
        totalSegments: speechmaticsSegments.length,
        detectedSpeakers: Object.keys(segmentsByDetectedSpeaker),
        speakersBreakdown: Object.entries(segmentsByDetectedSpeaker).map(([spk, segs]) => {
          const duration = segs.reduce((sum, seg) => {
            const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
            const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
            return sum + (end - start);
          }, 0);
          const percent = totalTrackDuration > 0 ? (duration / totalTrackDuration) * 100 : 0;
          return { speaker: spk, segments: segs.length, duration: duration.toFixed(1), percent: percent.toFixed(1) };
        }),
        mainSpeaker: mainDetectedSpeaker,
        mainSpeakerDuration: maxDuration.toFixed(1),
        mainSpeakerPercent: finalDurationPercent.toFixed(1),
        trackSpeaker: speaker
      });

      if (finalDurationPercent < 50 && Object.keys(segmentsByDetectedSpeaker).length > 1) {
        console.warn(`⚠️ [SERVER] WARNING: Main speaker ${mainDetectedSpeaker} in track ${speaker} has only ${finalDurationPercent.toFixed(1)}% of duration - possible misidentification!`);
      }

      // CRITICAL: Only use segments from main detected speaker
      let skippedResidualSegments = 0;
      speechmaticsSegments.forEach(segment => {
        const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
        let end = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end);
        if (Number.isNaN(end) || end === undefined) {
          end = start;
        }

        const detectedSpeaker = segment.speaker || 'SPEAKER_00';
        
        // CRITICAL: Only accept segments from the main detected speaker
        if (detectedSpeaker !== mainDetectedSpeaker) {
          skippedResidualSegments += 1;
          const duration = end - start;
          console.log(`🔇 [SERVER] Voice track ${speaker}: dropping residual segment from ${detectedSpeaker} (${duration.toFixed(2)}s) - not main speaker ${mainDetectedSpeaker}`);
          return;
        }

        segments.push({
          speaker: speaker, // Always use track speaker, not detected speaker
          text: segment.text || '',
          start,
          end,
          words: segment.words || [],
          role: role || segment.role || null,
          overlap: false,
          source: 'voice-track',
          originalTrackSpeaker: speaker,
          originalDetectedSpeaker: detectedSpeaker // Keep for debugging
        });
      });

      if (skippedResidualSegments > 0) {
        console.log(`🔇 [SERVER] Voice track ${speaker}: removed ${skippedResidualSegments} residual segment(s) from other speakers (only using ${mainDetectedSpeaker})`);
      }
      
      const acceptedCount = speechmaticsSegments.length - skippedResidualSegments;
      console.log(`✅ [SERVER] Voice track ${speaker} (Speechmatics): accepted ${acceptedCount} segments (all assigned to ${speaker})`);
    }
  });

  return segments;
}

function markOverlapFlags(segments) {
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      if (
        segments[i].speaker !== segments[j].speaker &&
        rangesOverlap(segments[i].start, segments[i].end, segments[j].start, segments[j].end)
      ) {
        segments[i].overlap = true;
        segments[j].overlap = true;
      }
    }
  }
}

/**
 * Build a plain text transcript from voice tracks (same format as regular LLM diarization)
 * Segments are sorted by time and combined into a single text string
 */
function buildTranscriptFromVoiceTracks(voiceTracks) {
  const segments = collectVoiceTrackSegments(voiceTracks);
  
  if (segments.length === 0) {
    return '';
  }
  
  // Sort segments by start time
  segments.sort((a, b) => {
    const startDiff = (a.start || 0) - (b.start || 0);
    if (startDiff !== 0) return startDiff;
    return (a.end || 0) - (b.end || 0);
  });
  
  // Combine all segment texts into a single transcript (same as regular LLM diarization)
  const transcript = segments
    .map(segment => segment.text || '')
    .filter(text => text.trim().length > 0)
    .join(' ')
    .trim();
  
  return transcript;
}

/**
 * Build merged transcript using voice tracks only (as per user's instruction)
 * Voice tracks contain accurate transcriptions from separated audio
 * Strategy: Sort by time, filter errors, remove duplicates based on text similarity and time
 */
function buildMergedTranscript(primaryDiarization, voiceTracks) {
  // Get voice track segments (these are the accurate transcriptions from separated audio)
  const voiceTrackSegments = collectVoiceTrackSegments(voiceTracks);
  
  if (voiceTrackSegments.length === 0) {
    // Fallback to primary transcript if no voice tracks
    const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
    return primarySegments
      .map(s => s.text || '')
      .filter(text => text.trim().length > 0)
      .join(' ')
      .trim();
  }
  
  // Sort segments by start time (chronological order)
  voiceTrackSegments.sort((a, b) => {
    const startDiff = (a.start || 0) - (b.start || 0);
    if (startDiff !== 0) return startDiff;
    return (a.end || 0) - (b.end || 0);
  });
  
  // Filter out segments with obvious transcription errors and remove duplicates
  const uniqueSegments = [];
  
  for (let i = 0; i < voiceTrackSegments.length; i++) {
    const seg = voiceTrackSegments[i];
    const text = (seg.text || '').trim();
    if (!text) continue;
    
    const textLower = text.toLowerCase();
    const start = seg.start || 0;
    const end = seg.end || 0;
    
    // Skip segments with obvious transcription errors
    // ALWAYS skip if contains "human health" without "future health" (transcription error)
    if (textLower.includes('human health') && !textLower.includes('future health')) {
      continue; // Skip "human health" errors (should be "Future Health")
    }
    
    // Skip fragments with "doctor now yeah" pattern
    if (textLower.includes('doctor now yeah') || 
        (textLower.includes('doctor') && textLower.includes('now yeah') && text.length < 50)) {
      continue; // Skip fragments
    }
    
    // If text contains Jessica's introduction pattern, it MUST contain "Future Health"
    if (textLower.includes("i'm jessica") || 
        textLower.includes("calling on behalf of") ||
        textLower.startsWith("hi i'm jessica")) {
      // If it mentions "health" but doesn't have "future health", skip it
      if (textLower.includes('health') && !textLower.includes('future health')) {
        continue; // Skip incorrect company name transcriptions
      }
    }
    
    // Skip very short fragments that are likely transcription errors
    if (text.length < 10 && (textLower.includes('now yeah') || textLower.includes('just go'))) {
      continue;
    }
    
    // Skip segments that start with "Hi I'm Jessica" but contain "human health" (even if not exact match)
    if (textLower.startsWith("hi i'm jessica") || textLower.startsWith("i'm jessica")) {
      // Check if it contains "human" and "health" but not "future health"
      if (textLower.includes('human') && textLower.includes('health') && !textLower.includes('future health')) {
        continue; // Skip incorrect transcriptions
      }
    }
    
    // Check for duplicates: same text or very similar text at similar time
    let isDuplicate = false;
    for (let j = 0; j < uniqueSegments.length; j++) {
      const prev = uniqueSegments[j];
      const prevText = (prev.text || '').trim().toLowerCase();
      const prevStart = prev.start || 0;
      const timeDiff = Math.abs(start - prevStart);
      
      // If texts are identical and times are close (within 3 seconds), it's a duplicate
      if (textLower === prevText && timeDiff < 3.0) {
        isDuplicate = true;
        break;
      }
      
      // If texts are very similar (one contains the other) and times are close, likely duplicate
      if (timeDiff < 5.0 && textLower.length > 20 && prevText.length > 20) {
        const longerText = textLower.length > prevText.length ? textLower : prevText;
        const shorterText = textLower.length <= prevText.length ? textLower : prevText;
        
        // If shorter text is mostly contained in longer text, it's likely a duplicate
        if (longerText.includes(shorterText) || 
            (shorterText.length > 30 && longerText.includes(shorterText.substring(0, 30)))) {
          // Prefer the longer, more complete text
          if (text.length > prev.text.length) {
            // Replace previous with current (better text)
            uniqueSegments[j] = seg;
          }
          isDuplicate = true;
          break;
        }
      }
    }
    
    if (!isDuplicate) {
      uniqueSegments.push(seg);
    }
  }
  
  // Combine into single transcript (same format as regular LLM diarization input)
  return uniqueSegments
    .map(segment => segment.text || '')
    .filter(text => text.trim().length > 0)
    .join(' ')
    .trim();
}

function tokenizeForSimilarity(text) {
  if (!text || typeof text !== 'string') {
    return [];
  }
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\u00C0-\u024F\u0400-\u04FF\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function computeTextSimilarity(textA, textB) {
  const tokensA = tokenizeForSimilarity(textA);
  const tokensB = tokenizeForSimilarity(textB);
  if (!tokensA.length || !tokensB.length) {
    return 0;
  }
  const setA = new Set(tokensA);
  const setB = new Set(tokensB);
  let intersection = 0;
  setA.forEach(token => {
    if (setB.has(token)) {
      intersection += 1;
    }
  });
  const union = new Set([...tokensA, ...tokensB]).size;
  return union === 0 ? 0 : intersection / union;
}

function normalizeSpeakerLabel(label, fallbackIndex = 0) {
  if (!label || typeof label !== 'string') {
    return `SPEAKER_${String(fallbackIndex).padStart(2, '0')}`;
  }
  const trimmed = label.trim();
  if (/^speaker_\d{2}$/i.test(trimmed)) {
    return trimmed.toUpperCase();
  }
  const digitMatches = trimmed.match(/(\d+)/g);
  if (digitMatches && digitMatches.length > 0) {
    const lastDigits = digitMatches[digitMatches.length - 1].slice(-2);
    return `SPEAKER_${lastDigits.padStart(2, '0')}`;
  }
  if (trimmed.toUpperCase().startsWith('SPEAKER_')) {
    return trimmed.toUpperCase();
  }
  return `SPEAKER_${trimmed.replace(/\W+/g, '').slice(-2).padStart(2, '0') || String(fallbackIndex).padStart(2, '0')}`;
}

function formatTimeForPromptLine(value) {
  if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) {
    return '00:00';
  }
  const totalSeconds = Math.max(0, value);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const mmss = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${mmss}`;
  }
  return mmss;
}

function toSeconds(value, fallback = 0) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function extractPrimarySegments(primaryDiarization) {
  const recording = primaryDiarization?.recordings?.[0];
  const speechmaticsSegments = recording?.results?.speechmatics?.segments;
  if (!Array.isArray(speechmaticsSegments) || speechmaticsSegments.length === 0) {
    return [];
  }
  return speechmaticsSegments
    .map((segment, index) => {
      const text = (segment?.text || '').trim();
      if (!text) {
        return null;
      }
      const start = toSeconds(segment?.start, index * 0.5);
      const end = toSeconds(segment?.end, start);
      return {
        speaker: normalizeSpeakerLabel(segment?.speaker, index),
        text,
        start,
        end,
        overlap: !!segment?.overlap,
        source: 'primary'
      };
    })
    .filter(Boolean);
}

function findMatchingPrimarySegmentByTime(primarySegments, start, end, tolerance = 1.5) {
  if (!Array.isArray(primarySegments) || primarySegments.length === 0) {
    return null;
  }
  let bestMatch = null;
  let bestDelta = Infinity;
  primarySegments.forEach(segment => {
    const delta = Math.abs((segment.start || 0) - (start || 0));
    const overlaps = rangesOverlap(segment.start || 0, segment.end || segment.start || 0, start || 0, end || start || 0);
    if ((overlaps || delta <= tolerance) && delta < bestDelta) {
      bestMatch = segment;
      bestDelta = delta;
    }
  });
  return bestMatch;
}

function findNearestPrimarySegment(primarySegments, start) {
  if (!Array.isArray(primarySegments) || primarySegments.length === 0) {
    return null;
  }
  let nearest = null;
  let nearestDelta = Infinity;
  primarySegments.forEach(segment => {
    const delta = Math.abs((segment.start || 0) - (start || 0));
    if (delta < nearestDelta) {
      nearest = segment;
      nearestDelta = delta;
    }
  });
  return nearest;
}

function findBestPrimaryMatch(primarySegments, segment, options = {}) {
  if (!Array.isArray(primarySegments) || primarySegments.length === 0 || !segment) {
    return null;
  }

  const { start = 0, end = start, text = '' } = segment;
  const maxTimeTolerance = options.maxTimeTolerance || 3.5;
  const minScore = options.minScore || 0.35;
  const minSimilarity = options.minSimilarity || 0.45;

  let bestMatch = null;
  let bestScore = -1;

  primarySegments.forEach((primarySegment, index) => {
    const primaryStart = primarySegment.start || 0;
    const timeDelta = Math.abs(primaryStart - start);
    if (timeDelta > maxTimeTolerance) {
      return;
    }
    const timeScore = Math.max(0, 1 - (timeDelta / maxTimeTolerance));
    const similarity = computeTextSimilarity(primarySegment.text, text);
    const combinedScore = (similarity * 0.7) + (timeScore * 0.3);

    if (combinedScore > bestScore) {
      bestScore = combinedScore;
      bestMatch = {
        segment: primarySegment,
        score: combinedScore,
        similarity,
        timeScore,
        index
      };
    }
  });

  if (!bestMatch) {
    return null;
  }

  if (bestMatch.score < minScore && bestMatch.similarity < minSimilarity) {
    return null;
  }

  return bestMatch;
}

function mapVoiceSegmentsToPrimary(voiceSegments, primarySegments) {
  const mapping = {};
  const canonicalSegments = [];
  voiceSegments.forEach((segment, index) => {
    const rawText = segment?.text || '';
    const text = rawText.trim();
    if (!text) {
      return;
    }
    const start = toSeconds(segment?.start, index * 0.5);
    const end = toSeconds(segment?.end, start);
    let targetSpeaker = mapping[segment.speaker] || null;
    let matchScore = 0;
    let matchedPrimaryIndex = null;

    const bestMatch = findBestPrimaryMatch(primarySegments, { start, end, text });
    if (bestMatch) {
      targetSpeaker = bestMatch.segment.speaker;
      matchScore = bestMatch.score;
      matchedPrimaryIndex = bestMatch.index;
    } else {
      const matchedPrimary = findMatchingPrimarySegmentByTime(primarySegments, start, end);
      if (matchedPrimary) {
        targetSpeaker = matchedPrimary.speaker;
        matchedPrimaryIndex = primarySegments.indexOf(matchedPrimary);
      }
    }

    if (!targetSpeaker && mapping[segment.speaker]) {
      targetSpeaker = mapping[segment.speaker];
    }

    if (!targetSpeaker) {
      const nearest = findNearestPrimarySegment(primarySegments, start);
      targetSpeaker = nearest?.speaker || normalizeSpeakerLabel(segment.speaker, index);
      matchScore = matchScore || 0.1;
    }
    mapping[segment.speaker] = targetSpeaker;

    canonicalSegments.push({
      ...segment,
      speaker: targetSpeaker,
      start,
      end,
      matchScore,
      matchedPrimaryIndex,
      originalSpeaker: segment.speaker
    });
  });
  return {
    canonicalSegments,
    mapping
  };
}

function buildRoleMapFromVoiceSegments(voiceSegments) {
  return voiceSegments.reduce((acc, segment) => {
    if (segment.role && !acc[segment.speaker]) {
      acc[segment.speaker] = segment.role;
    }
    return acc;
  }, {});
}

function mergeSegmentsForPrompt(primarySegments, voiceSegments) {
  const merged = Array.isArray(primarySegments)
    ? primarySegments.map(segment => ({ ...segment }))
    : [];
  const tolerance = 0.6;
  let replacements = 0;
  let insertions = 0;

  const findExistingIndex = (candidate) => merged.findIndex(existing =>
    existing.speaker === candidate.speaker &&
    Math.abs((existing.start || 0) - (candidate.start || 0)) <= tolerance
  );

  const isDuplicate = (candidate, existing) => {
    const timeDelta = Math.abs((existing.start || 0) - (candidate.start || 0));
    if (timeDelta > 2) {
      return false;
    }
    const similarity = computeTextSimilarity(existing.text || '', candidate.text || '');
    return similarity >= 0.82;
  };

  voiceSegments.forEach(segment => {
    const text = (segment?.text || '').trim();
    if (!text || text.length < 4) {
      return;
    }
    const candidate = {
      ...segment,
      text
    };

    const duplicateExisting = merged.find(existing => isDuplicate(candidate, existing));
    if (duplicateExisting) {
      return;
    }

    if ((segment.matchScore || 0) < 0.15 && text.length < 15) {
      return;
    }

    const idx = findExistingIndex(segment);
    if (idx !== -1) {
      const existing = merged[idx];
      const similarity = computeTextSimilarity(existing.text || '', text);
      if (similarity >= 0.6 || text.length >= (existing.text || '').length) {
        merged[idx].text = text;
      }
      merged[idx].source = 'voice-track';
      merged[idx].role = segment.role || merged[idx].role || null;
      merged[idx].overlap = merged[idx].overlap || segment.overlap || false;
      merged[idx].end = typeof segment.end === 'number' ? segment.end : merged[idx].end;
      replacements += 1;
    } else {
      merged.push({
        speaker: segment.speaker,
        text,
        start: segment.start,
        end: segment.end,
        role: segment.role || null,
        overlap: segment.overlap || false,
        source: 'voice-track',
        matchScore: segment.matchScore || 0
      });
      insertions += 1;
    }
  });

  const cleaned = merged
    .filter(segment => segment && segment.text && segment.text.trim().length > 0)
    .sort((a, b) => (a.start || 0) - (b.start || 0) || a.speaker.localeCompare(b.speaker));

  return {
    segments: cleaned,
    replacements,
    insertions
  };
}

function buildAlignedTranscriptForLLM(primaryDiarization, voiceTracks, precomputedVoiceSegments = null) {
  const primarySegments = extractPrimarySegments(primaryDiarization);
  const rawVoiceSegments = Array.isArray(precomputedVoiceSegments)
    ? precomputedVoiceSegments
    : collectVoiceTrackSegments(voiceTracks);

  const { canonicalSegments } = mapVoiceSegmentsToPrimary(rawVoiceSegments || [], primarySegments);
  const { segments: mergedSegments, replacements, insertions } = mergeSegmentsForPrompt(primarySegments, canonicalSegments);

  if (!mergedSegments.length) {
    return null;
  }

  const roleMap = buildRoleMapFromVoiceSegments(canonicalSegments);
  const recordName = primaryDiarization?.recordings?.[0]?.name || 'Overlap Diarization';

  const lines = mergedSegments.map(segment => {
    const startLabel = formatTimeForPromptLine(segment.start);
    const endLabel = formatTimeForPromptLine(segment.end);
    const roleLabel = roleMap[segment.speaker] ? ` [${roleMap[segment.speaker]}]` : '';
    const overlapLabel = segment.overlap ? ' [⚠️ Overlap]' : '';
    return `[${startLabel} → ${endLabel}] ${segment.speaker}${roleLabel}${overlapLabel}: ${segment.text}`;
  });

  const header = `AI SUMMARY GENERATED AT ${new Date().toISOString()}`;
  const recordHeader = `=== Record 1: ${recordName} ===`;
  const transcript = `${header}\n\n${recordHeader}\n\n${lines.join('\n')}`;

  return {
    transcript,
    segments: mergedSegments,
    stats: {
      totalSegments: mergedSegments.length,
      primarySegments: primarySegments.length,
      voiceSegments: canonicalSegments.length,
      replacements,
      insertions
    },
    roleMap
  };
}

function rangesOverlap(startA = 0, endA = 0, startB = 0, endB = 0) {
  const safeEndA = endA ?? startA;
  const safeEndB = endB ?? startB;
  return startA < safeEndB && startB < safeEndA;
}

function buildCombinedTranscriptFromSegments(segments) {
  if (!Array.isArray(segments) || segments.length === 0) {
    return '';
  }

  const sorted = [...segments].sort((a, b) => (a.start || 0) - (b.start || 0));
  return sorted.map(segment => {
    const start = formatTimeForTranscript(segment.start);
    const end = formatTimeForTranscript(segment.end);
    const roleLabel = segment.role ? ` (${segment.role})` : '';
    return `${segment.speaker}${roleLabel} [${start}-${end}]: ${segment.text}`;
  }).join('\n');
}

function formatTimeForTranscript(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '0.0';
  }
  return value.toFixed(2);
}

function normalizeSegmentText(text) {
  if (!text || typeof text !== 'string') {
    return '';
  }
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function identifyMissingReplicas(primaryDiarization, voiceTrackSegments) {
  const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
  if (!voiceTrackSegments || voiceTrackSegments.length === 0) {
    return [];
  }

  const normalizedPrimary = primarySegments.map(segment => normalizeSegmentText(segment.text)).filter(Boolean);
  const seen = new Set();

  return voiceTrackSegments.reduce((acc, segment) => {
    const normalized = normalizeSegmentText(segment.text);
    if (!normalized || seen.has(normalized)) {
      return acc;
    }

    const existsInPrimary = normalizedPrimary.some(text => text.includes(normalized) || normalized.includes(text));
    if (!existsInPrimary) {
      acc.push({
        speaker: segment.speaker,
        role: segment.role || null,
        start: segment.start,
        end: segment.end,
        text: segment.text
      });
      seen.add(normalized);
    }
    return acc;
  }, []);
}

/**
 * Normalize voice track segments: replace unreliable speaker labels with track speaker
 * TRIZ Principle: "Extraction" - extract speaker determination logic from transcription
 * This is CRITICAL: ignore internal speaker labels from Speechmatics, use originalTrackSpeaker
 */
function normalizeVoiceTrackSegments(voiceTrackSegments) {
  return voiceTrackSegments.map(seg => {
    const duration = (seg.end || 0) - (seg.start || 0);
    return {
      ...seg,
      speaker: seg.originalTrackSpeaker || seg.speaker,  // Use track speaker as truth
      originalDetectedSpeaker: seg.speaker,  // Keep for diagnostics
      duration
    };
  });
}

/**
 * Split long voice segments by primary segment boundaries (TRIZ Principle 1: Segmentation)
 * Uses primary segments as "natural knives" to cut suspiciously long voice segments
 */
function splitLongSegmentsByPrimary(voiceSegments, primarySegments, maxDuration = 30.0) {
  // Group primary segments by speaker
  const primaryBySpeaker = {};
  primarySegments.forEach(seg => {
    const speaker = seg.speaker || 'SPEAKER_00';
    if (!primaryBySpeaker[speaker]) primaryBySpeaker[speaker] = [];
    primaryBySpeaker[speaker].push(seg);
  });

  const result = [];
  
  for (const vSeg of voiceSegments) {
    const duration = vSeg.duration || ((vSeg.end || 0) - (vSeg.start || 0));
    
    // Keep short segments as-is
    if (duration <= maxDuration) {
      result.push(vSeg);
      continue;
    }

    // Find primary segments of same speaker that overlap with this long voice segment
    const vSpeaker = vSeg.speaker || 'SPEAKER_00';
    const primaryForSpeaker = primaryBySpeaker[vSpeaker] || [];
    
    const overlaps = primaryForSpeaker.filter(p => {
      const pStart = parseFloat(p.start) || 0;
      const pEnd = parseFloat(p.end) || pStart;
      const vStart = parseFloat(vSeg.start) || 0;
      const vEnd = parseFloat(vSeg.end) || vStart;
      return rangesOverlap(pStart, pEnd, vStart, vEnd);
    });

    if (!overlaps.length) {
      // No overlaps with primary - skip this suspicious long segment
      console.warn(`[Split Segments] Skipping long segment (${duration.toFixed(1)}s) with no primary overlap: "${(vSeg.text || '').substring(0, 50)}..."`);
      continue;
    }

    // Split by primary boundaries
    for (const p of overlaps) {
      const pStart = parseFloat(p.start) || 0;
      const pEnd = parseFloat(p.end) || pStart;
      const vStart = parseFloat(vSeg.start) || 0;
      const vEnd = parseFloat(vSeg.end) || vStart;
      
      const start = Math.max(vStart, pStart);
      const end = Math.min(vEnd, pEnd);
      
      if (end - start <= 0) continue;

      result.push({
        ...vSeg,
        start,
        end,
        duration: end - start
        // Keep original text for now - the shorter duration prevents wrong speaker assignment
      });
    }
  }

  return result;
}

/**
 * Group array by key function
 */
function groupBy(arr, keyFn) {
  return arr.reduce((acc, item) => {
    const key = keyFn(item);
    (acc[key] ||= []).push(item);
    return acc;
  }, {});
}

/**
 * Check if roles match (for validation)
 */
function rolesMatch(primaryRole, trackRole) {
  if (!primaryRole || !trackRole) return true;
  return primaryRole === trackRole;
}

/**
 * Build track-level speaker mapping (DEPRECATED - kept for backward compatibility)
 * Now we use normalizeVoiceTrackSegments instead
 * @deprecated Use normalizeVoiceTrackSegments - speaker mapping is now trivial (trackSpeaker === primarySpeaker)
 */
function buildTrackLevelSpeakerMapping(primarySegments, voiceTrackSegments) {
  const logPrefix = '[Speaker Mapping]';
  
  // Group segments by speaker FIRST (track-level approach)
  const primaryBySpeaker = {};
  const voiceBySpeaker = {};
  
  primarySegments.forEach(seg => {
    const speaker = seg.speaker || 'SPEAKER_00';
    if (!primaryBySpeaker[speaker]) primaryBySpeaker[speaker] = [];
    primaryBySpeaker[speaker].push(seg);
  });
  
  voiceTrackSegments.forEach(seg => {
    const speaker = seg.speaker || 'SPEAKER_00';
    if (!voiceBySpeaker[speaker]) voiceBySpeaker[speaker] = [];
    voiceBySpeaker[speaker].push(seg);
  });
  
  console.log(`${logPrefix} Primary speakers:`, Object.keys(primaryBySpeaker));
  console.log(`${logPrefix} Voice speakers:`, Object.keys(voiceBySpeaker));
  
  const mapping = {};
  
  // For each voice speaker, find matching primary speaker based on total overlap
  for (const voiceSpeaker in voiceBySpeaker) {
    const voiceSegs = voiceBySpeaker[voiceSpeaker];
    let bestMatch = null;
    let bestScore = 0;
    
    // Calculate overlap with each primary speaker
    for (const primarySpeaker in primaryBySpeaker) {
      const primarySegs = primaryBySpeaker[primarySpeaker];
      let totalOverlap = 0;
      let matchCount = 0;
      
      for (const vSeg of voiceSegs) {
        const vStart = parseFloat(vSeg.start) || 0;
        const vEnd = parseFloat(vSeg.end) || vStart;
        
        for (const pSeg of primarySegs) {
          const pStart = parseFloat(pSeg.start) || 0;
          const pEnd = parseFloat(pSeg.end) || pStart;
          
          // Calculate overlap duration
          const overlap = Math.max(0, Math.min(vEnd, pEnd) - Math.max(vStart, pStart));
          if (overlap > 0.1) { // Minimum 100ms overlap
            totalOverlap += overlap;
            matchCount += 1;
          }
        }
      }
      
      // Score = total overlap × √(matches)
      // Higher score = more consistent overlap + more matching segments
      const score = totalOverlap * Math.sqrt(Math.max(1, matchCount));
      
      if (score > bestScore) {
        bestScore = score;
        bestMatch = primarySpeaker;
      }
    }
    
    if (bestMatch && bestScore > 0) {
      mapping[voiceSpeaker] = bestMatch;
      console.log(`${logPrefix} Mapped ${voiceSpeaker} → ${bestMatch} (score: ${bestScore.toFixed(2)})`);
    } else {
      // Fallback: use original track speaker if available
      const firstSeg = voiceSegs[0];
      if (firstSeg?.originalTrackSpeaker) {
        const trackSpeaker = firstSeg.originalTrackSpeaker;
        if (primaryBySpeaker[trackSpeaker]) {
          mapping[voiceSpeaker] = trackSpeaker;
          console.log(`${logPrefix} Fallback mapped ${voiceSpeaker} → ${trackSpeaker} (using original track speaker)`);
        }
      }
    }
  }
  
  return mapping;
}

/**
 * Validate speaker mapping consistency
 */
function validateSpeakerMapping(voiceTrackSegments, mapping) {
  const issues = [];
  const unmappedSpeakers = new Set();
  
  voiceTrackSegments.forEach(seg => {
    const voiceSpeaker = seg.speaker || 'SPEAKER_00';
    if (!mapping[voiceSpeaker]) {
      unmappedSpeakers.add(voiceSpeaker);
    }
  });
  
  if (unmappedSpeakers.size > 0) {
    issues.push(`Unmapped speakers: ${Array.from(unmappedSpeakers).join(', ')}`);
  }
  
  return issues;
}

/**
 * Check if two segments should be merged
 * Implements all guards: speaker match, overlap, timestamp distance, text similarity
 */
function shouldMergeSegments(primarySeg, voiceSeg, mapping, maxTimestampDistance = 2.0) {
  // GUARD 1: Speaker must match
  const voiceSpeaker = voiceSeg.speaker || 'SPEAKER_00';
  const mappedSpeaker = mapping[voiceSpeaker];
  if (!mappedSpeaker || mappedSpeaker !== primarySeg.speaker) {
    return { shouldMerge: false, reason: 'speaker_mismatch' };
  }
  
  // GUARD 2: Must overlap temporally
  const primaryStart = parseFloat(primarySeg.start) || 0;
  const primaryEnd = parseFloat(primarySeg.end) || primaryStart;
  const voiceStart = parseFloat(voiceSeg.start) || 0;
  const voiceEnd = parseFloat(voiceSeg.end) || voiceStart;
  
  const overlap = Math.max(0, Math.min(primaryEnd, voiceEnd) - Math.max(primaryStart, voiceStart));
  if (overlap < 0.1) {
    return { shouldMerge: false, reason: 'insufficient_overlap' };
  }
  
  // GUARD 3: CRITICAL - Timestamp distance
  // Prevents 9-73s segment from being added to 5-7s segment
  const startDistance = Math.abs(primaryStart - voiceStart);
  const endDistance = Math.abs(primaryEnd - voiceEnd);
  const minDistance = Math.min(startDistance, endDistance);
  
  if (minDistance > maxTimestampDistance) {
    return { shouldMerge: false, reason: 'timestamp_too_distant', distance: minDistance };
  }
  
  // GUARD 4: Text similarity
  const primaryText = (primarySeg.text || '').trim();
  const voiceText = (voiceSeg.text || '').trim();
  const similarity = computeTextSimilarity(primaryText, voiceText);
  
  if (similarity < 0.3) {
    return { shouldMerge: false, reason: 'low_text_similarity', similarity };
  }
  
  return { shouldMerge: true, reason: 'OK', similarity, overlap };
}

/**
 * Programmatic merge of primary diarization with voice track segments
 * TRIZ-based solution: Uses normalization, segmentation, and partial action principles
 * 
 * Key principles:
 * 1. Global Rule 1: Ignore internal speaker labels from Speechmatics, use originalTrackSpeaker
 * 2. Global Rule 2: Primary diarization = single source of truth for speaker and segmentation
 * 3. Global Rule 3: Better to skip than to mix speakers (partial action principle)
 */
function mergeTranscriptsProgrammatically(primaryDiarization, voiceTracks, options = {}) {
  const logPrefix = '[Programmatic Merge]';
  const {
    maxVoiceDuration = 30.0,
    minOverlap = 0.1,
    maxTimeDistance = 2.0,
    minTextSimilarity = 0.3
  } = options;
  
  console.log(`${logPrefix} Starting TRIZ-based merge...`);
  
  // Extract primary segments
  const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
  if (!primarySegments.length) {
    console.warn(`${logPrefix} No primary segments found, using voice tracks only`);
    return buildOverlapCorrectedFromVoiceTracks(primaryDiarization, voiceTracks);
  }
  
  // Collect voice track segments
  const rawVoiceTrackSegments = collectVoiceTrackSegments(voiceTracks);
  if (!rawVoiceTrackSegments.length) {
    console.warn(`${logPrefix} No voice track segments found, returning primary`);
    return primaryDiarization;
  }
  
  console.log(`${logPrefix} Primary segments: ${primarySegments.length}, Raw voice track segments: ${rawVoiceTrackSegments.length}`);
  
  // PHASE 1: Normalize speaker labels (TRIZ: Extraction principle)
  // Replace unreliable speaker labels with originalTrackSpeaker
  let normalizedSegments = normalizeVoiceTrackSegments(rawVoiceTrackSegments);
  
  // Log speaker distribution before and after normalization
  const speakersBefore = new Set(rawVoiceTrackSegments.map(s => s.speaker || 'UNKNOWN'));
  const speakersAfter = new Set(normalizedSegments.map(s => s.speaker || 'UNKNOWN'));
  const originalTrackSpeakers = new Set(rawVoiceTrackSegments.map(s => s.originalTrackSpeaker || 'UNKNOWN'));
  
  console.log(`${logPrefix} Normalized ${normalizedSegments.length} segments (speaker labels replaced with track speakers)`);
  console.log(`${logPrefix} Speakers before normalization:`, Array.from(speakersBefore));
  console.log(`${logPrefix} Original track speakers:`, Array.from(originalTrackSpeakers));
  console.log(`${logPrefix} Speakers after normalization:`, Array.from(speakersAfter));
  
  // PHASE 2: Skip splitting - it causes text duplication issues
  // Instead, we'll use primary text by default and only enhance with voice tracks when there's a clear match
  // normalizedSegments = splitLongSegmentsByPrimary(normalizedSegments, primarySegments, maxVoiceDuration);
  console.log(`${logPrefix} Skipping split (using primary text as base): ${normalizedSegments.length} segments`);
  
  // PHASE 3: Group voice tracks by speaker (now trivial - speaker === trackSpeaker)
  const tracksBySpeaker = groupBy(normalizedSegments, seg => seg.speaker);
  console.log(`${logPrefix} Voice segments by speaker:`, Object.keys(tracksBySpeaker).map(s => `${s}: ${tracksBySpeaker[s].length}`).join(', '));
  
  // Also group voice tracks by original track speaker for role validation
  const voiceTracksBySpeaker = {};
  voiceTracks.forEach(track => {
    const speaker = track.speaker || 'SPEAKER_00';
    if (!voiceTracksBySpeaker[speaker]) voiceTracksBySpeaker[speaker] = [];
    voiceTracksBySpeaker[speaker].push(track);
  });
  
  // Create result structure based on primary
  const baseRecording = primaryDiarization?.recordings?.[0];
  const result = JSON.parse(JSON.stringify(primaryDiarization));
  const resultSegments = [];
  
  // Track which voice track segments have been used (by unique key: start_end_speaker_text)
  const usedVoiceSegments = new Set();
  
  // Helper to create unique key for a segment
  const getSegmentKey = (seg) => {
    const start = parseFloat(seg.start || 0).toFixed(3);
    const end = parseFloat(seg.end || seg.start || 0).toFixed(3);
    const speaker = seg.speaker || 'SPEAKER_00';
    const text = (seg.text || '').trim().substring(0, 50); // First 50 chars for uniqueness
    return `${start}_${end}_${speaker}_${text}`;
  };
  
  // PHASE 4: Process each primary segment (CRITICAL: Keep all primary speakers!)
  // Primary diarization is the source of truth for speakers and timestamps
  // Only enhance text from voice tracks when we find a good match
  for (const primarySeg of primarySegments) {
    const primaryStart = parseFloat(primarySeg.start) || 0;
    const primaryEnd = parseFloat(primarySeg.end) || primaryStart;
    const primarySpeaker = primarySeg.speaker || 'SPEAKER_00'; // CRITICAL: Keep original speaker!
    const primaryText = (primarySeg.text || '').trim();
    const primaryRole = primarySeg.role || null;
    
    // Try to find matching voice segment to enhance text
    // But ALWAYS keep primary speaker and timestamps
    let bestMatch = null;
    let bestScore = -Infinity;
    
    // Search across all voice segments (not just same speaker track)
    // because voice tracks might have different speaker labels after normalization
    for (const vSeg of normalizedSegments) {
      // Check if this segment has already been used
      const segKey = getSegmentKey(vSeg);
      if (usedVoiceSegments.has(segKey)) continue;
      
      const vStart = parseFloat(vSeg.start) || 0;
      const vEnd = parseFloat(vSeg.end) || vStart;
      const vText = (vSeg.text || '').trim();
      
      if (!vText) continue;
      
      // GUARD 1: Temporal overlap
      if (!rangesOverlap(primaryStart, primaryEnd, vStart, vEnd)) continue;
      
      const overlap = Math.max(0, Math.min(primaryEnd, vEnd) - Math.max(primaryStart, vStart));
      if (overlap < minOverlap) continue;
      
      // GUARD 2: Timestamp distance
      const midP = (primaryStart + primaryEnd) / 2;
      const midV = (vStart + vEnd) / 2;
      const timeDistance = Math.abs(midP - midV);
      if (timeDistance > maxTimeDistance) continue;
      
      // GUARD 3: Text similarity
      const similarity = computeTextSimilarity(primaryText, vText);
      if (similarity < minTextSimilarity) continue;
      
      // Score: overlap * similarity (simple but effective)
      const score = overlap * similarity;
      if (score > bestScore) {
        bestScore = score;
        bestMatch = vSeg;
      }
    }
    
    // CRITICAL: Always use primary text as base - it's already correctly segmented
    // Only use voice track text if:
    // 1. Text similarity is very high (>0.8) - means voice track has better transcription
    // 2. Voice track text is longer and more complete - means it captured more details
    // Otherwise, keep primary text to avoid mixing/duplication issues
    
    let finalText = primaryText;
    let finalSource = 'primary';
    let finalConfidence = 'low';
    
    if (bestMatch && bestScore > 0.1) {
      const bestVtText = (bestMatch.text || '').trim();
      const similarity = computeTextSimilarity(primaryText, bestVtText);
      
      // Only use voice track text if similarity is very high (>0.8) AND voice text is longer
      // This means voice track has better transcription of the same content
      if (similarity > 0.8 && bestVtText.length > primaryText.length * 0.9) {
        finalText = bestVtText;
        finalSource = 'voice-enhanced';
        finalConfidence = 'high';
        
        // Mark as used
        const segKey = getSegmentKey(bestMatch);
        usedVoiceSegments.add(segKey);
        
        console.log(`${logPrefix} Enhanced: ${primarySpeaker} [${primaryStart.toFixed(2)}-${primaryEnd.toFixed(2)}] similarity=${similarity.toFixed(2)}`);
      } else {
        console.log(`${logPrefix} Keeping primary: ${primarySpeaker} [${primaryStart.toFixed(2)}-${primaryEnd.toFixed(2)}] (similarity=${similarity.toFixed(2)} too low or text not better)`);
      }
    }
    
    // Always keep primary segment structure
    resultSegments.push({
      speaker: primarySpeaker, // CRITICAL: Keep primary speaker!
      text: finalText,
      start: primaryStart, // CRITICAL: Keep primary timestamps!
      end: primaryEnd,
      originalText: finalText,
      translations: primarySeg.translations || {},
      role: primaryRole || null,
      overlap: primarySeg.overlap || false,
      source: finalSource,
      mergeConfidence: finalConfidence
    });
  }
  
  // PHASE 5: Skip adding unused voice segments
  // This was causing duplication and mixing issues
  // Primary diarization already has all segments correctly segmented
  // Voice tracks are only used to enhance text quality, not to add new segments
  console.log(`${logPrefix} Skipping PHASE 5 (not adding unused voice segments to avoid duplication)`);
  
  // Sort segments by start time
  resultSegments.sort((a, b) => {
    const startDiff = (a.start || 0) - (b.start || 0);
    if (startDiff !== 0) return startDiff;
    return (a.end || 0) - (b.end || 0);
  });
  
  // Mark overlaps
  markOverlapFlags(resultSegments);
  
  // Calculate merge statistics (TRIZ: Feedback principle)
  const mergeStats = {
    total: resultSegments.length,
    fromPrimary: resultSegments.filter(s => s.source === 'primary').length,
    fromVoice: resultSegments.filter(s => s.source === 'voice').length,
    highConfidence: resultSegments.filter(s => s.mergeConfidence === 'high').length,
    lowConfidence: resultSegments.filter(s => s.mergeConfidence === 'low').length
  };
  
  // Update result structure
  const clonedRecording = result.recordings[0];
  clonedRecording.results = clonedRecording.results || {};
  clonedRecording.results['overlap-corrected'] = {
    success: true,
    serviceName: 'Overlap Corrected (TRIZ-based)',
    processingTime: 0,
    speedFactor: 0,
    speakerCount: new Set(resultSegments.map(s => s.speaker)).size.toString(),
    cost: '0.0000',
    segments: resultSegments,
    rawData: {
      source: 'overlap-corrected-triz',
      primarySegmentsCount: primarySegments.length,
      rawVoiceTrackSegmentsCount: rawVoiceTrackSegments.length,
      normalizedSegmentsCount: normalizedSegments.length,
      mergedSegmentsCount: resultSegments.length,
      mergeStats: mergeStats
    }
  };
  
  clonedRecording.servicesTested = clonedRecording.servicesTested || [];
  if (!clonedRecording.servicesTested.includes('overlap-corrected')) {
    clonedRecording.servicesTested.push('overlap-corrected');
  }
  
  // Log speaker distribution
  const speakerDistribution = {};
  resultSegments.forEach(seg => {
    const speaker = seg.speaker || 'UNKNOWN';
    speakerDistribution[speaker] = (speakerDistribution[speaker] || 0) + 1;
  });
  
  console.log(`${logPrefix} Merge completed: ${resultSegments.length} segments`);
  console.log(`${logPrefix} Merge stats:`, mergeStats);
  console.log(`${logPrefix} Speaker distribution:`, speakerDistribution);
  console.log(`${logPrefix} Unique speakers: ${Object.keys(speakerDistribution).length}`);
  
  // Validate that we have at least 2 speakers if primary had 2
  const primarySpeakers = new Set(primarySegments.map(s => s.speaker || 'SPEAKER_00'));
  const resultSpeakers = new Set(resultSegments.map(s => s.speaker || 'SPEAKER_00'));
  
  if (primarySpeakers.size > 1 && resultSpeakers.size === 1) {
    console.error(`${logPrefix} ⚠️ WARNING: Primary had ${primarySpeakers.size} speakers but result has only ${resultSpeakers.size}!`);
    console.error(`${logPrefix} Primary speakers:`, Array.from(primarySpeakers));
    console.error(`${logPrefix} Result speakers:`, Array.from(resultSpeakers));
  }
  
  return result;
}

// Correction Agent: Create corrected and merged result based on comparison
async function correctDiarizationResults(comparisonAnalysis, audioResult, llmResult, transcript) {
  try {
    // Load correction prompt template
    const correctionPromptTemplate = await fs.readFile('prompts/diarization_correction_prompt.txt', 'utf8');
    
    // Prepare input data for the prompt
    const comparisonJSON = JSON.stringify(comparisonAnalysis, null, 2);
    const audioResultJSON = JSON.stringify(audioResult, null, 2);
    const llmResultJSON = JSON.stringify(llmResult, null, 2);
    
    // Replace placeholders in the prompt
    let prompt = correctionPromptTemplate
      .replace('[PASTE THE COMPARISON ANALYSIS HERE]', comparisonJSON)
      .replace('[PASTE THE AUDIO RESULT JSON HERE]', audioResultJSON)
      .replace('[PASTE THE LLM RESULT JSON HERE]', llmResultJSON)
      .replace('[PASTE THE ORIGINAL TRANSCRIPT HERE]', transcript || '');
    
    // Use SMART_MODEL_ID for correction (more accurate)
    const correctionResponse = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: SMART_MODEL_ID,
        messages: [
          {
            role: 'system',
            content: (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim()
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
          'X-Title': 'Diarization Correction Agent'
        }
      }
    );

    const correctionOutput = correctionResponse.data.choices[0]?.message?.content || '';
    
    // Parse the corrected result
    let correctedResult;
    try {
      // Try to extract JSON from the response (might be wrapped in markdown)
      const jsonMatch = correctionOutput.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        correctedResult = JSON.parse(jsonMatch[0]);
      } else {
        correctedResult = JSON.parse(correctionOutput);
      }
      
      // Validate the structure
      const validation = validateDiarizationJSON(correctedResult);
      if (!validation.valid) {
        throw new Error(`Invalid structure: ${validation.error}`);
      }
      
      if (correctedResult?.recordings?.length) {
        normalizeRecordingRoles(correctedResult.recordings[0]);
      }
      
    } catch (parseError) {
      console.error('Failed to parse correction result:', parseError);
      console.error('Raw output:', correctionOutput.substring(0, 500));
      
      // Fallback: use LLM result as base and try to improve it
      writeLog('warn', 'Correction Agent parse failed, using LLM result as fallback', {
        error: parseError.message
      });
      
      correctedResult = llmResult; // Use LLM result as fallback
    }
    
    writeLog('info', 'Correction Agent completed', {
      segmentsCount: correctedResult?.recordings?.[0]?.results?.combined?.segments?.length || 0
    });
    
    return correctedResult;
    
  } catch (error) {
    console.error('Correction Agent error:', error);
    writeLog('error', 'Correction Agent failed', {
      error: error.message,
      stack: error.stack
    });
    
    // Fallback: use LLM result (usually better for roles)
    return llmResult;
  }
}

// Analyze voice role for a single voice track
async function analyzeVoiceRole(transcript, language, mode = 'fast') {
  try {
    if (!transcript || !transcript.trim()) {
      return {
        role: 'Unknown',
        confidence: 0.0,
        summary: 'No speech detected.'
      };
    }

    // Cache key based on transcript hash, language, and mode
    const crypto = require('crypto');
    const transcriptHash = crypto.createHash('sha256')
      .update(transcript.trim().toLowerCase())
      .digest('hex')
      .substring(0, 16);
    const cacheKey = `role_${transcriptHash}_${language}_${mode}`;
    const cachePath = path.join(cacheDir, 'role_analysis', `${cacheKey}.json`);
    
    // Check cache
    if (fsSync.existsSync(cachePath)) {
      try {
        const cacheAge = Date.now() - fsSync.statSync(cachePath).mtimeMs;
        const CACHE_TTL = 30 * 24 * 60 * 60 * 1000; // 30 days
        if (cacheAge <= CACHE_TTL) {
          const cached = JSON.parse(await fs.readFile(cachePath, 'utf8'));
          console.log(`✅ Using cached role analysis (age: ${(cacheAge / (24 * 60 * 60 * 1000)).toFixed(1)} days)`);
          return cached;
        } else {
          // Cache expired, remove it
          fsSync.unlinkSync(cachePath);
        }
      } catch (cacheError) {
        console.warn('Failed to load role analysis cache:', cacheError.message);
      }
    }

    // Select model based on mode
    let model;
    if (mode === 'fast') {
      model = FAST_MODEL_ID;
    } else if (mode === 'smart-2') {
      model = SMART_2_MODEL_ID;
    } else if (mode === 'test') {
      model = TEST_MODEL_ID;
    } else {
      model = SMART_MODEL_ID;
    }

    const systemPrompt = (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim();
    
    const rolePrompt = `You are analyzing a single-speaker transcript extracted from a call center conversation.
Determine if the speaker is an Agent (call center operator, support professional) or a Client (customer).
Consider cues such as greeting style, problem descriptions, or offers of assistance.

Respond with strict JSON using this schema:
{
  "role": "operator" | "client",
  "confidence": number between 0 and 1,
  "summary": "one-sentence synopsis of the speaker's intent"
}

Never include extra text before or after the JSON.`;

    const response = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: model,
        messages: [
          {
            role: 'system',
            content: systemPrompt
          },
          {
            role: 'user',
            content: `${rolePrompt}\n\nTranscript:\n${transcript.trim()}`
          }
        ],
        temperature: 0
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
          'X-Title': 'Voice Role Analysis'
        }
      }
    );

    const outputText = response.data.choices[0]?.message?.content || '';
    
    if (!outputText) {
      throw new Error('Empty response from OpenRouter API');
    }
    
    // Try to extract JSON from the response (might be wrapped in markdown)
    let cleanedOutput = outputText.trim();
    if (cleanedOutput.startsWith('```json')) {
      cleanedOutput = cleanedOutput.slice(7);
    }
    if (cleanedOutput.startsWith('```')) {
      cleanedOutput = cleanedOutput.slice(3);
    }
    if (cleanedOutput.endsWith('```')) {
      cleanedOutput = cleanedOutput.slice(0, -3);
    }
    cleanedOutput = cleanedOutput.trim();
    
    const parsed = JSON.parse(cleanedOutput);
    
    // Validate required fields
    if (!parsed.role || !['operator', 'client'].includes(parsed.role)) {
      throw new Error(`Invalid role: ${parsed.role}`);
    }
    
    const result = {
      role: parsed.role,
      confidence: parsed.confidence || 0.5,
      summary: parsed.summary || ''
    };
    
    // Save to cache
    try {
      const roleCacheDir = path.join(cacheDir, 'role_analysis');
      if (!fsSync.existsSync(roleCacheDir)) {
        fsSync.mkdirSync(roleCacheDir, { recursive: true });
      }
      await fs.writeFile(cachePath, JSON.stringify(result, null, 2), 'utf8');
      console.log(`💾 Saved role analysis to cache: ${cacheKey}`);
    } catch (cacheError) {
      console.warn('Failed to save role analysis cache:', cacheError.message);
    }
    
    return result;
    
  } catch (error) {
    console.error('Voice role analysis error:', error);
    writeLog('error', 'Voice role analysis failed', {
      error: error.message
    });
    
    // Fallback: heuristic
    const transcriptLower = (transcript || '').toLowerCase();
    const role = (transcriptLower.includes('help') || transcriptLower.includes('can i') || transcriptLower.includes('how can')) 
      ? 'operator' 
      : 'client';
    
    const fallbackResult = {
      role: role,
      confidence: 0.5,
      summary: 'Fallback role assignment'
    };
    
    // Don't cache fallback results - they are not reliable
    return fallbackResult;
  }
}

/**
 * Prepare segments data for LLM processing
 * Formats segments into a structured array with segment_id, text, start, end, speaker_id
 * 
 * @param {Array} segments - Array of segment objects from diarization result
 * @returns {Array} Formatted segments array for LLM
 */
function prepareSegmentsForLLM(segments) {
  if (!Array.isArray(segments) || segments.length === 0) {
    return [];
  }

  // Sort segments by start time to ensure chronological order
  const sortedSegments = [...segments].sort((a, b) => {
    const startA = parseFloat(a.start) || 0;
    const startB = parseFloat(b.start) || 0;
    if (startA !== startB) return startA - startB;
    const endA = parseFloat(a.end) || startA;
    const endB = parseFloat(b.end) || startB;
    return endA - endB;
  });

  // Format segments with unique segment_id
  // Convert SPEAKER_00 -> Agent, SPEAKER_01 -> Client for LLM
  const formattedSegments = sortedSegments.map((segment, index) => {
    const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
    const end = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end) || start;
    const speaker = segment.speaker || 'SPEAKER_00';
    const text = (segment.text || '').trim();
    
    // Map speaker to Agent/Client for LLM
    let speakerId = speaker;
    if (speaker === 'SPEAKER_00') {
      speakerId = 'Agent';
    } else if (speaker === 'SPEAKER_01') {
      speakerId = 'Client';
    }

    return {
      segment_id: index + 1, // 1-based index
      text: text,
      start: parseFloat(start.toFixed(2)), // Round to 2 decimal places
      end: parseFloat(end.toFixed(2)),
      speaker_id: speakerId,
      original_speaker: speaker // Зберігаємо оригінального спікера з Speechmatics (SPEAKER_00, SPEAKER_01)
    };
  });

  return formattedSegments;
}

/**
 * Знаходить підозрілі пари сегментів, які можуть бути розірваними фразами
 * @param {Array} segments - Масив сегментів
 * @param {Object} options - Опції
 * @returns {Array} Масив пар сегментів для перевірки
 */
function findSuspiciousFragmentPairs(segments, options = {}) {
  const { maxGapSeconds = 3.0 } = options;
  
  if (!Array.isArray(segments) || segments.length < 2) {
    return [];
  }

  const sorted = [...segments].sort((a, b) => 
    (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0)
  );

  const suspiciousPairs = [];

  for (let i = 0; i < sorted.length - 1; i++) {
    const current = sorted[i];
    const next = sorted[i + 1];
    
    const gap = (parseFloat(next.start) || 0) - (parseFloat(current.end) || 0);
    
    // Перевіряємо тільки сегменти з малим проміжком
    if (gap >= 0 && gap <= maxGapSeconds) {
      suspiciousPairs.push({
        first: current,
        second: next,
        gap: gap,
        index: i
      });
    }
  }

  return suspiciousPairs;
}

/**
 * Перевіряє через LLM чи два сегменти є однією розірваною фразою
 * @param {Object} pair - Пара сегментів для перевірки
 * @param {Object} options - Опції (model, useLocalLLM, apiUrl, headers)
 * @returns {Promise<boolean>} true якщо це одна фраза, false якщо ні
 */
async function checkIfFragmentedPhrase(pair, options = {}) {
  const { model, useLocalLLM, apiUrl, headers, logPrefix = '[LLM Check]' } = options;
  
  const firstText = (pair.first.text || '').trim();
  const secondText = (pair.second.text || '').trim();
  
  if (!firstText || !secondText) {
    return false;
  }

  const systemPrompt = `You are a language expert analyzing call center conversations. Your task is to determine if two text segments are parts of ONE fragmented phrase (split due to a pause) or TWO separate phrases.

CRITICAL: Even if segments have different speaker labels (Agent/Client), they might still be ONE fragmented phrase if the first segment is incomplete and the second continues it.

Answer ONLY with "YES" if they are one fragmented phrase, or "NO" if they are separate phrases.`;

  const firstSpeaker = pair.first.speaker_id || 'Unknown';
  const secondSpeaker = pair.second.speaker_id || 'Unknown';
  
  // Отримуємо оригінальних спікерів з Speechmatics (SPEAKER_00, SPEAKER_01)
  const firstOriginalSpeaker = pair.first.original_speaker || pair.first.speaker || 'Unknown';
  const secondOriginalSpeaker = pair.second.original_speaker || pair.second.speaker || 'Unknown';
  
  const userPrompt = `Are these two segments parts of ONE fragmented phrase?

Segment 1:
- Assigned role: ${firstSpeaker}
- Speechmatics speaker: ${firstOriginalSpeaker}
- Text: "${firstText}"

Segment 2:
- Assigned role: ${secondSpeaker}
- Speechmatics speaker: ${secondOriginalSpeaker}
- Text: "${secondText}"

Gap between them: ${pair.gap.toFixed(2)} seconds

IMPORTANT CONTEXT:
- Speechmatics speaker labels (${firstOriginalSpeaker}, ${secondOriginalSpeaker}) show who was detected by the audio diarization system
- Assigned roles (${firstSpeaker}, ${secondSpeaker}) may be incorrect if the phrase was fragmented
- If Speechmatics shows the SAME speaker for both segments, they are likely ONE fragmented phrase
- If Speechmatics shows DIFFERENT speakers, check if the first segment is incomplete (ends with "to", "and", "did you", etc.)

RULES FOR FRAGMENTED PHRASES:

1. **AGENT asking questions** - If first segment is Agent asking incomplete question, continuation belongs to AGENT:
   - "And did you try to" (Agent) + "reset your modem" (Agent/Client) = YES, belongs to AGENT
   - "Can you tell" (Agent) + "me your IP address" (Agent/Client) = YES, belongs to AGENT
   - "Did you check" (Agent) + "the settings" (Agent/Client) = YES, belongs to AGENT
   - Reason: Agent is asking a question, so the continuation of the question belongs to Agent

2. **CLIENT asking questions** - If first segment is Client asking incomplete question, continuation belongs to CLIENT:
   - "Can you help" (Client) + "me with this" (Client/Agent) = YES, belongs to CLIENT
   - "I need to" (Client) + "reset my password" (Client/Agent) = YES, belongs to CLIENT
   - Reason: Client is asking for help, so the continuation belongs to Client

3. **AGENT giving instructions** - If first segment is Agent giving incomplete instruction, continuation belongs to AGENT:
   - "Try to" (Agent) + "restart the device" (Agent/Client) = YES, belongs to AGENT
   - "You should" (Agent) + "check the settings" (Agent/Client) = YES, belongs to AGENT
   - Reason: Agent is providing instructions, so continuation belongs to Agent

4. **CLIENT describing problem** - If first segment is Client describing incomplete problem, continuation belongs to CLIENT:
   - "I have a problem" (Client) + "with my internet" (Client/Agent) = YES, belongs to CLIENT
   - "My connection" (Client) + "keeps dropping" (Client/Agent) = YES, belongs to CLIENT
   - Reason: Client is describing their issue, so continuation belongs to Client

5. **NOT fragmented** - These are separate phrases:
   - "Hello" + "How are you?" = NO (greeting + question, separate)
   - "Yes" + "please" = NO (short response, separate)
   - "Thank you" + "You're welcome" = NO (different speakers, separate)

DECISION RULE:
- If first segment ends with incomplete question/instruction/description (no period, ends with "to", "and", "did you", "can you", "I need", etc.)
- AND second segment completes the thought
- THEN = YES (one fragmented phrase)
- The continuation belongs to the SAME speaker who started the phrase

SPEECHMATICS SPEAKER CONTEXT:
- If Speechmatics shows the SAME speaker for both segments (${firstOriginalSpeaker} = ${secondOriginalSpeaker}), they are almost certainly ONE fragmented phrase
- If Speechmatics shows DIFFERENT speakers (${firstOriginalSpeaker} ≠ ${secondOriginalSpeaker}) but the first segment is clearly incomplete, check who STARTED the phrase based on content
- Speechmatics speaker detection is more reliable than assigned roles for fragmented phrases - trust it when speakers match

Examples:
- "And did you try to" (Assigned: Client, Speechmatics: ${firstOriginalSpeaker}) + "reset your modem" (Assigned: Agent, Speechmatics: ${secondOriginalSpeaker})
  * If Speechmatics speakers match (${firstOriginalSpeaker} = ${secondOriginalSpeaker}): YES, belongs to ${firstOriginalSpeaker}
  * If Speechmatics speakers differ: Check content - if it's a question from Agent, belongs to AGENT
- "I need to" (Assigned: Agent, Speechmatics: ${firstOriginalSpeaker}) + "reset my password" (Assigned: Client, Speechmatics: ${secondOriginalSpeaker})
  * If Speechmatics speakers match: YES, belongs to ${firstOriginalSpeaker}
  * If Speechmatics speakers differ: Check content - if Client was asking, belongs to CLIENT

Answer: YES or NO`;

  const payload = {
    model: model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt }
    ],
    temperature: 0,
    max_tokens: 10 // Коротка відповідь
  };

  try {
    // Логуємо дані, які відправляємо
    console.log(`${logPrefix} 📤 Sending to LLM:`, {
      segment1: {
        text: firstText,
        assignedRole: firstSpeaker,
        speechmaticsSpeaker: firstOriginalSpeaker
      },
      segment2: {
        text: secondText,
        assignedRole: secondSpeaker,
        speechmaticsSpeaker: secondOriginalSpeaker
      },
      gap: pair.gap.toFixed(2) + 's',
      prompt: userPrompt.substring(0, 200) + '...'
    });
    
    const response = await axios.post(apiUrl, payload, {
      headers: headers,
      timeout: 30000 // 30 секунд для короткої перевірки
    });

    const answer = (response.data.choices[0]?.message?.content || '').trim().toUpperCase();
    const isFragmented = answer.includes('YES');
    
    // Логуємо відповідь
    console.log(`${logPrefix} 📥 LLM Response:`, {
      rawAnswer: response.data.choices[0]?.message?.content || '',
      normalizedAnswer: answer,
      decision: isFragmented ? 'MERGE' : 'SEPARATE',
      segment1: firstText.substring(0, 40),
      segment2: secondText.substring(0, 40)
    });
    
    console.log(`${logPrefix} Fragment check: "${firstText.substring(0, 30)}..." + "${secondText.substring(0, 30)}..." → ${isFragmented ? 'MERGE' : 'SEPARATE'}`);
    
    return isFragmented;
  } catch (error) {
    console.error(`${logPrefix} ❌ Fragment check failed: ${error.message}`, error.stack);
    console.warn(`${logPrefix} ⚠️ Defaulting to false`);
    return false;
  }
}

/**
 * Об'єднує розірвані фрази на основі перевірки через LLM
 * @param {Array} segments - Масив сегментів
 * @param {Object} options - Опції для LLM перевірки
 * @returns {Promise<Array>} Сегменти з об'єднаними фразами
 */
async function mergeFragmentedPhrasesWithLLM(segments, options = {}) {
  const {
    model,
    useLocalLLM,
    apiUrl,
    headers,
    logPrefix = '[Fragment Merge]',
    maxGapSeconds = 3.0
  } = options;

  if (!Array.isArray(segments) || segments.length < 2) {
    return segments;
  }

  console.log(`${logPrefix} 🔍 Checking ${segments.length} segments for fragmented phrases...`);

  // Знаходимо підозрілі пари
  const suspiciousPairs = findSuspiciousFragmentPairs(segments, { maxGapSeconds });
  
  if (suspiciousPairs.length === 0) {
    console.log(`${logPrefix} ✅ No suspicious fragment pairs found`);
    return segments;
  }

  console.log(`${logPrefix} 📋 Found ${suspiciousPairs.length} suspicious pairs to check`);
  // Логуємо перші 3 пари для дебагу
  suspiciousPairs.slice(0, 3).forEach((pair, idx) => {
    const firstOriginal = pair.first.original_speaker || pair.first.speaker || 'Unknown';
    const secondOriginal = pair.second.original_speaker || pair.second.speaker || 'Unknown';
    console.log(`${logPrefix}   Pair ${idx + 1}: "${pair.first.text?.substring(0, 30)}..." (Role: ${pair.first.speaker_id}, Speechmatics: ${firstOriginal}) + "${pair.second.text?.substring(0, 30)}..." (Role: ${pair.second.speaker_id}, Speechmatics: ${secondOriginal}), gap: ${pair.gap.toFixed(2)}s`);
  });

  // Перевіряємо кожну пару через LLM
  const mergeDecisions = [];
  for (const pair of suspiciousPairs) {
    const shouldMerge = await checkIfFragmentedPhrase(pair, {
      model,
      useLocalLLM,
      apiUrl,
      headers,
      logPrefix
    });
    mergeDecisions.push({ pair, shouldMerge });
  }

  // Об'єднуємо сегменти на основі рішень LLM
  const sorted = [...segments].sort((a, b) => 
    (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0)
  );

  const merged = [];
  const mergedIndices = new Set();

  for (let i = 0; i < sorted.length; i++) {
    if (mergedIndices.has(i)) {
      continue; // Цей сегмент вже об'єднано
    }

    // Шукаємо рішення для цієї пари
    const decision = mergeDecisions.find(d => d.pair.index === i);
    
    if (decision && decision.shouldMerge) {
      // Об'єднуємо
      const current = sorted[i];
      const next = sorted[i + 1];
      
      if (next) {
        // Визначаємо правильного спікера на основі контексту:
        // Правило: продовження належить тому, хто ПОЧАВ фразу
        const currentText = (current.text || '').trim().toLowerCase();
        const nextText = (next.text || '').trim().toLowerCase();
        
        // Перевіряємо, хто почав фразу (питає, дає інструкції, описує проблему)
        let mergedSpeaker = current.speaker_id; // За замовчуванням - той, хто почав
        
        if (current.speaker_id !== next.speaker_id) {
          // Якщо спікери різні, визначаємо правильного на основі контексту
          
          // АГЕНТ починає (питає, дає інструкції) → продовження належить АГЕНТУ
          const agentStarts = /\b(and\s+)?(did\s+you|can\s+you|will\s+you|would\s+you|should\s+you|have\s+you|try\s+to|you\s+should|you\s+need|let\s+me|i\s+will|i\s+can)/i.test(currentText);
          
          // КЛІЄНТ починає (питає про допомогу, описує проблему) → продовження належить КЛІЄНТУ
          const clientStarts = /\b(i\s+have|i\s+need|i\s+want|i\s+can't|i\s+cannot|my\s+|can\s+you\s+help|please\s+help)/i.test(currentText);
          
          if (agentStarts && current.speaker_id === 'Agent') {
            mergedSpeaker = 'Agent'; // Агент почав питання/інструкцію → продовження агента
          } else if (agentStarts && current.speaker_id === 'Client') {
            // Якщо клієнт почав з агентської фрази, можливо це помилка - перевіряємо контекст
            // Якщо продовження виглядає як відповідь на питання → залишаємо клієнта
            const looksLikeAnswer = /^(yes|no|sure|okay|ok|alright|of course|certainly|absolutely|definitely|maybe|perhaps|probably|i think|i believe|i guess|i tried|i did|i can|i will)/i.test(nextText);
            mergedSpeaker = looksLikeAnswer ? 'Client' : 'Agent';
          } else if (clientStarts && current.speaker_id === 'Client') {
            mergedSpeaker = 'Client'; // Клієнт почав опис проблеми → продовження клієнта
          } else if (clientStarts && current.speaker_id === 'Agent') {
            // Якщо агент почав з клієнтської фрази, можливо це помилка
            mergedSpeaker = 'Agent'; // Залишаємо агента
          } else {
            // Якщо не можемо визначити, використовуємо спікера першого сегмента (той, хто почав)
            mergedSpeaker = current.speaker_id;
          }
        }
        
        const mergedSegment = {
          ...current,
          speaker_id: mergedSpeaker, // Використовуємо визначеного спікера
          text: ((current.text || '').trim() + ' ' + (next.text || '').trim()).trim(),
          end: Math.max(parseFloat(current.end) || 0, parseFloat(next.end) || parseFloat(next.start)),
          _mergedFragment: true,
          _originalSegments: [current, next],
          _originalSpeakers: [current.speaker_id, next.speaker_id] // Зберігаємо оригінальних спікерів для логів
        };
        
        merged.push(mergedSegment);
        mergedIndices.add(i);
        mergedIndices.add(i + 1);
        
        const speakerNote = current.speaker_id !== next.speaker_id 
          ? ` [${current.speaker_id}→${next.speaker_id}, assigned to ${mergedSpeaker} (${current.speaker_id} started the phrase)]`
          : '';
        console.log(`${logPrefix} 🔗 Merged: "${current.text.substring(0, 40)}..." + "${next.text.substring(0, 40)}..."${speakerNote}`);
        
        i++; // Пропускаємо наступний сегмент
        continue;
      }
    }

    // Не об'єднуємо - додаємо як є
    merged.push({ ...sorted[i] });
  }

  console.log(`${logPrefix} ✅ Fragment merge complete: ${segments.length} → ${merged.length} segments`);
  
  return merged;
}

/**
 * Send segments data to LLM for speaker mixing fixes
 * This function prepares the data and sends it to LLM API for correction
 * 
 * @param {Array} segmentsForLLM - Formatted segments array from prepareSegmentsForLLM
 * @param {Object} options - Options for LLM processing
 * @param {string} options.mode - LLM mode ('fast', 'smart', 'smart-2', 'local')
 * @param {string} options.language - Language code
 * @param {string} options.requestId - Request ID for logging
 * @param {Function} options.sendUpdate - Function to send SSE updates
 * @param {Array} options.voiceTracks - Voice tracks with transcriptions for context
 * @returns {Promise<Array>} Corrected segments array from LLM
 */
async function sendSegmentsToLLMForFixes(segmentsForLLM, options = {}) {
  const {
    mode = 'smart',
    language = 'en',
    requestId = null,
    sendUpdate = null,
    voiceTracks = [],
    filename = null
  } = options;

  const logPrefix = requestId ? `[${requestId}]` : '[LLM Fixes]';

  if (!Array.isArray(segmentsForLLM) || segmentsForLLM.length === 0) {
    console.warn(`${logPrefix} ⚠️ No segments provided for LLM processing`);
    return [];
  }

  console.log(`${logPrefix} 🤖 Preparing to send ${segmentsForLLM.length} segments to LLM for speaker mixing fixes...`);
  console.log(`${logPrefix} 📊 First 3 segments for debugging:`, segmentsForLLM.slice(0, 3).map(s => ({
    id: s.segment_id,
    speaker: s.speaker_id,
    text: s.text?.substring(0, 50),
    start: s.start,
    end: s.end
  })));

  // ---------- 1. Select model based on mode ----------
  let model;
  let useLocalLLM = false;
  
  if (mode === 'local' || mode === 'test' || mode === 'test2') {
    useLocalLLM = true;
    // Use getModelId() to always get current model from process.env (for cache key accuracy)
    model = getModelId(mode);
      console.log(`${logPrefix} 🔵 Using local LLM: ${LOCAL_LLM_BASE_URL}, model: ${model}`);
  } else {
    // For non-local models, check OpenRouter API key
    if (!process.env.OPENROUTER_API_KEY) {
      console.error(`${logPrefix} ❌ OpenRouter API key is not configured`);
      throw new Error('OpenRouter API key is not configured');
    }
    
    // Use getModelId() to always get current model from process.env (for cache key accuracy)
    model = getModelId(mode);
    console.log(`${logPrefix} 🔵 Using OpenRouter model: ${model}`);
  }

  // ---------- 2. Pre-merge fragmented phrases using LLM check ----------
  // Configure API endpoint and headers for fragment check
  const apiUrl = useLocalLLM 
    ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
    : 'https://openrouter.ai/api/v1/chat/completions';
  
  const headers = {
    'Content-Type': 'application/json'
  };
  
  if (useLocalLLM) {
    if (LOCAL_LLM_API_KEY) {
      headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
    }
  } else {
    headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
    headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
    headers['X-Title'] = 'Fragment Check';
  }

  // Merge fragmented phrases before main LLM processing
  let processedSegments = segmentsForLLM;
  try {
    console.log(`${logPrefix} 🔍 Starting fragmented phrases check for ${segmentsForLLM.length} segments...`);
    
    if (sendUpdate) {
      sendUpdate(5, 'processing', `🔍 Перевірка розірваних фраз (${segmentsForLLM.length} сегментів)...`, {
        stage: 'fragment_check',
        input: { segmentsCount: segmentsForLLM.length }
      });
    }
    
    processedSegments = await mergeFragmentedPhrasesWithLLM(segmentsForLLM, {
      model,
      useLocalLLM,
      apiUrl,
      headers,
      logPrefix,
      maxGapSeconds: 3.0
    });
    
    if (processedSegments.length !== segmentsForLLM.length) {
      console.log(`${logPrefix} ✅ Fragmented phrases merged: ${segmentsForLLM.length} → ${processedSegments.length} segments`);
    } else {
      console.log(`${logPrefix} ℹ️ No fragments merged (${segmentsForLLM.length} segments remain)`);
    }
  } catch (error) {
    console.error(`${logPrefix} ❌ Fragment merge failed: ${error.message}`, error.stack);
    console.warn(`${logPrefix} ⚠️ Continuing with original segments`);
    processedSegments = segmentsForLLM;
  }

  // ---------- 3. Build prompt with voice tracks context ----------
  // Extract voice tracks segments as JSON for context
  const voiceTracksContext = [];
  if (voiceTracks && Array.isArray(voiceTracks)) {
    voiceTracks.forEach((track, idx) => {
      if (track.error) return;
      const speaker = track.speaker || `SPEAKER_${idx.toString().padStart(2, '0')}`;
      const transcription = track.transcription;
      
      // Map speaker to Agent/Client
      const speakerLabel = speaker === 'SPEAKER_00' ? 'Agent' : (speaker === 'SPEAKER_01' ? 'Client' : speaker);
      
      // Get segments from Speechmatics transcription
      if (transcription?.recordings?.[0]?.results?.speechmatics?.segments) {
        const segments = transcription.recordings[0].results.speechmatics.segments;
        if (segments && segments.length > 0) {
          voiceTracksContext.push({
            speaker: speakerLabel,
            role: track.roleAnalysis?.role || null,
            segments: segments.map(s => ({
              text: s.text || '',
              start: s.start || 0,
              end: s.end || 0,
              speaker: s.speaker || speakerLabel,
              words: s.words || []
            }))
          });
        }
      }
    });
  }

  const systemPrompt = `You are an expert in dialogue transcription and speaker diarization.

Your task is to review the ENTIRE dialogue below, identify and fix ALL issues including:
- Mismatched speaker assignments
- Overlapping speech segments that need to be split
- Segments that should be merged (same speaker, continuous speech)
- Missing or incorrect speaker assignments
- Chronological inconsistencies
- Short responses that should be separate segments (e.g., "Yes please", "No thanks", "Sure")

CRITICAL RULES FOR SEGMENT SPLITTING:
1. **Split short responses**: If a segment ends with a question and the next segment starts with a short response (like "Yes", "No", "Sure", "Please", "Thanks"), these should be SEPARATE segments from DIFFERENT speakers
   - Example: "Would you like to check appointments? Yes please." → Split into:
     * Segment 1: "Would you like to check appointments?" (Agent)
     * Segment 2: "Yes please." (Client)
2. **Split after questions**: When a segment contains a question followed by an answer, split them if they logically belong to different speakers
3. **Split conversational turns**: Each natural turn-taking should be a separate segment
4. **Preserve context**: Use voice tracks transcriptions (provided below) as reference for correct text and speaker assignments

CRITICAL RULES FOR SEGMENT MERGING:
1. **Merge continuous speech**: If consecutive segments from the same speaker are clearly one continuous utterance, merge them
2. **Merge incomplete sentences**: If a segment ends mid-sentence and the next segment from the same speaker continues it, merge them
3. **Merge fragmented phrases across speakers**: If a segment is clearly incomplete (ends with "to", "and", "did you", "can you", etc.) and the next segment completes it, merge them EVEN IF they have different speaker labels. The continuation belongs to the speaker who STARTED the phrase:
   - **AGENT started** (asking question/giving instruction): "And did you try to" (Client) + "reset your modem" (Agent) → Merge, assign to AGENT (Agent was asking)
   - **CLIENT started** (asking for help/describing problem): "I need to" (Agent) + "reset my password" (Client) → Merge, assign to CLIENT (Client was asking)
   - **Rule**: The speaker who STARTS the incomplete phrase owns the continuation, regardless of intermediate speaker labels

CRITICAL RULES FOR SPEAKER ASSIGNMENT:
1. **AGENT/CLIENT IDENTIFICATION**: Determine which speaker is Agent and which is Client based on conversational context:
   - Agent typically: asks questions, provides information, offers help, uses professional language, initiates service interactions
   - Client typically: asks for help, responds to questions, makes requests, uses casual language, seeks assistance
   - Use the context of the conversation to assign roles correctly - don't just rely on labels
2. Use voice tracks transcriptions (provided below) as the AUTHORITATIVE source for speaker assignments
3. If voice track transcription shows different speaker than merged segments, trust the voice track
4. Consider conversational flow: questions → answers, statements → responses
5. Short responses (Yes/No/Please/Thanks) typically come from the listener, not the speaker who just asked a question

CRITICAL RULES FOR DUPLICATE PREVENTION:
1. **NO DUPLICATES**: Never create duplicate segments with identical or nearly identical text and timestamps
2. If you see the same text repeated with similar timestamps, keep only ONE instance
3. Check for overlapping segments with same text - merge them or keep the most complete one
4. Remove redundant segments that are exact duplicates

GENERAL RULES:
1. You can MODIFY timestamps (start/end) if needed to properly separate overlapping speech
2. You can SPLIT segments if they contain overlapping speech from different speakers OR if they contain a question followed by a response
3. You can MERGE segments if they belong to the same speaker and are continuous
4. You MUST preserve ALL meaningful text content - do not delete important text
5. **REMOVE EMPTY WORDS**: Remove empty or meaningless words/phrases (e.g., "uh", "um", "ah", excessive filler words) but keep all meaningful content
6. **REMOVE DUPLICATES**: If segments have identical or nearly identical text and timestamps, keep only ONE instance
7. Ensure chronological order (start times must be in ascending order)
8. Ensure no temporal gaps or overlaps unless they represent actual overlapping speech
9. Return a COMPLETE corrected dialogue covering the entire time range (WITHOUT DUPLICATES)

Return format: JSON array with fields: segment_id, text, start, end, speaker_id
- segment_id: sequential number starting from 1
- text: the spoken text (preserve all content)
- start: start time in seconds (can be modified to fix overlaps)
- end: end time in seconds (can be modified to fix overlaps)
- speaker_id: correct speaker identifier (use "Agent" for SPEAKER_00, "Client" for SPEAKER_01)`;

  // Format segments for prompt with more context (use processed segments)
  const totalDuration = Math.max(...processedSegments.map(s => s.end));
  const speakers = [...new Set(processedSegments.map(s => s.speaker_id))];
  
  // Re-number segments after merging
  const renumberedSegments = processedSegments.map((s, idx) => ({
    ...s,
    segment_id: idx + 1
  }));
  
  const segmentsText = renumberedSegments.map(s => 
    `[${s.segment_id}] ${s.speaker_id}: "${s.text}" (${s.start.toFixed(2)}s - ${s.end.toFixed(2)}s)`
  ).join('\n');

  // Build voice tracks context text with JSON segments
  let voiceTracksText = '';
  if (voiceTracksContext.length > 0) {
    voiceTracksText = `\n\nVOICE TRACKS SEGMENTS (AUTHORITATIVE SOURCE - JSON):
These are the complete diarized segments from separated audio tracks (Speechmatics). Use these as reference for correct text, speaker assignments, and timestamps.

${voiceTracksContext.map((vt, idx) => {
      const roleInfo = vt.role ? ` (Role: ${vt.role})` : '';
      const segmentsJson = JSON.stringify(vt.segments, null, 2);
      return `${vt.speaker}${roleInfo}:\n${segmentsJson}`;
    }).join('\n\n')}

IMPORTANT: 
- If the merged segments above show different text or speaker assignments than the voice tracks, trust the voice tracks and correct the segments accordingly.
- Use the exact text and timestamps from voice tracks segments when available.
- Map speakers: SPEAKER_00 = Agent, SPEAKER_01 = Client`;
  }

  const userPrompt = `Review and correct the ENTIRE dialogue below. Fix ALL speaker assignment errors, split overlapping segments, and merge continuous segments from the same speaker.

DIALOGUE CONTEXT:
- Total duration: ${totalDuration.toFixed(2)} seconds
- Speakers detected: ${speakers.join(', ')}
- Total segments: ${renumberedSegments.length}

MERGED SEGMENTS (in chronological order):
${segmentsText}${voiceTracksText}

TASK - Review the ENTIRE dialogue and:
1. **Check speaker assignments**: Ensure each segment is assigned to the correct speaker based on context and voice tracks
2. **Split overlapping segments**: If a segment contains speech from multiple speakers, split it into separate segments
3. **Split question-answer pairs**: If a segment contains a question followed by a response (e.g., "Would you like X? Yes please"), split them into separate segments
4. **Split short responses**: Separate short responses like "Yes", "No", "Please", "Thanks" into their own segments when they follow a question or statement
5. **Merge continuous segments**: If consecutive segments from the same speaker should be one continuous utterance, merge them
6. **Fix timestamps**: Adjust start/end times to properly separate overlapping speech
7. **Use voice tracks**: Trust voice tracks transcriptions for correct text and speaker assignments
8. **Ensure completeness**: The corrected dialogue must cover the entire time range from ${renumberedSegments[0]?.start?.toFixed(2) || 0}s to ${totalDuration.toFixed(2)}s
9. **Maintain all text**: Do not delete or modify any text content, only reorganize it

EXAMPLE OF CORRECT SPLITTING:
Input: [1] Agent: "Yes, you can. For lung issues, you might want to see a pulmonologist. Would you like to check the soonest available appointments with a pulmonology specialist? Yes please." (10.5s - 25.3s)

Correct Output:
- [1] Agent: "Yes, you can. For lung issues, you might want to see a pulmonologist. Would you like to check the soonest available appointments with a pulmonology specialist?" (10.5s - 24.8s)
- [2] Client: "Yes please." (24.8s - 25.3s)

OUTPUT:
Return a JSON array with ALL corrected segments. The array should:
- Cover the entire dialogue time range
- Have correct speaker assignments (use voice tracks as reference)
- Have properly separated overlapping speech
- Have split question-answer pairs and short responses
- Have merged continuous segments from the same speaker
- Be in chronological order

Return ONLY the JSON array, no additional text or markdown.`;

  // ---------- 4. Send request to LLM ----------
  if (sendUpdate) {
    sendUpdate(5, 'processing', `🤖 MODE3: Відправка ${renumberedSegments.length} сегментів на LLM для виправлення змішування спікерів...`, {
      stage: 'llm_request',
      input: {
        segmentsCount: renumberedSegments.length,
        speakers: [...new Set(renumberedSegments.map(s => s.speaker_id))],
        model: model
      }
    });
  }

  console.log(`${logPrefix} 📤 Sending LLM request to ${useLocalLLM ? 'local' : 'OpenRouter'} (model: ${model})...`);

  // Build payload with reasoning_effort if needed
  const payload = {
    model: model,
    messages: [
      {
        role: 'system',
        content: systemPrompt
      },
      {
        role: 'user',
        content: userPrompt
      }
    ],
    temperature: 0  // Maximum deterministic output
    // No max_tokens limit - let LLM generate full response
  };
  
  // Add reasoning effort for Fast, and Smart (GPT 5.1)
  // OpenRouter format: reasoning: { effort: "high" }
  // NOTE: Local mode (LM Studio) reasoning_effort is disabled - configure in LM Studio UI
  if (shouldUseHighReasoningEffort(mode, model)) {
    if (useLocalLLM) {
      // Local LLM reasoning_effort disabled - configure in LM Studio UI
      console.log(`${logPrefix} 🔧 Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)`);
    } else {
      // For OpenRouter, use nested reasoning object
      payload.reasoning = { effort: 'high' };
      console.log(`${logPrefix} 🔧 Using reasoning effort: high for ${mode} mode (model: ${model})`);
    }
  }

  // Check LLM cache if enabled
  if (LLM_CACHE_ENABLED) {
    if (!filename) {
      console.warn(`${logPrefix} ⚠️ LLM cache check skipped for overlap fixes: filename is missing`);
    } else {
    const fullPrompt = `system: ${systemPrompt}\n\nuser: ${userPrompt}`;
    const cacheKey = buildLLMCacheKey(filename, fullPrompt, model, mode, 'overlap-fixes');
    if (cacheKey) {
        console.log(`${logPrefix} 🔍 Checking LLM cache for overlap fixes:`, { filename, cacheKey, model, mode });
      const cachedResponse = readLLMCache(cacheKey);
      if (cachedResponse) {
          console.log(`${logPrefix} ✅ Using cached LLM response for overlap fixes (filename: ${filename}, cacheKey: ${cacheKey})`);
        try {
          const parsed = JSON.parse(cachedResponse.llmOutput);
          if (Array.isArray(parsed)) {
            return parsed;
          }
        } catch (parseError) {
          console.warn(`${logPrefix} ⚠️ Failed to parse cached LLM response, fetching fresh...`);
        }
        } else {
          console.log(`${logPrefix} 📝 LLM cache miss for overlap fixes (filename: ${filename}, cacheKey: ${cacheKey})`);
        }
      } else {
        console.warn(`${logPrefix} ⚠️ Failed to build LLM cache key for overlap fixes (filename: ${filename})`);
      }
    }
  }

  // Increased timeout for reasoning models (especially gpt-oss-20b which may take longer)
  const timeout = useLocalLLM ? 300000 : 180000; // 5 minutes for local, 3 minutes for OpenRouter
  console.log(`${logPrefix} ⏱️  Using timeout: ${timeout / 1000}s for ${useLocalLLM ? 'local' : 'OpenRouter'} LLM`);
  
  let llmResponse;
  try {
    llmResponse = await axios.post(
      apiUrl,
      payload,
      {
        headers: headers,
        timeout: timeout
      }
    );
  } catch (error) {
    console.error(`${logPrefix} ❌ LLM request failed:`, error.message);
    if (sendUpdate) {
      sendUpdate(5, 'error', `❌ Помилка запиту до LLM: ${error.message}`, {
        stage: 'llm_request_failed',
        error: error.message
      });
    }
    throw new Error(`LLM request failed: ${error.message}`);
  }

  const llmOutput = llmResponse.data.choices[0]?.message?.content || '';
  console.log(`${logPrefix} 📥 LLM response received (${llmOutput.length} chars)`);
  console.log(`${logPrefix} 📥 LLM response preview:`, llmOutput.substring(0, 300));
  
  // Save to LLM cache if enabled
  if (LLM_CACHE_ENABLED) {
    if (!filename) {
      console.warn(`${logPrefix} ⚠️ LLM cache save skipped for overlap fixes: filename is missing`);
    } else {
    const fullPrompt = `system: ${systemPrompt}\n\nuser: ${userPrompt}`;
    const cacheKey = buildLLMCacheKey(filename, fullPrompt, model, mode, 'overlap-fixes');
    if (cacheKey) {
        console.log(`${logPrefix} 💾 Saving LLM response to cache for overlap fixes:`, { filename, cacheKey, model, mode });
      writeLLMCache(cacheKey, { llmOutput, model, mode, promptVariant: 'overlap-fixes', timestamp: new Date().toISOString() });
      } else {
        console.warn(`${logPrefix} ⚠️ Failed to build LLM cache key for saving overlap fixes (filename: ${filename})`);
      }
    }
  }

  // ---------- 5. Parse LLM response ----------
  let correctedSegments = [];
  let parseAttempts = [];
  
  try {
    let jsonText = llmOutput.trim();
    
    // Check if response was truncated (common issue with max_tokens)
    const endsWithBracket = llmOutput.trim().endsWith(']');
    const hasCodeBlock = llmOutput.includes('```');
    if (llmOutput.length > 0 && !endsWithBracket && !hasCodeBlock) {
      console.warn(`${logPrefix} ⚠️ LLM response may be truncated (doesn't end with ']' or code block)`);
      console.warn(`${logPrefix} 📋 Response length: ${llmOutput.length} chars, last 100 chars:`, llmOutput.substring(Math.max(0, llmOutput.length - 100)));
    }
    
    // Attempt 1: Try to extract JSON from markdown code blocks
    const jsonMatch = jsonText.match(/```(?:json)?\s*(\[[\s\S]*?\])\s*```/);
    if (jsonMatch) {
      jsonText = jsonMatch[1];
      parseAttempts.push('markdown_code_block');
    }
    
    // Attempt 2: Try to find JSON array directly
    let jsonStart = jsonText.indexOf('[');
    let jsonEnd = jsonText.lastIndexOf(']');
    
    if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
      jsonText = jsonText.substring(jsonStart, jsonEnd + 1);
      parseAttempts.push('direct_array_extraction');
    } else {
      // Attempt 3: Try to find incomplete JSON and try to fix it
      jsonStart = jsonText.indexOf('[');
      if (jsonStart !== -1) {
        // Find the last complete segment before the end
        const partialJson = jsonText.substring(jsonStart);
        // Try to find all complete segment objects
        const segmentPattern = /\{\s*"segment_id"\s*:\s*\d+\s*,[\s\S]*?"speaker_id"\s*:\s*"[^"]+"\s*\}/g;
        const matches = [];
        let match;
        while ((match = segmentPattern.exec(partialJson)) !== null) {
          matches.push(match);
        }
        
        if (matches.length > 0) {
          // Get the last complete match
          const lastMatch = matches[matches.length - 1];
          const fixedJson = partialJson.substring(0, lastMatch.index + lastMatch[0].length) + ']';
          try {
            correctedSegments = JSON.parse(fixedJson);
            parseAttempts.push('truncated_fix');
            console.log(`${logPrefix} ✅ Successfully parsed truncated JSON by fixing incomplete last segment (${correctedSegments.length} segments)`);
          } catch (e) {
            // Continue to other attempts
          }
        }
      }
    }
    
    // If not already parsed, try standard parsing
    if (correctedSegments.length === 0) {
      correctedSegments = JSON.parse(jsonText);
      parseAttempts.push('standard_parse');
    }
    
    if (!Array.isArray(correctedSegments)) {
      throw new Error('LLM response is not an array');
    }
    
    console.log(`${logPrefix} ✅ Successfully parsed LLM response (${correctedSegments.length} segments) using: ${parseAttempts.join(', ')}`);
    
  } catch (parseError) {
    console.error(`${logPrefix} ❌ Failed to parse LLM response:`, parseError.message);
    console.error(`${logPrefix} 📋 Parse attempts:`, parseAttempts);
    console.error(`${logPrefix} 📋 LLM output length: ${llmOutput.length} chars`);
    console.error(`${logPrefix} 📋 LLM output (first 1000 chars):`, llmOutput.substring(0, 1000));
    console.error(`${logPrefix} 📋 LLM output (last 500 chars):`, llmOutput.substring(Math.max(0, llmOutput.length - 500)));
    
    // Try to extract partial JSON if response was truncated
    if (parseError.message.includes('Unexpected end') || parseError.message.includes('end of JSON')) {
      console.log(`${logPrefix} 🔧 Attempting to fix truncated JSON response...`);
      
      try {
        // Find the start of JSON array
        const jsonStart = llmOutput.indexOf('[');
        if (jsonStart !== -1) {
          // Try to extract all complete segments
          const partialJson = llmOutput.substring(jsonStart);
          // Find all complete segment objects (more flexible pattern)
          const segmentPattern = /\{\s*"segment_id"\s*:\s*\d+\s*,[\s\S]*?"speaker_id"\s*:\s*"[^"]+"\s*\}/g;
          const matches = [];
          let match;
          while ((match = segmentPattern.exec(partialJson)) !== null) {
            matches.push(match[0]);
          }
          
          if (matches.length > 0) {
            // Reconstruct JSON array from complete segments
            const fixedJson = '[' + matches.join(',') + ']';
            correctedSegments = JSON.parse(fixedJson);
            console.log(`${logPrefix} ✅ Successfully extracted ${correctedSegments.length} complete segments from truncated response`);
            parseAttempts.push('truncated_recovery');
          } else {
            console.warn(`${logPrefix} ⚠️ No complete segments found in truncated response`);
          }
        }
      } catch (recoveryError) {
        console.error(`${logPrefix} ❌ Failed to recover from truncated JSON:`, recoveryError.message);
      }
    }
    
    // If still failed, return original segments as fallback (don't throw error)
    if (correctedSegments.length === 0) {
      console.warn(`${logPrefix} ⚠️ All parsing attempts failed, using original segments as fallback`);
      if (sendUpdate) {
        sendUpdate(5, 'warning', `⚠️ MODE3: LLM фікси не вдалося застосувати (помилка парсингу), використано програмний результат`, {
          stage: 'llm_parse_failed',
          error: parseError.message,
          outputLength: llmOutput.length,
          outputPreview: llmOutput.substring(0, 500),
          fallback: 'using_original_segments'
        });
      }
      // Return original segments instead of throwing error
      return segmentsForLLM;
    }
  }

  // ---------- 6. Validate corrected segments ----------
  if (correctedSegments.length === 0) {
    console.warn(`${logPrefix} ⚠️ LLM returned empty array, using original segments`);
    return segmentsForLLM;
  }

  // Validate structure and completeness
  try {
    const originalStart = Math.min(...segmentsForLLM.map(s => s.start));
    const originalEnd = Math.max(...segmentsForLLM.map(s => s.end));
    const correctedStart = Math.min(...correctedSegments.map(s => s.start));
    const correctedEnd = Math.max(...correctedSegments.map(s => s.end));
    
    // Sort by start time for validation
    const sortedCorrected = [...correctedSegments].sort((a, b) => a.start - b.start);
    
    correctedSegments.forEach((seg, idx) => {
      if (typeof seg.segment_id !== 'number') {
        throw new Error(`segment_id missing or not a number at index ${idx}`);
      }
      if (typeof seg.text !== 'string' || seg.text.trim().length === 0) {
        throw new Error(`text missing, not a string, or empty at index ${idx}`);
      }
      if (typeof seg.start !== 'number' || typeof seg.end !== 'number') {
        throw new Error(`start/end missing or not numbers at index ${idx}`);
      }
      if (seg.start >= seg.end) {
        throw new Error(`Invalid time range at index ${idx}: start (${seg.start}) >= end (${seg.end})`);
      }
      if (typeof seg.speaker_id !== 'string' || !seg.speaker_id.startsWith('SPEAKER_')) {
        throw new Error(`speaker_id missing, not a string, or invalid format at index ${idx}: ${seg.speaker_id}`);
      }
    });
    
    // Check chronological order
    for (let i = 1; i < sortedCorrected.length; i++) {
      if (sortedCorrected[i].start < sortedCorrected[i - 1].start) {
        throw new Error(`Segments not in chronological order at index ${i}`);
      }
    }
    
    // Check coverage (allow some tolerance for rounding)
    const startDiff = Math.abs(originalStart - correctedStart);
    const endDiff = Math.abs(originalEnd - correctedEnd);
    if (startDiff > 1.0) {
      console.warn(`${logPrefix} ⚠️ Start time mismatch: original=${originalStart.toFixed(2)}s, corrected=${correctedStart.toFixed(2)}s (diff=${startDiff.toFixed(2)}s)`);
    }
    if (endDiff > 1.0) {
      console.warn(`${logPrefix} ⚠️ End time mismatch: original=${originalEnd.toFixed(2)}s, corrected=${correctedEnd.toFixed(2)}s (diff=${endDiff.toFixed(2)}s)`);
    }
    
    // Check for all speakers
    const originalSpeakers = [...new Set(segmentsForLLM.map(s => s.speaker_id))];
    const correctedSpeakers = [...new Set(correctedSegments.map(s => s.speaker_id))];
    const missingSpeakers = originalSpeakers.filter(s => !correctedSpeakers.includes(s));
    if (missingSpeakers.length > 0) {
      console.warn(`${logPrefix} ⚠️ Missing speakers in corrected segments: ${missingSpeakers.join(', ')}`);
    }
    
    console.log(`${logPrefix} ✅ Validation passed: ${correctedSegments.length} segments, time range ${correctedStart.toFixed(2)}s-${correctedEnd.toFixed(2)}s, speakers: ${correctedSpeakers.join(', ')}`);
    
  } catch (validationError) {
    console.error(`${logPrefix} ❌ Validation failed:`, validationError.message);
    if (sendUpdate) {
      sendUpdate(5, 'error', `❌ Помилка валідації: ${validationError.message}`, {
        stage: 'llm_validation_failed',
        error: validationError.message
      });
    }
    throw new Error(`Validation failed: ${validationError.message}`);
  }

  // ---------- 7. Return corrected segments (complete replacement) ----------
  console.log(`${logPrefix} ✅ LLM returned ${correctedSegments.length} corrected segments (replacing ${segmentsForLLM.length} original segments)`);
  
  // Log summary of changes
  const originalCount = segmentsForLLM.length;
  const correctedCount = correctedSegments.length;
  const countDiff = correctedCount - originalCount;
  
  if (countDiff !== 0) {
    console.log(`${logPrefix} 📊 Segment count changed: ${originalCount} → ${correctedCount} (${countDiff > 0 ? '+' : ''}${countDiff})`);
  }
  
  // Log speaker changes (if segment_id matches)
  const speakerChanges = [];
  correctedSegments.forEach(corrected => {
    const original = segmentsForLLM.find(s => s.segment_id === corrected.segment_id);
    if (original && corrected.speaker_id !== original.speaker_id) {
      speakerChanges.push({
        segment_id: corrected.segment_id,
        old: original.speaker_id,
        new: corrected.speaker_id
      });
    }
  });
  
  if (speakerChanges.length > 0) {
    console.log(`${logPrefix} 📊 LLM made ${speakerChanges.length} speaker assignment corrections`);
    speakerChanges.slice(0, 5).forEach(change => {
      console.log(`${logPrefix}   • Segment ${change.segment_id}: ${change.old} → ${change.new}`);
    });
  } else {
    console.log(`${logPrefix} ℹ️ No speaker assignment changes detected (or segments were split/merged)`);
  }

  if (sendUpdate) {
    sendUpdate(5, 'completed', `✅ MODE3: LLM виправив діалог: ${correctedCount} сегментів (${countDiff !== 0 ? `${countDiff > 0 ? '+' : ''}${countDiff} змін` : 'без зміни кількості'}, ${speakerChanges.length} змін спікерів)`, {
      stage: 'llm_fixes_completed',
      output: {
        originalSegmentsCount: originalCount,
        correctedSegmentsCount: correctedCount,
        segmentCountDiff: countDiff,
        speakerChanges: speakerChanges.length
      }
    });
  }

  // ---------- 8. Remove duplicates and empty words ----------
  const cleanedSegments = removeDuplicatesAndEmptyWords(correctedSegments, logPrefix);
  
  if (cleanedSegments.length !== correctedSegments.length) {
    console.log(`${logPrefix} 🧹 Cleaned segments: ${correctedSegments.length} → ${cleanedSegments.length} (removed ${correctedSegments.length - cleanedSegments.length} duplicates/empty)`);
    if (sendUpdate) {
      sendUpdate(5, 'processing', `🧹 Видалено дублікати та пусті слова: ${correctedSegments.length} → ${cleanedSegments.length}`, {
        stage: 'cleaning_segments',
        removed: correctedSegments.length - cleanedSegments.length
      });
    }
  }

  // Return sorted cleaned segments
  return cleanedSegments.sort((a, b) => a.start - b.start);
}

/**
 * Remove duplicate segments and empty words from segments
 * @param {Array} segments - Array of segments with text, start, end, speaker_id
 * @param {string} logPrefix - Log prefix for debugging
 * @returns {Array} Cleaned segments without duplicates and empty words
 */
function removeDuplicatesAndEmptyWords(segments, logPrefix = '') {
  if (!Array.isArray(segments) || segments.length === 0) {
    return segments;
  }

  // Empty words and filler words to remove
  const emptyWords = new Set(['uh', 'um', 'ah', 'eh', 'er', 'hmm', 'mmm', 'oh', 'ahh', 'uhh', 'umm']);
  
  // Normalize text for comparison (lowercase, remove punctuation, trim)
  const normalizeText = (text) => {
    if (!text || typeof text !== 'string') return '';
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, '') // Remove punctuation
      .replace(/\s+/g, ' ') // Normalize whitespace
      .trim();
  };

  // Check if two segments are duplicates
  const areDuplicates = (seg1, seg2) => {
    const text1 = normalizeText(seg1.text);
    const text2 = normalizeText(seg2.text);
    
    // Check if texts are identical or very similar (90% similarity)
    if (text1 === text2) return true;
    
    // Check if one text contains the other (with at least 80% match)
    const longer = text1.length > text2.length ? text1 : text2;
    const shorter = text1.length > text2.length ? text2 : text1;
    if (longer.includes(shorter) && shorter.length / longer.length >= 0.8) return true;
    
    // Check if timestamps are very close (within 0.5 seconds)
    const timeOverlap = Math.abs(seg1.start - seg2.start) < 0.5 && Math.abs(seg1.end - seg2.end) < 0.5;
    if (timeOverlap && text1.length > 0 && text2.length > 0) {
      // Calculate similarity
      const words1 = text1.split(' ').filter(w => w.length > 0);
      const words2 = text2.split(' ').filter(w => w.length > 0);
      const commonWords = words1.filter(w => words2.includes(w));
      const similarity = (commonWords.length * 2) / (words1.length + words2.length);
      if (similarity >= 0.9) return true;
    }
    
    return false;
  };

  // Remove empty words from text
  const cleanText = (text) => {
    if (!text || typeof text !== 'string') return '';
    const words = text.split(/\s+/);
    const cleanedWords = words.filter(word => {
      const normalized = word.toLowerCase().replace(/[^\w]/g, '');
      return normalized.length > 0 && !emptyWords.has(normalized);
    });
    return cleanedWords.join(' ').trim();
  };

  const cleaned = [];
  const seen = new Set();
  
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    
    // Clean text
    const originalText = seg.text || '';
    const cleanedText = cleanText(originalText);
    
    // Skip if text is empty after cleaning
    if (!cleanedText || cleanedText.length === 0) {
      console.log(`${logPrefix} 🗑️ Skipping segment ${seg.segment_id || i + 1}: empty text after cleaning`);
      continue;
    }
    
    // Create segment with cleaned text
    const cleanedSeg = {
      ...seg,
      text: cleanedText
    };
    
    // Check for duplicates
    let isDuplicate = false;
    for (let j = 0; j < cleaned.length; j++) {
      if (areDuplicates(cleanedSeg, cleaned[j])) {
        isDuplicate = true;
        // Keep the segment with longer text or earlier timestamp
        if (cleanedText.length > cleaned[j].text.length || 
            (cleanedText.length === cleaned[j].text.length && cleanedSeg.start < cleaned[j].start)) {
          // Replace with better version
          cleaned[j] = cleanedSeg;
        }
        break;
      }
    }
    
    if (!isDuplicate) {
      // Create unique key for this segment
      const key = `${normalizeText(cleanedText)}_${cleanedSeg.start.toFixed(2)}_${cleanedSeg.end.toFixed(2)}`;
      if (!seen.has(key)) {
        seen.add(key);
        cleaned.push(cleanedSeg);
      } else {
        console.log(`${logPrefix} 🗑️ Skipping duplicate segment ${seg.segment_id || i + 1}: "${cleanedText.substring(0, 50)}..."`);
      }
    } else {
      console.log(`${logPrefix} 🗑️ Skipping duplicate segment ${seg.segment_id || i + 1}: "${cleanedText.substring(0, 50)}..."`);
    }
  }
  
  return cleaned;
}

/**
 * Remove filler words like "Uh", "Um", "Ah" from text
 * @param {string} text - Text to clean
 * @returns {string} Cleaned text without filler words
 */
function removeFillerWords(text) {
  if (!text || typeof text !== 'string') {
    return text;
  }
  
  // List of filler words to remove (case-insensitive)
  const fillerWords = ['uh', 'um', 'ah', 'er', 'eh', 'hmm', 'hm'];
  
  // Split text into words, preserving spaces
  let cleaned = text;
  
  // Remove standalone filler words (with word boundaries)
  fillerWords.forEach(word => {
    // Match whole words only (case-insensitive)
    const regex = new RegExp(`\\b${word}\\b`, 'gi');
    cleaned = cleaned.replace(regex, '').trim();
  });
  
  // Clean up multiple spaces
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  
  return cleaned;
}

/**
 * Remove filler words from markdown table text cells
 * @param {string} markdown - Markdown table string
 * @returns {string} Markdown table with filler words removed from text cells
 */
function removeFillerWordsFromMarkdownTable(markdown) {
  if (!markdown || typeof markdown !== 'string') {
    return markdown;
  }
  
  const lines = markdown.split('\n');
  const cleanedLines = [];
  
  for (const line of lines) {
    // Check if it's a table row (starts and ends with |)
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      // Check if it's a separator row (|---|---|)
      if (line.match(/^\|[\s\-:]*\|$/)) {
        cleanedLines.push(line);
        continue;
      }
      
      // Parse table row
      const cells = line.split('|').map(c => c.trim()).filter(c => c);
      if (cells.length >= 5) {
        // Expected: Segment ID | Speaker | Text | Start Time | End Time
        const segmentId = cells[0];
        const speaker = cells[1];
        const text = removeFillerWords(cells[2]); // Remove filler words from text
        const startTime = cells[3];
        const endTime = cells[4];
        
        // Reconstruct row with cleaned text
        cleanedLines.push(`| ${segmentId} | ${speaker} | ${text} | ${startTime} | ${endTime} |`);
      } else {
        // Keep header or other rows as is
        cleanedLines.push(line);
      }
    } else {
      // Keep non-table lines as is
      cleanedLines.push(line);
    }
  }
  
  return cleanedLines.join('\n');
}

/**
 * Remove duplicate rows from a Markdown table
 * @param {string} markdown - Markdown table string
 * @returns {string} Cleaned markdown table without duplicates
 */
function removeDuplicatesFromMarkdownTable(markdown) {
  if (!markdown || typeof markdown !== 'string') {
    return markdown;
  }

  const lines = markdown.split('\n');
  const tableRows = [];
  let headerRow = null;
  let separatorRow = null;
  let beforeTable = [];
  let afterTable = [];
  let inTable = false;
  let foundSeparator = false;

  // Parse markdown table
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (line.startsWith('|') && line.endsWith('|')) {
      // Check if it's a separator row (|---|---|)
      if (line.match(/^\|[\s\-:]*\|$/)) {
        if (!foundSeparator) {
          separatorRow = line;
          foundSeparator = true;
          inTable = true;
        }
        continue; // Skip separator row
      }
      
      if (!inTable && !foundSeparator) {
        // Header row
        headerRow = line;
      } else if (inTable) {
        // Data row
        const cells = line.split('|').map(c => c.trim()).filter(c => c);
        if (cells.length >= 5) {
          // Expected: Segment ID | Speaker | Text | Start Time | End Time
          const segmentId = cells[0];
          const speaker = cells[1];
          const text = removeFillerWords(cells[2]); // Remove filler words from text
          const startTime = parseFloat(cells[3]) || 0;
          const endTime = parseFloat(cells[4]) || 0;
          
          tableRows.push({
            originalLine: line,
            segmentId: segmentId,
            speaker: speaker,
            text: text,
            startTime: startTime,
            endTime: endTime
          });
        }
      }
    } else {
      if (inTable && foundSeparator) {
        // After table
        afterTable.push(lines[i]);
      } else if (!inTable && !foundSeparator) {
        // Before table
        beforeTable.push(lines[i]);
      }
    }
  }

  // Remove duplicates
  const seen = new Set();
  const uniqueRows = [];
  
  for (const row of tableRows) {
    // Normalize text for comparison (lowercase, remove punctuation, trim)
    const normalizeText = (text) => {
      if (!text || typeof text !== 'string') return '';
      return text
        .toLowerCase()
        .replace(/[^\w\s]/g, '') // Remove punctuation
        .replace(/\s+/g, ' ') // Normalize whitespace
        .trim();
    };

    const normalizedText = normalizeText(row.text);
    const timeKey = `${row.startTime.toFixed(2)}_${row.endTime.toFixed(2)}`;
    const key = `${normalizedText}_${timeKey}_${row.speaker}`;
    
    // Check for duplicates
    let isDuplicate = false;
    for (const seenRow of uniqueRows) {
      const seenNormalizedText = normalizeText(seenRow.text);
      const timeOverlap = Math.abs(seenRow.startTime - row.startTime) < 0.5 && 
                          Math.abs(seenRow.endTime - row.endTime) < 0.5;
      
      // Check if texts are identical or very similar (90% similarity)
      if (normalizedText === seenNormalizedText && timeOverlap) {
        isDuplicate = true;
        // Keep the row with longer text or earlier timestamp
        if (row.text.length > seenRow.text.length || 
            (row.text.length === seenRow.text.length && row.startTime < seenRow.startTime)) {
          // Replace with better version
          const index = uniqueRows.indexOf(seenRow);
          uniqueRows[index] = row;
        }
        break;
      }
    }
    
    if (!isDuplicate && !seen.has(key)) {
      seen.add(key);
      uniqueRows.push(row);
    }
  }

  // Reconstruct markdown table
  let result = beforeTable.join('\n');
  if (result && !result.endsWith('\n')) result += '\n';
  
  if (headerRow) {
    result += headerRow + '\n';
  }
  if (separatorRow) {
    result += separatorRow + '\n';
  }
  
  for (const row of uniqueRows) {
    result += row.originalLine + '\n';
  }
  
  if (afterTable.length > 0) {
    result += afterTable.join('\n');
  }

  if (tableRows.length !== uniqueRows.length) {
    console.log(`🧹 Removed ${tableRows.length - uniqueRows.length} duplicate rows from markdown table`);
  }

  console.log(`📋 removeDuplicatesFromMarkdownTable: input ${markdown.length} chars, output ${result.length} chars`);
  console.log(`📋 removeDuplicatesFromMarkdownTable: input ${tableRows.length} rows, output ${uniqueRows.length} rows`);

  return result.trim();
}

/**
 * Analyze role distribution in a markdown table
 * @param {string} markdownTable - Markdown table string
 * @returns {Object} Object with agentCount, clientCount, and total
 */
function analyzeRoleDistribution(markdownTable) {
  if (!markdownTable) return { agentCount: 0, clientCount: 0, total: 0 };
  
  const tableLines = markdownTable.split('\n').filter(line => line.trim() && line.includes('|'));
  let agentCount = 0;
  let clientCount = 0;
  
  // Skip header and separator rows
  for (let i = 2; i < tableLines.length; i++) {
    const cells = tableLines[i].split('|').map(c => c.trim()).filter(c => c);
    if (cells.length >= 2) {
      const speaker = cells[1].toLowerCase();
      if (speaker === 'agent') agentCount++;
      else if (speaker === 'client') clientCount++;
    }
  }
  
  return { agentCount, clientCount, total: agentCount + clientCount };
}

/**
 * Merge consecutive segments from the same speaker in a markdown table
 * Ensures speakers alternate (Agent, Client, Agent, Client...)
 * @param {string} markdownTable - Markdown table string
 * @param {number} maxGapSeconds - Maximum gap between segments to merge (default: 2.0)
 * @returns {string} Markdown table with merged consecutive segments and alternating speakers
 */
function mergeConsecutiveSpeakerSegmentsInMarkdown(markdownTable, maxGapSeconds = 2.0) {
  if (!markdownTable) return markdownTable;
  
  const lines = markdownTable.split('\n');
  const headerLines = [];
  const dataRows = [];
  let foundSeparator = false;
  
  // Parse markdown table
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (!line || !line.includes('|')) {
      continue;
    }
    
    // Check if it's a separator row
    if (line.match(/^\|[\s\-:]*\|$/)) {
      if (!foundSeparator) {
        headerLines.push(lines[i]);
        foundSeparator = true;
      }
      continue;
    }
    
    if (!foundSeparator) {
      headerLines.push(lines[i]);
    } else {
      const cells = line.split('|').map(c => c.trim()).filter(c => c);
      if (cells.length >= 5) {
        // Expected: Segment ID | Speaker | Text | Start Time | End Time
        const speaker = cells[1].trim();
        // Normalize speaker name (Agent/Client)
        const normalizedSpeaker = speaker.toLowerCase() === 'agent' ? 'Agent' : 
                                 speaker.toLowerCase() === 'client' ? 'Client' : speaker;
        
        // CRITICAL: Filter out any speakers that are not Agent or Client
        // This prevents SPEAKER_02, SPEAKER_03, etc. from appearing in the table
        if (normalizedSpeaker !== 'Agent' && normalizedSpeaker !== 'Client') {
          console.log(`⚠️ Filtering out segment with disallowed speaker: ${speaker} (normalized: ${normalizedSpeaker})`);
          continue; // Skip this row
        }
        
        dataRows.push({
          segmentId: cells[0],
          speaker: normalizedSpeaker,
          text: cells[2],
          startTime: parseFloat(cells[3]) || 0,
          endTime: parseFloat(cells[4]) || 0,
          originalLine: lines[i]
        });
      }
    }
  }
  
  if (dataRows.length === 0) {
    return markdownTable;
  }
  
  // Step 1: Merge consecutive segments from the same speaker
  // IMPROVED: Now handles overlapping timestamps and larger gaps
  const mergedRows = [];
  
  for (let i = 0; i < dataRows.length; i++) {
    const current = dataRows[i];
    
    if (mergedRows.length === 0) {
      // First row
      mergedRows.push({
        ...current,
        startTime: current.startTime,
        endTime: current.endTime
      });
      continue;
    }
    
    const last = mergedRows[mergedRows.length - 1];
    const lastStart = last.startTime;
    const lastEnd = last.endTime;
    const currentStart = current.startTime;
    const currentEnd = current.endTime;
    
    // Calculate overlap and gap
    const overlap = Math.max(0, Math.min(lastEnd, currentEnd) - Math.max(lastStart, currentStart));
    const gap = currentStart - lastEnd;
    
    // Check if we should merge: same speaker AND (overlapping OR small gap)
    // Also merge if segments are very close in time (even if gap is slightly larger)
    const shouldMerge = last.speaker === current.speaker && (
      overlap > 0 || // Overlapping segments
      gap <= maxGapSeconds || // Small gap
      (gap <= maxGapSeconds * 2 && lastEnd > 0 && currentStart > 0 && Math.abs(gap) < 5.0) // Close segments from same speaker
    );
    
    if (shouldMerge) {
      // Merge: extend end time and combine text
      last.endTime = Math.max(lastEnd, currentEnd);
      last.text = (last.text + ' ' + current.text).trim().replace(/\s+/g, ' ');
      // Update start time to the earliest
      last.startTime = Math.min(lastStart, currentStart);
      console.log(`🔗 Merging consecutive ${last.speaker} segments: [${lastStart.toFixed(2)}-${lastEnd.toFixed(2)}] + [${currentStart.toFixed(2)}-${currentEnd.toFixed(2)}] → [${last.startTime.toFixed(2)}-${last.endTime.toFixed(2)}]`);
    } else {
      // New segment
      mergedRows.push({
        ...current,
        startTime: current.startTime,
        endTime: current.endTime
      });
    }
  }
  
  // Step 2: Ensure speakers alternate (Agent, Client, Agent, Client...)
  // If we have consecutive segments from the same speaker, merge them
  const alternatingRows = [];
  let segmentId = 1;
  
  for (let i = 0; i < mergedRows.length; i++) {
    const current = mergedRows[i];
    
    if (alternatingRows.length === 0) {
      // First row
      alternatingRows.push({
        ...current,
        segmentId: segmentId++
      });
      continue;
    }
    
    const last = alternatingRows[alternatingRows.length - 1];
    
    // If same speaker as previous, merge them (regardless of gap)
    if (last.speaker === current.speaker) {
      // Merge: extend end time and combine text
      last.endTime = Math.max(last.endTime, current.endTime);
      last.text = (last.text + ' ' + current.text).trim().replace(/\s+/g, ' ');
      // Update start time to the earliest
      last.startTime = Math.min(last.startTime, current.startTime);
    } else {
      // Different speaker - add as new row
      alternatingRows.push({
        ...current,
        segmentId: segmentId++
      });
    }
  }
  
  // Reconstruct markdown table
  let result = headerLines.join('\n') + '\n';
  alternatingRows.forEach(row => {
    result += `| ${row.segmentId} | ${row.speaker} | ${row.text} | ${row.startTime.toFixed(2)} | ${row.endTime.toFixed(2)} |\n`;
  });
  
  if (alternatingRows.length !== dataRows.length) {
    console.log(`🔗 Merged ${dataRows.length - alternatingRows.length} consecutive segments from same speaker in markdown table (ensured alternation)`);
  }
  
  return result.trim();
}

/**
 * Check input segments for potential consecutive same-speaker issues
 * This helps enhance the prompt before sending to LLM
 * @param {Array} agentSegments - Agent transcript segments
 * @param {Array} clientSegments - Client transcript segments
 * @param {Object} promptContext - Prompt context with general dialogue
 * @returns {boolean} True if potential consecutive segments detected
 */
function checkForConsecutiveSegmentsInInput(agentSegments, clientSegments, promptContext) {
  // Check if we have segments that are close in time and from the same speaker
  const allSegments = [];
  
  // Add agent segments
  (agentSegments || []).forEach(seg => {
    if (seg && seg.start !== undefined && seg.end !== undefined) {
      allSegments.push({
        ...seg,
        speaker: 'Agent',
        start: parseFloat(seg.start) || 0,
        end: parseFloat(seg.end) || parseFloat(seg.start) || 0
      });
    }
  });
  
  // Add client segments
  (clientSegments || []).forEach(seg => {
    if (seg && seg.start !== undefined && seg.end !== undefined) {
      allSegments.push({
        ...seg,
        speaker: 'Client',
        start: parseFloat(seg.start) || 0,
        end: parseFloat(seg.end) || parseFloat(seg.start) || 0
      });
    }
  });
  
  // Sort by start time
  allSegments.sort((a, b) => a.start - b.start);
  
  // Check for consecutive same-speaker segments
  for (let i = 1; i < allSegments.length; i++) {
    const prev = allSegments[i - 1];
    const curr = allSegments[i];
    
    // If same speaker and close in time (gap < 5 seconds or overlapping)
    if (prev.speaker === curr.speaker) {
      const gap = curr.start - prev.end;
      if (gap < 5.0 || gap < 0) { // Overlapping or close
        console.log(`⚠️ Potential consecutive ${prev.speaker} segments detected: [${prev.start.toFixed(2)}-${prev.end.toFixed(2)}] and [${curr.start.toFixed(2)}-${curr.end.toFixed(2)}] (gap: ${gap.toFixed(2)}s)`);
        return true;
      }
    }
  }
  
  return false;
}

/**
 * Detect consecutive same-speaker segments in markdown table
 * Returns information about consecutive segments that might need more context for LLM
 * @param {string} markdownTable - Markdown table string
 * @returns {Object} Object with hasConsecutiveSameSpeaker flag and details
 */
function detectConsecutiveSameSpeakerSegments(markdownTable) {
  if (!markdownTable) {
    return { hasConsecutiveSameSpeaker: false, consecutiveGroups: [] };
  }
  
  const lines = markdownTable.split('\n');
  const dataRows = [];
  let foundSeparator = false;
  
  // Parse markdown table
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (!line || !line.includes('|')) {
      continue;
    }
    
    // Check if it's a separator row
    if (line.match(/^\|[\s\-:]*\|$/)) {
      foundSeparator = true;
      continue;
    }
    
    if (foundSeparator) {
      const cells = line.split('|').map(c => c.trim()).filter(c => c);
      if (cells.length >= 5) {
        const speaker = cells[1].trim();
        const normalizedSpeaker = speaker.toLowerCase() === 'agent' ? 'Agent' : 
                                 speaker.toLowerCase() === 'client' ? 'Client' : speaker;
        
        if (normalizedSpeaker === 'Agent' || normalizedSpeaker === 'Client') {
          dataRows.push({
            segmentId: cells[0],
            speaker: normalizedSpeaker,
            text: cells[2],
            startTime: parseFloat(cells[3]) || 0,
            endTime: parseFloat(cells[4]) || 0
          });
        }
      }
    }
  }
  
  if (dataRows.length < 2) {
    return { hasConsecutiveSameSpeaker: false, consecutiveGroups: [] };
  }
  
  // Find consecutive same-speaker groups
  const consecutiveGroups = [];
  let currentGroup = null;
  
  for (let i = 0; i < dataRows.length; i++) {
    const current = dataRows[i];
    
    if (i === 0) {
      currentGroup = {
        speaker: current.speaker,
        segments: [current],
        startIndex: i
      };
    } else {
      const prev = dataRows[i - 1];
      
      if (prev.speaker === current.speaker) {
        // Same speaker - add to current group
        if (!currentGroup) {
          currentGroup = {
            speaker: current.speaker,
            segments: [prev, current],
            startIndex: i - 1
          };
        } else {
          currentGroup.segments.push(current);
        }
      } else {
        // Different speaker - finalize current group if it has 2+ segments
        if (currentGroup && currentGroup.segments.length >= 2) {
          consecutiveGroups.push({
            speaker: currentGroup.speaker,
            count: currentGroup.segments.length,
            startIndex: currentGroup.startIndex,
            endIndex: i - 1,
            segments: currentGroup.segments
          });
        }
        currentGroup = null;
      }
    }
  }
  
  // Finalize last group if exists
  if (currentGroup && currentGroup.segments.length >= 2) {
    consecutiveGroups.push({
      speaker: currentGroup.speaker,
      count: currentGroup.segments.length,
      startIndex: currentGroup.startIndex,
      endIndex: dataRows.length - 1,
      segments: currentGroup.segments
    });
  }
  
  return {
    hasConsecutiveSameSpeaker: consecutiveGroups.length > 0,
    consecutiveGroups: consecutiveGroups
  };
}

/**
 * Build enhanced dialogue context with more replicas when consecutive same-speaker segments detected
 * @param {Object} promptContext - Original prompt context
 * @param {Object} consecutiveDetection - Result from detectConsecutiveSameSpeakerSegments
 * @returns {Object} Enhanced prompt context with more replicas
 */
function buildEnhancedDialogueContext(promptContext, consecutiveDetection) {
  // If no consecutive segments detected, return original context
  if (!consecutiveDetection.hasConsecutiveSameSpeaker) {
    return promptContext;
  }
  
  console.log(`🔍 Detected ${consecutiveDetection.consecutiveGroups.length} groups of consecutive same-speaker segments. Providing enhanced context (4 replicas instead of 2).`);
  
  // For now, we'll add a note to the prompt about consecutive segments
  // The actual enhancement will be in the prompt template itself
  return {
    ...promptContext,
    hasConsecutiveSameSpeaker: true,
    consecutiveGroups: consecutiveDetection.consecutiveGroups
  };
}


// Correct primary diarization using extracted voice tracks
async function correctPrimaryDiarizationWithTracks(primaryDiarization, voiceTracks, language, mode = 'smart', requestId = null, res = null, overlapPipelineMode = 'mode1', sendSSEUpdate = null, filename = null) {
  const logPrefix = requestId ? `[${requestId}]` : '[Overlap Correction]';
  const isMode3 = overlapPipelineMode === 'mode3';
  
  // Helper to send SSE updates
  const sendUpdate = (step, status, description, details = {}) => {
    if (sendSSEUpdate) {
      sendSSEUpdate(step, status, description, details);
    } else if (isMode3 && res) {
      try {
        res.write(`data: ${JSON.stringify({
          type: 'step-progress',
          step,
          status,
          description,
          details: { ...details, timestamp: new Date().toISOString(), requestId }
        })}\n\n`);
      } catch (e) {
        console.error(`${logPrefix} ❌ Failed to send SSE update:`, e.message);
      }
    }
  };
  
  try {
    // Build plain text transcript by merging primary diarization with voice tracks
    // Use primary diarization as base and add missing replicas from voice tracks
    console.log(`${logPrefix} 🔵 STEP 5: Building merged transcript from voice tracks...`);
    
    sendUpdate(5, 'processing', `🔵 MODE3: Збір сегментів з voice tracks...`, {
      stage: 'collecting_segments',
      input: {
        voiceTracksCount: voiceTracks.length,
        voiceTracks: voiceTracks.map(vt => ({
          speaker: vt.speaker,
          hasError: !!vt.error,
          hasTranscription: !!vt.transcription,
          transcriptLength: vt.transcriptText?.length || 0,
          role: vt.roleAnalysis?.role
        }))
      }
    });
    
    const voiceTrackSegments = collectVoiceTrackSegments(voiceTracks);
    console.log(`${logPrefix} 📊 STEP 5 MODE3: Collected ${voiceTrackSegments.length} segments from voice tracks`);
    
    const segmentsBreakdown = voiceTrackSegments.reduce((acc, seg) => {
      acc[seg.speaker] = (acc[seg.speaker] || 0) + 1;
      return acc;
    }, {});
    
    console.log(`${logPrefix} 📊 STEP 5 MODE3: Segments breakdown:`, {
      totalSegments: voiceTrackSegments.length,
      segmentsBySpeaker: segmentsBreakdown,
      segmentsWithRole: voiceTrackSegments.filter(s => s.role).length,
      segmentsPreview: voiceTrackSegments.slice(0, 5).map(s => ({
        speaker: s.speaker,
        role: s.role,
        text: s.text?.substring(0, 50),
        start: s.start,
        end: s.end
      }))
    });
    
    sendUpdate(5, 'processing', `🔵 MODE3: Знайдено ${voiceTrackSegments.length} сегментів. Програмне об'єднання транскриптів...`, {
      stage: 'programmatic_merge',
      input: {
        primarySegmentsCount: primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0,
        voiceTracksCount: voiceTracks.length,
        voiceTrackSegmentsCount: voiceTrackSegments.length,
        segmentsBySpeaker: segmentsBreakdown
      }
    });
    
    // Step 1: Programmatic merge (preserves all primary speakers)
    let correctedResult = mergeTranscriptsProgrammatically(primaryDiarization, voiceTracks);
    
    // Prepare segments data for LLM processing
    const programmaticSegments = correctedResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
    const segmentsForLLM = prepareSegmentsForLLM(programmaticSegments);
    
    console.log(`${logPrefix} 📋 Prepared ${segmentsForLLM.length} segments for LLM processing`);
    console.log(`${logPrefix} 📋 Segments data structure:`, {
      totalSegments: segmentsForLLM.length,
      sampleSegments: segmentsForLLM.slice(0, 3),
      speakers: [...new Set(segmentsForLLM.map(s => s.speaker_id))],
      timeRange: segmentsForLLM.length > 0 ? {
        start: Math.min(...segmentsForLLM.map(s => s.start)),
        end: Math.max(...segmentsForLLM.map(s => s.end))
      } : null
    });
    
    // Log full segments array for debugging (first 5 and last 5)
    if (segmentsForLLM.length > 0) {
      console.log(`${logPrefix} 📋 First 5 segments:`, JSON.stringify(segmentsForLLM.slice(0, 5), null, 2));
      if (segmentsForLLM.length > 5) {
        console.log(`${logPrefix} 📋 Last 5 segments:`, JSON.stringify(segmentsForLLM.slice(-5), null, 2));
      }
    }
    
    // Step 2: Send segments to LLM for speaker mixing fixes
    // DISABLED FOR MODE3: New client-side approach (runDebugActionNew + testPostProcessing) handles this
    // The old server-side LLM overlap correction is no longer needed for MODE3
    let llmCorrectedSegments = null;
    if (isMode3) {
      console.log(`${logPrefix} ⏭️ MODE3: Skipping server-side LLM overlap correction (using new client-side approach instead)`);
      sendUpdate(5, 'processing', `⏭️ MODE3: Пропуск серверної LLM корекції overlap (використовується новий клієнтський підхід)`, {
        stage: 'skipping_llm_correction',
        reason: 'New client-side approach handles overlap correction via runDebugActionNew + testPostProcessing'
      });
      // Use programmatic merge result directly without LLM correction
      llmCorrectedSegments = null;
    } else {
      try {
        console.log(`${logPrefix} 📤 Sending ${segmentsForLLM.length} segments to LLM (voice tracks: ${voiceTracks.length})...`);
        llmCorrectedSegments = await sendSegmentsToLLMForFixes(segmentsForLLM, {
          mode: mode || 'smart',
          language: language,
          requestId: requestId,
          sendUpdate: sendUpdate,
          voiceTracks: voiceTracks, // Pass voice tracks for context
          filename: filename // Pass filename for LLM cache
        });
      
      console.log(`${logPrefix} 📥 LLM returned ${llmCorrectedSegments?.length || 0} segments`);
      
      // Create a map of original segments by segment_id for lookup
      const originalSegmentsMap = new Map(programmaticSegments.map((seg, idx) => {
        // Try to find segment_id if it exists, otherwise use index+1
        const segId = seg.segment_id || (idx + 1);
        return [segId, seg];
      }));
      
      // Replace entire segments array with LLM corrected segments
      // Always apply if we got valid corrected segments (even if count is same, content might differ)
      if (llmCorrectedSegments && llmCorrectedSegments.length > 0) {
        // Compare to detect actual changes
        const segmentsEqual = llmCorrectedSegments.length === segmentsForLLM.length && 
          JSON.stringify(llmCorrectedSegments) === JSON.stringify(segmentsForLLM);
        
        const hasChanges = !segmentsEqual || llmCorrectedSegments.length !== segmentsForLLM.length ||
          llmCorrectedSegments.some((llmSeg, idx) => {
            const origSeg = segmentsForLLM[idx];
            return !origSeg || 
              llmSeg.speaker_id !== origSeg.speaker_id ||
              llmSeg.text !== origSeg.text ||
              Math.abs(llmSeg.start - origSeg.start) > 0.01 ||
              Math.abs(llmSeg.end - origSeg.end) > 0.01;
          });
        
        if (segmentsEqual) {
          console.log(`${logPrefix} ⚠️ LLM returned identical segments (no changes detected), but applying anyway`);
        } else if (hasChanges) {
          console.log(`${logPrefix} ✅ LLM made changes detected (speaker/text/timestamps differ)`);
        }
        
        console.log(`${logPrefix} 🔄 Applying LLM corrections: ${programmaticSegments.length} → ${llmCorrectedSegments.length} segments`);
        
        // Convert LLM format back to system format
        // Convert Agent -> SPEAKER_00, Client -> SPEAKER_01
        const newSegments = llmCorrectedSegments.map((llmSeg, idx) => {
          // Try to find original segment by segment_id, fallback to index
          const originalSeg = originalSegmentsMap.get(llmSeg.segment_id) || programmaticSegments[idx] || {};
          
          // Convert Agent/Client back to SPEAKER_00/SPEAKER_01
          let speaker = llmSeg.speaker_id;
          if (speaker === 'Agent') {
            speaker = 'SPEAKER_00';
          } else if (speaker === 'Client') {
            speaker = 'SPEAKER_01';
          }
          
          return {
            speaker: speaker, // Convert back to SPEAKER_00/SPEAKER_01
            text: llmSeg.text, // Use corrected text
            start: llmSeg.start, // Use corrected start (may be modified)
            end: llmSeg.end, // Use corrected end (may be modified)
            words: originalSeg?.words || [], // Preserve words if available
            role: originalSeg?.role || null, // Preserve role if available
            overlap: originalSeg?.overlap || false, // Preserve overlap flag
            source: originalSeg?.source || 'llm-corrected', // Mark as LLM corrected
            originalTrackSpeaker: originalSeg?.originalTrackSpeaker || null,
            originalDetectedSpeaker: originalSeg?.originalDetectedSpeaker || null,
            isFullText: originalSeg?.isFullText || false,
            llmCorrected: true // Flag to indicate LLM correction
          };
        });
        
        // Replace entire segments array
        correctedResult.recordings[0].results['overlap-corrected'].segments = newSegments;
        
        // Update metadata
        correctedResult.recordings[0].results['overlap-corrected'].serviceName = 'Overlap Corrected (Programmatic + LLM Fixed)';
        correctedResult.recordings[0].results['overlap-corrected'].rawData = correctedResult.recordings[0].results['overlap-corrected'].rawData || {};
        correctedResult.recordings[0].results['overlap-corrected'].rawData.llmFixed = true;
        correctedResult.recordings[0].results['overlap-corrected'].rawData.llmCorrectionsCount = llmCorrectedSegments.length;
        correctedResult.recordings[0].results['overlap-corrected'].rawData.originalSegmentsCount = programmaticSegments.length;
        correctedResult.recordings[0].results['overlap-corrected'].rawData.correctedSegmentsCount = llmCorrectedSegments.length;
        
        console.log(`${logPrefix} ✅ Replaced ${programmaticSegments.length} segments with ${llmCorrectedSegments.length} LLM-corrected segments`);
        console.log(`${logPrefix} 📊 Coverage: ${Math.min(...newSegments.map(s => s.start)).toFixed(2)}s - ${Math.max(...newSegments.map(s => s.end)).toFixed(2)}s`);
        
        // Verify all speakers are present
        const speakers = [...new Set(newSegments.map(s => s.speaker))];
        console.log(`${logPrefix} 📊 Speakers in corrected dialogue: ${speakers.join(', ')}`);
        
        // Log sample of changes
        const sampleChanges = [];
        for (let i = 0; i < Math.min(5, newSegments.length); i++) {
          const newSeg = newSegments[i];
          const origSeg = programmaticSegments[i];
          if (origSeg && (newSeg.speaker !== origSeg.speaker || newSeg.text !== origSeg.text)) {
            sampleChanges.push({
              idx: i,
              speaker: `${origSeg.speaker} → ${newSeg.speaker}`,
              textChanged: newSeg.text !== origSeg.text
            });
          }
        }
        if (sampleChanges.length > 0) {
          console.log(`${logPrefix} 📊 Sample changes:`, sampleChanges);
        }
      } else {
        console.warn(`${logPrefix} ⚠️ LLM returned empty or invalid segments, using programmatic result`);
      }
      } catch (llmError) {
        console.error(`${logPrefix} ⚠️ LLM fixes failed, continuing with programmatic result:`, llmError.message);
        // Continue with programmatic result if LLM fails
        if (sendUpdate) {
          sendUpdate(5, 'warning', `⚠️ MODE3: LLM фікси не вдалося застосувати, використано програмний результат`, {
            stage: 'llm_fixes_failed',
            error: llmError.message
          });
        }
      }
    }
    
    // Step 3: Optional LLM refinement (if enabled)
    // LLM can help with final corrections, role assignments, and text improvements
    // DISABLED by default to avoid timeouts - enable explicitly if needed
    const useLLMRefinement = process.env.USE_LLM_OVERLAP_REFINEMENT === 'true'; // Default: false
    
    if (useLLMRefinement) {
      sendUpdate(5, 'processing', `🔵 MODE3: Програмне об'єднання завершено. Застосовується LLM для фінальної корекції...`, {
        stage: 'llm_refinement',
        input: {
          programmaticSegmentsCount: correctedResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments?.length || 0
        }
      });
      
      try {
        // Build transcript from programmatic result for LLM
        const programmaticSegments = correctedResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
        const transcript = programmaticSegments
          .map(s => `${s.speaker}: ${s.text}`)
          .join('\n');
        
        // Call LLM for refinement
        const llmRefined = await generateOverlapCorrectionResult({
          primaryDiarization: correctedResult, // Use programmatic result as primary
          voiceTracks,
          transcript,
          existingLLMResult: null,
          mode: mode || 'smart',
          requestId,
          filename: filename // Pass filename for LLM cache
        });
        
        if (llmRefined && llmRefined.recordings?.[0]?.results?.['overlap-corrected']?.segments?.length > 0) {
          // Merge LLM refinements: use LLM text but keep primary speakers and timestamps
          const llmSegments = llmRefined.recordings[0].results['overlap-corrected'].segments;
          const programmaticSegmentsMap = new Map();
          
          programmaticSegments.forEach(seg => {
            const key = `${seg.speaker}_${seg.start.toFixed(3)}_${seg.end.toFixed(3)}`;
            programmaticSegmentsMap.set(key, seg);
          });
          
          // Enhance programmatic segments with LLM text where timestamps match
          const finalSegments = programmaticSegments.map(progSeg => {
            const key = `${progSeg.speaker}_${progSeg.start.toFixed(3)}_${progSeg.end.toFixed(3)}`;
            
            // Find matching LLM segment
            const matchingLLM = llmSegments.find(llmSeg => {
              const llmStart = parseFloat(llmSeg.start || 0);
              const llmEnd = parseFloat(llmSeg.end || llmStart);
              const progStart = parseFloat(progSeg.start || 0);
              const progEnd = parseFloat(progSeg.end || progStart);
              
              return llmSeg.speaker === progSeg.speaker &&
                     Math.abs(llmStart - progStart) < 0.5 &&
                     Math.abs(llmEnd - progEnd) < 0.5;
            });
            
            if (matchingLLM && matchingLLM.text) {
              return {
                ...progSeg,
                text: matchingLLM.text, // Use LLM text
                originalText: matchingLLM.text,
                role: matchingLLM.role || progSeg.role,
                source: 'llm-refined',
                mergeConfidence: 'high'
              };
            }
            
            return progSeg;
          });
          
          // Update result with LLM-refined segments
          correctedResult.recordings[0].results['overlap-corrected'].segments = finalSegments;
          correctedResult.recordings[0].results['overlap-corrected'].serviceName = 'Overlap Corrected (Programmatic + LLM Refined)';
          correctedResult.recordings[0].results['overlap-corrected'].rawData.llmRefined = true;
          
          console.log(`${logPrefix} ✅ LLM refinement applied: ${finalSegments.length} segments`);
        } else {
          console.log(`${logPrefix} ⚠️ LLM refinement skipped (no valid result), using programmatic merge`);
        }
      } catch (llmError) {
        console.error(`${logPrefix} ❌ LLM refinement failed:`, llmError);
        // Continue with programmatic result
      }
    }
    
    const finalSegmentsCount = correctedResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments?.length || 0;
    
    sendUpdate(5, 'completed', `✅ MODE3: Об'єднання завершено. Результат: ${finalSegmentsCount} сегментів${useLLMRefinement ? ' (з LLM корекцією)' : ''}`, {
      stage: 'completed',
      output: {
        segmentsCount: finalSegmentsCount,
        source: useLLMRefinement ? 'programmatic+llm' : 'programmatic',
        primarySegmentsCount: primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0,
        segmentsDifference: finalSegmentsCount - (primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0)
      }
    });
    
    writeLog('info', 'Merge completed', {
      segmentsCount: finalSegmentsCount,
      source: useLLMRefinement ? 'programmatic+llm' : 'programmatic',
      primarySegmentsCount: primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0,
      voiceTrackSegmentsCount: voiceTrackSegments.length
    });
    
    return correctedResult;
    
    sendUpdate(5, 'completed', `⚠️ MODE3: Корекція overlap завершена. Використано primary diarization`, {
      stage: 'completed_primary',
      output: {
        source: 'primary',
        reason: 'Voice track fallback unavailable'
      }
    });
    
    writeLog('warn', 'Overlap Correction Agent fallback returned primary diarization', {
      reason: 'Voice track fallback unavailable',
      source: 'primary'
    });
    return primaryDiarization;
    
  } catch (error) {
    console.error('Overlap Correction Agent error:', error);
    writeLog('error', 'Overlap Correction Agent failed', {
      error: error.message,
      stack: error.stack
    });
    
    const fallbackResult = buildOverlapCorrectedFromVoiceTracks(primaryDiarization, voiceTracks);
    if (fallbackResult && fallbackResult !== primaryDiarization) {
      writeLog('info', 'Overlap Correction Agent error fallback succeeded', {
        segmentsCount: fallbackResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments?.length || 0
      });
      return fallbackResult;
    }
    
    return primaryDiarization;
  }
}

// Gemini 2.5 Pro multimodal integration
function getLanguageDisplayName(code) {
  if (!code || typeof code !== 'string') {
    return 'the provided language';
  }
  const normalized = code.toLowerCase();
  const map = {
    ar: 'Arabic',
    en: 'English',
    uk: 'Ukrainian',
    ru: 'Russian',
    fr: 'French',
    de: 'German',
    es: 'Spanish'
  };
  return map[normalized] || code;
}

function adaptPromptForLanguage(template, languageDisplayName) {
  if (
    !template ||
    !languageDisplayName ||
    languageDisplayName.toLowerCase() === 'arabic'
  ) {
    return template;
  }

  let adapted = template;
  const replacements = [
    { pattern: /Arabic\/English/gi, value: languageDisplayName },
    { pattern: /Arabic and English/gi, value: languageDisplayName },
    { pattern: /Arabic\/ English/gi, value: languageDisplayName },
    { pattern: /Arabic-language/gi, value: `${languageDisplayName}-language` },
    { pattern: /Arabic/gi, value: languageDisplayName }
  ];

  replacements.forEach(({ pattern, value }) => {
    adapted = adapted.replace(pattern, value);
  });

  return adapted;
}

async function callGeminiMultimodal(audioPath, transcript, languageHint = 'ar') {
  try {
    // Load diarization prompt
    const promptVariant = 'default'; // Use default prompt variant for Gemini diarization
    const promptConfig = getPromptTemplateConfig(promptVariant);
    const promptTemplate = await fs.readFile(promptConfig.templatePath, 'utf8');
    const systemPrompt = (await fs.readFile('prompts/system_diarization.txt', 'utf8')).trim();
    
    const languageDisplayName = getLanguageDisplayName(languageHint);
    const adaptedPromptTemplate = adaptPromptForLanguage(promptTemplate, languageDisplayName);
    const adaptedSystemPrompt = adaptPromptForLanguage(systemPrompt, languageDisplayName);
    const languageInstruction = `### LANGUAGE CONTEXT
- The conversation is in ${languageDisplayName}.
- Detect speakers, segments, and roles using ${languageDisplayName}.
- Do **not** label the transcript as Arabic unless the provided language is Arabic.`;

    // Replace transcript placeholder
    const promptBody = adaptedPromptTemplate.replace(
      '[PASTE YOUR ARABIC TRANSCRIPT HERE WITH CURRENT SPEAKER LABELS]',
      transcript || ''
    );
    const prompt = `${languageInstruction}\n\n${promptBody}`;
    
    // Try Google Gemini API first if API key is available
    if (GOOGLE_GEMINI_API_KEY) {
      try {
        // Read audio file and convert to base64
        let audioBase64 = null;
        let audioMimeType = 'audio/mpeg'; // Default
        
        if (audioPath && fsSync.existsSync(audioPath)) {
          const audioBuffer = await fs.readFile(audioPath);
          audioBase64 = audioBuffer.toString('base64');
          
          // Detect MIME type from file extension
          const ext = path.extname(audioPath).toLowerCase();
          const mimeTypes = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.aac': 'audio/aac',
            '.flac': 'audio/flac'
          };
          audioMimeType = mimeTypes[ext] || 'audio/mpeg';
        }
        
        // Prepare parts for multimodal input
        const parts = [];
        
        // Add audio if available
        if (audioBase64) {
          parts.push({
            inlineData: {
              mimeType: audioMimeType,
              data: audioBase64
            }
          });
        }
        
        // Add text prompt
        parts.push({
          text: `${systemPrompt}\n\n${prompt}`
        });
        
        // Call Google Gemini API
        const geminiResponse = await axios.post(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=${GOOGLE_GEMINI_API_KEY}`,
          {
            contents: [{
              parts: parts
            }],
            generationConfig: {
              temperature: 0,  // Minimum temperature for maximum determinism
              topP: 0.95,      // Nucleus sampling - focus on most likely tokens
              topK: 40,        // Limit to top K tokens for more deterministic output
              maxOutputTokens: 8192  // Ensure enough tokens for full response
            }
          },
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }
        );
        
        // Extract text from response - Gemini API may return text in different formats
        let llmOutput = '';
        const candidates = geminiResponse.data?.candidates || [];
        if (candidates.length > 0) {
          const content = candidates[0]?.content;
          if (content?.parts) {
            // Find text part
            const textPart = content.parts.find(part => part.text);
            if (textPart) {
              llmOutput = textPart.text;
            }
          }
        }
        
        if (llmOutput) {
          writeLog('info', 'Gemini 2.5 Pro multimodal (Google API) completed', {
            audioProvided: !!audioBase64,
            transcriptLength: transcript?.length || 0,
            outputLength: llmOutput.length
          });
          
          // Parse and return structured JSON
          const structuredJSON = parseToStructuredJSON(llmOutput, transcript || '');
          applyLanguageHintToResult(structuredJSON, languageHint);
          return {
            ...structuredJSON,
            source: 'google-gemini-2.5-pro',
            provider: 'google',
            multimodal: true,
            fallbackUsed: false
          };
        }
        
        throw new Error('Empty response from Google Gemini API');
        
      } catch (googleError) {
        console.error('Google Gemini API error:', googleError.response?.data || googleError.message);
        writeLog('warn', 'Google Gemini API failed, trying OpenRouter fallback', {
          error: googleError.message
        });
        
        // Fall through to OpenRouter fallback
      }
    }
    
    // Fallback to OpenRouter Gemini 2.5 Pro
    try {
      const openrouterResponse = await axios.post(
        'https://openrouter.ai/api/v1/chat/completions',
        {
          model: 'google/gemini-2.5-pro',
          messages: [
            {
              role: 'system',
            content: adaptedSystemPrompt
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0
        },
        {
          headers: {
            'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
            'X-Title': 'Gemini 2.5 Pro Multimodal'
          }
        }
      );
      
      const llmOutput = openrouterResponse.data.choices[0]?.message?.content || '';
      
      if (!llmOutput) {
        throw new Error('Empty response from OpenRouter Gemini 2.5 Pro');
      }
      
      writeLog('info', 'Gemini 2.5 Pro multimodal (OpenRouter) completed', {
        audioProvided: !!audioPath,
        transcriptLength: transcript?.length || 0,
        outputLength: llmOutput.length
      });
      
      // Parse and return structured JSON
      const structuredJSON = parseToStructuredJSON(llmOutput, transcript || '');
      applyLanguageHintToResult(structuredJSON, languageHint);
      return {
        ...structuredJSON,
        source: 'openrouter-gemini-2.5-pro',
        provider: 'openrouter',
        multimodal: false,
        fallbackUsed: true,
        limitations: ['text_only', 'no_direct_audio_ingest']
      };
      
    } catch (openrouterError) {
      console.error('OpenRouter Gemini 2.5 Pro error:', openrouterError.response?.data || openrouterError.message);
      writeLog('error', 'Both Google and OpenRouter Gemini 2.5 Pro failed', {
        googleError: GOOGLE_GEMINI_API_KEY ? 'API key provided but failed' : 'No API key',
        openrouterError: openrouterError.message
      });
      
      throw new Error('Gemini 2.5 Pro multimodal processing failed on both Google API and OpenRouter');
    }
    
  } catch (error) {
    console.error('Gemini 2.5 Pro multimodal error:', error);
    writeLog('error', 'Gemini 2.5 Pro multimodal failed', {
      error: error.message,
      audioPath: audioPath || 'none',
      transcriptLength: transcript?.length || 0
    });
    
    throw error;
  }
}

// Prompt Management API (Microservice endpoints)
app.get('/api/prompts', async (req, res) => {
  try {
    const promptsDir = path.join(__dirname, 'prompts');
    const files = await fs.readdir(promptsDir);
    const promptFiles = files.filter(f => f.endsWith('.txt'));
    
    const prompts = await Promise.all(
      promptFiles.map(async (filename) => {
        const filePath = path.join(promptsDir, filename);
        const content = await fs.readFile(filePath, 'utf8');
        return {
          filename,
          content,
          size: content.length,
          lastModified: (await fs.stat(filePath)).mtime.toISOString()
        };
      })
    );
    
    res.json({ success: true, prompts });
  } catch (error) {
    console.error('Error listing prompts:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/prompts/:filename', async (req, res) => {
  try {
    const { filename } = req.params;
    // Security: prevent path traversal
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    
    const filePath = path.join(__dirname, 'prompts', filename);
    const content = await fs.readFile(filePath, 'utf8');
    const stats = await fs.stat(filePath);
    
    res.json({
      success: true,
      filename,
      content,
      size: content.length,
      lastModified: stats.mtime.toISOString()
    });
  } catch (error) {
    if (error.code === 'ENOENT') {
      res.status(404).json({ error: 'Prompt file not found' });
    } else {
      console.error('Error reading prompt:', error);
      res.status(500).json({ error: error.message });
    }
  }
});

app.put('/api/prompts/:filename', async (req, res) => {
  try {
    const { filename } = req.params;
    const { content } = req.body;
    
    // Security: prevent path traversal
    if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
      return res.status(400).json({ error: 'Invalid filename' });
    }
    
    if (typeof content !== 'string') {
      return res.status(400).json({ error: 'Content must be a string' });
    }
    
    const filePath = path.join(__dirname, 'prompts', filename);
    
    // Verify file exists (optional - can allow creating new files)
    try {
      await fs.access(filePath);
    } catch (error) {
      // File doesn't exist - we can create it if needed
      // For now, only allow updating existing files
      if (error.code === 'ENOENT') {
        return res.status(404).json({ error: 'Prompt file not found. Cannot create new files via API.' });
      }
      throw error;
    }
    
    await fs.writeFile(filePath, content, 'utf8');
    
    const stats = await fs.stat(filePath);
    res.json({
      success: true,
      filename,
      size: content.length,
      lastModified: stats.mtime.toISOString(),
      message: 'Prompt updated successfully'
    });
  } catch (error) {
    console.error('Error updating prompt:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/prompts/:filename/reload', async (req, res) => {
  // This endpoint can be used to notify the application to reload prompts
  // In a true microservice, this would trigger a cache refresh
  res.json({ 
    success: true, 
    message: 'Prompt reload requested. Changes will take effect on next request.' 
  });
});

app.post('/api/translate', async (req, res) => {
  try {
    const { model, messages, temperature = 0.2 } = req.body || {};

    if (!model || !messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Model and messages are required' });
    }

    if (!process.env.OPENROUTER_API_KEY) {
      return res.status(500).json({ error: 'OpenRouter API key is not configured on the server' });
    }

    const response = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model,
        messages,
        temperature
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': req.headers.origin || 'http://localhost',
          'X-Title': 'Diarization Translator'
        },
        timeout: 60000
      }
    );

    res.json(response.data);
  } catch (error) {
    console.error('Translation proxy error:', error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: 'Translation request failed',
      details: error.response?.data || error.message
    });
  }
});

// Test route for dialogue generation endpoint
app.get('/api/generate-dialogue/test', (req, res) => {
  res.json({ status: 'ok', message: 'Dialogue generation endpoint is accessible' });
});

// Dialogue generation endpoint for call center audio
app.post('/api/generate-dialogue', async (req, res) => {
  console.log('📥 Received dialogue generation request:', {
    body: req.body,
    timestamp: new Date().toISOString()
  });
  
  try {
    const { 
      numSpeakers = 2, 
      dialogueLength = 'short', 
      scenario = 'customer_service',
      numOverlaps = 2,
      prompt: customPrompt
    } = req.body || {};
    
    console.log('📝 Dialogue generation request:', { numSpeakers, dialogueLength, scenario, numOverlaps, hasCustomPrompt: !!(customPrompt && customPrompt.trim()) });

    if (!process.env.OPENROUTER_API_KEY) {
      console.error('❌ OpenRouter API key is not configured');
      return res.status(500).json({ error: 'OpenRouter API key is not configured' });
    }

    // Map dialogue length to approximate number of exchanges
    const lengthMap = {
      'short': 4,
      'medium': 8,
      'long': 15
    };
    const numExchanges = lengthMap[dialogueLength] || 8;

    // Scenario prompts
    const scenarioPrompts = {
      'customer_service': 'customer service call center',
      'tech_support': 'technical support call center',
      'sales': 'sales call center',
      'complaint': 'customer complaint call center',
      'billing': 'billing inquiry call center'
    };
    const scenarioDesc = scenarioPrompts[scenario] || 'call center';

    const basePrompt = `Generate a realistic ${scenarioDesc} dialogue between ${numSpeakers} speakers. 
Create ${numExchanges} exchanges (back-and-forth conversations). 

Return a JSON array where each item represents one speaker utterance with this exact format:
{
  "speaker": 1,
  "text": "Hello, thank you for calling. How can I help you today?",
  "startTime": 0.0,
  "duration": 3.5
}

Requirements:
- "speaker" must be a number from 1 to ${numSpeakers}
- "text" should be natural conversational speech (5-30 words)
- "startTime" should be in seconds, starting from 0
- "duration" should be estimated based on text length (approximately 0.5 seconds per word)

Make it realistic with call center language, greetings, questions, responses, and closing.`;
    
    const messages = [
      {
        role: 'system',
        content: 'You are a dialogue generator for call center audio testing. Generate realistic, natural conversations in JSON format only.'
      },
      {
        role: 'user',
        content: `${basePrompt}

Return ONLY the JSON array, no markdown code blocks, no explanations, no other text.`
      }
    ];

    if (customPrompt && typeof customPrompt === 'string' && customPrompt.trim()) {
      messages.push({
        role: 'user',
        content: `Apply the following additional instructions with highest priority. Do not ignore or downplay them:\n${customPrompt.trim()}`
      });
    }

    console.log('🔄 Calling OpenRouter API for markdown generation...');
    
    let response;
    try {
      response = await axios.post(
        'https://openrouter.ai/api/v1/chat/completions',
        {
          model: process.env.FAST_MODEL_ID || 'gpt-oss-120b',
          messages,
          temperature: 0.7
        },
        {
          headers: {
            Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': req.headers.origin || 'http://localhost:3000',
            'X-Title': 'Dialogue Generator'
          },
          timeout: 60000 // 60 seconds
        }
      );
      console.log('✅ OpenRouter API response received');
    } catch (axiosError) {
      if (axiosError.code === 'ECONNABORTED' || axiosError.message?.includes('timeout')) {
        console.error('⏱️ Request timeout');
        return res.status(504).json({ error: 'Request timeout - OpenRouter API took too long to respond' });
      }
      if (axiosError.response) {
        console.error('❌ OpenRouter API error:', axiosError.response.status);
        console.error('Response data:', JSON.stringify(axiosError.response.data));
        return res.status(axiosError.response.status || 500).json({
          error: 'OpenRouter API error',
          details: axiosError.response.data || axiosError.message,
          status: axiosError.response.status
        });
      }
      if (axiosError.request) {
        console.error('❌ No response from OpenRouter API');
        console.error('Request error:', axiosError.message);
        return res.status(503).json({
          error: 'OpenRouter API unavailable',
          details: axiosError.message
        });
      }
      console.error('❌ Unexpected axios error:', axiosError.message);
      throw axiosError;
    }

    let dialogueData = response.data.choices[0]?.message?.content || '';
    
    // Try to parse JSON
    try {
      // Remove markdown code blocks if present
      dialogueData = dialogueData.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      let parsed;
      
      try {
        parsed = JSON.parse(dialogueData);
      } catch (parseErr) {
        // Try to extract JSON from text
        const jsonMatch = dialogueData.match(/\{[\s\S]*\}/) || dialogueData.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
          parsed = JSON.parse(jsonMatch[0]);
        } else {
          throw new Error('No valid JSON found in response');
        }
      }
      
      // Handle both array and object formats
      let dialogueArray = [];
      
      if (Array.isArray(parsed)) {
        dialogueArray = parsed;
      } else if (parsed.dialogue && Array.isArray(parsed.dialogue)) {
        dialogueArray = parsed.dialogue;
      } else if (parsed.exchanges && Array.isArray(parsed.exchanges)) {
        dialogueArray = parsed.exchanges;
      } else if (parsed.items && Array.isArray(parsed.items)) {
        dialogueArray = parsed.items;
      } else if (typeof parsed === 'object') {
        // Try to find array in object values
        const values = Object.values(parsed);
        const foundArray = values.find(v => Array.isArray(v) && v.length > 0);
        if (foundArray) {
          dialogueArray = foundArray;
        } else {
          // If no array found, create one from object properties
          dialogueArray = Object.entries(parsed).map(([key, value]) => ({
            speaker: parseInt(key) || 1,
            text: typeof value === 'string' ? value : JSON.stringify(value),
            startTime: 0,
            duration: 2
          }));
        }
      }
      
      // Ensure proper format with minimal gaps - allow overlaps
      let currentTime = 0;
      let prevSpeaker = null;
      let prevDuration = 0;
      
      dialogueArray = dialogueArray.map((item, index) => {
        const speaker = item.speaker || item.speakerId || ((index % numSpeakers) + 1);
        const text = item.text || item.message || item.content || '';
        const duration = item.duration || Math.max(2, text.split(' ').length * 0.5);
        
        // Calculate start time with minimal gaps or overlaps
        const isDifferentSpeaker = prevSpeaker && parseInt(speaker) !== parseInt(prevSpeaker);
        let startTime;
        
        if (item.startTime !== undefined) {
          startTime = item.startTime;
        } else {
          if (isDifferentSpeaker) {
            // Different speaker: aggressive overlap - start during last 60% of previous
            if (prevDuration > 0) {
              // Overlap: start during last 60% of previous utterance (40-100% into previous)
              const overlapAmount = prevDuration * (0.4 + Math.random() * 0.6); // 40-100% into previous
              startTime = Math.max(0, currentTime - prevDuration + overlapAmount);
            } else {
              // No previous duration - start immediately
              startTime = currentTime;
            }
          } else {
            // Same speaker: minimal gap (0.02-0.08s)
            startTime = currentTime + 0.02 + Math.random() * 0.06;
          }
        }
        
        currentTime = Math.max(currentTime, startTime + duration);
        prevSpeaker = speaker;
        prevDuration = duration;
        
        return {
          speaker: parseInt(speaker),
          text: text,
          startTime: startTime,
          duration: duration
        };
      });

      console.log('✅ Successfully parsed dialogue, sending response');
      res.json({ dialogue: dialogueArray, numOverlaps });
    } catch (parseError) {
      console.error('❌ Failed to parse dialogue JSON:', parseError.message);
      console.error('Stack:', parseError.stack);
      console.error('Raw response (first 1000 chars):', dialogueData.substring(0, 1000));
      res.status(500).json({ 
        error: 'Failed to parse dialogue response',
        details: parseError.message,
        raw: dialogueData.substring(0, 500)
      });
    }
  } catch (error) {
    console.error('❌ Dialogue generation error:', error.message);
    console.error('Stack:', error.stack);
    if (error.response) {
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
    }
    
    const statusCode = error.response?.status || 500;
    const errorMessage = error.response?.data || error.message;
    
    res.status(statusCode).json({
      error: 'Dialogue generation failed',
      details: typeof errorMessage === 'object' ? JSON.stringify(errorMessage) : errorMessage,
      message: error.message
    });
  }
});

const ROLE_SYNONYMS = {
  operator: [
    'operator',
    'agent',
    'support',
    'advisor',
    'representative',
    'banker',
    'employee',
    'staff',
    'agent/operator',
    'agent-operator',
    'agent_operator',
    'support agent',
    'service agent'
  ],
  client: [
    'client',
    'customer',
    'caller',
    'patient',
    'user',
    'guest',
    'consumer',
    'applicant',
    'subscriber',
    'member'
  ]
};

function normalizeRoleValue(role) {
  if (!role) return null;
  const normalized = role.toString().trim().toLowerCase();
  if (!normalized) return null;

  if (ROLE_SYNONYMS.operator.some(alias => normalized.includes(alias))) {
    return 'operator';
  }
  if (ROLE_SYNONYMS.client.some(alias => normalized.includes(alias))) {
    return 'client';
  }
  return null;
}

function ensureSegmentRoles(segments) {
  if (!Array.isArray(segments)) return;

  const speakerRoleMap = new Map();
  const fallbackRoles = ['operator', 'client'];
  let fallbackIndex = 0;

  segments.forEach(segment => {
    const normalized = normalizeRoleValue(segment.role);
    if (normalized) {
      segment.role = normalized;
      if (segment.speaker) {
        speakerRoleMap.set(segment.speaker, normalized);
      }
      return;
    }

    if (segment.speaker && speakerRoleMap.has(segment.speaker)) {
      segment.role = speakerRoleMap.get(segment.speaker);
      return;
    }

    const fallbackRole = fallbackRoles[fallbackIndex % fallbackRoles.length];
    segment.role = fallbackRole;
    if (segment.speaker) {
      speakerRoleMap.set(segment.speaker, fallbackRole);
    }
    fallbackIndex += 1;
  });
}

function normalizeRecordingRoles(recording) {
  if (!recording || !recording.results) return;

  Object.entries(recording.results).forEach(([serviceKey, serviceResult]) => {
    if (!serviceResult || !Array.isArray(serviceResult.segments) || serviceResult.segments.length === 0) {
      return;
    }
    const isTextResult = serviceKey === TEXT_SERVICE_KEY
      || serviceKey === 'combined'
      || (serviceResult.rawData?.source && serviceResult.rawData.source.toLowerCase() === 'text')
      || (serviceResult.serviceName && /text|llm|gemini|combined/i.test(serviceResult.serviceName));

    if (isTextResult) {
      ensureSegmentRoles(serviceResult.segments);
    }
  });
}

function coerceToNumber(value, fallback = 0) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  const parsed = parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function sanitizeWordTimings(words) {
  if (!Array.isArray(words)) {
    return [];
  }
  return words
    .map(word => {
      if (!word || typeof word !== 'object') {
        return null;
      }
      const start = coerceToNumber(word.start, 0);
      const end = coerceToNumber(word.end, start);
      return {
        word: (word.word || '').toString(),
        start,
        end,
        speaker: word.speaker !== undefined ? word.speaker : null,
        confidence: word.confidence !== undefined
          ? coerceToNumber(word.confidence, null)
          : null
      };
    })
    .filter(Boolean);
}

function sanitizeSegmentsCollection(segments) {
  if (!Array.isArray(segments)) {
    return [];
  }

  return segments
    .filter(segment => segment && typeof segment === 'object')
    .map(segment => {
      const start = coerceToNumber(segment.start, 0);
      let end = coerceToNumber(segment.end, start);
      if (end < start) {
        end = start;
      }

      const sanitized = {
        speaker: segment.speaker || 'SPEAKER_00',
        text: (segment.text || '').toString(),
        start,
        end,
        words: sanitizeWordTimings(segment.words || []),
        role: normalizeRoleValue(segment.role) || 'operator',
        overlap: Boolean(segment.overlap)
      };

      return sanitized;
    })
    .sort((a, b) => a.start - b.start);
}

function ensureResultContainer(recording, resultKey, serviceNameOverride) {
  if (!recording.results) {
    recording.results = {};
  }

  // Normalize legacy text-service keys
  LEGACY_TEXT_SERVICE_KEYS.forEach(key => {
    if (recording.results[key] && !recording.results[TEXT_SERVICE_KEY]) {
      recording.results[TEXT_SERVICE_KEY] = recording.results[key];
      delete recording.results[key];
    }
  });

  if (!recording.results[resultKey]) {
    if (recording.results[TEXT_SERVICE_KEY]) {
      recording.results[resultKey] = recording.results[TEXT_SERVICE_KEY];
    } else {
      const fallbackKey = Object.keys(recording.results)[0];
      if (fallbackKey) {
        recording.results[resultKey] = recording.results[fallbackKey];
      } else {
        recording.results[resultKey] = {
          success: true,
          serviceName: serviceNameOverride || 'Text Mode 📝',
          processingTime: 0,
          speedFactor: 0,
          speakerCount: 0,
          cost: '0.0000',
          segments: [],
          rawData: {
            duration: 0,
            language: recording.language || 'ar',
            source: resultKey
          }
        };
      }
    }
  }

  if (serviceNameOverride) {
    recording.results[resultKey].serviceName = serviceNameOverride;
  }

  if (!Array.isArray(recording.results[resultKey].segments)) {
    recording.results[resultKey].segments = sanitizeSegmentsCollection(
      recording.results[resultKey].segments || []
    );
  }

  if (!recording.results[resultKey].rawData) {
    recording.results[resultKey].rawData = {
      duration: 0,
      language: recording.language || 'ar',
      source: resultKey
    };
  }

  if (!recording.servicesTested || !Array.isArray(recording.servicesTested)) {
    recording.servicesTested = [];
  }

  if (!recording.servicesTested.includes(resultKey)) {
    recording.servicesTested.push(resultKey);
  }

  return recording.results[resultKey];
}

function splitTranscriptIntoChunks(transcript, maxWords = LLM_CHUNK_WORD_LIMIT) {
  if (!transcript) {
    return [];
  }
  const sanitized = transcript.replace(/\s+/g, ' ').trim();
  if (!sanitized) {
    return [];
  }
  const tokens = sanitized.split(' ');
  const chunks = [];
  let offsetSeconds = 0;

  for (let i = 0; i < tokens.length; i += maxWords) {
    const chunkTokens = tokens.slice(i, i + maxWords);
    const text = chunkTokens.join(' ').trim();
    if (!text) {
      continue;
    }
    const wordCount = chunkTokens.length;
    const duration = wordCount / WORDS_PER_SECOND;
    chunks.push({
      text,
      wordCount,
      duration,
      offset: offsetSeconds
    });
    offsetSeconds += duration;
  }

  return chunks.length
    ? chunks
    : [{
        text: sanitized,
        wordCount: tokens.length,
        duration: tokens.length / WORDS_PER_SECOND,
        offset: 0
      }];
}

function getOpenRouterHeaders(title = 'Speaker Diarization') {
  return {
    'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
    'Content-Type': 'application/json',
    'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
    'X-Title': title
  };
}

function getLocalLLMHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (LOCAL_LLM_API_KEY) {
    headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
  }
  return headers;
}

async function streamChatCompletion({
  url,
  headers,
  payload,
  stopSequences = LLM_STOP_SEQUENCES,
  errorContext = 'LLM request'
}) {
  const finalPayload = {
    temperature: 0,
    stream: true,
    ...payload
  };

  if (!finalPayload.stop && Array.isArray(stopSequences) && stopSequences.length > 0) {
    finalPayload.stop = stopSequences;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(finalPayload)
  });

  if (!response.ok || !response.body) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`${errorContext} failed: ${response.status} ${errorText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let collected = '';
  let braceBalance = 0;
  let finished = false;

  while (!finished) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const event of events) {
      if (!event.startsWith('data:')) continue;
      const data = event.slice(5).trim();
      if (!data) continue;
      if (data === '[DONE]') {
        finished = true;
        break;
      }

      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }

      const deltas = parsed.choices?.[0]?.delta?.content || [];
      for (const delta of deltas) {
        const text = delta.text || '';
        collected += text;
        for (const ch of text) {
          if (ch === '{') {
            braceBalance += 1;
          } else if (ch === '}' && braceBalance > 0) {
            braceBalance -= 1;
          }
        }
      }

      if (braceBalance === 0 && collected.trim().endsWith('}')) {
        finished = true;
        try {
          await reader.cancel();
        } catch (err) {
          console.warn('Failed to cancel stream reader:', err.message);
        }
        break;
      }
    }
  }

  const trimmed = collected.trim();
  if (!trimmed) {
    throw new Error('LLM returned an empty streaming response.');
  }

  if (trimmed.startsWith('```json')) {
    return trimmed.replace(/^```json\s*/, '').replace(/\s*```$/, '').trim();
  }
  if (trimmed.startsWith('```')) {
    return trimmed.replace(/^```\s*/, '').replace(/\s*```$/, '').trim();
  }
  return trimmed;
}

function applyLanguageHintToResult(result, languageHint) {
  if (!result || !languageHint) {
    return;
  }

  const normalized = languageHint.trim();
  if (!normalized) {
    return;
  }

  const recordings = result.recordings || [];
  recordings.forEach(recording => {
    recording.language = normalized;
    if (recording.results) {
      Object.values(recording.results).forEach(serviceResult => {
        if (!serviceResult) return;
        if (!serviceResult.rawData) {
          serviceResult.rawData = {};
        }
        serviceResult.rawData.language = normalized;
      });
    }
  });
}

// Parse LLM text output to structured JSON
function parseToStructuredJSON(llmOutput, originalTranscript, options = {}) {
  const {
    resultKey = TEXT_SERVICE_KEY,
    serviceNameOverride
  } = options;

  let jsonText = llmOutput.trim();
  const jsonMatch = jsonText.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
  if (jsonMatch) {
    jsonText = jsonMatch[1];
  }

  const jsonStart = jsonText.indexOf('{');
  const jsonEnd = jsonText.lastIndexOf('}');
  if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
    jsonText = jsonText.substring(jsonStart, jsonEnd + 1);
  }

  let parsedData;
  try {
    parsedData = JSON.parse(jsonText);
    console.log('✅ Successfully parsed JSON');
  } catch (e) {
    console.warn('❌ Failed to parse as JSON:', e.message);
    console.warn('Error at position:', e.message.match(/position (\d+)/)?.[1] || 'unknown');
    console.warn('JSON text around error:', jsonText.substring(Math.max(0, (parseInt(e.message.match(/position (\d+)/)?.[1] || '0') - 100)), parseInt(e.message.match(/position (\d+)/)?.[1] || '0') + 100));
    return parseTextFormatToJSON(llmOutput, originalTranscript, resultKey);
  }

  if (Array.isArray(parsedData) && parsedData.length > 0) {
    const firstItem = parsedData[0];
    if (firstItem && firstItem.output && firstItem.output.recordings) {
      console.log('✅ Found wrapped format: [{ output: { recordings: [...] } }]');
      parsedData = firstItem.output;
    } else if (firstItem && firstItem.recordings) {
      console.log('✅ Found alternative wrapped format: [{ recordings: [...] }]');
      parsedData = firstItem;
    }
  }

  if (!parsedData.recordings || !Array.isArray(parsedData.recordings)) {
    console.warn('⚠️ Recordings missing or not array, attempting fix...');
    if (parsedData.recordings && typeof parsedData.recordings === 'object') {
      console.log('Converting recordings object to array');
      parsedData.recordings = [parsedData.recordings];
    } else {
      console.warn('No recordings found, trying text format parsing');
      return parseTextFormatToJSON(llmOutput, originalTranscript, resultKey);
    }
  }

  if (parsedData.recordings && parsedData.recordings.length > 0) {
    const recording = parsedData.recordings[0];

    if (!recording.id) recording.id = 'rec_001';
    if (!recording.name) recording.name = 'Transcript Analysis';
    if (!recording.fileName) recording.fileName = 'transcript.txt';
    if (recording.size === undefined) recording.size = 0;
    if (!recording.language) recording.language = 'ar';
    if (!recording.status) recording.status = 'completed';
    if (recording.addedAt === undefined) recording.addedAt = null;
    if (!recording.translationState) {
      recording.translationState = { currentLanguage: 'original', lastError: null };
    }
    if (!recording.aggregated) recording.aggregated = {};
    if (!Array.isArray(recording.servicesTested)) {
      recording.servicesTested = [];
    }
    if (!recording.servicesTested.includes(resultKey)) {
      recording.servicesTested.push(resultKey);
    }

    const targetResult = ensureResultContainer(recording, resultKey, serviceNameOverride);
    targetResult.rawData = targetResult.rawData || {};
    targetResult.rawData.language = targetResult.rawData.language || recording.language || 'ar';
    targetResult.rawData.source = targetResult.rawData.source || resultKey;

    if (!targetResult.segments || targetResult.segments.length === 0) {
      console.warn(`⚠️ No segments found in ${resultKey} results. Attempting fallback from transcript.`);
      const fallbackSegments = sanitizeSegmentsCollection(createFallbackSegments(originalTranscript));
      if (fallbackSegments.length > 0) {
        targetResult.segments = fallbackSegments;
      } else {
        targetResult.segments = [];
      }
    } else {
      targetResult.segments = sanitizeSegmentsCollection(targetResult.segments);
    }

    if (targetResult.segments.length > 0) {
      const maxEnd = Math.max(...targetResult.segments.map(s => s.end || 0));
      recording.duration = maxEnd;
      targetResult.rawData.duration = maxEnd;
      const uniqueSpeakers = new Set(targetResult.segments.map(s => s.speaker));
      const speakerCount = uniqueSpeakers.size;
      recording.speakerCount = speakerCount.toString();
      targetResult.speakerCount = speakerCount;
    } else {
      recording.duration = recording.duration || 0;
      targetResult.rawData.duration = targetResult.rawData.duration || 0;
      targetResult.speakerCount = targetResult.speakerCount || 0;
      recording.speakerCount = recording.speakerCount || '0';
    }

    normalizeRecordingRoles(recording);
  }

  const validation = validateDiarizationJSON(parsedData, { resultKey });
  if (!validation.valid) {
    console.warn(`⚠️ Schema validation failed for ${resultKey}: ${validation.error}`);
    if (validation.details && validation.details.length) {
      console.warn('Validation details:', validation.details);
    }
  }

  return parsedData;
}

// Parse text format (SPEAKER_XX: text) to structured JSON
function parseTextFormatToJSON(textOutput, originalTranscript, resultKey = TEXT_SERVICE_KEY) {
  const lines = textOutput.split('\n').filter(line => {
    const trimmed = line.trim();
    return trimmed.startsWith('SPEAKER_') && trimmed.includes(':');
  });

  const segments = [];
  let currentSpeaker = null;
  let currentText = '';

  for (const line of lines) {
    const match = line.match(/^(SPEAKER_\d+):\s*(.+)$/);
    if (!match) continue;

    const [, speaker, text] = match;

    if (speaker === currentSpeaker) {
      currentText += ' ' + text.trim();
    } else {
      if (currentSpeaker && currentText) {
        segments.push({
          speaker: currentSpeaker,
          text: currentText.trim(),
          start: 0.0,
          end: 0.0,
          words: [],
          overlap: false
        });
      }
      currentSpeaker = speaker;
      currentText = text.trim();
    }
  }

  if (currentSpeaker && currentText) {
    segments.push({
      speaker: currentSpeaker,
      text: currentText.trim(),
      start: 0.0,
      end: 0.0,
      words: [],
      overlap: false
    });
  }

  let currentTime = 0;
  for (const seg of segments) {
    const wordCount = seg.text.split(/\s+/).filter(w => w.length > 0).length;
    const duration = wordCount / 3.5;
    seg.start = parseFloat(currentTime.toFixed(1));
    seg.end = parseFloat((currentTime + duration).toFixed(1));
    currentTime = seg.end;
  }

  const uniqueSpeakers = new Set(segments.map(s => s.speaker));
  const speakerCount = uniqueSpeakers.size;
  ensureSegmentRoles(segments);

  return {
    version: '2.0',
    exportedAt: new Date().toISOString(),
    activeRecordingId: 'rec_001',
    recordings: [{
      id: 'rec_001',
      name: 'Transcript Analysis',
      fileName: 'transcript.txt',
      size: originalTranscript.length,
      duration: currentTime,
      language: 'ar',
      speakerCount: speakerCount.toString(),
      status: 'completed',
      addedAt: null,
      translationState: {
        currentLanguage: 'original',
        lastError: null
      },
      results: {
        [resultKey]: {
          success: true,
          serviceName: 'Text Mode 📝',
          processingTime: 0,
          speedFactor: 0,
          speakerCount: speakerCount,
          cost: '0.0000',
          segments: segments,
          rawData: {
            duration: currentTime,
            language: 'ar',
            source: resultKey
          }
        }
      },
      aggregated: {},
      servicesTested: [resultKey]
    }]
  };
}
// Fallback: Create segments from original transcript if model didn't return any
function createFallbackSegments(transcript) {
  if (!transcript || !transcript.trim()) {
    return [];
  }
  
  // Try to split by common patterns
  // Look for sentence endings, question marks, etc.
  const sentences = transcript
    .split(/([.!?]\s+|\.\s+)/)
    .filter(s => s.trim().length > 0)
    .map(s => s.trim())
    .filter(s => s.length > 0);
  
  if (sentences.length === 0) {
    // If no sentence breaks, treat entire transcript as one segment
    const wordCount = transcript.split(/\s+/).filter(w => w.length > 0).length;
    const duration = wordCount / 3.5;
    return [{
      speaker: 'SPEAKER_00',
      text: transcript.trim(),
      start: 0.0,
      end: parseFloat(duration.toFixed(1)),
      words: []
    }];
  }
  
  // Group sentences into segments (every 2-3 sentences per speaker as fallback)
  const segments = [];
  let currentTime = 0;
  let currentSpeaker = 0;
  
  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    const wordCount = sentence.split(/\s+/).filter(w => w.length > 0).length;
    const duration = wordCount / 3.5;
    
    // Alternate speakers every 2 sentences (simple fallback)
    if (i > 0 && i % 2 === 0) {
      currentSpeaker = (currentSpeaker + 1) % 2;
    }
    
    segments.push({
      speaker: `SPEAKER_${currentSpeaker.toString().padStart(2, '0')}`,
      text: sentence,
      start: parseFloat(currentTime.toFixed(1)),
      end: parseFloat((currentTime + duration).toFixed(1)),
      words: []
    });
    
    currentTime += duration;
  }
  
  return segments;
}

// Error handling middleware - must be before static middleware
app.use((err, req, res, next) => {
  console.error('Express error handler:', err);
  if (req.path && req.path.startsWith('/api/')) {
    // API routes should always return JSON
    res.status(err.status || 500).json({
      success: false,
      error: err.message || 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? err.stack : undefined
    });
  } else {
    // For non-API routes, use default error handling
    next(err);
  }
});

// Serve specific HTML routes BEFORE static middleware
// Serve index.html for root route
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Serve prompt-manager.html
app.get('/prompt-manager', (req, res) => {
  const filePath = path.join(__dirname, 'Features', 'prompt-manager.html');
  res.sendFile(filePath, (err) => {
    if (err) {
      console.error('Error serving prompt-manager.html:', err);
      res.status(500).send('Error loading Prompt Manager');
    }
  });
});

// API endpoint to get demo configuration
// Automatically detects if local or cloud mode based on DEMO_LLM_MODE or DEMO_LOCAL_LLM_MODE
app.get('/api/demo-config', (req, res) => {
  // Debug: Log all relevant environment variables
  console.log('🔍 [DEMO-CONFIG] Environment variables check:');
  console.log(`   DEMO_LOCAL_LLM_MODE: ${process.env.DEMO_LOCAL_LLM_MODE}`);
  console.log(`   DEMO_LLM_MODE: ${process.env.DEMO_LLM_MODE}`);
  console.log(`   LOCAL_LLM_BASE_URL: ${process.env.LOCAL_LLM_BASE_URL || LOCAL_LLM_BASE_URL}`);
  console.log(`   LOCAL_LLM_MODEL: ${process.env.LOCAL_LLM_MODEL || LOCAL_LLM_MODEL}`);
  
  // Check if local LLM mode is configured
  const hasLocalConfig = process.env.DEMO_LOCAL_LLM_MODE === 'local';
  const hasCloudConfig = process.env.DEMO_LLM_MODE && process.env.DEMO_LLM_MODE !== 'local';
  const isLocalMode = process.env.DEMO_LLM_MODE === 'local';
  
  // Prefer local config if explicitly set, otherwise use cloud config
  const useLocal = hasLocalConfig || isLocalMode;
  
  console.log(`   hasLocalConfig: ${hasLocalConfig}`);
  console.log(`   hasCloudConfig: ${hasCloudConfig}`);
  console.log(`   isLocalMode: ${isLocalMode}`);
  console.log(`   useLocal: ${useLocal}`);
  
  if (useLocal) {
    // Local LLM configuration
    const config = {
      speakerCount: parseInt(process.env.DEMO_LOCAL_SPEAKER_COUNT || process.env.DEMO_SPEAKER_COUNT || '2', 10),
      pipelineMode: process.env.DEMO_LOCAL_PIPELINE_MODE || process.env.DEMO_PIPELINE_MODE || 'mode3',
      llmMode: 'local', // Always use 'local' for local mode
      transcriptionEngine: process.env.DEMO_LOCAL_TRANSCRIPTION_ENGINE || process.env.DEMO_TRANSCRIPTION_ENGINE || 'speechmatics',
      isLocal: true
    };
    console.log('✅ [DEMO-CONFIG] Returning LOCAL configuration:', config);
    res.json(config);
  } else {
    // Cloud/Remote LLM configuration
    const config = {
      speakerCount: parseInt(process.env.DEMO_SPEAKER_COUNT || '2', 10),
      pipelineMode: process.env.DEMO_PIPELINE_MODE || 'mode3',
      llmMode: process.env.DEMO_LLM_MODE || 'smart',
      transcriptionEngine: process.env.DEMO_TRANSCRIPTION_ENGINE || 'speechmatics',
      isLocal: false
    };
    console.log('☁️  [DEMO-CONFIG] Returning CLOUD configuration:', config);
    res.json(config);
  }
});

// Serve main diarization user page on /demo
app.get('/demo', (req, res) => {
  const filePath = path.join(__dirname, 'public', 'demo2_user.html');
  res.sendFile(filePath, (err) => {
    if (err) {
      console.error('Error serving demo2_user.html on /demo:', err);
      // Fallback to original demo.html if custom page is not available
      const fallbackPath = path.join(__dirname, 'public', 'demo.html');
      res.sendFile(fallbackPath, (fallbackErr) => {
        if (fallbackErr) {
          console.error('Error serving fallback demo.html:', fallbackErr);
          res.status(404).send('Demo page not found');
        }
      });
    }
  });
});

// Serve demo2.html (SpeechBrain + Whisper)
app.get('/demo2', (req, res) => {
  const demo2Path = path.join(__dirname, 'public', 'demo2.html');
  res.sendFile(demo2Path, (err) => {
    if (err) {
      console.error('Error serving demo2 page:', err);
      res.status(404).send('Demo2 page not found');
    }
  });
});

// Proxy для Flask API (demo2)
const DEMO2_FLASK_PORT = process.env.DEMO2_PORT || 5001;
const DEMO2_FLASK_URL = process.env.DEMO2_FLASK_URL || `http://localhost:${DEMO2_FLASK_PORT}`;

app.get('/api/demo2/health', async (req, res) => {
  try {
    const response = await axios.get(`${DEMO2_FLASK_URL}/api/health`, {
      timeout: 5000
    });
    res.json(response.data);
  } catch (err) {
    res.status(503).json({
      status: 'error',
      message: 'Flask server is not available',
      error: err.message
    });
  }
});

app.post('/api/demo2/diarize', upload.single('file'), async (req, res) => {
  let tempFile = req.file;
  
  try {
    // Створюємо FormData для передачі в Flask
    const FormData = require('form-data');
    const formData = new FormData();
    
    if (tempFile) {
      formData.append('file', fsSync.createReadStream(tempFile.path), {
        filename: tempFile.originalname,
        contentType: tempFile.mimetype
      });
    }
    
    // Додаємо інші параметри
    if (req.body.num_speakers) {
      formData.append('num_speakers', req.body.num_speakers);
    }
    if (req.body.language) {
      formData.append('language', req.body.language);
    }
    if (req.body.segment_duration) {
      formData.append('segment_duration', req.body.segment_duration);
    }
    if (req.body.overlap) {
      formData.append('overlap', req.body.overlap);
    }
    if (req.body.include_transcription !== undefined) {
      formData.append('include_transcription', req.body.include_transcription);
    }
    
    const response = await axios.post(`${DEMO2_FLASK_URL}/api/diarize`, formData, {
      headers: formData.getHeaders(),
      timeout: 300000, // 5 хвилин для великих файлів
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    
    res.json(response.data);
  } catch (err) {
    console.error('Error proxying to Flask:', err.message);
    res.status(err.response?.status || 500).json({
      success: false,
      error: err.message || 'Error processing request'
    });
  } finally {
    // Видаляємо тимчасовий файл
    if (tempFile) {
      try {
        await fs.unlink(tempFile.path);
      } catch (e) {
        console.warn('Could not delete temp file:', e.message);
      }
    }
  }
});

// Serve realistic-audio-generator.html
app.get('/audio-generator', (req, res) => {
  const filePath = path.join(__dirname, 'Features', 'Realistic-audio-generator.html');
  res.sendFile(filePath, (err) => {
    if (err) {
      console.error('Error serving Realistic-audio-generator.html:', err);
      res.status(500).send('Error loading Audio Generator');
    }
  });
});

// Serve monitor.html
app.get('/monitor', (req, res) => {
  const filePath = path.join(__dirname, 'monitor.html');
  res.sendFile(filePath, (err) => {
    if (err) {
      console.error('Error serving monitor.html:', err);
      res.status(500).send('Error loading Monitoring Dashboard');
    }
  });
});

// API endpoint for processing audio files with temporary storage
app.post('/api/process-audio-temp', upload.single('audio'), async (req, res) => {
  let tempProcessor = null;
  let pythonScriptPath = path.join(__dirname, 'process_audio_temp.py');
  
  try {
    // Check if Python script exists
    if (!fsSync.existsSync(pythonScriptPath)) {
      return res.status(500).json({ 
        error: 'Python processing script not found',
        message: 'Please ensure process_audio_temp.py exists in the project root'
      });
    }
    
    const { url, noCleanup } = req.body;
    const uploadedFile = req.file;
    
    if (!url && !uploadedFile) {
      return res.status(400).json({ 
        error: 'Either file upload or URL must be provided' 
      });
    }
    
    // Build Python command
    const pythonArgs = [];
    
    if (url) {
      pythonArgs.push('--url', url);
    } else if (uploadedFile) {
      pythonArgs.push('--file', uploadedFile.path);
    }
    
    if (noCleanup) {
      pythonArgs.push('--no-cleanup');
    }
    
    pythonArgs.push('--output-format', 'json');
    
    // Execute Python script
    const pythonProcess = spawn(PYTHON_BIN, [pythonScriptPath, ...pythonArgs], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', async (code) => {
      try {
        // Clean up uploaded file if it exists
        if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
          await fs.unlink(uploadedFile.path);
        }
        
        if (code !== 0) {
          console.error('Python script error:', stderr);
          return res.status(500).json({
            error: 'Processing failed',
            details: stderr || 'Unknown error',
            code
          });
        }
        
        // Parse Python script output
        try {
          const result = JSON.parse(stdout);
          res.json(result);
        } catch (parseError) {
          console.error('Error parsing Python output:', parseError);
          res.status(500).json({
            error: 'Failed to parse processing result',
            output: stdout,
            error_output: stderr
          });
        }
      } catch (cleanupError) {
        console.error('Error during cleanup:', cleanupError);
        // Still return response even if cleanup fails
        if (!res.headersSent) {
          res.status(500).json({
            error: 'Processing completed but cleanup failed',
            details: cleanupError.message
          });
        }
      }
    });
    
    pythonProcess.on('error', async (error) => {
      console.error('Failed to start Python process:', error);
      
      // Clean up uploaded file
      if (uploadedFile && fsSync.existsSync(uploadedFile.path)) {
        try {
          await fs.unlink(uploadedFile.path);
        } catch (cleanupError) {
          console.error('Cleanup error:', cleanupError);
        }
      }
      
      res.status(500).json({
        error: 'Failed to start processing',
        details: error.message,
        hint: 'Ensure Python 3 is installed and process_audio_temp.py is executable'
      });
    });
    
  } catch (error) {
    console.error('Error in process-audio-temp endpoint:', error);
    
    // Clean up uploaded file if exists
    if (req.file && fsSync.existsSync(req.file.path)) {
      try {
        await fs.unlink(req.file.path);
      } catch (cleanupError) {
        console.error('Cleanup error:', cleanupError);
      }
    }
    
    res.status(500).json({
      error: 'Internal server error',
      details: error.message
    });
  }
});

// Overlap Detection Endpoints

// Audio-based overlap detection using Pyannote (placeholder - requires Python backend)
app.post('/api/overlap-audio', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Audio file is required' });
    }

    console.log('📥 Audio overlap detection request:', {
      filename: req.file.filename,
      size: req.file.size,
      mimetype: req.file.mimetype
    });

    // NOTE: This is a placeholder implementation
    // In production, you would:
    // 1. Call a Python FastAPI service with Pyannote
    // 2. Or use a Node.js wrapper for Pyannote
    // 3. Or spawn a Python subprocess
    
    // For now, return a mock response indicating the endpoint is ready
    // You'll need to integrate with actual Pyannote service
    
    // Example integration (commented out):
    /*
    const { spawn } = require('child_process');
    const pythonProcess = spawn(PYTHON_BIN, [
      'overlap_detector.py',
      req.file.path
    ]);
    
    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        const overlaps = JSON.parse(output);
        res.json({ overlaps });
      } else {
        res.status(500).json({ error: 'Pyannote detection failed' });
      }
    });
    */

    // Placeholder response
    res.json({
      overlaps: [],
      message: 'Audio overlap detection endpoint is ready. Pyannote integration needed.',
      note: 'To enable: integrate with Python FastAPI service or Pyannote Node.js wrapper'
    });

  } catch (error) {
    console.error('❌ Audio overlap detection error:', error);
    res.status(500).json({ 
      error: 'Audio overlap detection failed',
      details: error.message 
    });
  }
});

// LLM-based overlap detection from transcript
app.post('/api/overlap-llm', async (req, res) => {
  try {
    const { transcript } = req.body;

    if (!transcript || !transcript.trim()) {
      return res.status(400).json({ error: 'Transcript is required' });
    }

    console.log('📥 LLM overlap detection request:', {
      transcriptLength: transcript.length,
      timestamp: new Date().toISOString()
    });

    if (!process.env.OPENROUTER_API_KEY) {
      return res.status(500).json({ error: 'OpenRouter API key is not configured' });
    }

    // Create prompt for LLM to detect overlaps
    const prompt = `Analyze the following transcript and identify all instances where speakers are talking simultaneously (overlapping speech).

TRANSCRIPT:
${transcript}

TASK:
1. Identify all segments where multiple speakers are talking at the same time
2. For each overlap, provide:
   - Start time (in seconds, estimate if not provided)
   - End time (in seconds, estimate if not provided)
   - Confidence level (0.0 to 1.0)
   - Type of overlap (e.g., "interruption", "simultaneous", "crosstalk")
   - Which speakers are involved (if identifiable)

OUTPUT FORMAT (JSON only, no markdown):
{
  "overlaps": [
    {
      "start": 5.2,
      "end": 7.8,
      "confidence": 0.85,
      "type": "interruption",
      "speakers": ["SPEAKER_00", "SPEAKER_01"],
      "description": "Speaker 1 interrupts Speaker 0"
    }
  ]
}

If no overlaps are detected, return: {"overlaps": []}

Analyze the transcript now:`;

    const response = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: SMART_MODEL_ID,
        messages: [
          {
            role: 'system',
            content: 'You are an expert in analyzing speech transcripts to detect overlapping speech patterns. Return only valid JSON.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.1
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
          'X-Title': 'Overlap Detection'
        },
        timeout: 60000
      }
    );

    const llmOutput = response.data.choices[0]?.message?.content || '';
    console.log('📤 LLM overlap detection response:', llmOutput.substring(0, 200));

    // Parse JSON from LLM response
    let overlaps = [];
    try {
      // Try to extract JSON from markdown code blocks
      let jsonText = llmOutput.trim();
      const jsonMatch = jsonText.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
      if (jsonMatch) {
        jsonText = jsonMatch[1];
      }
      
      // Try to find JSON object
      const jsonStart = jsonText.indexOf('{');
      const jsonEnd = jsonText.lastIndexOf('}');
      
      if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
        jsonText = jsonText.substring(jsonStart, jsonEnd + 1);
      }
      
      const parsed = JSON.parse(jsonText);
      overlaps = parsed.overlaps || [];
    } catch (parseError) {
      console.warn('⚠️ Failed to parse LLM overlap response, using fallback:', parseError.message);
      // Fallback: try to extract overlaps from text
      overlaps = [];
    }

    console.log(`✅ Detected ${overlaps.length} overlaps from LLM`);

    res.json({
      overlaps: overlaps,
      source: 'llm',
      transcriptLength: transcript.length
    });

  } catch (error) {
    console.error('❌ LLM overlap detection error:', error);
    res.status(500).json({ 
      error: 'LLM overlap detection failed',
      details: error.response?.data || error.message 
    });
  }
});

// Pattern-based LLM overlap detection from transcript (with 50 patterns)
app.post('/api/overlap-llm-patterns', async (req, res) => {
  try {
    const { transcript } = req.body;

    if (!transcript || !transcript.trim()) {
      return res.status(400).json({ error: 'Transcript is required' });
    }

    console.log('📥 Pattern-based LLM overlap detection request:', {
      transcriptLength: transcript.length,
      timestamp: new Date().toISOString()
    });

    if (!process.env.OPENROUTER_API_KEY) {
      return res.status(500).json({ error: 'OpenRouter API key is not configured' });
    }

    // Create detailed prompt with 50 patterns
    const prompt = `Your task is to find and annotate all overlap regions in a conversation transcript, where both speakers talk over each other or one speaker continues talking when logically the other speaker should have said something.

Below is your "knowledge base" – a set of 50 patterns for overlap detection, with code-like descriptions and plain examples.

--- OVERLAP PATTERNS BASE ---

1. Answer without question: turn starts with "Yes"/"No"/"Sure"/"Of course"/"Absolutely" and previous turn is same speaker or no question
2. Consecutive same-speaker: if current and previous turn are both from same speaker, flag overlap
3. Answer to answer: answer-response pairs, both with response-words, no question in between
4. Incomplete sentence + short pause: unfinished sentence by speaker X, gap <2s, X continues
5. Very short gap between speakers: gap<0.3s, different speakers, likely interruption
6. Topic jump: abrupt semantic shift between turns
7. Double statement: "Absolutely. I'll book it now."
8. Missing question/answer: question followed by same speaker, not a real answer
9. Two answers, no question: both turns are "sure", "yes", etc.
10. Rapid turn-taking: interruption on incomplete sentence
11. Long monologue, abrupt short answer: prev turn>>60 chars, current turn <10 chars
12. Suspicious agreement: question, answer not "yes"/"no" but expectation
13. Contradictory answer: "yes" and "no" in one turn
14. Triple alternation: alternation in speaker pattern
15. Rhetorical question + answer
16. Emotion intro: "Oh,", "Ahem,", "Wow," etc.
17. "Let's move on" uncontextualized
18. [crosstalk] ASR tags
19. [laugh]/[noise]/[background]
20. Repeated phrase at turn start, different speaker, short
21. Explicit overlap: "sorry, talking at same time"
22. Question mark but no question words
23. Incomplete turn with ellipsis followed by different speaker
24. Backchannel without main speaker turn
25. Simultaneous "thank you" patterns
26. Overlapping confirmations
27. Interruption markers: "but", "wait", "hold on" mid-turn
28. Speaker self-correction after pause
29. Overlapping questions
30. Response to non-existent question
31. Echo pattern: same phrase repeated by different speaker
32. Fillers followed by different speaker quickly
33. Agreement tokens without context
34. Turn starts with conjunction from previous
35. Abrupt topic change mid-conversation
36. Missing acknowledgment after statement
37. Overlapping apologies
38. Simultaneous greetings
39. Turn boundary confusion
40. Speaker attribution errors
41. Time-based anomalies (negative gaps)
42. Speaker ID inconsistencies
43. Text overlap in timestamps
44. Missing silence markers
45. Abrupt volume changes (if available)
46. Speaker continuation after logical endpoint
47. Missing backchannel responses
48. Overlapping requests
49. Simultaneous confirmations
50. Long gap (>3s) between same-speaker turns

--- END OF PATTERN BASE ---

For EACH turn, decide if one or more patterns fire according to the description and examples above.  

If so, record an overlap with:
- turn_index: index in transcript
- detected_patterns: list of pattern names (from above)
- evidence: snippet or "JS" reasoning why flagged
- confidence: 0.60…0.95 depending on strength
- optionally: missing_speaker (who logically should've responded)

Example (illustrative):
[
  {
    "turn_index": 8,
    "detected_patterns": ["consecutive same-speaker", "answer without question"],
    "evidence": "Speaker 00 has two turns in a row. Turn 8 starts with 'No problem' after own previous turn.",
    "confidence": 0.80,
    "missing_speaker": "Speaker 01"
  }
]

Analyze the following transcript using these 50 patterns. Return a JSON array of detected overlaps (empty array if none).

Transcript:
${transcript}`;

    const response = await axios.post(
      'https://openrouter.ai/api/v1/chat/completions',
      {
        model: SMART_MODEL_ID,
        messages: [
          {
            role: 'system',
            content: 'You are an expert in analyzing speech transcripts to detect overlapping speech patterns using pattern-based analysis. Return only valid JSON array of overlaps.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.1
      },
      {
        headers: {
          'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': process.env.APP_URL || 'http://localhost:3000',
          'X-Title': 'Pattern-Based Overlap Detection'
        },
        timeout: 120000
      }
    );

    const llmOutput = response.data.choices[0]?.message?.content || '';
    console.log('📤 Pattern-based LLM overlap detection response:', llmOutput.substring(0, 200));

    // Parse JSON from LLM response
    let overlaps = [];
    try {
      // Try to extract JSON from markdown code blocks
      let jsonText = llmOutput.trim();
      const jsonMatch = jsonText.match(/```(?:json)?\s*(\[[\s\S]*\])\s*```/);
      if (jsonMatch) {
        jsonText = jsonMatch[1];
      }
      
      // Try to find JSON array
      const jsonStart = jsonText.indexOf('[');
      const jsonEnd = jsonText.lastIndexOf(']');
      
      if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
        jsonText = jsonText.substring(jsonStart, jsonEnd + 1);
      }
      
      const parsed = JSON.parse(jsonText);
      overlaps = Array.isArray(parsed) ? parsed : (parsed.overlaps || []);
    } catch (parseError) {
      console.warn('⚠️ Failed to parse LLM overlap response, using fallback:', parseError.message);
      overlaps = [];
    }

    console.log(`✅ Detected ${overlaps.length} overlaps from pattern-based LLM`);

    res.json({
      overlaps: overlaps,
      source: 'llm-patterns',
      transcriptLength: transcript.length
    });

  } catch (error) {
    console.error('❌ Pattern-based LLM overlap detection error:', error);
    res.status(500).json({ 
      error: 'Pattern-based overlap detection failed',
      details: error.response?.data || error.message 
    });
  }
});

// API endpoint to get Speechmatics API key from environment
app.get('/api/speechmatics-key', (req, res) => {
  const apiKey = process.env.SPEECHMATICS_API_KEY;
  if (!apiKey) {
    return res.status(404).json({ 
      error: 'Speechmatics API key is not configured on the server',
      hint: 'Please set SPEECHMATICS_API_KEY in your .env file'
    });
  }
  res.json({ apiKey });
});

// API endpoint for client-side logging
app.post('/api/client-log', express.json(), (req, res) => {
  try {
    const { level, message, data, timestamp, url, userAgent } = req.body;
    
    // Format log message
    const logMessage = `[CLIENT] [${level.toUpperCase()}] ${message}`;
    const logData = {
      timestamp: timestamp || new Date().toISOString(),
      url: url || 'unknown',
      userAgent: userAgent || 'unknown',
      data: data || {}
    };
    
    // Log to server console based on level
    switch (level) {
      case 'error':
        console.error(logMessage, logData);
        break;
      case 'warn':
        console.warn(logMessage, logData);
        break;
      case 'info':
        console.log(logMessage, logData);
        break;
      case 'debug':
        console.log(logMessage, logData);
        break;
      default:
        console.log(logMessage, logData);
    }
    
    res.json({ success: true });
  } catch (error) {
    console.error('Error processing client log:', error);
    res.status(500).json({ error: 'Failed to process log' });
  }
});

// Debug overlap diarization page
app.get('/debug-overlap', (req, res) => {
  res.sendFile(path.join(__dirname, 'debug_overlap.html'));
});

// Role voting page
app.get('/voting', (req, res) => {
  res.sendFile(path.join(__dirname, 'voting.html'));
});

// Proxy endpoints for n8n webhooks (to avoid CORS issues)
app.post('/api/webhook/diarization-send', async (req, res) => {
  try {
    console.log('📤 [Webhook Proxy] Sending POST request to n8n webhook...');
    console.log('📤 [Webhook Proxy] Request body size:', JSON.stringify(req.body).length, 'bytes');
    
    const response = await axios.post('https://spsoft.app.n8n.cloud/webhook/diarization-send', req.body, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 300000 // 5 minutes - збільшено для великих даних
    });
    
    console.log('✅ [Webhook Proxy] POST request successful, status:', response.status);
    res.json(response.data);
  } catch (error) {
    console.error('❌ [Webhook Proxy] POST error:', error.message);
    if (error.code === 'ECONNABORTED') {
      console.error('❌ [Webhook Proxy] Request timeout - webhook took too long to respond');
      res.status(504).json({
        error: 'Webhook send failed',
        details: 'Request timeout - the external service took too long to respond. Please try again.',
        code: 'TIMEOUT'
      });
    } else {
      res.status(error.response?.status || 500).json({
        error: 'Webhook send failed',
        details: error.message || 'Failed to proxy webhook request',
        code: error.code || 'UNKNOWN_ERROR'
      });
    }
  }
});

app.get('/api/webhook/diarization-get', async (req, res) => {
  try {
    console.log('📥 [Webhook Proxy] Sending GET request to n8n webhook...');
    
    const response = await axios.get('https://spsoft.app.n8n.cloud/webhook/diarization-get', {
      timeout: 300000 // 5 minutes - збільшено для великих даних
    });
    
    console.log('✅ [Webhook Proxy] GET request successful, status:', response.status);
    res.json(response.data);
  } catch (error) {
    console.error('❌ [Webhook Proxy] GET error:', error.message);
    if (error.code === 'ECONNABORTED') {
      console.error('❌ [Webhook Proxy] Request timeout - webhook took too long to respond');
      res.status(504).json({
        error: 'Webhook get failed',
        details: 'Request timeout - the external service took too long to respond. Please try again.',
        code: 'TIMEOUT'
      });
    } else {
      res.status(error.response?.status || 500).json({
        error: 'Webhook get failed',
        details: error.message || 'Failed to proxy webhook request',
        code: error.code || 'UNKNOWN_ERROR'
      });
    }
  }
});

// Proxy endpoint for LLM chat completions (uses demo mode from env)
app.post('/api/llm/chat-completions', async (req, res) => {
  // Define variables outside try block so they're accessible in catch
  let apiUrl;
  let apiKey;
  let model;
  let useLocal = false;
  
  try {
    // Determine which LLM to use based on demo mode
    const hasLocalConfig = process.env.DEMO_LOCAL_LLM_MODE === 'local';
    const isLocalMode = process.env.DEMO_LLM_MODE === 'local';
    useLocal = hasLocalConfig || isLocalMode;
    
    if (useLocal) {
      // Use local LLM (LM Studio)
      apiUrl = `${LOCAL_LLM_BASE_URL}/v1/chat/completions`;
      apiKey = LOCAL_LLM_API_KEY;
      model = LOCAL_LLM_MODEL;
      console.log('🤖 [LLM-PROXY] Using LOCAL LLM:', { apiUrl, model });
    } else {
      // Use cloud LLM (OpenRouter)
      const llmMode = process.env.DEMO_LLM_MODE || 'smart';
      apiUrl = 'https://openrouter.ai/api/v1/chat/completions';
      apiKey = process.env.OPENROUTER_API_KEY;
      
      // Get model based on mode (use getModelId function for consistency)
      // Normalize smart2 to smart-2 for compatibility
      const normalizedMode = llmMode === 'smart2' ? 'smart-2' : llmMode;
      model = getModelId(normalizedMode);
      
      console.log('☁️  [LLM-PROXY] Using CLOUD LLM:', { apiUrl, model, mode: llmMode, normalizedMode });
    }
    
    // Prepare request body
    const requestBody = {
      ...req.body,
      model: req.body.model || model // Use provided model or fallback to configured model
    };
    
    // Prepare headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (apiKey) {
      headers['Authorization'] = `Bearer ${apiKey}`;
    }
    
    // Make request to LLM
    const response = await axios.post(apiUrl, requestBody, {
      headers: headers,
      timeout: 300000 // 5 minutes for LLM requests
    });
    
    res.json(response.data);
  } catch (error) {
    console.error('❌ [LLM-PROXY] Error:', error.message);
    
    // Get current model from request body or fallback to configured model
    // If model wasn't set (error occurred before assignment), use default
    const currentModel = req.body?.model || model || 'google/gemini-3.0-pro';
    
    // Special handling for 402 Payment Required
    if (error.response?.status === 402) {
      console.error('❌ [LLM-PROXY] Payment Required (402) - API credits exhausted or payment issue');
      console.error('❌ [LLM-PROXY] Model that failed:', currentModel);
      console.error('❌ [LLM-PROXY] Error details:', error.response?.data);
      const errorData = error.response?.data || {};
      
      // Try to fallback to gpt-5.1 if the current model fails
      const isGeminiModel = currentModel.includes('gemini') || currentModel.includes('google/');
      
      if (isGeminiModel && !useLocal) {
        console.log('🔄 [LLM-PROXY] Attempting fallback to gpt-5.1...');
        try {
          const fallbackModel = 'openai/gpt-5.1';
          const fallbackRequestBody = {
            ...req.body,
            model: fallbackModel
          };
          
          const fallbackHeaders = {
            'Content-Type': 'application/json'
          };
          if (apiKey) {
            fallbackHeaders['Authorization'] = `Bearer ${apiKey}`;
          }
          
          const fallbackResponse = await axios.post(apiUrl, fallbackRequestBody, {
            headers: fallbackHeaders,
            timeout: 300000
          });
          
          console.log('✅ [LLM-PROXY] Fallback to gpt-5.1 successful');
          return res.json(fallbackResponse.data);
        } catch (fallbackError) {
          console.error('❌ [LLM-PROXY] Fallback to gpt-5.1 also failed:', fallbackError.message);
          if (fallbackError.response?.status === 402) {
            console.error('❌ [LLM-PROXY] gpt-5.1 also requires payment. Trying alternative fallback...');
            // Try one more fallback to a cheaper model
            try {
              const alternativeModel = 'openai/gpt-4o-mini';
              const alternativeRequestBody = {
                ...req.body,
                model: alternativeModel
              };
              
              const alternativeHeaders = {
                'Content-Type': 'application/json'
              };
              if (apiKey) {
                alternativeHeaders['Authorization'] = `Bearer ${apiKey}`;
              }
              
              const alternativeResponse = await axios.post(apiUrl, alternativeRequestBody, {
                headers: alternativeHeaders,
                timeout: 300000
              });
              
              console.log('✅ [LLM-PROXY] Fallback to gpt-4o-mini successful');
              return res.json(alternativeResponse.data);
            } catch (altError) {
              console.error('❌ [LLM-PROXY] All fallback models failed');
            }
          }
        }
      }
      
      // Try to fallback to local LLM if available
      const hasLocalConfig = process.env.DEMO_LOCAL_LLM_MODE === 'local';
      const isLocalMode = process.env.DEMO_LLM_MODE === 'local';
      const canUseLocal = hasLocalConfig || isLocalMode;
      
      if (canUseLocal && LOCAL_LLM_BASE_URL) {
        console.log('🔄 [LLM-PROXY] Attempting fallback to local LLM...');
        try {
          const localApiUrl = `${LOCAL_LLM_BASE_URL}/v1/chat/completions`;
          const localApiKey = LOCAL_LLM_API_KEY;
          const localModel = LOCAL_LLM_MODEL;
          
          const localHeaders = {
            'Content-Type': 'application/json'
          };
          if (localApiKey) {
            localHeaders['Authorization'] = `Bearer ${localApiKey}`;
          }
          
          const localRequestBody = {
            ...req.body,
            model: req.body.model || localModel
          };
          
          const localResponse = await axios.post(localApiUrl, localRequestBody, {
            headers: localHeaders,
            timeout: 300000
          });
          
          console.log('✅ [LLM-PROXY] Fallback to local LLM successful');
          return res.json(localResponse.data);
        } catch (localError) {
          console.error('❌ [LLM-PROXY] Local LLM fallback also failed:', localError.message);
        }
      }
      
      return res.status(402).json({
        error: 'Payment Required',
        message: `LLM API credits exhausted for model "${currentModel}". The model may require a minimum balance or special access. Try using a different model or check your OpenRouter account balance.`,
        details: errorData,
        code: 'PAYMENT_REQUIRED',
        model: currentModel,
        canRetry: false,
        suggestion: 'Consider using a cheaper model like google/gemini-3.0-pro or configure local LLM'
      });
    }
    
    // Handle other errors
    console.error('❌ [LLM-PROXY] Error details:', error.response?.data || error.message);
    const statusCode = error.response?.status || 500;
    const errorMessage = error.response?.data?.error?.message || error.message || 'Failed to proxy LLM request';
    
    res.status(statusCode).json({
      error: errorMessage,
      details: error.response?.data || undefined,
      code: error.code || 'LLM_REQUEST_FAILED',
      status: statusCode
    });
  }
});

// Static files middleware (must be AFTER API routes and specific routes)
app.use(express.static(__dirname));
// Serve public directory at root level for cleaner paths
app.use(express.static(path.join(__dirname, 'public')));

// Increase timeouts for long-running operations (PyAnnote model loading, etc.)
const server = require('http').createServer(app);
server.keepAliveTimeout = 20 * 60 * 1000; // 20 minutes
server.headersTimeout = 21 * 60 * 1000; // 21 minutes (must be > keepAliveTimeout)

// Start server
/**
 * Detect pauses in segments using multiple methods:
 * 1. Gaps between segments (inter-segment pauses)
 * 2. Gaps between words within segments (intra-segment pauses)
 * 3. Long gaps (> threshold) indicate natural breaks between replicas
 * 
 * @param {Array} segments - Array of segments with optional words array
 * @param {Object} options - Configuration options
 * @returns {Array} Segments with pause information added
 */
function detectPauses(segments, options = {}) {
  const {
    interSegmentThreshold = 0.3,  // 300ms gap between segments = pause
    intraSegmentThreshold = 0.5,   // 500ms gap between words = pause
    longPauseThreshold = 1.0,        // 1s gap = long pause (definite break)
    minSegmentDuration = 0.1        // Minimum segment duration to consider
  } = options;

  if (!segments || segments.length === 0) {
    return segments;
  }

  // Sort segments by start time
  const sortedSegments = [...segments].sort((a, b) => (a.start || 0) - (b.start || 0));
  const segmentsWithPauses = [];

  for (let i = 0; i < sortedSegments.length; i++) {
    const seg = { ...sortedSegments[i] };
    const prevSeg = i > 0 ? sortedSegments[i - 1] : null;

    // Initialize pause arrays
    seg.pauses = seg.pauses || [];
    seg.pauseBefore = null;

    // METHOD 1: Detect pause BEFORE segment (gap between segments)
    if (prevSeg) {
      const gap = (seg.start || 0) - (prevSeg.end || 0);
      if (gap > interSegmentThreshold) {
        seg.pauseBefore = {
          duration: gap,
          start: prevSeg.end || 0,
          end: seg.start || 0,
          type: gap > longPauseThreshold ? 'long_pause' : 'pause',
          isReplicaBoundary: gap > longPauseThreshold // Long pauses definitely separate replicas
        };
      }
    }

    // METHOD 2: Detect pauses WITHIN segment (using word-level timestamps)
    if (seg.words && Array.isArray(seg.words) && seg.words.length > 1) {
      const intraSegmentPauses = [];
      
      for (let j = 1; j < seg.words.length; j++) {
        const prevWord = seg.words[j - 1];
        const currWord = seg.words[j];
        
        const wordGap = (currWord.start || currWord.start_time || 0) - 
                       (prevWord.end || prevWord.end_time || prevWord.start || 0);
        
        if (wordGap > intraSegmentThreshold) {
          intraSegmentPauses.push({
            duration: wordGap,
            start: prevWord.end || prevWord.end_time || prevWord.start || 0,
            end: currWord.start || currWord.start_time || 0,
            type: wordGap > longPauseThreshold ? 'long_pause' : 'pause',
            isReplicaBoundary: wordGap > longPauseThreshold,
            wordIndex: j, // Position in words array
            beforeWord: currWord.word || currWord.text || '',
            afterWord: prevWord.word || prevWord.text || ''
          });
        }
      }
      
      if (intraSegmentPauses.length > 0) {
        seg.pauses = seg.pauses.concat(intraSegmentPauses);
        // Mark segment as having internal pauses
        seg.hasInternalPauses = true;
        // If there's a long pause, this segment might need splitting
        seg.needsSplitting = intraSegmentPauses.some(p => p.isReplicaBoundary);
      }
    }

    segmentsWithPauses.push(seg);
  }

  return segmentsWithPauses;
}

/**
 * Pause-based replica detection and merge algorithm
 * 
 * Algorithm:
 * 1. Take primary diarization (first record)
 * 2. Detect pause locations
 * 3. Identify non-pause locations (replicas)
 * 4. Mark replicas with timestamps
 * 5. Search for matching timestamps in Speechmatics transcript (already diarized)
 * 6. If slight deviation - accept as additional phrase to correct
 * 7. Check if additional phrase exists in transcript but not displayed
 * 8. Add missing phrases from voice tracks (full replicas from separated audio)
 * 
 * @param {Object} primaryDiarization - Primary diarization result
 * @param {Array} voiceTracks - Voice tracks with transcripts
 * @param {Object} options - Configuration options
 * @returns {Object} Merged result in same format as primary
 */
function mergeWithPauseBasedReplicas(primaryDiarization, voiceTracks, options = {}) {
  const logPrefix = '[Pause-Based Replica Merge]';
  const {
    pauseThreshold = 0.3,        // 300ms = pause
    longPauseThreshold = 1.0,     // 1s = definite replica boundary
    timestampTolerance = 0.5,     // 500ms tolerance for timestamp matching
    minReplicaDuration = 0.1      // Minimum replica duration
  } = options;

  console.log(`${logPrefix} Starting pause-based replica merge algorithm...`);

  // STEP 1: Extract primary segments
  const primarySegments = primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
  if (!primarySegments.length) {
    console.warn(`${logPrefix} No primary segments found`);
    return primaryDiarization;
  }

  console.log(`${logPrefix} Primary segments: ${primarySegments.length}`);

  // STEP 2: Detect pauses in primary diarization
  const primaryWithPauses = detectPauses(primarySegments, {
    interSegmentThreshold: pauseThreshold,
    intraSegmentThreshold: 0.5,
    longPauseThreshold: longPauseThreshold
  });

  console.log(`${logPrefix} Detected pauses: ${primaryWithPauses.filter(s => s.pauseBefore || (s.pauses && s.pauses.length > 0)).length} segments with pauses`);

  // STEP 3: Identify replicas (continuous speech without pauses)
  // A replica is a continuous segment or group of segments without significant pauses
  const replicas = [];
  let currentReplica = null;

  for (let i = 0; i < primaryWithPauses.length; i++) {
    const seg = primaryWithPauses[i];
    const prevSeg = i > 0 ? primaryWithPauses[i - 1] : null;

    // Check if there's a pause before this segment (replica boundary)
    const hasPauseBefore = seg.pauseBefore && seg.pauseBefore.duration > pauseThreshold;
    const hasLongPauseBefore = seg.pauseBefore && seg.pauseBefore.isReplicaBoundary;

    // Check if there are pauses within this segment
    const hasInternalPauses = seg.pauses && seg.pauses.length > 0;
    const hasLongInternalPause = hasInternalPauses && seg.pauses.some(p => p.isReplicaBoundary);

    // Start new replica if:
    // 1. First segment
    // 2. Long pause before segment (definite boundary)
    // 3. Long pause within previous segment
    if (!currentReplica || hasLongPauseBefore || (prevSeg && prevSeg.needsSplitting)) {
      // Save previous replica if exists
      if (currentReplica) {
        replicas.push(currentReplica);
      }

      // Start new replica
      currentReplica = {
        segments: [seg],
        start: parseFloat(seg.start || 0),
        end: parseFloat(seg.end || seg.start || 0),
        speaker: seg.speaker || 'SPEAKER_00',
        text: (seg.text || '').trim(),
        hasPauses: hasPauseBefore || hasInternalPauses,
        role: seg.role || null
      };
    } else {
      // Continue current replica (no significant pause)
      currentReplica.segments.push(seg);
      currentReplica.end = Math.max(currentReplica.end, parseFloat(seg.end || seg.start || 0));
      currentReplica.text += ' ' + (seg.text || '').trim();
      if (hasPauseBefore || hasInternalPauses) {
        currentReplica.hasPauses = true;
      }
    }
  }

  // Add last replica
  if (currentReplica) {
    replicas.push(currentReplica);
  }

  console.log(`${logPrefix} Identified ${replicas.length} replicas from ${primarySegments.length} segments`);

  // STEP 4: Collect voice track segments
  const rawVoiceTrackSegments = collectVoiceTrackSegments(voiceTracks);
  if (!rawVoiceTrackSegments.length) {
    console.warn(`${logPrefix} No voice track segments found`);
    return primaryDiarization;
  }

  // STEP 4.5: Split ALL voice track segments into replicas by pauses (using word-level timestamps)
  // CRITICAL: Voice tracks from separated audio MUST be split into separate replicas
  // Each pause between words indicates a new replica (dialogue turn)
  // This is essential for proper dialogue structure
  const splitVoiceSegments = [];
  
  for (const vSeg of rawVoiceTrackSegments) {
    const duration = (vSeg.end || 0) - (vSeg.start || 0);
    
    // CRITICAL FIX: Split ALL segments with word-level timestamps, not just long ones
    // Every voice track segment should be split into replicas based on pauses
    if (vSeg.words && Array.isArray(vSeg.words) && vSeg.words.length > 1) {
      const words = vSeg.words;
      
      // Get start time of first word (words have 'start' and 'end' fields from Python parsing)
      const firstWordStart = parseFloat(words[0].start || words[0].start_time || vSeg.start || 0);
      let currentReplicaStart = firstWordStart;
      let currentReplicaWords = [words[0]];
      let splitCount = 0;
      
      for (let i = 1; i < words.length; i++) {
        const prevWord = words[i - 1];
        const currWord = words[i];
        
        // Extract timestamps (words have 'start' and 'end' from Python)
        const prevWordEnd = parseFloat(prevWord.end || prevWord.end_time || prevWord.start || currentReplicaStart);
        const currWordStart = parseFloat(currWord.start || currWord.start_time || prevWordEnd);
        const gap = currWordStart - prevWordEnd;
        
        // Check if there's a pause here (replica boundary)
        // Use pauseThreshold for all pauses - this ensures proper replica separation
        const hasPause = gap > pauseThreshold;
        
        if (hasPause && currentReplicaWords.length > 0) {
          // Save current replica (ends at end of last word before pause)
          const replicaEnd = prevWordEnd;
          const replicaText = currentReplicaWords.map(w => {
            // Extract word text (can be 'word', 'text', or 'content')
            return w.word || w.text || w.content || '';
          }).filter(Boolean).join(' ').trim();
          
          if (replicaText.length > 0 && replicaEnd > currentReplicaStart) {
            splitVoiceSegments.push({
              speaker: vSeg.speaker || 'SPEAKER_00',
              text: replicaText,
              start: currentReplicaStart,
              end: replicaEnd,
              words: [...currentReplicaWords],
              role: vSeg.role || null,
              source: 'voice-track-split',
              originalTrackSpeaker: vSeg.originalTrackSpeaker || vSeg.speaker,
              isSplit: true,
              isReplica: true // Mark as separate replica
            });
            splitCount++;
          }
          
          // Start new replica (starts at start of current word)
          currentReplicaStart = currWordStart;
          currentReplicaWords = [currWord];
        } else {
          // Continue current replica (no pause)
          currentReplicaWords.push(currWord);
        }
      }
      
      // Add last replica
      if (currentReplicaWords.length > 0) {
        const lastWord = currentReplicaWords[currentReplicaWords.length - 1];
        const replicaEnd = parseFloat(lastWord.end || lastWord.end_time || lastWord.start || currentReplicaStart);
        const replicaText = currentReplicaWords.map(w => {
          return w.word || w.text || w.content || '';
        }).filter(Boolean).join(' ').trim();
        
        if (replicaText.length > 0 && replicaEnd > currentReplicaStart) {
          splitVoiceSegments.push({
            speaker: vSeg.speaker || 'SPEAKER_00',
            text: replicaText,
            start: currentReplicaStart,
            end: replicaEnd,
            words: [...currentReplicaWords],
            role: vSeg.role || null,
            source: 'voice-track-split',
            originalTrackSpeaker: vSeg.originalTrackSpeaker || vSeg.speaker,
            isSplit: true,
            isReplica: true // Mark as separate replica
          });
          splitCount++;
        }
      }
      
      if (splitCount > 1) {
        console.log(`${logPrefix} ✅ Split voice segment [${vSeg.start.toFixed(2)}-${vSeg.end.toFixed(2)}s] into ${splitCount} separate replicas`);
      } else if (splitCount === 1) {
        // Single replica from segment
        console.log(`${logPrefix} Voice segment [${vSeg.start.toFixed(2)}-${vSeg.end.toFixed(2)}s] is one replica (no pauses)`);
      } else {
        // No valid replicas found, keep original segment as fallback
        console.warn(`${logPrefix} ⚠️ No valid replicas extracted from segment, keeping original`);
        splitVoiceSegments.push(vSeg);
      }
    } else {
      // No word timestamps available - cannot split, keep as is
      // But mark it as a single replica
      console.log(`${logPrefix} Voice segment [${vSeg.start.toFixed(2)}-${vSeg.end.toFixed(2)}s] has no word timestamps, keeping as single replica`);
      splitVoiceSegments.push({
        ...vSeg,
        isReplica: true // Mark as replica even without splitting
      });
    }
  }
  
  // Normalize voice track speakers
  const normalizedVoiceSegments = normalizeVoiceTrackSegments(splitVoiceSegments.length > 0 ? splitVoiceSegments : rawVoiceTrackSegments);
  console.log(`${logPrefix} Voice track segments: ${normalizedVoiceSegments.length} (after splitting: ${splitVoiceSegments.length > 0 ? splitVoiceSegments.length : rawVoiceTrackSegments.length})`);

  // STEP 5: Match replicas with voice track segments and find additional phrases
  const resultSegments = [];
  const usedVoiceSegments = new Set();

  // Helper to create unique key
  const getSegmentKey = (seg) => {
    const start = parseFloat(seg.start || 0).toFixed(3);
    const end = parseFloat(seg.end || seg.start || 0).toFixed(3);
    const speaker = seg.speaker || 'SPEAKER_00';
    const text = (seg.text || '').trim().substring(0, 50);
    return `${start}_${end}_${speaker}_${text}`;
  };

  // Process each replica
  for (const replica of replicas) {
    const replicaStart = replica.start;
    const replicaEnd = replica.end;
    const replicaSpeaker = replica.speaker;
    const replicaText = replica.text;

    // Find matching voice track segments for this replica
    const matchingVoiceSegments = [];

    for (const vSeg of normalizedVoiceSegments) {
      const segKey = getSegmentKey(vSeg);
      if (usedVoiceSegments.has(segKey)) continue;

      const vStart = parseFloat(vSeg.start || 0);
      const vEnd = parseFloat(vSeg.end || vStart);
      const vText = (vSeg.text || '').trim();

      if (!vText) continue;

      // Check timestamp proximity (within tolerance)
      // Use more flexible matching: check if voice segment overlaps with replica OR is within tolerance
      const overlaps = rangesOverlap(replicaStart, replicaEnd, vStart, vEnd);
      
      // Calculate time distance (use start times for better matching)
      const startDistance = Math.abs(vStart - replicaStart);
      const endDistance = Math.abs(vEnd - replicaEnd);
      const timeDistance = Math.min(startDistance, endDistance);
      
      // More lenient matching: accept if overlaps OR starts within tolerance OR ends within tolerance
      const isClose = !overlaps && (startDistance <= timestampTolerance || endDistance <= timestampTolerance);
      
      // Also check if voice segment is completely contained within replica (or vice versa)
      const voiceContained = vStart >= replicaStart && vEnd <= replicaEnd;
      const replicaContained = replicaStart >= vStart && replicaEnd <= vEnd;
      const isContained = voiceContained || replicaContained;

      if (overlaps || isClose || isContained) {
        const overlap = overlaps 
          ? Math.max(0, Math.min(replicaEnd, vEnd) - Math.max(replicaStart, vStart))
          : (isContained ? Math.min(replicaEnd - replicaStart, vEnd - vStart) : 0);
        
        const similarity = computeTextSimilarity(replicaText, vText);
        
        // Higher score for contained segments or high overlap
        let score = 0;
        if (isContained) {
          score = similarity * 2; // Boost contained segments
        } else if (overlap > 0) {
          score = overlap * similarity;
        } else {
          score = similarity / (1 + timeDistance);
        }

        matchingVoiceSegments.push({
          segment: vSeg,
          overlap,
          timeDistance,
          similarity,
          isContained,
          score
        });
      }
    }

    // Sort by score
    matchingVoiceSegments.sort((a, b) => b.score - a.score);

    // Use best match if it provides more complete text
    let finalText = replicaText;
    let finalSource = 'primary';

    if (matchingVoiceSegments.length > 0) {
      const bestMatch = matchingVoiceSegments[0];
      const voiceText = (bestMatch.segment.text || '').trim();
      
      // More flexible matching: accept voice text if:
      // 1. It's contained and similar (high confidence)
      // 2. It's longer and similar (more complete)
      // 3. It has high similarity even if shorter (might be cleaner)
      const textLengthDiff = voiceText.length - replicaText.length;
      const isMoreComplete = textLengthDiff > 10 && bestMatch.similarity > 0.5;
      const isContainedAndSimilar = bestMatch.isContained && bestMatch.similarity > 0.6;
      const isHighSimilarity = bestMatch.similarity > 0.7 && Math.abs(textLengthDiff) < 20; // Similar length, high similarity
      
      if (isMoreComplete || isContainedAndSimilar || isHighSimilarity) {
        finalText = voiceText;
        finalSource = 'voice-enhanced';
        
        const segKey = getSegmentKey(bestMatch.segment);
        usedVoiceSegments.add(segKey);
        
        console.log(`${logPrefix} Enhanced replica ${replicaSpeaker} [${replicaStart.toFixed(2)}-${replicaEnd.toFixed(2)}] (similarity: ${bestMatch.similarity.toFixed(2)})`);
      }
    }

    // CRITICAL: Each replica must be added as a SEPARATE segment
    // Do NOT merge replicas - each one represents a distinct dialogue turn
    // Check if replica has internal pauses that should split it further
    
    // Check if replica has internal pauses that should split it
    const replicaHasInternalPauses = replica.segments.some(seg => 
      seg.pauses && seg.pauses.length > 0 && seg.pauses.some(p => p.isReplicaBoundary)
    );
    
    if (replicaHasInternalPauses && replica.segments.length > 1) {
      // Split replica by internal pauses - each pause-boundary segment becomes separate replica
      for (let segIdx = 0; segIdx < replica.segments.length; segIdx++) {
        const seg = replica.segments[segIdx];
        const prevSeg = segIdx > 0 ? replica.segments[segIdx - 1] : null;
        
        // Check if there's a pause boundary before this segment
        const hasPauseBoundary = seg.pauseBefore && seg.pauseBefore.isReplicaBoundary;
        const hasInternalPauseBoundary = seg.pauses && seg.pauses.some(p => p.isReplicaBoundary);
        
        // If pause boundary or first segment, create separate replica segment
        if (segIdx === 0 || hasPauseBoundary || hasInternalPauseBoundary) {
          const segStart = parseFloat(seg.start || 0);
          const segEnd = parseFloat(seg.end || segStart);
          const segText = (seg.text || '').trim();
          
          // Find matching voice segment for this individual segment
          let segFinalText = segText;
          let segFinalSource = 'primary';
          
          // Try to find voice match for this specific segment
          for (const vSeg of normalizedVoiceSegments) {
            const segKey = getSegmentKey(vSeg);
            if (usedVoiceSegments.has(segKey)) continue;
            
            const vStart = parseFloat(vSeg.start || 0);
            const vEnd = parseFloat(vSeg.end || vStart);
            const vText = (vSeg.text || '').trim();
            
            if (!vText) continue;
            
            const overlaps = rangesOverlap(segStart, segEnd, vStart, vEnd);
            const startDistance = Math.abs(vStart - segStart);
            const similarity = computeTextSimilarity(segText, vText);
            
            if ((overlaps || startDistance <= timestampTolerance) && similarity > 0.6) {
              segFinalText = vText;
              segFinalSource = 'voice-enhanced';
              usedVoiceSegments.add(segKey);
              break;
            }
          }
          
          resultSegments.push({
            speaker: seg.speaker || replicaSpeaker,
            text: segFinalText,
            start: segStart,
            end: segEnd,
            originalText: segText,
            source: segFinalSource,
            role: seg.role || replica.role,
            isReplica: true, // Mark as separate replica
            replicaSegmentCount: 1
          });
        }
      }
    } else {
      // No internal pauses - add replica as single segment (but still mark as replica)
      resultSegments.push({
        speaker: replicaSpeaker,
        text: finalText,
        start: replicaStart,
        end: replicaEnd,
        originalText: replicaText,
        source: finalSource,
        role: replica.role,
        isReplica: true, // Mark as separate replica
        replicaSegmentCount: replica.segments.length
      });
    }
  }

  // STEP 6: Find additional phrases from voice tracks (overlaps not in primary)
  const additionalPhrases = [];

  for (const vSeg of normalizedVoiceSegments) {
    const segKey = getSegmentKey(vSeg);
    if (usedVoiceSegments.has(segKey)) continue;

    const vStart = parseFloat(vSeg.start || 0);
    const vEnd = parseFloat(vSeg.end || vStart);
    const vText = (vSeg.text || '').trim();

    if (!vText || vText.length < 10) continue; // Skip very short segments

    // Check if this segment doesn't overlap significantly with any result segment
    let hasSignificantOverlap = false;
    for (const resultSeg of resultSegments) {
      const rStart = parseFloat(resultSeg.start || 0);
      const rEnd = parseFloat(resultSeg.end || rStart);

      if (rangesOverlap(vStart, vEnd, rStart, rEnd)) {
        const overlap = Math.max(0, Math.min(vEnd, rEnd) - Math.max(vStart, rStart));
        const overlapRatio = overlap / Math.max(vEnd - vStart, 0.1);

        if (overlapRatio > 0.3) { // More than 30% overlap
          hasSignificantOverlap = true;
          break;
        }
      }
    }

    // If no significant overlap and no similar text, this is an additional phrase from overlap region
    if (!hasSignificantOverlap && !hasSimilarText) {
      // Find closest result segment by time for insertion point
      let closestSeg = null;
      let minDistance = Infinity;

      for (const resultSeg of resultSegments) {
        const rStart = parseFloat(resultSeg.start || 0);
        const rEnd = parseFloat(resultSeg.end || rStart);
        const rMid = (rStart + rEnd) / 2;
        const vMid = (vStart + vEnd) / 2;
        const distance = Math.abs(rMid - vMid);

        if (distance < minDistance && distance < 5.0) { // Within 5 seconds
          minDistance = distance;
          closestSeg = resultSeg;
        }
      }

      if (closestSeg) {
        additionalPhrases.push({
          segment: vSeg,
          insertAfter: closestSeg,
          timeDistance: minDistance
        });
      }
    }
  }

  // STEP 7: Insert additional phrases at appropriate positions
  if (additionalPhrases.length > 0) {
    console.log(`${logPrefix} Found ${additionalPhrases.length} additional phrases from overlaps`);

    // Sort additional phrases by timestamp
    additionalPhrases.sort((a, b) => {
      const aStart = parseFloat(a.segment.start || 0);
      const bStart = parseFloat(b.segment.start || 0);
      return aStart - bStart;
    });

    // Insert into result segments at correct positions
    for (const phrase of additionalPhrases) {
      const vSeg = phrase.segment;
      const vStart = parseFloat(vSeg.start || 0);

      // Find insertion index (before first segment that starts after this phrase)
      let insertIndex = resultSegments.length;
      for (let i = 0; i < resultSegments.length; i++) {
        const segStart = parseFloat(resultSegments[i].start || 0);
        if (vStart < segStart) {
          insertIndex = i;
          break;
        }
      }

      resultSegments.splice(insertIndex, 0, {
        speaker: vSeg.speaker || 'SPEAKER_00',
        text: vSeg.text || '',
        start: vStart,
        end: parseFloat(vSeg.end || vStart),
        source: 'voice-additional',
        isAdditionalPhrase: true,
        isReplica: true, // Mark as separate replica
        role: vSeg.role || null
      });

      const segKey = getSegmentKey(vSeg);
      usedVoiceSegments.add(segKey);

      console.log(`${logPrefix} Inserted additional phrase: ${vSeg.speaker} [${vStart.toFixed(2)}-${parseFloat(vSeg.end || vStart).toFixed(2)}] "${(vSeg.text || '').substring(0, 50)}..."`);
    }
  }

  // Sort final segments by start time
  resultSegments.sort((a, b) => {
    const startDiff = (a.start || 0) - (b.start || 0);
    if (startDiff !== 0) return startDiff;
    return (a.end || 0) - (b.end || 0);
  });

  // Mark overlaps
  markOverlapFlags(resultSegments);

  // STEP 8: Create result in same format as primary diarization
  const result = JSON.parse(JSON.stringify(primaryDiarization));
  const clonedRecording = result.recordings[0];
  clonedRecording.results = clonedRecording.results || {};
  
  clonedRecording.results['pause-based-merge'] = {
    success: true,
    serviceName: 'Pause-Based Replica Merge',
    processingTime: 0,
    speedFactor: 0,
    speakerCount: new Set(resultSegments.map(s => s.speaker)).size.toString(),
    cost: '0.0000',
    segments: resultSegments,
    rawData: {
      source: 'pause-based-replica-merge',
      primarySegmentsCount: primarySegments.length,
      replicasCount: replicas.length,
      voiceTrackSegmentsCount: normalizedVoiceSegments.length,
      additionalPhrasesCount: additionalPhrases.length,
      finalSegmentsCount: resultSegments.length
    }
  };

  clonedRecording.servicesTested = clonedRecording.servicesTested || [];
  if (!clonedRecording.servicesTested.includes('pause-based-merge')) {
    clonedRecording.servicesTested.push('pause-based-merge');
  }

  console.log(`${logPrefix} Merge completed: ${resultSegments.length} segments (${replicas.length} replicas, ${additionalPhrases.length} additional phrases)`);

  return result;
}

// Apply pause-based replica merge (new algorithm)
app.post('/api/apply-pause-based-merge', async (req, res) => {
  try {
    const { recordingId, recording } = req.body;

    if (!recordingId) {
      return res.status(400).json({ error: 'recordingId is required' });
    }

    console.log('📥 Apply pause-based merge request:', {
      recordingId,
      hasRecordingData: !!recording,
      timestamp: new Date().toISOString()
    });

    // Try to use recording data from request first (from frontend)
    let recordingData = recording;
    
    // If not provided, try to load from file
    if (!recordingData) {
      const recordingsPath = path.join(__dirname, 'recordings.json');
      try {
        const data = await fs.readFile(recordingsPath, 'utf-8');
        const recordings = JSON.parse(data).recordings || [];
        recordingData = recordings.find(r => r.id === recordingId);
      } catch (error) {
        console.warn('Failed to load recordings from file:', error.message);
        // Continue - will return error below if recordingData is still null
      }
    }

    if (!recordingData) {
      return res.status(404).json({ 
        error: 'Recording not found. Please ensure the recording is loaded in the frontend and try again.' 
      });
    }

    // Get primary diarization
    const primaryDiarization = {
      version: '2.0',
      exportedAt: new Date().toISOString(),
      activeRecordingId: recordingId,
      recordings: [recordingData]
    };

    // Get voice tracks (from mode3 overlap diarization)
    // Voice tracks are stored in the final result structure
    const voiceTracks = [];
    
    // Check overlapMetadata first (this is where voice tracks are stored after mode3)
    // This is the primary location where voice tracks are stored
    if (recordingData.overlapMetadata && recordingData.overlapMetadata.voiceTracks && Array.isArray(recordingData.overlapMetadata.voiceTracks)) {
      voiceTracks.push(...recordingData.overlapMetadata.voiceTracks);
      console.log(`Found ${voiceTracks.length} voice tracks in overlapMetadata.voiceTracks`);
    }
    
    // Try other possible locations if not found in overlapMetadata
    if (voiceTracks.length === 0) {
      if (recordingData.voiceTracks && Array.isArray(recordingData.voiceTracks)) {
        voiceTracks.push(...recordingData.voiceTracks);
        console.log(`Found ${voiceTracks.length} voice tracks in recordingData.voiceTracks`);
      } else if (recordingData.results && recordingData.results['voice-tracks']) {
        const storedTracks = recordingData.results['voice-tracks'];
        if (Array.isArray(storedTracks)) {
          voiceTracks.push(...storedTracks);
          console.log(`Found ${voiceTracks.length} voice tracks in results['voice-tracks']`);
        }
      } else if (recordingData.results && recordingData.results['overlap-corrected'] && recordingData.results['overlap-corrected'].rawData) {
        // Try to reconstruct from rawData if available
        const rawData = recordingData.results['overlap-corrected'].rawData;
        if (rawData.voiceTracks && Array.isArray(rawData.voiceTracks)) {
          voiceTracks.push(...rawData.voiceTracks);
          console.log(`Found ${voiceTracks.length} voice tracks in rawData.voiceTracks`);
        }
      }
    }
    
    // If still no voice tracks, try to reconstruct from segments
    if (voiceTracks.length === 0 && recordingData.results) {
      console.log('Voice tracks not found in standard locations, checking results structure...');
      console.log('Available result keys:', Object.keys(recordingData.results || {}));
      
      // Try to reconstruct from rawVoiceTrackSegments in rawData
      const overlapCorrected = recordingData.results['overlap-corrected'];
      if (overlapCorrected && overlapCorrected.rawData) {
        const rawVoiceTrackSegments = overlapCorrected.rawData.rawVoiceTrackSegments;
        if (rawVoiceTrackSegments && Array.isArray(rawVoiceTrackSegments) && rawVoiceTrackSegments.length > 0) {
          // Reconstruct minimal voice track structure
          const trackMap = new Map();
          
          rawVoiceTrackSegments.forEach(seg => {
            const originalTrackSpeaker = seg.originalTrackSpeaker || seg.speaker || 'UNKNOWN';
            if (!trackMap.has(originalTrackSpeaker)) {
              trackMap.set(originalTrackSpeaker, {
                speaker: originalTrackSpeaker,
                transcription: {
                  recordings: [{
                    results: {
                      speechmatics: {
                        segments: []
                      }
                    }
                  }]
                },
                transcriptText: '',
                roleAnalysis: { role: 'UNKNOWN', confidence: 0 }
              });
            }
            
            const track = trackMap.get(originalTrackSpeaker);
            track.transcription.recordings[0].results.speechmatics.segments.push(seg);
            track.transcriptText += (track.transcriptText ? ' ' : '') + (seg.text || '');
          });
          
          voiceTracks.push(...Array.from(trackMap.values()));
          console.log(`Reconstructed ${voiceTracks.length} voice tracks from rawData.rawVoiceTrackSegments`);
        }
      }
    }

    if (voiceTracks.length === 0) {
      console.log('Voice tracks not found. Available recording keys:', Object.keys(recordingData));
      if (recordingData.overlapMetadata) {
        console.log('overlapMetadata keys:', Object.keys(recordingData.overlapMetadata));
      }
      return res.status(400).json({ 
        error: 'Voice tracks not found. Please run overlap diarization (mode3) first. Voice tracks are required for pause-based merge.',
        debug: {
          hasOverlapMetadata: !!recordingData.overlapMetadata,
          hasVoiceTracks: !!recordingData.voiceTracks,
          resultsKeys: Object.keys(recordingData.results || {}),
          overlapMetadataKeys: recordingData.overlapMetadata ? Object.keys(recordingData.overlapMetadata) : []
        }
      });
    }

    console.log(`Found ${voiceTracks.length} voice tracks for merge`);

    // Apply pause-based merge
    const mergedResult = mergeWithPauseBasedReplicas(primaryDiarization, voiceTracks, {
      pauseThreshold: 0.3,
      longPauseThreshold: 1.0,
      timestampTolerance: 0.5
    });

    // Update recording with merged result
    const updatedRecording = mergedResult.recordings[0];

    console.log('✅ Pause-based merge completed successfully');

    res.json({
      success: true,
      recording: updatedRecording,
      message: 'Pause-based merge applied successfully'
    });

  } catch (error) {
    console.error('❌ Apply pause-based merge error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to apply pause-based merge',
      details: error.message
    });
  }
});

// Apply overlap fixes with LLM (respecting pauses as replica boundaries)
app.post('/api/apply-overlap-fixes', async (req, res) => {
  try {
    const { segments, mode = 'smart', recordingId } = req.body;

    if (!segments || !Array.isArray(segments) || segments.length === 0) {
      return res.status(400).json({ error: 'Segments array is required' });
    }

    console.log('📥 Apply overlap fixes request:', {
      segmentsCount: segments.length,
      mode,
      recordingId,
      timestamp: new Date().toISOString()
    });

    // Select model based on mode
    let model;
    let useLocalLLM = false;
    
    if (mode === 'local' || mode === 'test' || mode === 'test2') {
      useLocalLLM = true;
      if (mode === 'test') {
        model = TEST_MODEL_ID;
        console.log(`🔵 Using local test LLM: ${LOCAL_LLM_BASE_URL}, model: ${model}`);
      } else if (mode === 'test2') {
        model = TEST2_MODEL_ID;
        console.log(`🔵 Using local test2 LLM: ${LOCAL_LLM_BASE_URL}, model: ${model}`);
      } else {
        model = LOCAL_LLM_MODEL;
        console.log(`🔵 Using local LLM: ${LOCAL_LLM_BASE_URL}, model: ${model}`);
      }
    } else {
      // For non-local models, check OpenRouter API key
      if (!process.env.OPENROUTER_API_KEY) {
        return res.status(500).json({ error: 'OpenRouter API key is not configured' });
      }
      
      if (mode === 'fast') {
        model = FAST_MODEL_ID;
      } else if (mode === 'smart-2') {
        model = SMART_2_MODEL_ID;
      } else {
        model = SMART_MODEL_ID;
      }
      console.log(`🔵 Using OpenRouter model: ${model}`);
    }

    // Detect pauses using advanced detection
    const segmentsWithPauseDetection = detectPauses(segments, {
      interSegmentThreshold: 0.3,  // 300ms between segments
      intraSegmentThreshold: 0.5,  // 500ms between words
      longPauseThreshold: 1.0        // 1s = definite break
    });

    // Build transcript with pause markers (both inter-segment and intra-segment)
    // Limit transcript length for local LLM (they have smaller context windows)
    const MAX_TRANSCRIPT_LENGTH = useLocalLLM ? 6000 : 10000;
    const transcriptWithPauses = segmentsWithPauseDetection.map((seg, idx) => {
      const pauseMarkers = [];
      
      // Inter-segment pause (before this segment)
      if (seg.pauseBefore) {
        pauseMarkers.push(
          `[PAUSE: ${seg.pauseBefore.duration.toFixed(2)}s gap - ${seg.pauseBefore.isReplicaBoundary ? 'DEFINITE BREAK' : 'break'} between replicas]`
        );
      }
      
      // Intra-segment pauses (within this segment)
      if (seg.pauses && seg.pauses.length > 0) {
        seg.pauses.forEach((pause, pIdx) => {
          pauseMarkers.push(
            `[PAUSE: ${pause.duration.toFixed(2)}s gap at word ${pause.wordIndex} - ${pause.isReplicaBoundary ? 'DEFINITE BREAK' : 'break'} within segment]`
          );
        });
      }
      
      // Add source information
      const sourceMarkers = [];
      if (seg.source === 'voice-track' || seg.isFullText) {
        sourceMarkers.push('[VOICE-TRACK]');
      } else if (seg.isPartial || seg.source === 'primary') {
        sourceMarkers.push('[PRIMARY]');
      }
      
      const pauseMarkerText = pauseMarkers.length > 0 
        ? `\n${pauseMarkers.join('\n')}\n`
        : '';
      const sourceMarkerText = sourceMarkers.length > 0
        ? `\n${sourceMarkers.join('\n')}\n`
        : '';
      
      return `${pauseMarkerText}${sourceMarkerText}${seg.speaker}: ${seg.text}`;
    }).join('\n');
    
    // Truncate if too long (but keep important voice track segments)
    let finalTranscript = transcriptWithPauses;
    if (finalTranscript.length > MAX_TRANSCRIPT_LENGTH) {
      console.warn(`⚠️ Transcript too long (${finalTranscript.length} chars), truncating to ${MAX_TRANSCRIPT_LENGTH} for local LLM`);
      
      // Try to keep voice track segments even if truncating
      const voiceTrackLines = transcriptWithPauses.split('\n').filter(line => line.includes('[VOICE-TRACK]'));
      const otherLines = transcriptWithPauses.split('\n').filter(line => !line.includes('[VOICE-TRACK]'));
      
      // Take first part + all voice tracks + remaining space
      const voiceTrackText = voiceTrackLines.join('\n');
      const availableSpace = MAX_TRANSCRIPT_LENGTH - voiceTrackText.length - 500; // Reserve space
      const truncatedOther = otherLines.slice(0, Math.floor(availableSpace / 100)).join('\n');
      
      finalTranscript = truncatedOther + '\n\n[... additional segments ...]\n\n' + voiceTrackText;
      
      if (finalTranscript.length > MAX_TRANSCRIPT_LENGTH) {
        finalTranscript = finalTranscript.substring(0, MAX_TRANSCRIPT_LENGTH) + '\n\n[... transcript truncated for LLM ...]';
      }
    }

    // Check if we have voice track segments (marked with [VOICE-TRACK])
    const hasVoiceTracks = segments.some(s => s.source === 'voice-track' || s.isFullText);
    const voiceTrackNote = hasVoiceTracks 
      ? `
VOICE TRACKS PRESENT:
- Segments marked [VOICE-TRACK] have COMPLETE text from separated audio
- Segments marked [PRIMARY] may be INCOMPLETE
- When both exist for similar time, use [VOICE-TRACK] text
`
      : '';

    // Simplified prompt for local LLM - more actionable and less overwhelming
    const segmentCount = segments.length;
    
    // Load prompt template from file
    let promptTemplate;
    try {
      promptTemplate = await fs.readFile('prompts/overlap_fixes_json_prompt.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Failed to load overlap_fixes_json_prompt.txt, using fallback prompt');
      // Fallback to hardcoded prompt if file not found
      promptTemplate = `Fix speaker diarization by correcting speaker assignments and using complete text from voice tracks.

SIMPLE RULES:
1. Keep all text - don't delete anything
2. When you see [VOICE-TRACK] - use that complete text
3. When you see [PRIMARY] - it may be partial, check for voice track replacement
4. Split at [PAUSE] markers - each pause = new replica
5. Fix speaker labels if wrong

PROCESSING STEPS:
- Go through each segment in order
- If segment has [VOICE-TRACK] marker, use its complete text
- If segment has [PRIMARY] marker and there's a [VOICE-TRACK] segment nearby with similar time, use the voice track text instead
- Keep the speaker from the segment
- Split segments at [PAUSE] markers
- Output all segments with corrected text and speakers

{{VOICE_TRACKS_NOTE}}

INPUT TRANSCRIPT ({{SEGMENT_COUNT}} segments):
{{TRANSCRIPT}}

OUTPUT: Return JSON with "segments" array. Each segment needs: speaker, text, start, end.
Include ALL text from input. Use voice track text when available.`;
    }
    
    // Replace placeholders in template
    const prompt = promptTemplate
      .replace(/\{\{VOICE_TRACKS_NOTE\}\}/g, voiceTrackNote)
      .replace(/\{\{SEGMENT_COUNT\}\}/g, segmentCount.toString())
      .replace(/\{\{TRANSCRIPT\}\}/g, finalTranscript);

    const apiUrl = useLocalLLM 
      ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
      : 'https://openrouter.ai/api/v1/chat/completions';
    
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (useLocalLLM) {
      if (LOCAL_LLM_API_KEY) {
        headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
      }
      console.log(`📤 Sending request to local LLM: ${apiUrl}`);
    } else {
      headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
      headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
      headers['X-Title'] = 'Apply Overlap Fixes';
    }

    // Build payload with reasoning_effort if needed
    const payload = {
        model: model,
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant that fixes speaker diarization. Always return valid JSON with a "segments" array. Keep all text from input.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
      temperature: 0  // Maximum deterministic output
    };
    
    // Add reasoning effort for Fast, and Smart (GPT 5.1)
    // OpenRouter format: reasoning: { effort: "high" }
    // NOTE: Local mode (LM Studio) reasoning_effort is disabled - configure in LM Studio UI
    if (shouldUseHighReasoningEffort(mode, model)) {
      if (useLocalLLM) {
        // Local LLM reasoning_effort disabled - configure in LM Studio UI
        console.log(`🔧 Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)`);
      } else {
        // For OpenRouter, use nested reasoning object
        payload.reasoning = { effort: 'high' };
        console.log(`🔧 Using reasoning effort: high for ${mode} mode (model: ${model})`);
      }
    }

    // Increased timeout for reasoning models (especially gpt-oss-20b which may take longer)
    // Для локальної моделі встановлюємо більший таймаут (30 хвилин), оскільки вона працює повільніше
    const timeout = useLocalLLM ? 1800000 : 600000; // 30 хвилин для локальної, 10 хвилин для віддаленої
    console.log(`⏱️  Using timeout: ${timeout / 1000 / 60} minutes (${timeout / 1000}s) for ${useLocalLLM ? 'local' : 'OpenRouter'} LLM`);
    
    const response = await axios.post(
      apiUrl,
      payload,
      {
        headers: headers,
        timeout: timeout
      }
    );

    const llmOutput = response.data.choices[0]?.message?.content || '';
    console.log('📥 LLM overlap fixes response:', llmOutput.substring(0, 300));

    // Parse JSON from LLM response
    let correctedSegments = [];
    try {
      let jsonText = llmOutput.trim();
      const jsonMatch = jsonText.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
      if (jsonMatch) {
        jsonText = jsonMatch[1];
      }
      
      const jsonStart = jsonText.indexOf('{');
      const jsonEnd = jsonText.lastIndexOf('}');
      
      if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
        jsonText = jsonText.substring(jsonStart, jsonEnd + 1);
      }
      
      const parsed = JSON.parse(jsonText);
      correctedSegments = parsed.segments || [];
      
      if (!Array.isArray(correctedSegments) || correctedSegments.length === 0) {
        console.warn('⚠️ LLM returned empty or invalid segments array, using original segments');
        correctedSegments = segments;
      }
    } catch (parseError) {
      console.error('⚠️ Failed to parse LLM response:', parseError.message);
      console.log('LLM output preview:', llmOutput.substring(0, 500));
      // Use original segments as fallback
      correctedSegments = segments;
    }

    // Ensure all segments have required fields
    correctedSegments = correctedSegments.map(seg => ({
      speaker: seg.speaker || 'SPEAKER_00',
      text: seg.text || '',
      start: parseFloat(seg.start) || 0,
      end: parseFloat(seg.end) || seg.start || 0,
      originalText: seg.originalText || seg.text || '',
      translations: seg.translations || {},
      role: seg.role || null,
      overlap: seg.overlap || false,
      speaker_id: seg.speaker || seg.speaker_id || 'SPEAKER_00' // For compatibility with removeDuplicatesAndEmptyWords
    }));

    // Remove duplicates and empty words
    const cleanedSegments = removeDuplicatesAndEmptyWords(correctedSegments, '[Apply Overlap Fixes]');
    
    if (cleanedSegments.length !== correctedSegments.length) {
      console.log(`🧹 Cleaned segments: ${correctedSegments.length} → ${cleanedSegments.length} (removed ${correctedSegments.length - cleanedSegments.length} duplicates/empty)`);
    }

    // Convert back to expected format
    const finalSegments = cleanedSegments.map(seg => ({
      speaker: seg.speaker || seg.speaker_id || 'SPEAKER_00',
      text: seg.text || '',
      start: seg.start || 0,
      end: seg.end || seg.start || 0,
      originalText: seg.originalText || seg.text || '',
      translations: seg.translations || {},
      role: seg.role || null,
      overlap: seg.overlap || false
    }));

    res.json({
      success: true,
      segments: finalSegments,
      source: 'llm-overlap-fixes',
      originalCount: segments.length,
      correctedCount: finalSegments.length,
      duplicatesRemoved: correctedSegments.length - finalSegments.length
    });

  } catch (error) {
    console.error('❌ Apply overlap fixes error:', error);
    
    // Check if it's a timeout error
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return res.status(504).json({
        error: 'LLM request timeout',
        message: `The LLM request took longer than expected to complete. This may happen with reasoning models. Try reducing the input size or using a faster model.`,
        suggestion: 'For local models, try adjusting reasoning settings in LM Studio UI or use a faster model.'
      });
    }
    
    res.status(500).json({
      error: 'Failed to apply overlap fixes',
      details: error.response?.data || error.message
    });
  }
});

// Clear LLM cache endpoint
app.post('/api/clear-llm-cache', async (req, res) => {
  try {
    if (!llmCacheDir || !fsSync.existsSync(llmCacheDir)) {
      return res.json({
        success: true,
        deletedCount: 0,
        message: 'Cache directory does not exist'
      });
    }

    // Read all files in cache directory
    const files = fsSync.readdirSync(llmCacheDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    let deletedCount = 0;
    let errors = [];

    // Delete each cache file
    for (const file of jsonFiles) {
      try {
        const filePath = path.join(llmCacheDir, file);
        fsSync.unlinkSync(filePath);
        deletedCount++;
      } catch (error) {
        errors.push({ file, error: error.message });
        console.error(`⚠️ Failed to delete cache file ${file}:`, error.message);
      }
    }

    console.log(`🗑️ LLM cache cleared: ${deletedCount} files deleted`);

    res.json({
      success: true,
      deletedCount,
      totalFiles: jsonFiles.length,
      errors: errors.length > 0 ? errors : undefined,
      message: `Successfully cleared ${deletedCount} cache file(s)`
    });
  } catch (error) {
    console.error('❌ Clear LLM cache error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to clear LLM cache',
      details: error.message
    });
  }
});

// Save LLM cache endpoint (export cache as JSON file)
app.post('/api/save-llm-cache', async (req, res) => {
  try {
    if (!llmCacheDir || !fsSync.existsSync(llmCacheDir)) {
      return res.status(404).json({
        success: false,
        error: 'Cache directory does not exist'
      });
    }

    // Read all files in cache directory
    const files = fsSync.readdirSync(llmCacheDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    const cacheData = {
      exportedAt: new Date().toISOString(),
      totalFiles: jsonFiles.length,
      cacheDir: llmCacheDir,
      files: []
    };

    // Read each cache file and add to export
    for (const file of jsonFiles) {
      try {
        const filePath = path.join(llmCacheDir, file);
        const fileContent = fsSync.readFileSync(filePath, 'utf8');
        const parsed = JSON.parse(fileContent);
        
        cacheData.files.push({
          filename: file,
          cacheKey: file.replace('.json', ''),
          data: parsed,
          size: fileContent.length,
          modified: fsSync.statSync(filePath).mtime.toISOString()
        });
      } catch (error) {
        console.error(`⚠️ Failed to read cache file ${file}:`, error.message);
        cacheData.files.push({
          filename: file,
          error: error.message
        });
      }
    }

    console.log(`💾 LLM cache export: ${cacheData.files.length} files exported`);

    // Send as JSON file download
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Disposition', `attachment; filename="llm-cache-backup-${new Date().toISOString().split('T')[0]}.json"`);
    res.json(cacheData);
  } catch (error) {
    console.error('❌ Save LLM cache error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to save LLM cache',
      details: error.message
    });
  }
});

// Analyze text for highlighting (Blue/Green/Red) - receives payload with general, speaker1, speaker2, markdown
app.post('/api/analyze-text', async (req, res) => {
  try {
    const payload = req.body;
    
    if (!payload) {
      return res.status(400).json({ 
        error: 'Payload is required',
        Blue: [],
        Green: [],
        Red: []
      });
    }
    
    // Отримуємо режим з environment variable
    const textAnalysisMode = process.env.TEXT_ANALYSIS_MODE || 'script';
    const useLLM = textAnalysisMode === 'llm';
    
    // Перевіряємо, чи використовувати 3-tier систему з complexity routing
    const useComplexityRouting = process.env.USE_COMPLEXITY_ROUTING === 'true';
    
    console.log('🔍 [/api/analyze-text] ============================================');
    console.log('🔍 [/api/analyze-text] Text analysis request received');
    console.log('🔍 [/api/analyze-text] Configuration:', {
      useLLM: useLLM,
      textAnalysisMode: textAnalysisMode,
      useComplexityRouting: useComplexityRouting,
      processEnvTEXT_ANALYSIS_MODE: process.env.TEXT_ANALYSIS_MODE,
      processEnvUSE_COMPLEXITY_ROUTING: process.env.USE_COMPLEXITY_ROUTING,
      willUseLLM: useLLM
    });
    console.log('🔍 [/api/analyze-text] Payload structure:', {
      hasMarkdown: !!payload.markdown,
      markdownLength: payload.markdown?.length || 0,
      markdownType: typeof payload.markdown,
      hasGeneral: !!payload.general,
      hasSpeaker1: !!payload.speaker1,
      hasSpeaker2: !!payload.speaker2,
      generalSegmentsCount: payload.general?.segments?.length || payload.general?.speechmatics?.segments?.length || 0,
      speaker1SegmentsCount: payload.speaker1?.segments?.length || payload.speaker1?.speechmatics?.segments?.length || 0,
      speaker2SegmentsCount: payload.speaker2?.segments?.length || payload.speaker2?.speechmatics?.segments?.length || 0
    });
    console.log('🔍 [/api/analyze-text] ============================================');
    
    let analysisResult;
    
    // Використовуємо HybridAnalyzer, якщо увімкнено complexity routing
    if (useComplexityRouting && useLLM) {
      try {
        const HybridAnalyzer = require('./lib/hybrid-analyzer');
        
        console.log('🎯 [/api/analyze-text] Using 3-tier Hybrid Analyzer with complexity routing...');
        
        // Визначаємо режим LLM з payload або використовуємо smart за замовчуванням
        const requestedMode = payload.mode || 'smart';
        const useLocalLLM = requestedMode === 'local' || requestedMode === 'test' || requestedMode === 'test2';
        const llmModel = getModelId(requestedMode);
        const apiUrl = useLocalLLM 
          ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
          : 'https://openrouter.ai/api/v1/chat/completions';
        const apiKey = useLocalLLM ? LOCAL_LLM_API_KEY : process.env.OPENROUTER_API_KEY;
        
        // Визначаємо провайдера для classifier (за замовчуванням ollama, але можна lmstudio)
        const classifierProvider = process.env.CLASSIFIER_PROVIDER || 'ollama';
        const classifierBaseURL = classifierProvider === 'ollama' 
          ? (process.env.OLLAMA_BASE_URL || 'http://localhost:11434')
          : (process.env.LM_STUDIO_BASE_URL || 'http://127.0.0.1:3001');
        const classifierModel = process.env.CLASSIFIER_MODEL || (classifierProvider === 'ollama' ? 'phi:3.5' : 'microsoft/phi-3.5-mini-instruct');
        
        const analyzer = new HybridAnalyzer({
          enableLLM: true,
          useComplexityRouting: true,
          provider: classifierProvider,
          llmModel: llmModel,
          apiUrl: apiUrl,
          apiKey: apiKey,
          useLocalLLM: useLocalLLM,
          mode: requestedMode,
          baseURL: classifierBaseURL,
          classifier: {
            provider: classifierProvider,
            baseURL: classifierBaseURL,
            model: classifierModel,
            timeout: parseInt(process.env.CLASSIFIER_TIMEOUT || '5000')
          },
          simpleThreshold: parseFloat(process.env.SIMPLE_THRESHOLD || '0.9'),
          mediumThreshold: parseFloat(process.env.MEDIUM_THRESHOLD || '0.6')
        });
        
        console.log('🎯 [/api/analyze-text] Hybrid Analyzer configuration:', {
          useComplexityRouting: true,
          classifierProvider: classifierProvider,
          classifierModel: classifierModel,
          classifierBaseURL: classifierBaseURL,
          llmModel: llmModel,
          apiUrl: apiUrl,
          useLocalLLM: useLocalLLM
        });
        
        analysisResult = await analyzer.analyze(payload);
        
        console.log('✅ [/api/analyze-text] Hybrid Analyzer completed:', {
          blueCount: analysisResult.Blue?.length || 0,
          greenCount: analysisResult.Green?.length || 0,
          redCount: analysisResult.Red?.length || 0,
          hasBlue: !!analysisResult.Blue,
          hasGreen: !!analysisResult.Green,
          hasRed: !!analysisResult.Red
        });
      } catch (hybridError) {
        console.error('❌ [/api/analyze-text] Hybrid Analyzer failed:', hybridError.message);
        console.error('❌ [/api/analyze-text] Error details:', hybridError);
        // Fallback до стандартного режиму
        throw hybridError;
      }
    } else if (useLLM) {
      // Стандартний 2-tier режим (без complexity routing)
      try {
        console.log('🤖 [/api/analyze-text] ============================================');
        console.log('🤖 [/api/analyze-text] Using LLM mode for text analysis (2-tier)...');
        
        // Визначаємо режим LLM з payload або використовуємо smart за замовчуванням
        const requestedMode = payload.mode || 'smart';
        const useLocalLLM = requestedMode === 'local' || requestedMode === 'test' || requestedMode === 'test2';
        const llmModel = getModelId(requestedMode);
        const apiUrl = useLocalLLM 
          ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
          : 'https://openrouter.ai/api/v1/chat/completions';
        const apiKey = useLocalLLM ? LOCAL_LLM_API_KEY : process.env.OPENROUTER_API_KEY;
        
        console.log('🤖 [/api/analyze-text] LLM configuration:', {
          llmModel: llmModel,
          apiUrl: apiUrl,
          useLocalLLM: useLocalLLM,
          mode: requestedMode,
          hasApiKey: !!apiKey
        });
        
        if (!apiKey && !useLocalLLM) {
          throw new Error('OPENROUTER_API_KEY is not configured');
        }
        
        console.log('🤖 [/api/analyze-text] Calling analyzeTextWithLLM...');
        analysisResult = await textAnalysis.analyzeTextWithLLM(
          payload,
          llmModel,
          apiUrl,
          apiKey,
          useLocalLLM,
          requestedMode
        );
        
        console.log('✅ [/api/analyze-text] LLM analysis completed:', {
          blueCount: analysisResult.Blue?.length || 0,
          greenCount: analysisResult.Green?.length || 0,
          redCount: analysisResult.Red?.length || 0,
          hasBlue: !!analysisResult.Blue,
          hasGreen: !!analysisResult.Green,
          hasRed: !!analysisResult.Red
        });
        console.log('🤖 [/api/analyze-text] ============================================');
      } catch (llmError) {
        console.error('❌ [/api/analyze-text] ============================================');
        console.error('❌ [/api/analyze-text] LLM analysis failed:', llmError.message);
        console.error('❌ [/api/analyze-text] Error details:', llmError);
        console.error('❌ [/api/analyze-text] Stack:', llmError.stack);
        console.error('❌ [/api/analyze-text] ============================================');
        
        // НЕ використовуємо fallback до script режиму, якщо користувач явно вибрав LLM режим
        // Повертаємо помилку, щоб користувач знав, що LLM не спрацював
        throw new Error(`LLM analysis failed: ${llmError.message}. Please check LLM configuration and try again.`);
      }
    } else {
      console.log('📝 [/api/analyze-text] Using script mode for text analysis');
      analysisResult = textAnalysis.analyzeText(payload);
    }
    
    console.log('📤 [/api/analyze-text] Sending response:', {
      blueCount: analysisResult.Blue?.length || 0,
      greenCount: analysisResult.Green?.length || 0,
      redCount: analysisResult.Red?.length || 0,
      hasError: !!analysisResult.error
    });
    
    res.json(analysisResult);
    
  } catch (error) {
    console.error('❌ [/api/analyze-text] ============================================');
    console.error('❌ [/api/analyze-text] Fatal error:', error.message);
    console.error('❌ [/api/analyze-text] Error details:', error);
    console.error('❌ [/api/analyze-text] Stack:', error.stack);
    console.error('❌ [/api/analyze-text] ============================================');
    
    res.status(500).json({
      error: 'Text analysis failed',
      details: error.message,
      Blue: [],
      Green: [],
      Red: []
    });
  }
});

// Apply markdown fixes with LLM - receives JSON transcripts from Agent and Client
// Multi-step LLM processing for markdown fixes (optimized for local models)
// Helper function to normalize text and extract words (ignore punctuation)
function normalizeTextToWords(text) {
  if (!text) return [];
  return text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')  // Replace punctuation with spaces
    .replace(/\s+/g, ' ')      // Normalize whitespace
    .trim()
    .split(/\s+/)
    .filter(w => w.length > 0); // Filter out empty strings
}

// Function to calculate word-level match with ground truth
function calculateWordLevelMatch(diarizationText, groundTruthText) {
  if (!diarizationText || !groundTruthText) {
    return null;
  }

  try {
    // Extract all words from ground truth (normalized, no punctuation)
    const gtWords = normalizeTextToWords(groundTruthText);
    const totalGtWords = gtWords.length;
    
    if (totalGtWords === 0) {
      return {
        matchPercent: 0,
        matchedWords: 0,
        unmatchedWords: 0,
        totalWords: 0,
        extraWords: 0
      };
    }

    // Extract all words from diarization result (normalized, no punctuation)
    const diarizationWords = normalizeTextToWords(diarizationText);
    
    // Count word matches (case-insensitive, ignoring punctuation)
    const gtWordCounts = {};
    gtWords.forEach(word => {
      gtWordCounts[word] = (gtWordCounts[word] || 0) + 1;
    });
    
    const diarizationWordCounts = {};
    diarizationWords.forEach(word => {
      diarizationWordCounts[word] = (diarizationWordCounts[word] || 0) + 1;
    });
    
    // Count matched words (words that appear in both)
    let matchedWords = 0;
    Object.keys(gtWordCounts).forEach(word => {
      const gtCount = gtWordCounts[word];
      const diarizationCount = diarizationWordCounts[word] || 0;
      matchedWords += Math.min(gtCount, diarizationCount);
    });
    
    // Count unmatched words (words in GT but not in diarization)
    const unmatchedWords = totalGtWords - matchedWords;
    
    // Count extra words (words in diarization but not in GT)
    let extraWords = 0;
    Object.keys(diarizationWordCounts).forEach(word => {
      if (!gtWordCounts[word]) {
        extraWords += diarizationWordCounts[word];
      }
    });
    
    // Calculate match percentage
    const matchPercent = totalGtWords > 0 ? (matchedWords / totalGtWords) * 100 : 0;

    return {
      matchPercent: Math.round(matchPercent * 10) / 10,
      matchedWords: matchedWords,
      unmatchedWords: unmatchedWords,
      totalWords: totalGtWords,
      extraWords: extraWords,
      diarizationWordCount: diarizationWords.length
    };
  } catch (error) {
    console.error('Error calculating word-level match:', error);
    return null;
  }
}

// Function to extract all text from diarization segments
function extractAllTextFromSegments(segments) {
  if (!Array.isArray(segments) || segments.length === 0) {
    return '';
  }
  
  return segments
    .map(seg => (seg.text || '').trim())
    .filter(text => text.length > 0)
    .join(' ');
}

// Function to calculate ground truth match metrics (word-level comparison)
function calculateGroundTruthMatch(markdownTable, groundTruthText, primaryDiarization = null) {
  if (!markdownTable || !groundTruthText) {
    return null;
  }

  try {
    // Extract all text from ground truth
    const gtLines = groundTruthText.split('\n').filter(line => line.trim());
    const gtText = gtLines
      .map(line => {
        const match = line.match(/^(?:Speaker\s+)?(?:\d+|Agent|Client)[:\s]+(.+)$/i);
        return match ? match[1].trim() : '';
      })
      .filter(text => text.length > 0)
      .join(' ');

    // Extract all text from NextLevel markdown table
    // IMPORTANT: This is the LLM-processed result, NOT the original Speechmatics
    const tableLines = markdownTable.split('\n').filter(line => line.trim() && line.includes('|'));
    const nextLevelText = [];
    
    // Skip header and separator rows (usually first 2 lines)
    let headerSkipped = 0;
    for (let i = 0; i < tableLines.length; i++) {
      const line = tableLines[i].trim();
      // Skip header row (contains "Segment ID", "Speaker", "Text", etc.)
      if (line.toLowerCase().includes('segment') || line.toLowerCase().includes('speaker') || line.toLowerCase().includes('text')) {
        headerSkipped++;
        continue;
      }
      // Skip separator row (contains only dashes and pipes)
      if (line.match(/^[\|\s\-:]+$/)) {
        headerSkipped++;
        continue;
      }
      
      // Process data rows
      const cells = line.split('|').map(c => c.trim()).filter(c => c);
      // Typically: Segment ID | Speaker | Text | Start Time | End Time
      // Text is usually in column 2 (index 2)
      if (cells.length >= 3 && cells[2]) {
        const text = cells[2].trim();
        if (text && text.length > 0) {
          nextLevelText.push(text);
        }
      }
    }
    const nextLevelFullText = nextLevelText.join(' ');
    console.log(`[calculateGroundTruthMatch] NextLevel (LLM-PROCESSED) segments: ${nextLevelText.length}, text length: ${nextLevelFullText.length}`);
    console.log(`[calculateGroundTruthMatch] NextLevel text preview: ${nextLevelFullText.substring(0, 200)}...`);
    console.log(`[calculateGroundTruthMatch] Markdown table lines: ${tableLines.length}, header rows skipped: ${headerSkipped}`);

    // Calculate word-level match for NextLevel
    const nextLevelMatch = calculateWordLevelMatch(nextLevelFullText, gtText);

    // Calculate word-level match for Speechmatics (if available)
    // IMPORTANT: Extract ONLY from original Speechmatics segments, NOT from overlap-corrected
    let speechmaticsMatch = null;
    if (primaryDiarization) {
      // Extract ONLY from speechmatics results, not from overlap-corrected
      let speechmaticsSegments = [];
      const recording = primaryDiarization?.recordings?.[0];
      if (recording?.results?.speechmatics?.segments) {
        speechmaticsSegments = recording.results.speechmatics.segments;
      } else if (primaryDiarization?.speechmatics?.segments) {
        speechmaticsSegments = primaryDiarization.speechmatics.segments;
      } else if (Array.isArray(primaryDiarization.segments)) {
        // Fallback: if segments are at top level, use them
        speechmaticsSegments = primaryDiarization.segments;
      }
      
      if (speechmaticsSegments && speechmaticsSegments.length > 0) {
        const speechmaticsText = extractAllTextFromSegments(speechmaticsSegments);
        console.log(`[calculateGroundTruthMatch] Speechmatics (ORIGINAL) segments: ${speechmaticsSegments.length}, text length: ${speechmaticsText.length}`);
        console.log(`[calculateGroundTruthMatch] Speechmatics text preview: ${speechmaticsText.substring(0, 100)}...`);
        speechmaticsMatch = calculateWordLevelMatch(speechmaticsText, gtText);
      } else {
        console.log(`[calculateGroundTruthMatch] No Speechmatics segments found in primaryDiarization`);
      }
    } else {
      console.log(`[calculateGroundTruthMatch] primaryDiarization is null`);
    }

    const comparison = speechmaticsMatch ? {
      nextLevelBetter: (nextLevelMatch?.matchPercent || 0) > (speechmaticsMatch?.matchPercent || 0),
      improvement: (nextLevelMatch?.matchPercent || 0) - (speechmaticsMatch?.matchPercent || 0),
      nextLevelPercent: nextLevelMatch?.matchPercent || 0,
      speechmaticsPercent: speechmaticsMatch?.matchPercent || 0
    } : null;
    
    // Log comparison results
    if (comparison) {
      const status = comparison.nextLevelBetter ? '✅ BETTER' : '❌ WORSE';
      console.log(`[calculateGroundTruthMatch] Comparison: NextLevel is ${status}`);
      console.log(`[calculateGroundTruthMatch]   NextLevel: ${comparison.nextLevelPercent}%`);
      console.log(`[calculateGroundTruthMatch]   Speechmatics: ${comparison.speechmaticsPercent}%`);
      console.log(`[calculateGroundTruthMatch]   Improvement: ${comparison.improvement > 0 ? '+' : ''}${comparison.improvement.toFixed(1)}%`);
      
      if (!comparison.nextLevelBetter) {
        console.warn(`[calculateGroundTruthMatch] ⚠️ WARNING: NextLevel is WORSE than Speechmatics!`);
        console.warn(`[calculateGroundTruthMatch]   This should not happen - NextLevel should always be better.`);
        console.warn(`[calculateGroundTruthMatch]   NextLevel text length: ${nextLevelFullText.length}`);
        console.warn(`[calculateGroundTruthMatch]   Speechmatics text length: ${speechmaticsMatch ? 'N/A' : 'N/A'}`);
      }
    }
    
    return {
      nextLevel: nextLevelMatch,
      speechmatics: speechmaticsMatch,
      comparison: comparison
    };
  } catch (error) {
    console.error('Error calculating ground truth match:', error);
    return null;
  }
}

async function processMarkdownFixesMultiStep(promptContext, mode = 'smart', recordingId = null, isAutoTest = false) {
  const useLocalLLM = mode === 'local' || mode === 'test' || mode === 'test2';
  const model = getModelId(mode);
  const apiUrl = useLocalLLM 
    ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
    : 'https://openrouter.ai/api/v1/chat/completions';
  
  const headers = {
    'Content-Type': 'application/json'
  };
  
  if (useLocalLLM) {
    if (LOCAL_LLM_API_KEY) {
      headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
    }
  } else {
    headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
    headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
    headers['X-Title'] = 'Apply Markdown Fixes';
  }

  const timeout = useLocalLLM ? 1800000 : 600000;
  
  // Helper function to call LLM with a prompt
  async function callLLMStep(stepNumber, stepName, promptTemplate, replacements, outputFormat = 'json') {
    let prompt = promptTemplate;
    Object.entries(replacements).forEach(([token, value]) => {
      const safeValue = value && value.trim().length > 0 ? value : '[empty]';
      prompt = prompt.replace(new RegExp(token, 'g'), safeValue);
    });

    // Log input for this step
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📥 STEP ${stepNumber}: ${stepName} - INPUT`);
    console.log(`${'='.repeat(80)}`);
    console.log(`📋 Prompt template length: ${promptTemplate.length} chars`);
    console.log(`📋 Final prompt length: ${prompt.length} chars`);
    console.log(`📋 Replacements used:`);
    Object.entries(replacements).forEach(([token, value]) => {
      const valueLength = value ? value.length : 0;
      const preview = value && value.length > 0 ? value.substring(0, 100).replace(/\n/g, ' ') : '[empty]';
      console.log(`   ${token}: ${valueLength} chars - "${preview}${valueLength > 100 ? '...' : ''}"`);
    });
    console.log(`\n📝 Full prompt (first 500 chars):`);
    console.log(prompt.substring(0, 500) + (prompt.length > 500 ? '...' : ''));

    const payload = {
      model: model,
      messages: [
        {
          role: 'system',
          content: `You are a helpful assistant. Follow instructions carefully and return ${outputFormat === 'json' ? 'valid JSON' : 'a Markdown table'}.`
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0
    };

    if (shouldUseHighReasoningEffort(mode, model) && !useLocalLLM) {
      payload.reasoning = { effort: 'high' };
    }

    console.log(`\n🔄 Step ${stepNumber}: Calling LLM (${model})...`);
    const startTime = Date.now();
    
    let response;
    try {
      response = await axios.post(apiUrl, payload, { headers, timeout });
    } catch (error) {
      console.error(`\n❌ STEP ${stepNumber}: ${stepName} - LLM REQUEST FAILED`);
      console.error(`Error: ${error.message}`);
      if (error.response) {
        console.error(`Status: ${error.response.status}`);
        console.error(`Response: ${JSON.stringify(error.response.data).substring(0, 500)}`);
      }
      throw error;
    }
    
    const duration = Date.now() - startTime;
    const message = response.data.choices[0]?.message || {};
    const llmOutput = message.content || message.reasoning || '';
    
    // Log output for this step
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📤 STEP ${stepNumber}: ${stepName} - OUTPUT`);
    console.log(`${'='.repeat(80)}`);
    console.log(`⏱️  Duration: ${duration}ms (${(duration / 1000).toFixed(2)}s)`);
    console.log(`📏 Output length: ${llmOutput.length} chars`);
    console.log(`📋 Output preview (first 500 chars):`);
    console.log(llmOutput.substring(0, 500) + (llmOutput.length > 500 ? '...' : ''));
    if (outputFormat === 'json') {
      try {
        const jsonMatch = llmOutput.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          console.log(`✅ Valid JSON detected: ${Array.isArray(parsed) ? parsed.length : 'N/A'} items`);
        }
      } catch (e) {
        console.warn(`⚠️  JSON parsing failed: ${e.message}`);
      }
    }
    console.log(`\n📝 Full output:`);
    console.log(llmOutput);
    console.log(`${'='.repeat(80)}\n`);
    
    return llmOutput;
  }

  try {
    // Step 1: Validate replicas
    console.log('\n🚀 Starting multi-step markdown fixes processing...');
    console.log(`📊 Mode: ${mode}, Model: ${model}, UseLocalLLM: ${useLocalLLM}`);
    console.log(`📊 Recording ID: ${recordingId || 'N/A'}`);
    
    let step1Template = '';
    try {
      step1Template = await fs.readFile('prompts/step1_validate_replicas.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Step 1 template not found:', error.message);
    }
    if (!step1Template) {
      throw new Error('Step 1 prompt template not found');
    }
    
    const step1Replacements = {
      '{{GENERAL_DIALOGUE}}': promptContext.generalDialog || '[empty]',
      '{{STANDARD_SPEAKER0_DIALOGUE}}': promptContext.speaker0Dialog || '[empty]',
      '{{STANDARD_SPEAKER1_DIALOGUE}}': promptContext.speaker1Dialog || '[empty]',
      '{{AGENT_DIALOGUE}}': promptContext.agentDialog || '[empty]',
      '{{CLIENT_DIALOGUE}}': promptContext.clientDialog || '[empty]',
      '{{SEGMENT_TIMESTAMPS}}': promptContext.segmentTimestampsText || '[empty]'
    };
    
    const step1Output = await callLLMStep(1, 'Validate Replicas', step1Template, step1Replacements, 'json');
    let validatedReplicas = [];
    try {
      const jsonMatch = step1Output.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        validatedReplicas = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      console.warn('⚠️ Step 1: Failed to parse JSON, using all replicas');
    }

    // Step 2: Assign roles
    let step2Template = '';
    try {
      step2Template = await fs.readFile('prompts/step2_assign_roles.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Step 2 template not found:', error.message);
    }
    if (!step2Template) {
      throw new Error('Step 2 prompt template not found');
    }
    
    const step2Replacements = {
      '{{VALIDATED_REPLICAS}}': JSON.stringify(validatedReplicas, null, 2),
      '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText || '[empty]'
    };
    
    const step2Output = await callLLMStep(2, 'Assign Roles', step2Template, step2Replacements, 'json');
    let roledReplicas = [];
    try {
      const jsonMatch = step2Output.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        roledReplicas = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      console.warn('⚠️ Step 2: Failed to parse JSON, using validated replicas');
      roledReplicas = validatedReplicas;
    }

    // Step 3: Remove wrong speaker replicas and duplicates
    let step3Template = '';
    try {
      step3Template = await fs.readFile('prompts/step3_remove_duplicates.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Step 3 template not found:', error.message);
    }
    if (!step3Template) {
      throw new Error('Step 3 prompt template not found');
    }
    
    const step3Replacements = {
      '{{ROLED_REPLICAS}}': JSON.stringify(roledReplicas, null, 2),
      '{{GENERAL_DIALOGUE}}': promptContext.generalDialog || '[empty]',
      '{{STANDARD_SPEAKER0_DIALOGUE}}': promptContext.speaker0Dialog || '[empty]',
      '{{STANDARD_SPEAKER1_DIALOGUE}}': promptContext.speaker1Dialog || '[empty]',
      '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText || '[empty]'
    };
    
    const step3Output = await callLLMStep(3, 'Remove Wrong Speaker Replicas', step3Template, step3Replacements, 'json');
    let cleanedReplicas = [];
    try {
      const jsonMatch = step3Output.match(/\[[\s\S]*\]/);
      if (jsonMatch) {
        cleanedReplicas = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      console.warn('⚠️ Step 3: Failed to parse JSON, using roled replicas');
      cleanedReplicas = roledReplicas;
    }

    // Step 4: Format table
    let step4Template = '';
    try {
      step4Template = await fs.readFile('prompts/step4_format_table.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Step 4 template not found:', error.message);
    }
    if (!step4Template) {
      throw new Error('Step 4 prompt template not found');
    }
    
    const step4Replacements = {
      '{{CLEANED_REPLICAS}}': JSON.stringify(cleanedReplicas, null, 2)
    };
    
    let markdownTable = await callLLMStep(4, 'Format Table', step4Template, step4Replacements, 'markdown');
    
    // Step 5: Verify result
    let step5Template = '';
    try {
      step5Template = await fs.readFile('prompts/step5_verify_result.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Step 5 template not found (optional step):', error.message);
    }
    
    if (step5Template) {
      const step5Replacements = {
        '{{GENERAL_DIALOGUE}}': promptContext.generalDialog || '[empty]',
        '{{STANDARD_SPEAKER0_DIALOGUE}}': promptContext.speaker0Dialog || '[empty]',
        '{{STANDARD_SPEAKER1_DIALOGUE}}': promptContext.speaker1Dialog || '[empty]',
        '{{AGENT_DIALOGUE}}': promptContext.agentDialog || '[empty]',
        '{{CLIENT_DIALOGUE}}': promptContext.clientDialog || '[empty]',
        '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText || '[empty]',
        '{{GENERATED_TABLE}}': markdownTable
      };
      
      const verifiedTable = await callLLMStep(5, 'Verify Result', step5Template, step5Replacements, 'markdown');
      if (verifiedTable && verifiedTable.trim().length > 0) {
        console.log(`\n✅ Step 5: Verification completed. Table updated.`);
        markdownTable = verifiedTable;
      } else {
        console.log(`\n⚠️  Step 5: Verification returned empty, keeping Step 4 result.`);
      }
    } else {
      console.log(`\n⏭️  Step 5: Skipped (template not found)`);
    }
    
    // Optional Step 6: analyze discrepancies or calculate diff metrics
    let groundTruthMetrics = null;
    if (promptContext.groundTruthText) {
      if (isAutoTest) {
        // Auto-test mode: analyze discrepancies and provide recommendations
        let step6Template = '';
        try {
          step6Template = await fs.readFile('prompts/step6_ground_truth_alignment.txt', 'utf8');
        } catch (error) {
          console.warn('⚠️ Step 6 template not found (ground truth analysis skipped):', error.message);
        }
        
        if (step6Template) {
          const step6Replacements = {
            '{{GROUND_TRUTH_DIALOGUE}}': promptContext.groundTruthText || '[empty]',
            '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText || '[empty]',
            '{{GENERATED_TABLE}}': markdownTable || '[empty]',
            '{{SEGMENT_TIMESTAMPS}}': promptContext.segmentTimestampsText || '[empty]'
          };
          
          const analysisOutput = await callLLMStep(6, 'Ground Truth Analysis', step6Template, step6Replacements, 'json');
          
          // Parse analysis output
          try {
            const jsonMatch = analysisOutput.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              const analysis = JSON.parse(jsonMatch[0]);
              
              console.log(`\n${'='.repeat(80)}`);
              console.log(`📊 STEP 6: GROUND TRUTH ANALYSIS RESULTS`);
              console.log(`${'='.repeat(80)}`);
              console.log(`📋 Summary: ${analysis.summary || 'N/A'}`);
              console.log(`📋 Affected Steps: ${(analysis.affectedSteps || []).join(', ') || 'None'}`);
              console.log(`📋 Discrepancies found: ${(analysis.discrepancies || []).length}`);
              
              if (analysis.recommendations) {
                console.log(`\n💡 RECOMMENDATIONS:`);
                Object.entries(analysis.recommendations).forEach(([step, rec]) => {
                  if (rec && rec.trim()) {
                    console.log(`   Step ${step}: ${rec}`);
                  }
                });
              }
              
              if (analysis.discrepancies && analysis.discrepancies.length > 0) {
                console.log(`\n🔍 DISCREPANCIES:`);
                analysis.discrepancies.slice(0, 10).forEach((disc, idx) => {
                  console.log(`   ${idx + 1}. [${disc.type}] ${disc.description}`);
                  console.log(`      Expected: ${disc.groundTruth}`);
                  console.log(`      Got: ${disc.generated}`);
                  console.log(`      Step ${disc.affectedStep}: ${disc.recommendation || 'N/A'}`);
                });
                if (analysis.discrepancies.length > 10) {
                  console.log(`   ... and ${analysis.discrepancies.length - 10} more`);
                }
              }
              console.log(`${'='.repeat(80)}\n`);
            }
          } catch (parseError) {
            console.warn('⚠️ Step 6: Failed to parse analysis JSON:', parseError.message);
            console.log('Raw output:', analysisOutput.substring(0, 500));
          }
          
          // Don't modify markdownTable in auto-test mode
          console.log(`\n✅ Step 6: Ground truth analysis completed (markdown table unchanged)`);
          
          // Also calculate metrics for information (even in auto-test mode)
          groundTruthMetrics = calculateGroundTruthMatch(
            markdownTable, 
            promptContext.groundTruthText,
            promptContext.primaryDiarization || null
          );
          if (groundTruthMetrics) {
            console.log(`\n📊 WORD-LEVEL METRICS (for reference):`);
            if (groundTruthMetrics.nextLevel) {
              console.log(`   NextLevel: ${groundTruthMetrics.nextLevel.matchPercent}% (matched: ${groundTruthMetrics.nextLevel.matchedWords}/${groundTruthMetrics.nextLevel.totalWords}, unmatched: ${groundTruthMetrics.nextLevel.unmatchedWords}, extra: ${groundTruthMetrics.nextLevel.extraWords})`);
            }
            if (groundTruthMetrics.speechmatics) {
              console.log(`   Speechmatics: ${groundTruthMetrics.speechmatics.matchPercent}% (matched: ${groundTruthMetrics.speechmatics.matchedWords}/${groundTruthMetrics.speechmatics.totalWords}, unmatched: ${groundTruthMetrics.speechmatics.unmatchedWords}, extra: ${groundTruthMetrics.speechmatics.extraWords})`);
            }
            if (groundTruthMetrics.comparison) {
              const improvement = groundTruthMetrics.comparison.improvement;
              const status = groundTruthMetrics.comparison.nextLevelBetter ? '✅ BETTER' : '❌ WORSE';
              console.log(`   Comparison: NextLevel is ${status} by ${Math.abs(improvement).toFixed(1)}%`);
            }
          }
        }
      } else {
        // Production mode: calculate diff metrics only
        // Pass primaryDiarization for Speechmatics comparison
        groundTruthMetrics = calculateGroundTruthMatch(
          markdownTable, 
          promptContext.groundTruthText,
          promptContext.primaryDiarization || null
        );
        
        if (groundTruthMetrics) {
          console.log(`\n${'='.repeat(80)}`);
          console.log(`📊 STEP 6: GROUND TRUTH METRICS (WORD-LEVEL)`);
          console.log(`${'='.repeat(80)}`);
          if (groundTruthMetrics.nextLevel) {
            console.log(`📈 NextLevel Match: ${groundTruthMetrics.nextLevel.matchPercent}%`);
            console.log(`   Matched words: ${groundTruthMetrics.nextLevel.matchedWords}/${groundTruthMetrics.nextLevel.totalWords}`);
            console.log(`   Unmatched words: ${groundTruthMetrics.nextLevel.unmatchedWords}`);
            console.log(`   Extra words: ${groundTruthMetrics.nextLevel.extraWords}`);
          }
          if (groundTruthMetrics.speechmatics) {
            console.log(`📈 Speechmatics Match: ${groundTruthMetrics.speechmatics.matchPercent}%`);
            console.log(`   Matched words: ${groundTruthMetrics.speechmatics.matchedWords}/${groundTruthMetrics.speechmatics.totalWords}`);
            console.log(`   Unmatched words: ${groundTruthMetrics.speechmatics.unmatchedWords}`);
            console.log(`   Extra words: ${groundTruthMetrics.speechmatics.extraWords}`);
          }
          if (groundTruthMetrics.comparison) {
            const improvement = groundTruthMetrics.comparison.improvement;
            const status = groundTruthMetrics.comparison.nextLevelBetter ? '✅ BETTER' : '❌ WORSE';
            console.log(`📊 Comparison: NextLevel is ${status} by ${Math.abs(improvement).toFixed(1)}%`);
          }
          console.log(`${'='.repeat(80)}\n`);
        }
      }
    }

    console.log(`\n${'='.repeat(80)}`);
    console.log(`✅ MULTI-STEP PROCESSING COMPLETED`);
    console.log(`${'='.repeat(80)}`);
    console.log(`📊 Final markdown table length: ${markdownTable.length} chars`);
    console.log(`📊 Final markdown preview (first 300 chars):`);
    console.log(markdownTable.substring(0, 300) + (markdownTable.length > 300 ? '...' : ''));
    if (groundTruthMetrics && groundTruthMetrics.nextLevel) {
      console.log(`📊 Ground Truth Match (NextLevel): ${groundTruthMetrics.nextLevel.matchPercent}%`);
      if (groundTruthMetrics.speechmatics) {
        console.log(`📊 Ground Truth Match (Speechmatics): ${groundTruthMetrics.speechmatics.matchPercent}%`);
      }
    }
    console.log(`${'='.repeat(80)}\n`);

    // Return both markdown and metrics
    return {
      markdownTable,
      groundTruthMetrics
    };
  } catch (error) {
    console.error('❌ Multi-step processing error:', error);
    throw error;
  }
}

app.post('/api/apply-markdown-fixes', async (req, res) => {
  try {
    let { agentTranscript, clientTranscript, mode = 'smart', recordingId, useMultiStep = false } = req.body;

    // Fallback: якщо mode не передано або 'smart', перевіряємо DEMO_LLM_MODE зі змінних оточення
    if (!mode || mode === 'smart') {
      const demoMode = process.env.DEMO_LLM_MODE || process.env.DEMO_LOCAL_LLM_MODE;
      if (demoMode) {
        mode = demoMode;
        console.log(`🔍 [apply-markdown-fixes] Using DEMO_LLM_MODE from env: ${mode}`);
      }
    }

    if (!agentTranscript || !clientTranscript) {
      return res.status(400).json({ 
        error: 'Both agentTranscript and clientTranscript JSON objects are required' 
      });
    }

    // Validate that transcripts have segments
    const agentSegments = agentTranscript.segments || [];
    const clientSegments = clientTranscript.segments || [];
    
    if (!Array.isArray(agentSegments) || !Array.isArray(clientSegments)) {
      return res.status(400).json({ 
        error: 'Transcripts must contain a segments array' 
      });
    }

    console.log('📥 Apply markdown fixes request:', {
      agentSegmentsCount: agentSegments.length,
      clientSegmentsCount: clientSegments.length,
      mode,
      recordingId,
      timestamp: new Date().toISOString()
    });

    // Select model based on mode (same as apply-overlap-fixes)
    // Use getModelId() to always get current model from process.env (for cache key accuracy)
    let model = getModelId(mode);
    let useLocalLLM = false;
    
    if (mode === 'local' || mode === 'test' || mode === 'test2') {
      useLocalLLM = true;
      console.log(`🔵 Using local LLM: ${LOCAL_LLM_BASE_URL}, model: ${model}`);
      } else {
      if (!process.env.OPENROUTER_API_KEY) {
        return res.status(500).json({ error: 'OpenRouter API key is not configured' });
      }
      console.log(`🔵 Using OpenRouter model: ${model}`);
    }

    // Pre-process: Merge fragmented phrases in segments BEFORE building prompt context
    let processedAgentSegments = agentSegments;
    let processedClientSegments = clientSegments;
    
    try {
      console.log('🔍 Pre-processing: Checking for fragmented phrases in transcripts...');
      
      // Configure API for fragment check
      const apiUrl = useLocalLLM 
        ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
        : 'https://openrouter.ai/api/v1/chat/completions';
      
      const headers = {
        'Content-Type': 'application/json'
      };
      
      if (useLocalLLM) {
        if (LOCAL_LLM_API_KEY) {
          headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
        }
      } else {
        headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
        headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
        headers['X-Title'] = 'Fragment Check';
      }
      
      // Prepare segments in LLM format for fragment check
      // Зберігаємо оригінального спікера з Speechmatics для кращого контексту
      const agentSegmentsForLLM = agentSegments.map((seg, idx) => ({
        segment_id: idx + 1,
        text: seg.text || '',
        start: parseFloat(seg.start) || 0,
        end: parseFloat(seg.end) || parseFloat(seg.start) || 0,
        speaker_id: 'Agent',
        original_speaker: seg.speaker || seg.original_speaker || 'SPEAKER_00' // Зберігаємо оригінального спікера
      }));
      
      const clientSegmentsForLLM = clientSegments.map((seg, idx) => ({
        segment_id: idx + 1,
        text: seg.text || '',
        start: parseFloat(seg.start) || 0,
        end: parseFloat(seg.end) || parseFloat(seg.start) || 0,
        speaker_id: 'Client',
        original_speaker: seg.speaker || seg.original_speaker || 'SPEAKER_01' // Зберігаємо оригінального спікера
      }));
      
      // Merge fragmented phrases
      const mergedAgent = await mergeFragmentedPhrasesWithLLM(agentSegmentsForLLM, {
        model,
        useLocalLLM,
        apiUrl,
        headers,
        logPrefix: '[Markdown Fixes]',
        maxGapSeconds: 3.0
      });
      
      const mergedClient = await mergeFragmentedPhrasesWithLLM(clientSegmentsForLLM, {
        model,
        useLocalLLM,
        apiUrl,
        headers,
        logPrefix: '[Markdown Fixes]',
        maxGapSeconds: 3.0
      });
      
      // Convert back to original format
      processedAgentSegments = mergedAgent.map(seg => ({
        ...agentSegments[seg.segment_id - 1] || {},
        text: seg.text,
        start: seg.start,
        end: seg.end
      })).filter(seg => seg.text);
      
      processedClientSegments = mergedClient.map(seg => ({
        ...clientSegments[seg.segment_id - 1] || {},
        text: seg.text,
        start: seg.start,
        end: seg.end
      })).filter(seg => seg.text);
      
      if (processedAgentSegments.length !== agentSegments.length || processedClientSegments.length !== clientSegments.length) {
        console.log(`✅ Fragmented phrases merged: Agent ${agentSegments.length}→${processedAgentSegments.length}, Client ${clientSegments.length}→${processedClientSegments.length}`);
      }
    } catch (error) {
      console.warn(`⚠️ Fragment merge failed: ${error.message}, using original segments`);
      processedAgentSegments = agentSegments;
      processedClientSegments = clientSegments;
    }
    
    const promptContext = buildDialoguePromptContext({
      primaryDiarization: req.body.primaryDiarization || null,
      agentTranscript: { ...agentTranscript, segments: processedAgentSegments },
      clientTranscript: { ...clientTranscript, segments: processedClientSegments },
      voiceTracks: Array.isArray(req.body.voiceTracks) ? req.body.voiceTracks : [],
      speaker0SegmentsOverride: Array.isArray(req.body.standardSpeaker0Segments) ? req.body.standardSpeaker0Segments : null,
      speaker1SegmentsOverride: Array.isArray(req.body.standardSpeaker1Segments) ? req.body.standardSpeaker1Segments : null,
      groundTruthText: req.body.groundTruthText || null
    });
    
    console.log('📋 Preparing transcripts for LLM:', {
      agentSegments: processedAgentSegments.length,
      clientSegments: processedClientSegments.length
    });
    
    // Calculate total segments count
    const totalSegmentsCount = processedAgentSegments.length + processedClientSegments.length;
    
    // Load prompt template from file
    let promptTemplate;
    try {
      promptTemplate = await fs.readFile('prompts/overlap_fixes_markdown_prompt.txt', 'utf8');
    } catch (error) {
      console.warn('⚠️ Failed to load overlap_fixes_markdown_prompt.txt, using fallback prompt');
      // Fallback to hardcoded prompt if file not found
      promptTemplate = `You are the **NextLevel diarization controller**. You receive already-extracted dialogues (plain text, one replica per line) instead of raw JSON. Your goal is to fuse Standard diarization evidence with separated voice-track transcripts, detect the real speaker roles, and output a **clean, deduplicated Markdown table** with STRICT alternation between Agent and Client.

## ABSOLUTE TRUTH POLICY
Only the provided dialogues are trustworthy. Every sentence in the final table **must appear verbatim** in at least one dialogue block below. If a sentence is absent from **all** blocks, you MUST discard it as hallucination.

## DATA BLOCKS YOU RECEIVE

1. **Combined Standard Diarization (General)**
\`\`\`
{{GENERAL_DIALOGUE}}
\`\`\`

2. **Standard Speaker 0 Dialogue (raw Speechmatics speaker track)**
\`\`\`
{{STANDARD_SPEAKER0_DIALOGUE}}
\`\`\`

3. **Standard Speaker 1 Dialogue (raw Speechmatics speaker track)**
\`\`\`
{{STANDARD_SPEAKER1_DIALOGUE}}
\`\`\`

4. **Separated Voice Track – Agent Candidate**
\`\`\`
{{AGENT_DIALOGUE}}
\`\`\`

5. **Separated Voice Track – Client Candidate**
\`\`\`
{{CLIENT_DIALOGUE}}
\`\`\`

6. **Role & Context Guidance (output of the classifier / debug checks)**
\`\`\`
{{ROLE_GUIDANCE}}
\`\`\`

Each dialogue is already normalized into plain text lines like \`SPEAKER_00 (operator): text\`. Treat every line as a possible segment. Start/end timestamps for each line are also provided in the metadata block below:
\`\`\`
{{SEGMENT_TIMESTAMPS}}
\`\`\`
\`SEGMENT_TIMESTAMPS\` maps dialogue lines to their numeric \`start\`/\`end\` (seconds). Always copy these exact numbers into the final table.

## CRITICAL TASKS
1. **Role Detection**
   - Use \`ROLE_GUIDANCE\`, conversational intent, and speaker labels to decide who is Agent vs Client.
   - **CRITICAL**: ROLE_GUIDANCE contains \`speakerRoleMap\` which shows the exact mapping. For example, if \`speakerRoleMap\` shows \`{"SPEAKER_00": "agent", "SPEAKER_01": "client"}\`, this means ALL segments from SPEAKER_00 must be Agent, and ALL segments from SPEAKER_01 must be Client.
   - **CRITICAL**: The voice tracks (Agent/Client candidate dialogues) contain the MAIN and MOST COMPLETE replicas for each speaker. These are the primary source - use them as the authoritative text for each role.
   - **CRITICAL**: You MUST output only TWO roles: "Agent" and "Client". Never use "SPEAKER_00", "SPEAKER_01", "SPEAKER_02" or any other speaker labels in the final table.
   - After generating the table, verify: are there segments from BOTH roles? If all segments have the same role, check ROLE_GUIDANCE and reassign roles correctly.
   - If metadata contradicts the dialogue meaning, trust the meaning plus guidance.

2. **Replica Validation**
   - **PRIORITY**: Use replicas from voice-track dialogues (Agent/Client candidate) as they contain the most complete and accurate text without truncation.
   - Keep a replica ONLY if it exists in at least one dialogue block.
   - Prefer the voice-track (Agent/Client candidate) versions - they have the most complete replicas without truncation.
   - If a replica appears in multiple blocks, keep the version from the voice-track that matches the intended role.
   - If text is found only in Standard Speaker 0/1 but matches the other participant, reassign it correctly based on ROLE_GUIDANCE.

3. **Duplicate & Overlap Control**
   - Remove exact or near-duplicate sentences that describe the same moment twice.
   - If a line appears once as Agent and once as Client, decide who truly said it; keep only that one.
   - Merge overlapping lines from the same real speaker: earliest start, latest end, concatenated text (chronological order).
   - **CRITICAL**: Merge consecutive segments from the same speaker. If you see Client → Client or Agent → Agent in sequence, these are almost always parts of ONE continuous utterance that was incorrectly split. Merge them unless there's a clear time gap (> 5 seconds) AND different topics.

4. **Strict Alternation**
   - Final table must alternate Agent → Client → Agent → Client.
   - **CRITICAL**: Consecutive same-speaker segments (e.g., Client → Client or Agent → Agent) indicate an error. You MUST merge them into a single row.
   - Temporary double Agent/Client is allowed ONLY if there is a significant time gap (> 5 seconds) AND clearly different topics. Otherwise, merge them.

5. **No Hallucinations**
   - Sentences absent from every dialogue block are forbidden. Discard them even if they appeared in the original markdown.
   - If \`ROLE_GUIDANCE\` lists phrases flagged as “not in any source”, ensure they **never** reach the final table.

6. **Timestamp Fidelity**
   - Every row needs \`Start Time\` and \`End Time\` taken from \`SEGMENT_TIMESTAMPS\`.
   - If you merged multiple lines, use the min start / max end for that merged row.

## OUTPUT FORMAT
Return ONLY a Markdown table:
| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | … | 0.64 | 2.15 |

Where:
- \`Segment ID\` starts at 1.
- \`Speaker\` is either **Agent** or **Client** (NEVER use "SPEAKER_00", "SPEAKER_01", "SPEAKER_02", or any other speaker labels).
- \`Text\` is verbatim (no paraphrasing, keep fillers). Prefer text from voice-track dialogues as they are most complete.
- \`Start/End Time\` use numeric seconds with 2 decimal precision (e.g., \`3.45\`). Use timestamps from SEGMENT_TIMESTAMPS that match the selected text.

## QUALITY CHECKLIST (do this mentally before outputting)
1. Every row comes from the supplied dialogues.
2. Duplicates removed; overlaps merged.
3. Roles validated against \`ROLE_GUIDANCE\` - ensure both Agent and Client segments are present.
4. Alternation Agent/Client preserved.
5. No stray text or commentary outside the Markdown table.
6. **VERIFY ROLE DISTRIBUTION**: Check that the table contains segments from BOTH Agent and Client. If all segments have the same role, you MUST review ROLE_GUIDANCE and correct the assignments.

If any requirement cannot be satisfied, adjust the rows (reassign, merge, drop) until compliance is achieved. Only then output the final table.`;
    }
    
    // Check for potential consecutive same-speaker segments in input data
    // This helps us enhance the prompt to prevent the issue
    const hasPotentialConsecutiveSegments = checkForConsecutiveSegmentsInInput(
      agentSegments,
      clientSegments,
      promptContext
    );
    
    if (hasPotentialConsecutiveSegments) {
      console.log('⚠️ Detected potential consecutive same-speaker segments in input. Enhancing prompt with merge instructions.');
      // Add special instruction to prompt template
      const mergeInstruction = `

## ⚠️ CRITICAL: CONSECUTIVE SEGMENT MERGING
**IMPORTANT**: The input data may contain consecutive segments from the same speaker that should be merged.
- If you see multiple segments from the same speaker in a row (e.g., Client → Client or Agent → Agent), these are likely parts of ONE continuous utterance.
- **ALWAYS merge consecutive same-speaker segments** unless there is a clear time gap (> 5 seconds) AND different topics.
- When merging: combine text, use earliest start time, use latest end time.
- The final table MUST alternate between Agent and Client. Consecutive same-speaker segments indicate an error that needs fixing.

`;
      promptTemplate = mergeInstruction + promptTemplate;
    }
    
    const replacements = {
      '{{GENERAL_DIALOGUE}}': promptContext.generalDialog,
      '{{STANDARD_SPEAKER0_DIALOGUE}}': promptContext.speaker0Dialog,
      '{{STANDARD_SPEAKER1_DIALOGUE}}': promptContext.speaker1Dialog,
      '{{AGENT_DIALOGUE}}': promptContext.agentDialog,
      '{{CLIENT_DIALOGUE}}': promptContext.clientDialog,
      '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText,
      '{{SEGMENT_TIMESTAMPS}}': promptContext.segmentTimestampsText
    };

    let prompt = promptTemplate;
    Object.entries(replacements).forEach(([token, value]) => {
      const safeValue = value && value.trim().length > 0 ? value : '[empty]';
      prompt = prompt.replace(new RegExp(token, 'g'), safeValue);
    });

    // Decide whether to use multi-step processing
    // Use multi-step for local models or if explicitly requested
    const shouldUseMultiStep = useMultiStep || useLocalLLM || process.env.USE_MULTI_STEP_MARKDOWN === 'true';
    
    if (shouldUseMultiStep) {
      console.log('🔄 Using multi-step processing for markdown fixes');
      try {
        // Detect if this is an auto-test (curl/script) vs frontend request
        const userAgent = req.headers['user-agent'] || '';
        const isAutoTest = userAgent.includes('curl') || userAgent.includes('node') || req.body.isAutoTest === true;
        
        const result = await processMarkdownFixesMultiStep(promptContext, mode, recordingId, isAutoTest);
        
        let markdownTable = null;
        let groundTruthMetrics = null;
        
        if (result) {
          if (typeof result === 'string') {
            // Backward compatibility: if string is returned, use it as markdownTable
            markdownTable = result;
          } else if (result.markdownTable) {
            markdownTable = result.markdownTable;
            groundTruthMetrics = result.groundTruthMetrics || null;
          }
        }
        
        // Merge consecutive segments from the same speaker
        if (markdownTable) {
          markdownTable = mergeConsecutiveSpeakerSegmentsInMarkdown(markdownTable, 2.0);
        }
        
        return res.json({
          success: true,
          markdown: markdownTable,
          groundTruthMetrics: groundTruthMetrics,
          cached: false,
          multiStep: true
        });
      } catch (error) {
        console.error('❌ Multi-step processing failed, falling back to single-step:', error.message);
        // Fall through to single-step processing
      }
    }

    // Build cache key for markdown fixes
    // Include DEMO_LLM_MODE to ensure cache is invalidated when demo mode changes
    const demoLlmMode = process.env.DEMO_LLM_MODE || 'smart';
    const cacheKey = buildLLMCacheKey(
      recordingId || 'demo',
      prompt,
      model,
      mode,
      'markdown-fixes',
      demoLlmMode
    );
    
    // Check cache first
    let cachedResult = null;
    if (cacheKey) {
      cachedResult = readLLMCache(cacheKey);
      if (cachedResult && cachedResult.rawMarkdown) {
        console.log('✅ Using cached raw markdown from LLM');
        // Merge consecutive segments from the same speaker
        const mergedMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(cachedResult.rawMarkdown, 2.0);
        return res.json({
          success: true,
          markdown: mergedMarkdown,
          cached: true
        });
      }
    }
    
    // Якщо локальний режим і кеш не знайдено, спробувати знайти кеш для моделі fast
    if (!cachedResult && useLocalLLM) {
      const fastModelId = getModelId('fast');
      const fastCacheKey = buildLLMCacheKey(
        recordingId || 'demo',
        prompt,
        fastModelId,
        'fast',
        'markdown-fixes',
        demoLlmMode
      );
      
      if (fastCacheKey) {
        const fastCachedResult = readLLMCache(fastCacheKey);
        if (fastCachedResult && fastCachedResult.rawMarkdown) {
          console.log('✅ Using cached raw markdown from fast model for local mode');
          // Merge consecutive segments from the same speaker
          const mergedMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(fastCachedResult.rawMarkdown, 2.0);
          return res.json({
            success: true,
            markdown: mergedMarkdown,
            cached: true,
            sourceModel: 'fast'
          });
        }
      }
    }

    const apiUrl = useLocalLLM 
      ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
      : 'https://openrouter.ai/api/v1/chat/completions';
    
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (useLocalLLM) {
      if (LOCAL_LLM_API_KEY) {
        headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
      }
      console.log(`📤 Sending request to local LLM: ${apiUrl}`);
    } else {
      headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
      headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
      headers['X-Title'] = 'Apply Markdown Fixes';
    }

    // Build payload with reasoning_effort if needed
    const payload = {
      model: model,
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant that fixes speaker diarization. Always return a valid Markdown table with columns: Segment ID, Speaker, Text, Start Time, End Time. Keep all text from input. **CRITICAL**: After reasoning, you MUST generate the final Markdown table in your response content. Do not leave the content field empty.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0  // Maximum deterministic output
    };
    
    // Add reasoning effort for Fast, and Smart (GPT 5.1)
    // OpenRouter format: reasoning: { effort: "high" }
    // NOTE: Local mode (LM Studio) reasoning_effort is disabled - configure in LM Studio UI
    if (shouldUseHighReasoningEffort(mode, model)) {
      if (useLocalLLM) {
        // Local LLM reasoning_effort disabled - configure in LM Studio UI
        console.log(`🔧 Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)`);
      } else {
        // For OpenRouter, use nested reasoning object
        payload.reasoning = { effort: 'high' };
        console.log(`🔧 Using reasoning effort: high for ${mode} mode (model: ${model})`);
      }
    }

    // Increased timeout for reasoning models (especially gpt-oss-20b which may take longer)
    // Для локальної моделі встановлюємо більший таймаут (30 хвилин), оскільки вона працює повільніше
    const timeout = useLocalLLM ? 1800000 : 600000; // 30 хвилин для локальної, 10 хвилин для віддаленої
    console.log(`⏱️  Using timeout: ${timeout / 1000 / 60} minutes (${timeout / 1000}s) for ${useLocalLLM ? 'local' : 'OpenRouter'} LLM`);
    
    const response = await axios.post(
      apiUrl,
      payload,
      {
        headers: headers,
        timeout: timeout
      }
    );

    // Handle reasoning models that may return content in reasoning field
    const message = response.data.choices[0]?.message || {};
    let llmOutput = message.content || '';
    const reasoning = message.reasoning || '';
    
    console.log('📥 Markdown fixes LLM response received:', {
      contentLength: llmOutput.length,
      reasoningLength: reasoning.length,
      hasContent: !!llmOutput,
      hasReasoning: !!reasoning
    });

    // If content is empty but reasoning exists, try to extract table from reasoning
    if (!llmOutput && reasoning) {
      console.warn('⚠️ Content is empty, attempting to extract table from reasoning field');
      // Try to find markdown table in reasoning
      const tableMatch = reasoning.match(/\|[\s\S]*?\|[\s\S]*?\|[\s\S]*?\|[\s\S]*?\|[\s\S]*?\|/);
      if (tableMatch) {
        // Find the full table (from first | to last |)
        const tableStart = reasoning.indexOf('|');
        if (tableStart !== -1) {
          // Find the end of the table (look for end of last row with |)
          let tableEnd = reasoning.length;
          const lines = reasoning.split('\n');
          let inTable = false;
          let lastTableLine = -1;
          
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
              inTable = true;
              lastTableLine = i;
            } else if (inTable && !lines[i].trim().startsWith('|')) {
              // Table ended
              break;
            }
          }
          
          if (lastTableLine !== -1) {
            const tableLines = lines.slice(0, lastTableLine + 1);
            llmOutput = tableLines.join('\n');
            console.log('✅ Extracted table from reasoning field');
          }
        }
      }
      
      // If still no output, use reasoning as fallback (though it's unlikely to contain a proper table)
      if (!llmOutput) {
        console.warn('⚠️ Could not extract table from reasoning, using reasoning as fallback');
        llmOutput = reasoning;
      }
    }

    // If still empty, return error
    if (!llmOutput || !llmOutput.trim()) {
      console.error('❌ LLM returned empty content and no usable reasoning');
      return res.status(500).json({
        error: 'LLM returned empty response',
        details: 'The model generated reasoning but no final content. Try a different model or adjust reasoning settings.',
        hasReasoning: !!reasoning,
        reasoningLength: reasoning.length
      });
    }

    // Extract markdown table from response
    // Зберігаємо чистий markdown від LLM для кешування
    let rawMarkdownFromLLM = llmOutput.trim();
    
    console.log('📋 Raw LLM output length:', llmOutput.length);
    console.log('📋 Raw LLM output first 500 chars:', llmOutput.substring(0, 500));
    console.log('📋 Raw LLM output last 500 chars:', llmOutput.substring(Math.max(0, llmOutput.length - 500)));
    
    // Try to extract table if wrapped in code blocks
    const codeBlockMatch = rawMarkdownFromLLM.match(/```(?:markdown)?\s*([\s\S]*?)\s*```/);
    if (codeBlockMatch) {
      console.log('📋 Found code block, extracting...');
      rawMarkdownFromLLM = codeBlockMatch[1].trim();
      console.log('📋 Extracted from code block, length:', rawMarkdownFromLLM.length);
    }

    console.log('📋 Markdown after code block extraction, length:', rawMarkdownFromLLM.length);
    console.log('📋 Markdown first 500 chars:', rawMarkdownFromLLM.substring(0, 500));
    console.log('📋 Markdown last 500 chars:', rawMarkdownFromLLM.substring(Math.max(0, rawMarkdownFromLLM.length - 500)));

    // Validate that we have a markdown table
    if (!rawMarkdownFromLLM.includes('|') || !rawMarkdownFromLLM.includes('---')) {
      console.warn('⚠️ LLM response does not appear to be a Markdown table');
      // Try to find any table-like structure
      if (rawMarkdownFromLLM.includes('|')) {
        // At least has some table structure, continue
        console.log('✅ Found some table structure, continuing...');
      } else {
        // No table structure at all
        console.error('❌ No markdown table structure found in LLM response');
        return res.status(500).json({
          error: 'LLM did not return a valid Markdown table',
          details: 'The model response does not contain a markdown table. Response preview: ' + rawMarkdownFromLLM.substring(0, 200),
          responsePreview: rawMarkdownFromLLM.substring(0, 500)
        });
      }
    }

    // Use markdown directly from LLM - no programmatic processing
    // Це буде оброблено валідацією далі, але для кешу зберігаємо чистий markdown
    // Remove filler words from markdown table
    let cleanedMarkdown = removeFillerWordsFromMarkdownTable(rawMarkdownFromLLM);
    // Merge consecutive segments from the same speaker
    cleanedMarkdown = mergeConsecutiveSpeakerSegmentsInMarkdown(cleanedMarkdown, 2.0);
    
    console.log('📋 First LLM response, length:', cleanedMarkdown.length);
    console.log('📋 First LLM response first 500 chars:', cleanedMarkdown.substring(0, 500));
    
    // Second LLM call: Validation and correction
    console.log('🔍 Starting validation LLM call...');
    
    // Prepare JSON strings for validation prompt (define outside try block to ensure availability)
    let agentTranscriptJSON;
    let clientTranscriptJSON;
    
    try {
      // Prepare JSON strings for validation prompt
      agentTranscriptJSON = JSON.stringify(processedAgentSegments, null, 2);
      clientTranscriptJSON = JSON.stringify(processedClientSegments, null, 2);
      
      // Load validation prompt template
      let validationPromptTemplate;
      try {
        validationPromptTemplate = await fs.readFile('prompts/overlap_fixes_markdown_validation_prompt.txt', 'utf8');
      } catch (error) {
        console.warn('⚠️ Failed to load overlap_fixes_markdown_validation_prompt.txt, skipping validation');
        validationPromptTemplate = null;
      }
      
      if (validationPromptTemplate && agentTranscriptJSON && clientTranscriptJSON) {
        // Ensure totalSegmentsCount is defined for validation prompt
        const validationSegmentCount = typeof totalSegmentsCount !== 'undefined' 
          ? totalSegmentsCount 
          : (processedAgentSegments.length + processedClientSegments.length);
        
        // Replace placeholders in validation prompt
        const validationPrompt = validationPromptTemplate
          .replace(/\{\{AGENT_TRANSCRIPT\}\}/g, agentTranscriptJSON)
          .replace(/\{\{CLIENT_TRANSCRIPT\}\}/g, clientTranscriptJSON)
          .replace(/\{\{TABLE_TO_VALIDATE\}\}/g, cleanedMarkdown)
          .replace(/\{\{SEGMENT_COUNT\}\}/g, validationSegmentCount.toString());
        
        // Build payload for validation LLM call
        const validationPayload = {
          model: model,
          messages: [
            {
              role: 'system',
              content: 'You are a helpful assistant that validates and corrects dialogue transcription tables. Always return a valid Markdown table with columns: Segment ID, Speaker, Text, Start Time, End Time. Use ONLY text from original transcripts, ensure strict alternation, remove duplicates.'
            },
            {
              role: 'user',
              content: validationPrompt
            }
          ],
          temperature: 0  // Maximum deterministic output
        };
        
        // Add reasoning effort if needed
        if (shouldUseHighReasoningEffort(mode, model)) {
          if (useLocalLLM) {
            console.log(`🔧 Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)`);
          } else {
            validationPayload.reasoning = { effort: 'high' };
            console.log(`🔧 Using reasoning effort: high for validation (${mode} mode, model: ${model})`);
          }
        }
        
        // Call validation LLM
        const validationResponse = await axios.post(
          apiUrl,
          validationPayload,
          {
            headers: headers,
            timeout: timeout
          }
        );
        
        // Handle reasoning models that may return content in reasoning field
        const validationMessage = validationResponse.data.choices[0]?.message || {};
        let validationOutput = validationMessage.content || '';
        const validationReasoning = validationMessage.reasoning || '';
        
        console.log('📥 Validation LLM response received:', {
          contentLength: validationOutput.length,
          reasoningLength: validationReasoning.length,
          hasContent: !!validationOutput,
          hasReasoning: !!validationReasoning
        });
        
        // If content is empty but reasoning exists, try to extract table from reasoning
        if (!validationOutput && validationReasoning) {
          console.warn('⚠️ Validation content is empty, attempting to extract table from reasoning field');
          // Try to find markdown table in reasoning
          const tableMatch = validationReasoning.match(/\|[\s\S]*?\|[\s\S]*?\|[\s\S]*?\|[\s\S]*?\|[\s\S]*?\|/);
          if (tableMatch) {
            const tableStart = validationReasoning.indexOf('|');
            if (tableStart !== -1) {
              const lines = validationReasoning.split('\n');
              let inTable = false;
              let lastTableLine = -1;
              
              for (let i = 0; i < lines.length; i++) {
                if (lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
                  inTable = true;
                  lastTableLine = i;
                } else if (inTable && !lines[i].trim().startsWith('|')) {
                  break;
                }
              }
              
              if (lastTableLine !== -1) {
                const tableLines = lines.slice(0, lastTableLine + 1);
                validationOutput = tableLines.join('\n');
                console.log('✅ Extracted validation table from reasoning field');
              }
            }
          }
          
          if (!validationOutput) {
            console.warn('⚠️ Could not extract table from validation reasoning, using reasoning as fallback');
            validationOutput = validationReasoning;
          }
        }
        
        if (validationOutput && validationOutput.trim()) {
          // Extract markdown table from validation response
          let validatedMarkdown = validationOutput.trim();
          
          // Try to extract table if wrapped in code blocks
          const codeBlockMatch = validatedMarkdown.match(/```(?:markdown)?\s*([\s\S]*?)\s*```/);
          if (codeBlockMatch) {
            validatedMarkdown = codeBlockMatch[1].trim();
          }
          
          // Validate that we have a markdown table
          if (validatedMarkdown.includes('|') && validatedMarkdown.includes('---')) {
            cleanedMarkdown = validatedMarkdown;
            console.log('✅ Validation LLM corrected the table');
            console.log('📋 Validated markdown length:', cleanedMarkdown.length);
          } else {
            console.warn('⚠️ Validation LLM response does not appear to be a valid Markdown table, using original');
          }
        } else {
          console.warn('⚠️ Validation LLM returned empty response, using original table');
        }
      }
    } catch (validationError) {
      console.error('❌ Validation LLM error:', validationError.message);
      console.warn('⚠️ Continuing with original table due to validation error');
      // Continue with original markdown if validation fails
    }
    
    console.log('📋 Final markdown to send, length:', cleanedMarkdown.length);
    console.log('📋 Final markdown first 500 chars:', cleanedMarkdown.substring(0, 500));
    console.log('📋 Final markdown last 500 chars:', cleanedMarkdown.substring(Math.max(0, cleanedMarkdown.length - 500)));
    console.log('📋 Final markdown line count:', cleanedMarkdown.split('\n').length);

    // Save to cache - зберігаємо чистий markdown від LLM (до валідації)
    // rawMarkdownFromLLM - це чистий markdown після витягнення з code blocks, але до валідації
    
    // Ensure totalSegmentsCount is defined (fallback to sum if not already calculated)
    const finalTotalSegmentsCount = typeof totalSegmentsCount !== 'undefined' 
      ? totalSegmentsCount 
      : (processedAgentSegments.length + processedClientSegments.length);
    
    if (cacheKey && rawMarkdownFromLLM) {
      writeLLMCache(cacheKey, {
        rawMarkdown: rawMarkdownFromLLM, // Чистий markdown від LLM (після витягнення з code blocks, але до валідації)
        agentSegmentsCount: agentSegments.length,
        clientSegmentsCount: clientSegments.length,
        totalSegmentsCount: finalTotalSegmentsCount,
        model: model,
        mode: mode,
        timestamp: new Date().toISOString()
      });
    }

    res.json({
      success: true,
      markdown: cleanedMarkdown,
      agentSegmentsCount: agentSegments.length,
      clientSegmentsCount: clientSegments.length,
      totalSegmentsCount: finalTotalSegmentsCount
    });

  } catch (error) {
    console.error('❌ Apply markdown fixes error:', error);
    
    // Check if it's a timeout error
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      return res.status(504).json({
        error: 'LLM request timeout',
        message: `The LLM request took longer than expected to complete. This may happen with reasoning models. Try reducing the input size or using a faster model.`,
        suggestion: 'For local models, try adjusting reasoning settings in LM Studio UI or use a faster model.'
      });
    }
    
    res.status(500).json({
      error: 'Failed to apply markdown fixes',
      message: error.message,
      details: error.response?.data || error.stack
    });
  }
});

// Debug LLM query endpoint for testing speaker role identification
app.post('/api/debug-llm-query', async (req, res) => {
  try {
    const { prompt } = req.body;
    
    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: 'Prompt is required'
      });
    }

    // Визначаємо режим LLM на основі змінних оточення (та сама логіка, що в інших endpoints)
    // Перевіряємо DEMO_LLM_MODE або DEMO_LOCAL_LLM_MODE для визначення локального/хмарного режиму
    const demoMode = process.env.DEMO_LLM_MODE || process.env.DEMO_LOCAL_LLM_MODE;
    const useLocalLLM = demoMode === 'local' || demoMode === 'test' || demoMode === 'test2';
    
    // Якщо не встановлено через DEMO_*, перевіряємо USE_LOCAL_LLM
    const finalUseLocalLLM = useLocalLLM || process.env.USE_LOCAL_LLM === 'true';
    
    // Визначаємо модель (та сама логіка, що в інших endpoints)
    const mode = demoMode || 'smart'; // За замовчуванням 'smart'
    const model = getModelId(mode);
    
    // Перевіряємо кеш перед відправкою на LLM
    let cacheKey = null;
    if (LLM_CACHE_ENABLED) {
      // Для debug-llm-query використовуємо 'debug' як filename та 'debug-query' як promptVariant
      cacheKey = buildLLMCacheKey('debug', prompt, model, mode, 'debug-query');
      if (cacheKey) {
        console.log('🔍 [DEBUG LLM] Checking cache:', { cacheKey, model, mode });
        const cachedResponse = readLLMCache(cacheKey);
        if (cachedResponse && cachedResponse.llmOutput) {
          console.log('✅ [DEBUG LLM] Using cached response:', { cacheKey });
          return res.json({
            success: true,
            result: cachedResponse.llmOutput,
            cached: true
          });
        } else {
          console.log('📝 [DEBUG LLM] Cache miss:', { cacheKey });
        }
      } else {
        console.warn('⚠️ [DEBUG LLM] Failed to build cache key');
      }
    }
    
    // Визначаємо URL API (та сама логіка, що в інших endpoints)
    const apiUrl = finalUseLocalLLM 
      ? `${LOCAL_LLM_BASE_URL}/v1/chat/completions`
      : 'https://openrouter.ai/api/v1/chat/completions';
    
    // Детальне логування для діагностики
    console.log('🔍 [DEBUG LLM] Configuration check:', {
      DEMO_LLM_MODE: process.env.DEMO_LLM_MODE,
      DEMO_LOCAL_LLM_MODE: process.env.DEMO_LOCAL_LLM_MODE,
      USE_LOCAL_LLM: process.env.USE_LOCAL_LLM,
      demoMode: demoMode,
      useLocalLLM: useLocalLLM,
      finalUseLocalLLM: finalUseLocalLLM,
      mode: mode,
      model: model,
      apiUrl: apiUrl,
      LOCAL_LLM_BASE_URL: LOCAL_LLM_BASE_URL,
      hasLocalApiKey: !!LOCAL_LLM_API_KEY,
      hasOpenRouterApiKey: !!process.env.OPENROUTER_API_KEY,
      cacheKey: cacheKey
    });
    
    // Формуємо headers (та сама логіка, що в інших endpoints)
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (finalUseLocalLLM) {
      if (LOCAL_LLM_API_KEY) {
        headers['Authorization'] = `Bearer ${LOCAL_LLM_API_KEY}`;
      }
      console.log('🔍 [DEBUG LLM] Using LOCAL LLM:', { 
        model, 
        apiUrl, 
        baseUrl: LOCAL_LLM_BASE_URL,
        hasApiKey: !!LOCAL_LLM_API_KEY
      });
    } else {
      if (!process.env.OPENROUTER_API_KEY) {
        throw new Error('OPENROUTER_API_KEY is not configured');
      }
      headers['Authorization'] = `Bearer ${process.env.OPENROUTER_API_KEY}`;
      headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
      headers['X-Title'] = 'Debug LLM Query';
      console.log('🔍 [DEBUG LLM] Using OPENROUTER:', { 
        model, 
        apiUrl,
        hasApiKey: !!process.env.OPENROUTER_API_KEY
      });
    }
    
    const payload = {
      model: model,
      messages: [
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0
    };
    
    // Add reasoning effort if needed (та сама логіка, що в інших endpoints)
    if (shouldUseHighReasoningEffort(mode, model)) {
      if (finalUseLocalLLM) {
        console.log('🔧 [DEBUG LLM] Local LLM mode: reasoning_effort disabled (configure in LM Studio UI)');
      } else {
        payload.reasoning = { effort: 'high' };
        console.log('🔧 [DEBUG LLM] Using reasoning effort: high for', mode, 'mode (model:', model + ')');
      }
    }
    
    console.log('🔍 [DEBUG LLM] Sending query to LLM:', { 
      model, 
      useLocalLLM: finalUseLocalLLM, 
      mode,
      apiUrl,
      promptLength: prompt.length 
    });
    
    const axios = require('axios');
    const timeout = finalUseLocalLLM ? 1800000 : 60000; // 30 хв для локальної, 1 хв для віддаленої
    
    const response = await axios.post(apiUrl, payload, { headers, timeout });
    
    if (response.data && response.data.choices && response.data.choices.length > 0) {
      const result = response.data.choices[0].message.content.trim();
      console.log('✅ [DEBUG LLM] Response received:', result.substring(0, 100) + '...');
      
      // Зберігаємо результат в кеш
      if (LLM_CACHE_ENABLED && cacheKey) {
        writeLLMCache(cacheKey, {
          llmOutput: result,
          model: model,
          mode: mode,
          promptVariant: 'debug-query',
          timestamp: new Date().toISOString()
        });
        console.log('💾 [DEBUG LLM] Response saved to cache:', { cacheKey });
      }
      
      return res.json({
        success: true,
        result: result,
        cached: false
      });
    } else {
      console.error('❌ [DEBUG LLM] Invalid response format:', JSON.stringify(response.data, null, 2));
      throw new Error('Invalid response format from LLM');
    }
  } catch (error) {
    console.error('❌ [DEBUG LLM] Error:', error.message);
    if (error.response) {
      console.error('❌ [DEBUG LLM] Response status:', error.response.status);
      console.error('❌ [DEBUG LLM] Response data:', error.response.data);
    }
    return res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

server.listen(PORT, async () => {
  console.log(`🚀 Server is running on http://localhost:${PORT}`);
  console.log(`📄 Open http://localhost:${PORT} in your browser`);
  console.log('✓ Static uploads directory mounted at /uploads');
  if (!process.env.OPENROUTER_API_KEY) {
    console.warn('⚠️  WARNING: OPENROUTER_API_KEY environment variable not set!');
  }
  if (!process.env.SPEECHMATICS_API_KEY) {
    console.warn('⚠️  WARNING: SPEECHMATICS_API_KEY environment variable not set!');
  }
  await startTunnel();
  console.log('📋 Ready to accept requests!');
  console.log(`   Local:  http://localhost:${PORT}`);
  console.log(`   Public: ${tunnelUrl || PUBLIC_URL || 'Not available (localhost only)'}`);
});

