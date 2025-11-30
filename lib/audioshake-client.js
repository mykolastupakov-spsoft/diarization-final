const API_BASE_URL = 'https://api.audioshake.ai';
const DEFAULT_MODEL = 'multi_voice';
const DEFAULT_FORMATS = ['wav'];
const DEFAULT_VARIANT = 'n_speaker';
const DEFAULT_LANGUAGE = 'en';

function getApiKey() {
  const key = process.env.AUDIOSHAKE_API_KEY;
  if (!key) {
    throw new Error('AudioShake API key is not configured. Set AUDIOSHAKE_API_KEY in your environment.');
  }
  return key;
}

async function httpRequest(endpoint, options = {}) {
  if (typeof fetch !== 'function') {
    throw new Error('Global fetch is not available. AudioShake integration requires Node.js 18 or newer.');
  }
  
  // Check if AbortController is available (Node.js 15+)
  if (typeof AbortController === 'undefined') {
    throw new Error('AbortController is not available. AudioShake integration requires Node.js 15 or newer.');
  }
  
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = options.headers || {};
  
  // Set timeout (default 60 seconds, or from options)
  const timeoutMs = options.timeout || 60000;
  const abortController = new AbortController();
  const timeoutId = setTimeout(() => {
    abortController.abort();
  }, timeoutMs);
  
  const requestInit = {
    ...options,
    signal: abortController.signal,
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': getApiKey(),
      ...headers,
    },
  };

  const callId = `http_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const method = requestInit.method || 'GET';
  console.log(`[${callId}] 🌐 AudioShake: ${method} ${url} (timeout: ${timeoutMs}ms)`);
  console.log(`[${callId}] 🌐 AudioShake: Headers:`, Object.keys(requestInit.headers));
  if (options.body) {
    console.log(`[${callId}] 🌐 AudioShake: Body length: ${options.body.length} chars`);
    console.log(`[${callId}] 🌐 AudioShake: Body preview:`, options.body.substring(0, 200));
  }
  
  const startedAt = Date.now();
  console.log(`[${callId}] 🌐 AudioShake: Sending request...`);

  try {
    const response = await fetch(url, requestInit);
    clearTimeout(timeoutId);
    const durationMs = Date.now() - startedAt;
    const durationSec = (durationMs / 1000).toFixed(2);
    console.log(`[${callId}] ⏱️ AudioShake: Response received in ${durationSec}s (status ${response.status})`);

    if (!response.ok) {
      let errorBody = '';
      let errorData = null;
      try {
        errorBody = await response.text();
        console.error(`[${callId}] ❌ AudioShake: Error response body:`, errorBody.substring(0, 500));
        
        // Try to parse as JSON
        try {
          errorData = JSON.parse(errorBody);
          console.error(`[${callId}] ❌ AudioShake: Parsed error data:`, errorData);
        } catch (parseErr) {
          // Not JSON, use as text
          errorData = { message: errorBody };
        }
      } catch (err) {
        errorBody = `<failed to read body: ${err.message}>`;
        errorData = { message: errorBody };
        console.error(`[${callId}] ❌ AudioShake: Failed to read error body:`, err.message);
      }

      // Check for specific error types
      let message = errorBody || `AudioShake API returned ${response.status}`;
      
      if (errorData) {
        // Check for insufficient credits
        if (errorData.message && errorData.message.includes('Insufficient credits')) {
          message = 'Insufficient credits in speaker separation service account. Please add credits to continue.';
        } else if (errorData.message) {
          message = errorData.message;
        } else if (errorData.error && errorData.message) {
          message = `${errorData.error}: ${errorData.message}`;
        }
      }
      
      const error = new Error(message);
      error.status = response.status;
      error.statusCode = errorData?.statusCode || response.status;
      error.body = errorBody;
      error.errorData = errorData;
      throw error;
    }

    // Some endpoints may return empty body on success
    if (response.status === 204) {
      console.log(`[${callId}] ✅ AudioShake: 204 No Content - returning null`);
      return null;
    }

    console.log(`[${callId}] 🌐 AudioShake: Parsing JSON response...`);
    const parseStart = Date.now();
    const jsonData = await response.json();
    const parseDuration = ((Date.now() - parseStart) / 1000).toFixed(2);
    console.log(`[${callId}] ✅ AudioShake: JSON parsed successfully in ${parseDuration}s`);
    console.log(`[${callId}] ✅ AudioShake: Response data keys:`, Object.keys(jsonData));
    
    const totalDuration = ((Date.now() - startedAt) / 1000).toFixed(2);
    console.log(`[${callId}] ✅ AudioShake: Total request time: ${totalDuration}s`);
    
    return jsonData;
  } catch (error) {
    clearTimeout(timeoutId);
    const durationMs = Date.now() - startedAt;
    const durationSec = (durationMs / 1000).toFixed(2);
    
    // Check if it's a timeout/abort error
    if (error.name === 'AbortError' || error.message.includes('aborted')) {
      const timeoutError = new Error(`AudioShake API request timed out after ${timeoutMs}ms (${durationSec}s)`);
      timeoutError.name = 'TimeoutError';
      timeoutError.status = 408;
      console.error(`[${callId}] ⏱️ AudioShake: Request timed out after ${durationSec}s`);
      throw timeoutError;
    }
    
    console.error(`[${callId}] ❌ AudioShake: Request failed after ${durationSec}s:`, error);
    console.error(`[${callId}] ❌ AudioShake: Error type:`, error.constructor.name);
    console.error(`[${callId}] ❌ AudioShake: Error message:`, error.message);
    if (error.stack) {
      console.error(`[${callId}] ❌ AudioShake: Error stack:`, error.stack.substring(0, 500));
    }
    throw error;
  }
}

function validateHttpsUrl(audioUrl) {
  try {
    const parsed = new URL(audioUrl);
    if (parsed.protocol !== 'https:') {
      throw new Error('AudioShake requires audioUrl to be an HTTPS URL.');
    }
    return parsed.toString();
  } catch (error) {
    throw new Error(`Invalid audio URL provided: ${error.message}`);
  }
}

async function createTask(audioUrl, options = {}) {
  const callId = `createTask_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  console.log(`[${callId}] 🆔 AudioShake: Starting createTask`);
  console.log(`[${callId}] 🆔 AudioShake: Audio URL: ${audioUrl.substring(0, 80)}...`);
  console.log(`[${callId}] 🆔 AudioShake: Options:`, {
    model: options.model || DEFAULT_MODEL,
    language: options.language || DEFAULT_LANGUAGE,
    variant: options.variant || DEFAULT_VARIANT
  });
  
  const startTime = Date.now();
  try {
    console.log(`[${callId}] 🆔 AudioShake: Validating URL...`);
    const validatedUrl = validateHttpsUrl(audioUrl);
    console.log(`[${callId}] 🆔 AudioShake: URL validated: ${validatedUrl.substring(0, 80)}...`);
    
    const payload = {
      url: validatedUrl,
      targets: [
        {
          model: options.model || DEFAULT_MODEL,
          formats: options.formats || DEFAULT_FORMATS,
          variant: options.variant || DEFAULT_VARIANT,
          language: options.language || DEFAULT_LANGUAGE,
        },
      ],
    };

    console.log(`[${callId}] 🆔 AudioShake: Sending POST request to /tasks...`);
    console.log(`[${callId}] 🆔 AudioShake: Payload:`, JSON.stringify(payload, null, 2));
    
    const requestStart = Date.now();
    const task = await httpRequest('/tasks', {
      method: 'POST',
      body: JSON.stringify(payload),
      timeout: 120000, // 2 minutes timeout for task creation
    });
    const requestDuration = ((Date.now() - requestStart) / 1000).toFixed(2);
    
    console.log(`[${callId}] ✅ AudioShake: Request completed in ${requestDuration}s`);
    console.log(`[${callId}] 🆔 AudioShake: Response:`, JSON.stringify(task, null, 2).substring(0, 500));

    if (!task?.id) {
      console.error(`[${callId}] ❌ AudioShake: Task response missing ID:`, task);
      throw new Error('AudioShake createTask did not return a task id.');
    }

    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`[${callId}] ✅ AudioShake: Task created successfully in ${totalDuration}s: ${task.id}`);
    return task;
  } catch (error) {
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`[${callId}] ❌ AudioShake: createTask failed after ${totalDuration}s:`, error);
    console.error(`[${callId}] ❌ AudioShake: Error details:`, {
      message: error.message,
      stack: error.stack,
      status: error.status,
      body: error.body
    });
    throw error;
  }
}

async function getTaskStatus(taskId) {
  if (!taskId) {
    throw new Error('AudioShake task id is required.');
  }
  return httpRequest(`/tasks/${taskId}`, { 
    method: 'GET',
    timeout: 30000, // 30 seconds timeout for status check
  });
}

async function pollTaskCompletion(taskId, maxAttempts = 120, intervalMs = 5000) {
  const startTime = Date.now();
  const fsSync = require('fs');
  const path = require('path');
  const logFile = path.join(__dirname, '..', 'server_debug.log');
  
  console.log(`🔄 Starting AudioShake polling for task ${taskId} (max ${maxAttempts} attempts, ${intervalMs}ms interval)`);
  try {
    fsSync.appendFileSync(logFile, JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'info',
      message: `AudioShake: Starting polling for task ${taskId}`,
      data: { taskId, maxAttempts, intervalMs }
    }) + '\n', 'utf8');
  } catch (e) {}
  
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const task = await getTaskStatus(taskId);
      const target = task?.targets?.[0];
      const status = target?.status;
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      const progress = target?.progress;

      console.log(`🔁 AudioShake polling [${attempt}/${maxAttempts}] - status: ${status}${progress ? `, progress: ${progress}%` : ''} (elapsed: ${elapsed}s)`);
      
      // Log to file every 5 attempts or on status change
      if (attempt % 5 === 0 || status === 'completed' || status === 'failed' || (status && status !== 'pending' && status !== 'processing')) {
        try {
          fsSync.appendFileSync(logFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `AudioShake: Polling attempt ${attempt}/${maxAttempts}`,
            data: { taskId, attempt, maxAttempts, status, progress, elapsed }
          }) + '\n', 'utf8');
        } catch (e) {}
      }

      if (status === 'completed') {
        const totalTime = Math.round((Date.now() - startTime) / 1000);
        console.log(`✅ AudioShake task ${taskId} completed in ${totalTime}s after ${attempt} attempts`);
        try {
          fsSync.appendFileSync(logFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `AudioShake: Task ${taskId} completed`,
            data: { taskId, totalTime, attempts: attempt }
          }) + '\n', 'utf8');
        } catch (e) {}
        return task;
      }

      if (status === 'failed') {
        const failure = target?.error?.message || target?.error || 'Unknown failure';
        console.error(`❌ AudioShake task ${taskId} failed: ${failure}`);
        try {
          fsSync.appendFileSync(logFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'error',
            message: `AudioShake: Task ${taskId} failed`,
            data: { taskId, failure, attempt }
          }) + '\n', 'utf8');
        } catch (e) {}
        throw new Error(`AudioShake task ${taskId} failed: ${failure}`);
      }

      // Log intermediate statuses
      if (status && status !== 'pending' && status !== 'processing') {
        console.log(`ℹ️ AudioShake task ${taskId} status: ${status}`);
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    } catch (error) {
      // If it's a network error, log it but continue polling
      if (error.message && error.message.includes('Network')) {
        console.warn(`⚠️ Network error during polling [${attempt}/${maxAttempts}]: ${error.message}, retrying...`);
        try {
          fsSync.appendFileSync(logFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            level: 'warn',
            message: `AudioShake: Network error during polling`,
            data: { taskId, attempt, error: error.message }
          }) + '\n', 'utf8');
        } catch (e) {}
        await new Promise((resolve) => setTimeout(resolve, intervalMs));
        continue;
      }
      // For other errors, rethrow
      try {
        fsSync.appendFileSync(logFile, JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `AudioShake: Polling error`,
          data: { taskId, attempt, error: error.message, stack: error.stack }
        }) + '\n', 'utf8');
      } catch (e) {}
      throw error;
    }
  }

  const totalTime = Math.round((Date.now() - startTime) / 1000);
  const errorMsg = `AudioShake task ${taskId} polling timed out after ${totalTime}s (${maxAttempts} attempts)`;
  console.error(`⏱️ ${errorMsg}`);
  try {
    fsSync.appendFileSync(logFile, JSON.stringify({
      timestamp: new Date().toISOString(),
      level: 'error',
      message: `AudioShake: Polling timeout`,
      data: { taskId, totalTime, maxAttempts }
    }) + '\n', 'utf8');
  } catch (e) {}
  throw new Error(errorMsg);
}

function normalizeSpeakers(task) {
  const target = task?.targets?.[0];
  if (!target || !Array.isArray(target.output)) {
    return [];
  }

  return target.output.map((output) => ({
    name: output.name || 'speaker',
    format: output.format,
    url: output.link,
    isBackground: output.name === '_background',
  }));
}

async function separateSpeakers(audioUrl, options = {}) {
  const callId = `separateSpeakers_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const startTime = Date.now();
  console.log(`[${callId}] 🎵 AudioShake: Starting separateSpeakers`);
  console.log(`[${callId}] 🎵 AudioShake: Audio URL: ${audioUrl.substring(0, 80)}...`);
  console.log(`[${callId}] 🎵 AudioShake: Options:`, options);
  
  // Write to log file as well
  const fsSync = require('fs');
  const path = require('path');
  try {
    const logDir = path.join(__dirname, '..');
    const logFile = path.join(logDir, 'server_debug.log');
    const logEntry = {
      timestamp: new Date().toISOString(),
      level: 'info',
      message: `[${callId}] AudioShake: Starting separateSpeakers`,
      data: {
        callId,
        audioUrl: audioUrl.substring(0, 100),
        options
      }
    };
    fsSync.appendFileSync(logFile, JSON.stringify(logEntry) + '\n', 'utf8');
  } catch (logError) {
    console.error(`[${callId}] Failed to write log:`, logError.message);
  }
  
  try {
    // Step 1: Create task
    console.log(`[${callId}] 🎵 AudioShake: Step 1 - Creating task...`);
    const fsSync = require('fs');
    const path = require('path');
    try {
      const logFile = path.join(__dirname, '..', 'server_debug.log');
      fsSync.appendFileSync(logFile, JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: `[${callId}] AudioShake: Step 1 - Creating task`,
        data: { callId, audioUrl: audioUrl.substring(0, 100), timeout: '120s' }
      }) + '\n', 'utf8');
    } catch (e) {}
    
    const createTaskStart = Date.now();
    console.log(`[${callId}] 🎵 AudioShake: Calling createTask with 2 minute timeout...`);
    let task;
    try {
      task = await createTask(audioUrl, options);
    } catch (createTaskError) {
      const createTaskDuration = ((Date.now() - createTaskStart) / 1000).toFixed(2);
      console.error(`[${callId}] ❌ AudioShake: createTask failed after ${createTaskDuration}s:`, createTaskError);
      const fsSync = require('fs');
      const path = require('path');
      try {
        const logFile = path.join(__dirname, '..', 'server_debug.log');
        fsSync.appendFileSync(logFile, JSON.stringify({
          timestamp: new Date().toISOString(),
          level: 'error',
          message: `[${callId}] AudioShake: createTask failed`,
          data: { 
            callId, 
            duration: `${createTaskDuration}s`,
            error: createTaskError.message,
            errorType: createTaskError.constructor?.name,
            errorName: createTaskError.name
          }
        }) + '\n', 'utf8');
      } catch (e) {}
      throw createTaskError;
    }
    const createTaskDuration = ((Date.now() - createTaskStart) / 1000).toFixed(2);
    console.log(`[${callId}] ✅ AudioShake: Task created in ${createTaskDuration}s, taskId: ${task.id}`);
    
    try {
      const logFile = path.join(__dirname, '..', 'server_debug.log');
      fsSync.appendFileSync(logFile, JSON.stringify({
        timestamp: new Date().toISOString(),
        level: 'info',
        message: `[${callId}] AudioShake: Task created`,
        data: { callId, taskId: task.id, duration: createTaskDuration }
      }) + '\n', 'utf8');
    } catch (e) {}
    
    // Step 2: Poll for completion
    console.log(`[${callId}] 🎵 AudioShake: Step 2 - Starting polling...`);
    const pollStart = Date.now();
    const completedTask = await pollTaskCompletion(
      task.id, 
      options.maxAttempts || 120, 
      options.intervalMs || 5000
    );
    const pollDuration = ((Date.now() - pollStart) / 1000).toFixed(2);
    console.log(`[${callId}] ✅ AudioShake: Polling completed in ${pollDuration}s`);
    
    // Step 3: Normalize speakers
    console.log(`[${callId}] 🎵 AudioShake: Step 3 - Normalizing speakers...`);
    const speakers = normalizeSpeakers(completedTask);
    const target = completedTask.targets?.[0];
    
    console.log(`[${callId}] ✅ AudioShake: Found ${speakers.length} speakers`);
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`[${callId}] ✅ AudioShake: separateSpeakers completed in ${totalDuration}s`);

    return {
      taskId: completedTask.id,
      speakers,
      cost: target?.cost ?? null,
      duration: target?.duration ?? null,
      raw: completedTask,
    };
  } catch (error) {
    const totalDuration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`[${callId}] ❌ AudioShake: separateSpeakers failed after ${totalDuration}s:`, error);
    console.error(`[${callId}] ❌ AudioShake: Error details:`, {
      message: error.message,
      stack: error.stack,
      status: error.status,
      body: error.body
    });
    throw error;
  }
}

module.exports = {
  createTask,
  getTaskStatus,
  pollTaskCompletion,
  separateSpeakers,
};

