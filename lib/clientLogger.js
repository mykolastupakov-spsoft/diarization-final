/**
 * Client-side logging utility
 * Sends logs to server instead of browser console
 * NOTE: On demo page (/demo), logging is disabled
 */

(function() {
  'use strict';
  
  // Check if we're on demo page
  const isDemoPage = () => {
    if (typeof window === 'undefined') return false;
    return window.location.pathname === '/demo' || window.location.pathname.includes('/demo');
  };
  
  // Queue for logs (in case server is not ready)
  let logQueue = [];
  let isServerReady = true;
  
  // Send log to server
  const sendLogToServer = (level, message, data = {}) => {
    // Don't log on demo page
    if (isDemoPage()) {
      return;
    }
    
    // If server is not ready, queue the log
    if (!isServerReady) {
      logQueue.push({ level, message, data, timestamp: new Date().toISOString() });
      return;
    }
    
    // Send log to server
    fetch('/api/client-log', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        level,
        message,
        data,
        timestamp: new Date().toISOString(),
        url: window.location.href,
        userAgent: navigator.userAgent
      })
    }).catch(error => {
      // If server is not available, mark as not ready and queue logs
      isServerReady = false;
      logQueue.push({ level, message, data, timestamp: new Date().toISOString() });
    });
  };
  
  // Process queued logs
  const processLogQueue = () => {
    if (logQueue.length === 0) return;
    
    const logs = [...logQueue];
    logQueue = [];
    
    logs.forEach(log => {
      sendLogToServer(log.level, log.message, log.data);
    });
  };
  
  // Try to process queue periodically
  setInterval(() => {
    if (logQueue.length > 0 && isServerReady) {
      processLogQueue();
    }
  }, 5000);
  
  // Client logger object
  const clientLogger = {
    log: (message, ...args) => {
      sendLogToServer('info', message, { args });
    },
    
    warn: (message, ...args) => {
      sendLogToServer('warn', message, { args });
    },
    
    error: (message, ...args) => {
      sendLogToServer('error', message, { args });
    },
    
    info: (message, ...args) => {
      sendLogToServer('info', message, { args });
    },
    
    debug: (message, ...args) => {
      sendLogToServer('debug', message, { args });
    }
  };
  
  // Export to window if in browser
  if (typeof window !== 'undefined') {
    window.clientLogger = clientLogger;
  }
  
  // Export for Node.js (if needed)
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = clientLogger;
  }
})();

