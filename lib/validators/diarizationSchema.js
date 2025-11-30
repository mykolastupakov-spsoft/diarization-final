const Ajv = require('ajv');

const ajv = new Ajv({
  allErrors: true,
  strict: false
});

const WORD_SCHEMA = {
  type: 'object',
  required: ['word', 'start', 'end'],
  properties: {
    word: { type: 'string' },
    start: { type: 'number' },
    end: { type: 'number' },
    speaker: { type: ['string', 'number', 'null'] },
    confidence: { type: ['number', 'null'] }
  },
  additionalProperties: true
};

const SEGMENT_SCHEMA = {
  type: 'object',
  required: ['speaker', 'text', 'start', 'end', 'words', 'role', 'overlap'],
  properties: {
    speaker: { type: 'string' },
    text: { type: 'string' },
    start: { type: 'number' },
    end: { type: 'number' },
    words: {
      type: 'array',
      items: WORD_SCHEMA,
      default: []
    },
    role: {
      type: 'string',
      enum: ['operator', 'client']
    },
    overlap: { type: 'boolean' }
  },
  additionalProperties: true
};

const RAW_DATA_SCHEMA = {
  type: 'object',
  required: ['language', 'source'],
  properties: {
    duration: { type: ['number', 'null'] },
    language: { type: 'string' },
    source: { type: 'string' }
  },
  additionalProperties: true
};

const SERVICE_RESULT_SCHEMA = (serviceKey) => ({
  type: 'object',
  required: ['success', 'serviceName', 'segments', 'rawData', 'speakerCount'],
  properties: {
    success: { type: 'boolean' },
    serviceName: { type: 'string' },
    processingTime: { type: ['number', 'null'] },
    speedFactor: { type: ['number', 'null'] },
    speakerCount: { type: ['number', 'string'] },
    cost: { type: ['string', 'number', 'null'] },
    segments: {
      type: 'array',
      items: SEGMENT_SCHEMA
    },
    rawData: RAW_DATA_SCHEMA,
    metadata: { type: 'object', additionalProperties: true }
  },
  additionalProperties: true,
  errorMessage: {
    required: {
      segments: `${serviceKey} segments array is required`
    }
  }
});

const RECORDING_SCHEMA = (serviceKey) => ({
  type: 'object',
  required: [
    'id',
    'name',
    'fileName',
    'language',
    'status',
    'translationState',
    'results'
  ],
  properties: {
    id: { type: 'string' },
    name: { type: 'string' },
    fileName: { type: 'string' },
    size: { type: ['number', 'null'] },
    duration: { type: ['number', 'null'] },
    language: { type: 'string' },
    speakerCount: { type: ['string', 'number'] },
    status: { type: 'string' },
    translationState: {
      type: 'object',
      required: ['currentLanguage', 'lastError'],
      properties: {
        currentLanguage: { type: 'string' },
        lastError: { type: ['string', 'null'] }
      },
      additionalProperties: true
    },
    aggregated: { type: 'object', additionalProperties: true },
    servicesTested: {
      type: 'array',
      items: { type: 'string' },
      minItems: 1
    },
    results: {
      type: 'object',
      required: [serviceKey],
      properties: {
        [serviceKey]: SERVICE_RESULT_SCHEMA(serviceKey)
      },
      additionalProperties: true
    }
  },
  additionalProperties: true
});

const buildSchema = (serviceKey) => ({
  type: 'object',
  required: ['version', 'exportedAt', 'activeRecordingId', 'recordings'],
  properties: {
    version: { type: 'string' },
    exportedAt: { type: 'string' },
    activeRecordingId: { type: 'string' },
    recordings: {
      type: 'array',
      minItems: 1,
      items: RECORDING_SCHEMA(serviceKey)
    }
  },
  additionalProperties: false
});

const validatorCache = new Map();

function getValidator(serviceKey = 'text-service') {
  if (!validatorCache.has(serviceKey)) {
    validatorCache.set(serviceKey, ajv.compile(buildSchema(serviceKey)));
  }
  return validatorCache.get(serviceKey);
}

function validateDiarizationPayload(payload, serviceKey = 'text-service') {
  const validator = getValidator(serviceKey);
  const valid = validator(payload);
  return { valid, errors: validator.errors || [] };
}

function formatAjvErrors(errors = []) {
  if (!errors || errors.length === 0) {
    return '';
  }
  return errors
    .map((err) => `${err.instancePath || '(root)'} ${err.message}`)
    .join('; ');
}

module.exports = {
  validateDiarizationPayload,
  formatAjvErrors
};

