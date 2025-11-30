#!/usr/bin/env node
/**
 * Quick regression tester for the Overlap (Mode 3 / SpeechBrain) pipeline.
 *
 * Usage:
 *   node test_overlap_mode3.js                                 # auto-discovers WAV/MP3 in audio examples
 *   node test_overlap_mode3.js "audio examples/Call center 1.wav"
 *
 * Environment overrides:
 *   TEST_SERVER_URL   (default: http://localhost:3000)
 *   TEST_AUDIO_DIR    (default: <repo>/audio examples)
 *   TEST_LANGUAGE     (default: en)
 *   TEST_MODEL_TIER   (default: smart)
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const SERVER_URL = process.env.TEST_SERVER_URL || 'http://localhost:3000';
const AUDIO_DIR = process.env.TEST_AUDIO_DIR || path.join(__dirname, 'audio examples');
const DEFAULT_LANGUAGE = process.env.TEST_LANGUAGE || 'en';
const DEFAULT_MODEL = process.env.TEST_MODEL_TIER || 'smart';
const PIPELINE_MODE = process.env.TEST_PIPELINE_MODE || 'mode3';

async function main() {
  const files = gatherFiles(process.argv.slice(2));
  if (!files.length) {
    console.error('No audio files found. Pass explicit paths or populate the audio examples directory.');
    process.exit(1);
  }

  console.log(`🚀 Testing overlap diarization (${PIPELINE_MODE}) against ${SERVER_URL}`);
  console.log(`Language=${DEFAULT_LANGUAGE}, Model=${DEFAULT_MODEL}`);

  for (const filePath of files) {
    await runSingleTest(filePath);
  }

  console.log('\n✅ Completed SpeechBrain Mode 3 regression run.');
}

function gatherFiles(cliArgs) {
  if (cliArgs.length) {
    return cliArgs.map(p => path.resolve(p)).filter(fs.existsSync);
  }

  if (!fs.existsSync(AUDIO_DIR)) return [];
  const candidates = fs.readdirSync(AUDIO_DIR)
    .filter(name => /\.(wav|mp3|m4a|ogg)$/i.test(name))
    .map(name => path.join(AUDIO_DIR, name));

  return candidates.filter(fs.existsSync);
}

async function runSingleTest(filePath) {
  process.stdout.write(`→ ${path.basename(filePath)} ... `);
  const startTime = Date.now();

  try {
    const response = await postOverlapRequest(filePath);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`done (${elapsed}s)`);
    printSummary(response.data);
  } catch (error) {
    console.log('failed');
    if (error.response) {
      console.error(`HTTP ${error.response.status}: ${JSON.stringify(error.response.data, null, 2)}`);
    } else {
      console.error(error.message);
    }
  }
}

async function postOverlapRequest(filePath) {
  const form = new FormData();
  form.append('audio', fs.createReadStream(filePath));
  form.append('language', DEFAULT_LANGUAGE);
  form.append('mode', DEFAULT_MODEL);
  form.append('pipelineMode', PIPELINE_MODE);

  return axios.post(`${SERVER_URL}/api/diarize-overlap`, form, {
    headers: form.getHeaders(),
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
    timeout: 1000 * 60 * 30 // 30 minutes to cover first-time model downloads
  });
}

function printSummary(result) {
  if (!result || !result.success) {
    console.log('⚠️ Request reported failure.');
    return;
  }

  const speakers = result?.separation?.speakers?.filter(s => !s.isBackground) || [];
  const overlaps = result?.correctedDiarization?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
  const overlapCount = overlaps.filter(seg => seg.overlap).length;

  console.log(`   • Speakers separated: ${speakers.length}`);
  speakers.forEach((speaker, idx) => {
    console.log(`      - ${speaker.name || `SPEAKER_${idx.toString().padStart(2, '0')}`}: ${speaker.audioUrl || speaker.downloadUrl}`);
  });

  console.log(`   • Corrected segments: ${overlaps.length} (${overlapCount} flagged as overlap)`);
  console.log(`   • Steps:`);
  Object.entries(result.steps || {}).forEach(([key, value]) => {
    console.log(`      - ${key}: ${value.status} (${value.duration || 'n/a'})`);
  });
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});

