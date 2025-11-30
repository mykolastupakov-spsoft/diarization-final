#!/usr/bin/env node
/**
 * Integration test for the "✅ Apply Overlap Fixes" flow.
 * Runs the overlap diarization pipeline against a known audio sample,
 * rebuilds the merged segments via OverlapMergeUtils, and asserts that
 * no duplicate speaker segments remain.
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const overlapUtils = require('./overlap_merge_utils');
const textSimilarityUtils = require('./text_similarity_utils');

const SERVER_URL = process.env.TEST_SERVER_URL || 'http://localhost:3000';
const DEFAULT_AUDIO = path.join(__dirname, 'audio examples', 'OverlappingCallCenterWithoutBackground.MP3');
const AUDIO_PATH = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_AUDIO;
const LANGUAGE = process.env.TEST_LANGUAGE || 'en';
const MODE = process.env.TEST_MODEL_TIER || 'smart';
const PIPELINE_MODE = process.env.TEST_PIPELINE_MODE || 'mode3';
const ENGINE = process.env.TEST_ENGINE || 'azure_realtime';
const DEBUG = process.env.DEBUG_OVERLAP_TEST === '1';

async function main() {
  if (!fs.existsSync(AUDIO_PATH)) {
    console.error(`❌ Audio file not found: ${AUDIO_PATH}`);
    process.exit(1);
  }

  console.log('🎧 Running overlap fixes regression test');
  console.log(`   • Server: ${SERVER_URL}`);
  console.log(`   • Audio: ${AUDIO_PATH}`);
  console.log(`   • Engine: ${ENGINE}, Mode: ${MODE}, Pipeline: ${PIPELINE_MODE}`);

  const response = await runOverlapRequest(AUDIO_PATH);
  if (!response || !response.success) {
    const reason = response?.error || response?.message || 'unknown error';
    throw new Error(`Overlap diarization request failed: ${reason}`);
  }

  const recording =
    response.correctedDiarization?.recordings?.[0] ||
    response.primaryDiarization?.recordings?.[0];
  if (!recording) {
    throw new Error('Response does not include a recording payload');
  }

  const primarySegments = recording.results?.speechmatics?.segments || [];
  const voiceTracks =
    response.voiceTracks ||
    response.overlapMetadata?.voiceTracks ||
    recording.overlapMetadata?.voiceTracks ||
    [];

  if (!voiceTracks.length) {
    throw new Error('No voice tracks returned by the pipeline. Ensure overlap diarization completed successfully.');
  }

  const logger = createLogger(DEBUG);
  const voiceSegments = overlapUtils.collectVoiceTrackSegments(voiceTracks, primarySegments, { logger });
  const mergedSegments = overlapUtils.mergeVoiceTrackSegments(voiceSegments, primarySegments, recording, {
    logger,
    areTextsSimilar: textSimilarityUtils.areTextsSimilar
  });

  const duplicates = detectDuplicates(mergedSegments);
  if (duplicates.length > 0) {
    console.error('❌ Duplicate segments detected after overlap fixes:');
    duplicates.slice(0, 10).forEach(dup => {
      console.error(
        `   • ${dup.speaker} @ ${dup.times.map(t => t.toFixed(2)).join(' / ')}s → "${dup.texts[0]}" || "${dup.texts[1]}"`
      );
    });
    if (duplicates.length > 10) {
      console.error(`   ...and ${duplicates.length - 10} more`);
    }
    process.exit(1);
  }

  console.log(`✅ Overlap fixes test passed. ${mergedSegments.length} merged segments, ${new Set(mergedSegments.map(s => s.speaker || 'UNKNOWN')).size} speakers.`);
}

function createLogger(verbose) {
  if (verbose) {
    return console;
  }
  return {
    log: () => {},
    warn: (...args) => console.warn('[test_overlap_fixes]', ...args),
    error: (...args) => console.error('[test_overlap_fixes]', ...args)
  };
}

async function runOverlapRequest(filePath) {
  const form = new FormData();
  form.append('audio', fs.createReadStream(filePath));
  form.append('language', LANGUAGE);
  form.append('mode', MODE);
  form.append('pipelineMode', PIPELINE_MODE);
  form.append('engine', ENGINE);

  const response = await axios.post(`${SERVER_URL}/api/diarize-overlap`, form, {
    headers: form.getHeaders(),
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
    timeout: 1000 * 60 * 60 // allow long runs for first-time model downloads
  });

  return response.data;
}

function detectDuplicates(segments) {
  const duplicates = [];
  for (let i = 0; i < segments.length; i++) {
    const segA = segments[i];
    const speakerA = segA.speaker || 'SPEAKER_00';
    const startA = parseFloat(segA.start) || 0;
    const endA = parseFloat(segA.end) || startA;
    const durationA = Math.max(0.01, endA - startA);
    const textA = (segA.text || '').trim().toLowerCase();

    for (let j = i + 1; j < segments.length; j++) {
      const segB = segments[j];
      const speakerB = segB.speaker || 'SPEAKER_00';
      if (speakerA !== speakerB) continue;

      const startB = parseFloat(segB.start) || 0;
      const endB = parseFloat(segB.end) || startB;
      const durationB = Math.max(0.01, endB - startB);
      const overlap = Math.max(0, Math.min(endA, endB) - Math.max(startA, startB));
      if (overlap <= 0) continue;

      const coverage = overlap / Math.min(durationA, durationB);
      const textB = (segB.text || '').trim().toLowerCase();
      const textSimilar = textSimilarityUtils.areTextsSimilar(segA.text, segB.text, {
        minLevenshteinSim: 0.85,
        minJaccardSim: 0.75
      });

      if (coverage > 0.6 && textSimilar) {
        duplicates.push({
          speaker: speakerA,
          times: [startA, startB],
          texts: [segA.text || '', segB.text || '']
        });
      }
    }
  }
  return duplicates;
}

main().catch(error => {
  if (error.response) {
    console.error('❌ Request failed:', JSON.stringify(error.response.data, null, 2));
  } else {
    console.error('❌ Error:', error.message);
  }
  process.exit(1);
});

