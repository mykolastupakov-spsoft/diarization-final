#!/usr/bin/env node
/**
 * Analyze Azure speaker mapping and segment distribution
 * This script helps identify issues with speaker assignment in Azure results
 */

const fs = require('fs');
const path = require('path');

function analyzeAzureResults(recordingPath) {
  console.log('🔍 Analyzing Azure speaker mapping...\n');
  
  let recording;
  try {
    const data = fs.readFileSync(recordingPath, 'utf8');
    recording = JSON.parse(data);
  } catch (error) {
    console.error('❌ Error reading recording:', error.message);
    process.exit(1);
  }
  
  const recordingData = recording?.recordings?.[0] || recording;
  const azureResult = recordingData?.results?.azure;
  const speechmaticsResult = recordingData?.results?.speechmatics;
  
  // Check if this is Azure result
  const engine = recordingData?._engine || 
                 speechmaticsResult?.engine ||
                 azureResult?.engine;
  
  const isAzure = engine && (engine.includes('azure') || engine.includes('Azure'));
  
  if (!isAzure) {
    console.log('⚠️  This is not an Azure result. Engine:', engine);
    return;
  }
  
  console.log('✅ Azure result detected. Engine:', engine);
  console.log('📊 Service:', azureResult?.serviceName || speechmaticsResult?.serviceName || 'Unknown');
  console.log('');
  
  // Get raw Azure data
  const azureRawData = azureResult?.rawData || speechmaticsResult?.rawData;
  
  if (!azureRawData || !azureRawData.segments) {
    console.log('❌ No Azure raw data found');
    return;
  }
  
  console.log('📋 Raw Azure Data Analysis:');
  console.log('='.repeat(80));
  
  // Analyze speaker IDs from Azure
  const azureSpeakerIds = new Set();
  const azureSegmentsBySpeakerId = {};
  
  azureRawData.segments.forEach(seg => {
    if (seg.type === 'final' && seg.speakerId) {
      azureSpeakerIds.add(seg.speakerId);
      if (!azureSegmentsBySpeakerId[seg.speakerId]) {
        azureSegmentsBySpeakerId[seg.speakerId] = [];
      }
      azureSegmentsBySpeakerId[seg.speakerId].push(seg);
    }
  });
  
  console.log('\n🔤 Azure Speaker IDs found:', Array.from(azureSpeakerIds).sort());
  console.log('\n📊 Segments per Azure Speaker ID:');
  Object.keys(azureSegmentsBySpeakerId).sort().forEach(speakerId => {
    const count = azureSegmentsBySpeakerId[speakerId].length;
    console.log(`   ${speakerId}: ${count} segments`);
  });
  
  // Analyze speaker map
  const speakerMap = azureRawData.speakerMap || {};
  console.log('\n🗺️  Speaker Map:', speakerMap);
  
  // Analyze mapped segments
  const mappedSegmentsBySpeaker = {};
  azureRawData.segments.forEach(seg => {
    if (seg.type === 'final' && seg.speakerId) {
      const azureSpeakerId = seg.speakerId;
      const mappedSpeaker = speakerMap[azureSpeakerId] || `SPEAKER_${azureSpeakerId}`;
      if (!mappedSegmentsBySpeaker[mappedSpeaker]) {
        mappedSegmentsBySpeaker[mappedSpeaker] = [];
      }
      mappedSegmentsBySpeaker[mappedSpeaker].push(seg);
    }
  });
  
  console.log('\n📊 Segments per Mapped Speaker:');
  Object.keys(mappedSegmentsBySpeaker).sort().forEach(speaker => {
    const count = mappedSegmentsBySpeaker[speaker].length;
    console.log(`   ${speaker}: ${count} segments`);
  });
  
  // Analyze parsed segments (from getTextServiceResult)
  const parsedSegments = recordingData?.results?.speechmatics?.segments || 
                         recordingData?.results?.azure?.segments || [];
  
  console.log('\n📊 Parsed Segments Analysis:');
  const parsedSegmentsBySpeaker = {};
  parsedSegments.forEach(seg => {
    const speaker = seg.speaker || 'Unknown';
    if (!parsedSegmentsBySpeaker[speaker]) {
      parsedSegmentsBySpeaker[speaker] = [];
    }
    parsedSegmentsBySpeaker[speaker].push(seg);
  });
  
  Object.keys(parsedSegmentsBySpeaker).sort().forEach(speaker => {
    const count = parsedSegmentsBySpeaker[speaker].length;
    console.log(`   ${speaker}: ${count} segments`);
  });
  
  // Check for inconsistencies
  console.log('\n🔍 Consistency Check:');
  console.log('='.repeat(80));
  
  const azureSpeakerCounts = Object.keys(azureSegmentsBySpeakerId).map(id => ({
    id,
    count: azureSegmentsBySpeakerId[id].length
  })).sort((a, b) => b.count - a.count);
  
  const mappedSpeakerCounts = Object.keys(mappedSegmentsBySpeaker).map(speaker => ({
    speaker,
    count: mappedSegmentsBySpeaker[speaker].length
  })).sort((a, b) => b.count - a.count);
  
  const parsedSpeakerCounts = Object.keys(parsedSegmentsBySpeaker).map(speaker => ({
    speaker,
    count: parsedSegmentsBySpeaker[speaker].length
  })).sort((a, b) => b.count - a.count);
  
  console.log('\n📈 Azure Speaker IDs (sorted by segment count):');
  azureSpeakerCounts.forEach(({ id, count }) => {
    const mapped = speakerMap[id] || 'NOT MAPPED';
    console.log(`   ${id} → ${mapped}: ${count} segments`);
  });
  
  console.log('\n📈 Mapped Speakers (sorted by segment count):');
  mappedSpeakerCounts.forEach(({ speaker, count }) => {
    console.log(`   ${speaker}: ${count} segments`);
  });
  
  console.log('\n📈 Parsed Speakers (sorted by segment count):');
  parsedSpeakerCounts.forEach(({ speaker, count }) => {
    console.log(`   ${speaker}: ${count} segments`);
  });
  
  // Check if mapping is consistent
  if (mappedSpeakerCounts.length !== parsedSpeakerCounts.length) {
    console.log('\n⚠️  WARNING: Mapped speakers count differs from parsed speakers count!');
    console.log(`   Mapped: ${mappedSpeakerCounts.length}, Parsed: ${parsedSpeakerCounts.length}`);
  }
  
  // Check if speaker order is consistent
  const mappedOrder = mappedSpeakerCounts.map(s => s.speaker);
  const parsedOrder = parsedSpeakerCounts.map(s => s.speaker);
  
  if (JSON.stringify(mappedOrder) !== JSON.stringify(parsedOrder)) {
    console.log('\n⚠️  WARNING: Speaker order differs between mapped and parsed!');
    console.log(`   Mapped order: ${mappedOrder.join(', ')}`);
    console.log(`   Parsed order: ${parsedOrder.join(', ')}`);
  }
  
  // Show first few segments for each speaker to verify
  console.log('\n📝 Sample Segments (first 3 per speaker):');
  console.log('='.repeat(80));
  
  Object.keys(mappedSegmentsBySpeaker).sort().forEach(speaker => {
    const segments = mappedSegmentsBySpeaker[speaker].slice(0, 3);
    console.log(`\n${speaker}:`);
    segments.forEach((seg, idx) => {
      const text = (seg.text || '').substring(0, 60);
      console.log(`   ${idx + 1}. [${seg.offset?.toFixed(2)}s] ${text}...`);
    });
  });
  
  console.log('\n✅ Analysis complete!\n');
}

// Main
const recordingPath = process.argv[2];
if (!recordingPath) {
  console.error('Usage: node analyze_azure_speakers.js <path-to-recording.json>');
  console.error('Example: node analyze_azure_speakers.js recordings.json');
  process.exit(1);
}

analyzeAzureResults(recordingPath);











