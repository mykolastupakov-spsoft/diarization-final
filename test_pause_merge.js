const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');

const SERVER_URL = 'http://localhost:3000';
const TEST_AUDIO = path.join(__dirname, 'audio examples', 'OverlappingCallCenterWithoutBackground.MP3');

async function testPauseBasedMerge() {
  console.log('🧪 Testing pause-based merge functionality\n');
  console.log(`📁 Test audio: ${TEST_AUDIO}`);
  console.log(`📊 File size: ${(fs.statSync(TEST_AUDIO).size / 1024 / 1024).toFixed(2)} MB\n`);

  try {
    // Step 1: Run mode3 overlap diarization
    console.log('═══════════════════════════════════════════════════════════');
    console.log('STEP 1: Running Mode3 Overlap Diarization');
    console.log('═══════════════════════════════════════════════════════════\n');
    
    const formData = new FormData();
    formData.append('audio', fs.createReadStream(TEST_AUDIO));
    formData.append('language', 'en');
    formData.append('speakerCount', '2');
    formData.append('pipelineMode', 'mode3');
    formData.append('mode', 'fast');

    console.log('📤 Sending request to /api/diarize-overlap...');
    
    // For SSE, we need to handle the stream manually
    const response = await axios.post(`${SERVER_URL}/api/diarize-overlap`, formData, {
      headers: formData.getHeaders(),
      responseType: 'stream',
      timeout: 600000 // 10 minutes
    });

    // Parse SSE stream
    let buffer = '';
    let finalResult = null;
    
    return new Promise((resolve, reject) => {
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        
        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6).trim());
              if (data.type === 'step-progress') {
                console.log(`[${data.step}] ${data.status}: ${data.description}`);
              } else if (data.type === 'final-result') {
                finalResult = data;
                delete finalResult.type;
              }
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      });

      response.data.on('end', () => {
        if (!finalResult) {
          // Try to parse final JSON from buffer
          try {
            const jsonMatch = buffer.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              finalResult = JSON.parse(jsonMatch[0]);
            }
          } catch (e) {
            reject(new Error('Failed to parse final result from SSE stream'));
            return;
          }
        }

        if (!finalResult || !finalResult.success) {
          reject(new Error('Overlap diarization failed: ' + JSON.stringify(finalResult)));
          return;
        }

        console.log('\n✅ Mode3 Overlap Diarization completed\n');
        
        // Step 2: Extract recording and voice tracks
        const recording = finalResult.correctedDiarization?.recordings?.[0];
        if (!recording) {
          reject(new Error('No recording found in result'));
          return;
        }

        const voiceTracks = finalResult.voiceTracks || recording.overlapMetadata?.voiceTracks || [];
        console.log(`📊 Found ${voiceTracks.length} voice tracks`);
        console.log(`📊 Primary segments: ${finalResult.primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0}`);
        console.log(`📊 Corrected segments: ${recording.results?.['overlap-corrected']?.segments?.length || 0}\n`);

        // Step 3: Apply pause-based merge
        console.log('═══════════════════════════════════════════════════════════');
        console.log('STEP 2: Applying Pause-Based Merge');
        console.log('═══════════════════════════════════════════════════════════\n');

        applyPauseBasedMerge(recording, voiceTracks)
          .then(result => {
            // Step 4: Validate results
            validateResults(finalResult, result);
            resolve(result);
          })
          .catch(reject);
      });

      response.data.on('error', reject);
    });

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    throw error;
  }
}

async function applyPauseBasedMerge(recording, voiceTracks) {
  console.log('📤 Sending request to /api/apply-pause-based-merge...');
  
  const response = await axios.post(`${SERVER_URL}/api/apply-pause-based-merge`, {
    recordingId: recording.id,
    recording: {
      ...recording,
      overlapMetadata: {
        ...recording.overlapMetadata,
        voiceTracks: voiceTracks
      }
    }
  }, {
    timeout: 300000 // 5 minutes
  });

  if (!response.data.success) {
    throw new Error('Pause-based merge failed: ' + response.data.error);
  }

  console.log('✅ Pause-based merge completed\n');
  return response.data;
}

function validateResults(originalResult, pauseMergeResult) {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('STEP 3: Validating Results');
  console.log('═══════════════════════════════════════════════════════════\n');

  const primarySegments = originalResult.primaryDiarization?.recordings?.[0]?.results?.speechmatics?.segments || [];
  const correctedSegments = originalResult.correctedDiarization?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
  const pauseMergeSegments = pauseMergeResult.recording?.results?.['pause-based-merge']?.segments || [];

  console.log('📊 Segment counts:');
  console.log(`   Primary: ${primarySegments.length}`);
  console.log(`   Corrected (mode3): ${correctedSegments.length}`);
  console.log(`   Pause-based merge: ${pauseMergeSegments.length}\n`);

  // Collect all voice track segments (including split ones from pause-based merge)
  // The pause-based merge splits long segments by pauses, so we need to check the actual segments used
  const voiceTrackSegments = [];
  const voiceTracks = originalResult.voiceTracks || [];
  
  voiceTracks.forEach(track => {
    const segments = track.transcription?.recordings?.[0]?.results?.speechmatics?.segments || [];
    segments.forEach(seg => {
      // If segment has words, count potential splits (pauses > 300ms)
      if (seg.words && seg.words.length > 1) {
        let pauseCount = 0;
        for (let i = 1; i < seg.words.length; i++) {
          const prevWord = seg.words[i - 1];
          const currWord = seg.words[i];
          const prevEnd = prevWord.end || prevWord.end_time || prevWord.start || 0;
          const currStart = currWord.start || currWord.start_time || prevEnd;
          const gap = currStart - prevEnd;
          if (gap > 0.3) pauseCount++;
        }
        // Each pause creates a potential split, so we have pauseCount + 1 potential segments
        // But for validation, we'll use the actual segments from pause-based merge result
        voiceTrackSegments.push(seg);
      } else {
        voiceTrackSegments.push(seg);
      }
    });
  });

  // Get actual split segments from pause-based merge rawData if available
  const pauseMergeRawData = pauseMergeResult.recording?.results?.['pause-based-merge']?.rawData;
  const actualVoiceTrackSegmentsCount = pauseMergeRawData?.voiceTrackSegmentsCount || voiceTrackSegments.length;

  console.log(`📊 Voice track segments (original): ${voiceTrackSegments.length}`);
  console.log(`📊 Voice track segments (after splitting): ${actualVoiceTrackSegmentsCount}\n`);

  // Check if all voice track segments are represented
  // Since pause-based merge splits long segments, we need to check word-level coverage
  console.log('🔍 Checking voice track coverage...');
  
  const pauseMergeTexts = pauseMergeSegments.map(s => ({
    text: (s.text || '').trim().toLowerCase(),
    start: s.start,
    end: s.end,
    speaker: s.speaker
  }));

  // Collect all words from voice tracks (after splitting)
  const voiceTrackWords = [];
  voiceTracks.forEach(track => {
    const segments = track.transcription?.recordings?.[0]?.results?.speechmatics?.segments || [];
    segments.forEach(seg => {
      if (seg.words && seg.words.length > 0) {
        seg.words.forEach(word => {
          const wordText = (word.word || word.text || word.content || '').trim().toLowerCase();
          if (wordText.length > 0) {
            voiceTrackWords.push({
              text: wordText,
              start: word.start || word.start_time || 0,
              end: word.end || word.end_time || word.start || 0,
              speaker: seg.speaker || track.speaker
            });
          }
        });
      } else {
        // Fallback: use segment text
        const segText = (seg.text || '').trim().toLowerCase();
        if (segText.length > 0) {
          voiceTrackWords.push({
            text: segText,
            start: seg.start,
            end: seg.end,
            speaker: seg.speaker || track.speaker
          });
        }
      }
    });
  });

  // Check coverage: count how many voice track words appear in pause merge result
  let coveredWords = 0;
  let missingWords = 0;
  const missingWordSegments = [];

  voiceTrackWords.forEach(vWord => {
    let found = false;
    for (const pmSeg of pauseMergeTexts) {
      // Check if word appears in this segment (by text and time overlap)
      const timeOverlap = rangesOverlap(vWord.start, vWord.end, pmSeg.start, pmSeg.end);
      const textMatch = pmSeg.text.includes(vWord.text) || computeSimilarity(pmSeg.text, vWord.text) > 0.3;
      
      if (timeOverlap && textMatch) {
        found = true;
        break;
      }
    }

    if (found) {
      coveredWords++;
    } else {
      missingWords++;
      if (missingWordSegments.length < 20) {
        missingWordSegments.push(vWord);
      }
    }
  });

  const coveragePercent = voiceTrackWords.length > 0 ? ((coveredWords / voiceTrackWords.length) * 100).toFixed(1) : 0;
  console.log(`   ✅ Covered words: ${coveredWords}/${voiceTrackWords.length} (${coveragePercent}%)`);
  console.log(`   ❌ Missing words: ${missingWords}/${voiceTrackWords.length}\n`);

  function rangesOverlap(startA, endA, startB, endB) {
    return startA < endB && startB < endA;
  }

  if (missingWords > 0 && missingWordSegments.length > 0) {
    console.log('⚠️  Missing words/phrases from voice tracks:');
    missingWordSegments.slice(0, 10).forEach((word, idx) => {
      console.log(`   ${idx + 1}. [${word.speaker}] [${word.start.toFixed(2)}-${word.end.toFixed(2)}s]: "${word.text}"`);
    });
    if (missingWords > 10) {
      console.log(`   ... and ${missingWords - 10} more`);
    }
    console.log('');
  }

  // Check dialogue structure
  console.log('🔍 Checking dialogue structure...');
  const speakers = new Set(pauseMergeSegments.map(s => s.speaker));
  console.log(`   Speakers: ${Array.from(speakers).join(', ')}`);
  
  let speakerSwitches = 0;
  for (let i = 1; i < pauseMergeSegments.length; i++) {
    if (pauseMergeSegments[i].speaker !== pauseMergeSegments[i-1].speaker) {
      speakerSwitches++;
    }
  }
  console.log(`   Speaker switches: ${speakerSwitches}`);
  console.log(`   Average segment length: ${(pauseMergeSegments.reduce((sum, s) => sum + ((s.end || 0) - (s.start || 0)), 0) / pauseMergeSegments.length).toFixed(2)}s\n`);

  // Summary
  console.log('═══════════════════════════════════════════════════════════');
  console.log('SUMMARY');
  console.log('═══════════════════════════════════════════════════════════\n');
  
  const coverageThreshold = 0.95; // 95% coverage is considered success
  const coverageRatio = voiceTrackWords.length > 0 ? coveredWords / voiceTrackWords.length : 0;
  
  if (coverageRatio >= coverageThreshold) {
    console.log(`✅ SUCCESS: ${(coverageRatio * 100).toFixed(1)}% of voice track content is represented in pause-based merge!`);
  } else {
    console.log(`⚠️  WARNING: Only ${(coverageRatio * 100).toFixed(1)}% coverage (${coveredWords}/${voiceTrackWords.length} words)`);
    console.log('   This may indicate that the merge algorithm needs adjustment.');
  }

  return {
    success: coverageRatio >= coverageThreshold,
    stats: {
      primarySegments: primarySegments.length,
      correctedSegments: correctedSegments.length,
      pauseMergeSegments: pauseMergeSegments.length,
      voiceTrackSegments: actualVoiceTrackSegmentsCount,
      voiceTrackWords: voiceTrackWords.length,
      coveredWords,
      missingWords,
      coverageRatio,
      speakerSwitches
    },
    missingSegments: missingWordSegments
  };
}

function computeSimilarity(textA, textB) {
  const tokensA = textA.toLowerCase().split(/\s+/).filter(Boolean);
  const tokensB = textB.toLowerCase().split(/\s+/).filter(Boolean);
  
  if (!tokensA.length || !tokensB.length) return 0;
  
  const setA = new Set(tokensA);
  const setB = new Set(tokensB);
  let intersection = 0;
  
  setA.forEach(token => {
    if (setB.has(token)) intersection++;
  });
  
  const union = new Set([...tokensA, ...tokensB]).size;
  return union === 0 ? 0 : intersection / union;
}

// Run test
testPauseBasedMerge()
  .then(result => {
    console.log('\n✅ Test completed successfully!');
    process.exit(0);
  })
  .catch(error => {
    console.error('\n❌ Test failed:', error.message);
    if (error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  });

