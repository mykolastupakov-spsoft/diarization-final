/**
 * Test script to verify the complete diarization flow
 * Simulates frontend actions: upload audio → diarization → upload dialogue → comparison
 */

const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');

const BASE_URL = process.env.TEST_BASE_URL || 'http://localhost:3000';
const TEST_AUDIO_FILE = process.env.TEST_AUDIO_FILE || path.join(__dirname, 'audio examples', 'Call centre example.MP3');
const TEST_DIALOGUE_FILE = process.env.TEST_DIALOGUE_FILE || null; // Can be set to a dialogue file path

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function uploadAudio(audioFilePath) {
  console.log('\n📤 Step 1: Uploading audio file...');
  console.log(`   File: ${audioFilePath}`);
  
  if (!fs.existsSync(audioFilePath)) {
    throw new Error(`Audio file not found: ${audioFilePath}`);
  }
  
  const formData = new FormData();
  formData.append('audio', fs.createReadStream(audioFilePath));
  formData.append('language', 'en');
  formData.append('speakerCount', '2');
  formData.append('pipelineMode', 'mode3');
  formData.append('mode', 'smart');
  formData.append('engine', 'speechmatics');
  
  try {
    const response = await axios.post(`${BASE_URL}/api/diarize-overlap`, formData, {
      headers: formData.getHeaders(),
      timeout: 300000, // 5 minutes timeout
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    
    console.log('✅ Audio uploaded and diarization started');
    console.log(`   Response keys: ${Object.keys(response.data).join(', ')}`);
    
    return response.data;
  } catch (error) {
    if (error.response) {
      console.error('❌ Error response:', error.response.status, error.response.data);
    } else {
      console.error('❌ Error:', error.message);
    }
    throw error;
  }
}

async function uploadDialogue(dialogueText) {
  console.log('\n📤 Step 2: Uploading dialogue text...');
  console.log(`   Text length: ${dialogueText.length} characters`);
  
  try {
    const response = await axios.post(`${BASE_URL}/api/dialogue-scripts`, {
      text: dialogueText,
      lines: dialogueText.split('\n').map(line => line.trim()).filter(Boolean),
      meta: {
        uploadedAt: new Date().toISOString(),
        source: 'test_flow.js'
      }
    });
    
    console.log('✅ Dialogue uploaded');
    return response.data;
  } catch (error) {
    console.error('❌ Error uploading dialogue:', error.message);
    throw error;
  }
}

async function compareWithDialogue(diarizationData, dialogueText) {
  console.log('\n🔄 Step 3: Comparing diarization with dialogue...');
  
  try {
    const response = await axios.post(`${BASE_URL}/api/compare-with-dialogue`, {
      groundTruthText: dialogueText,
      diarizationData: diarizationData
    });
    
    console.log('✅ Comparison completed');
    
    if (response.data.groundTruthMetrics) {
      const metrics = response.data.groundTruthMetrics;
      
      console.log('\n📊 Comparison Results:');
      if (metrics.nextLevel) {
        console.log(`   NextLevel: ${metrics.nextLevel.matchPercent}%`);
        console.log(`     Matched: ${metrics.nextLevel.matchedWords}/${metrics.nextLevel.totalWords}`);
        console.log(`     Unmatched: ${metrics.nextLevel.unmatchedWords}`);
        console.log(`     Extra: ${metrics.nextLevel.extraWords}`);
      }
      
      if (metrics.speechmatics) {
        console.log(`   Speechmatics: ${metrics.speechmatics.matchPercent}%`);
        console.log(`     Matched: ${metrics.speechmatics.matchedWords}/${metrics.speechmatics.totalWords}`);
        console.log(`     Unmatched: ${metrics.speechmatics.unmatchedWords}`);
        console.log(`     Extra: ${metrics.speechmatics.extraWords}`);
      }
      
      if (metrics.comparison) {
        const comp = metrics.comparison;
        const status = comp.nextLevelBetter ? '✅ BETTER' : '❌ WORSE';
        console.log(`   ${status}: NextLevel is ${comp.nextLevelBetter ? 'BETTER' : 'WORSE'} by ${Math.abs(comp.improvement).toFixed(1)}%`);
        
        if (!comp.nextLevelBetter) {
          console.error('\n❌ FAILURE: NextLevel should be better than Speechmatics!');
          return false;
        } else {
          console.log('\n✅ SUCCESS: NextLevel is better than Speechmatics!');
          return true;
        }
      }
    }
    
    return true;
  } catch (error) {
    console.error('❌ Error comparing:', error.message);
    if (error.response) {
      console.error('   Response:', error.response.data);
    }
    throw error;
  }
}

async function runTest() {
  console.log('🚀 Starting diarization flow test...');
  console.log(`   Base URL: ${BASE_URL}`);
  console.log(`   Audio file: ${TEST_AUDIO_FILE}`);
  
  try {
    // Step 1: Upload audio and get diarization results
    const diarizationData = await uploadAudio(TEST_AUDIO_FILE);
    
    // Wait a bit for processing to complete
    await sleep(2000);
    
    // Step 2: Upload dialogue if provided
    if (TEST_DIALOGUE_FILE && fs.existsSync(TEST_DIALOGUE_FILE)) {
      const dialogueText = fs.readFileSync(TEST_DIALOGUE_FILE, 'utf8');
      await uploadDialogue(dialogueText);
      
      // Step 3: Compare
      const isSuccess = await compareWithDialogue(diarizationData, dialogueText);
      
      if (isSuccess) {
        console.log('\n✅ All tests passed!');
        process.exit(0);
      } else {
        console.log('\n❌ Tests failed!');
        process.exit(1);
      }
    } else {
      console.log('\n⚠️  No dialogue file provided. Skipping comparison test.');
      console.log('   Set TEST_DIALOGUE_FILE environment variable to test comparison.');
      console.log('✅ Audio diarization test passed!');
      process.exit(0);
    }
  } catch (error) {
    console.error('\n❌ Test failed with error:', error.message);
    process.exit(1);
  }
}

// Run the test
if (require.main === module) {
  runTest();
}

module.exports = { uploadAudio, uploadDialogue, compareWithDialogue };



