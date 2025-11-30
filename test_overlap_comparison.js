#!/usr/bin/env node
/**
 * Test script to compare overlap diarization results with expected format
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const SERVER_URL = process.env.TEST_SERVER_URL || 'http://localhost:3000';
const TEST_FILE = process.argv[2] || path.join(__dirname, 'audio examples', 'Call center 1.wav');

// Expected transcript format (from user's example)
const EXPECTED_TRANSCRIPT = `Hi I'm Jessica I'm calling on behalf of Future Health to reconfirm your appointment with our doctor Are you available for a short conversation right now

Yeah Jessica I do available What do you have for me

Great To ensure I have the correct information for you could you please confirm your date of birth for me please

Uh April 29th 1980

Thank you I see you have an appointment scheduled with Doctor Eleanor Vance in general practitioner GP

Can you please make the correction in my date of birth

I'm not able to update your date of birth directly but I can connect you with our team Who can help with that Would you like me to transfer you to a human agent right now

No let's Let's uh move forward to the schedule

No problem Are you looking to reschedule your existing appointment with Doctor Eleanor Vance or would you like to book a completely new appointment Of course Your appointment is with Doctor Eleanor Vance General practitioner GP on June 25th at 1:00 pm at Future Health Harley Street Clinic 120 Harley Street London one seven JS Please arrive 15 minutes early and bring your insurance card and any necessary documents Would you like me to send you a reminder via SMS or email

I want to reschedule this one

Sure we can do that Would you like to see available dates with Doctor Eleanor Vance or are you open to another General Practitioner GP at a different location

I'd like this appointment to be the soonest possible

Got it The soonest available dates with doctor Eleanor Vance General Practitioner GP hour June 27th June 20th

9th

Or July 3rd Would any of these days work for you Absolutely You can Would you like to stick with a general practitioner GP or are you looking for a different specialty If GP is fine I can check the soonest dates with other available GPS for you

I have some issues with lungs Can I visit the doctor

Yes you can For lung issues you might want to see a pulmonologist Would you like to check the soonest available appointments with a pulmonology specialist

Yes please

The soonest available dates with our pulmonologist doctor Chloe Adams are June 30th July 2nd or July 6th Do any of these days work for you

Any of these works

Great for June 30th The available times with Doctor Chloe Adams are 9:15 a.m. 10 a.m. or 10:45 a.m. Which of these times would suit you best Perfect I've booked your appointment with Doctor Chloe Adams pulmonologist on June 30th at 10 a.m. at Future Health City Medical Hub 15 Bishopsgate London EC 2N3 AR Would you like me to send you a reminder for this appointment via SMS or email

Yes SMS would be great

All set You'll get an SMS reminder for your appointment with Doctor Chloe Adams on June 30th No problem I'll send you both an SMS and an email reminder for your appointment Do you have all the information you need or is there anything else I can help with today

We are good for today

Glad to hear everything's good for today We look forward to seeing you at your appointment Have a great day

Thanks Bye`;

async function main() {
  console.log('🧪 Testing overlap diarization and comparing results...\n');
  
  try {
    // Step 1: Run overlap diarization
    console.log('📤 Step 1: Uploading audio and running overlap diarization...');
    const result = await runOverlapDiarization(TEST_FILE);
    
    if (!result || !result.success) {
      console.error('❌ Overlap diarization failed');
      console.error(JSON.stringify(result, null, 2));
      return;
    }
    
    // Step 2: Extract and compare transcript
    console.log('\n📝 Step 2: Extracting transcript from voice tracks...');
    const voiceTracks = result.voiceTracks || [];
    const combinedTranscript = extractTranscriptFromVoiceTracks(voiceTracks);
    
    console.log(`\n📊 Transcript length: ${combinedTranscript.length} characters`);
    console.log(`📊 Expected length: ${EXPECTED_TRANSCRIPT.length} characters`);
    
    // Step 3: Get final diarization result
    console.log('\n📋 Step 3: Analyzing final diarization result...');
    const finalResult = result.correctedDiarization;
    const segments = finalResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
    
    console.log(`\n📊 Final segments count: ${segments.length}`);
    console.log(`📊 Overlap segments: ${segments.filter(s => s.overlap).length}`);
    
    // Step 4: Compare with expected format
    console.log('\n🔍 Step 4: Comparing transcript format...');
    compareTranscripts(combinedTranscript, EXPECTED_TRANSCRIPT);
    
    // Step 5: Save results for inspection
    const outputPath = path.join(__dirname, 'test_overlap_result.json');
    fs.writeFileSync(outputPath, JSON.stringify({
      transcript: combinedTranscript,
      segments: segments,
      voiceTracks: voiceTracks.map(vt => ({
        speaker: vt.speaker,
        segmentsCount: vt.transcription?.recordings?.[0]?.results?.speechmatics?.segments?.length || 0
      })),
      steps: result.steps
    }, null, 2));
    console.log(`\n💾 Results saved to: ${outputPath}`);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('Response:', JSON.stringify(error.response.data, null, 2));
    }
    process.exit(1);
  }
}

async function runOverlapDiarization(filePath) {
  const form = new FormData();
  form.append('audio', fs.createReadStream(filePath));
  form.append('language', 'en');
  form.append('mode', 'smart');
  form.append('pipelineMode', 'mode3');

  const response = await axios.post(`${SERVER_URL}/api/diarize-overlap`, form, {
    headers: form.getHeaders(),
    maxContentLength: Infinity,
    maxBodyLength: Infinity,
    timeout: 1000 * 60 * 30 // 30 minutes
  });

  return response.data;
}

function extractTranscriptFromVoiceTracks(voiceTracks) {
  const segments = [];
  
  voiceTracks.forEach(track => {
    if (!track || track.error) return;
    const recording = track.transcription?.recordings?.[0];
    const speechmaticsSegments = recording?.results?.speechmatics?.segments || [];
    
    speechmaticsSegments.forEach(segment => {
      segments.push({
        text: segment.text || '',
        start: typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0,
        end: typeof segment.end === 'number' ? segment.end : parseFloat(segment.end) || 0
      });
    });
  });
  
  // Sort by start time
  segments.sort((a, b) => {
    const startDiff = (a.start || 0) - (b.start || 0);
    if (startDiff !== 0) return startDiff;
    return (a.end || 0) - (b.end || 0);
  });
  
  // Combine into single transcript
  return segments
    .map(segment => segment.text || '')
    .filter(text => text.trim().length > 0)
    .join(' ')
    .trim();
}

function compareTranscripts(actual, expected) {
  // Normalize both transcripts (remove extra spaces, lowercase for comparison)
  const normalize = (text) => text.toLowerCase().replace(/\s+/g, ' ').trim();
  
  const actualNormalized = normalize(actual);
  const expectedNormalized = normalize(expected);
  
  if (actualNormalized === expectedNormalized) {
    console.log('✅ Transcripts match exactly!');
    return;
  }
  
  // Find differences
  const actualWords = actualNormalized.split(' ');
  const expectedWords = expectedNormalized.split(' ');
  
  console.log(`⚠️ Transcripts differ:`);
  console.log(`   Actual words: ${actualWords.length}`);
  console.log(`   Expected words: ${expectedWords.length}`);
  
  // Show first 200 characters of each
  console.log(`\n📄 Actual (first 200 chars):`);
  console.log(`   ${actual.substring(0, 200)}...`);
  console.log(`\n📄 Expected (first 200 chars):`);
  console.log(`   ${expected.substring(0, 200)}...`);
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});

