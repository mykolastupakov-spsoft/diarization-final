#!/usr/bin/env node
/**
 * Automated testing and fixing script for overlap diarization
 * Runs tests and compares results until they match expected format
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const TEST_FILE = path.join(__dirname, 'audio examples', 'Call center 1.wav');
const MAX_ITERATIONS = 5;

console.log('🔧 Starting automated testing and fixing...\n');

// Kill any existing server
try {
  execSync('pkill -f "node.*server.js"', { stdio: 'ignore' });
  console.log('✅ Stopped existing server');
} catch (e) {
  // Ignore if no server running
}

// Start server
console.log('🚀 Starting server...');
const serverProcess = execSync('npm start > /dev/null 2>&1 &', { cwd: __dirname });
await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds

console.log('🧪 Running test...\n');

// Run test
try {
  const result = execSync(`node test_overlap_comparison.js "${TEST_FILE}"`, {
    cwd: __dirname,
    encoding: 'utf8'
  });
  
  console.log(result);
  
  // Check if results match
  const resultData = JSON.parse(fs.readFileSync(path.join(__dirname, 'test_overlap_result.json'), 'utf8'));
  const transcript = resultData.transcript;
  
  // Expected transcript (normalized)
  const expected = `Hi I'm Jessica I'm calling on behalf of Future Health to reconfirm your appointment with our doctor Are you available for a short conversation right now

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

  const normalize = (text) => text.toLowerCase().replace(/\s+/g, ' ').trim();
  const transcriptNormalized = normalize(transcript);
  const expectedNormalized = normalize(expected);
  
  if (transcriptNormalized === expectedNormalized) {
    console.log('\n✅ SUCCESS! Transcripts match exactly!');
  } else {
    console.log('\n⚠️ Transcripts still differ');
    console.log(`   Actual length: ${transcript.length}`);
    console.log(`   Expected length: ${expected.length}`);
  }
  
} catch (error) {
  console.error('❌ Error:', error.message);
} finally {
  // Cleanup
  try {
    execSync('pkill -f "node.*server.js"', { stdio: 'ignore' });
  } catch (e) {
    // Ignore
  }
}











