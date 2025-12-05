/**
 * Тест для діагностики класифікації
 */

const { analyzeText } = require('./lib/textAnalysis');

// Тест 1: Простий збіг
console.log('=== TEST 1: Simple match ===');
const mdText = "Hello world";
const general = { segments: [{ text: "Hello world", start: 1, end: 2 }] };
const speaker1 = { segments: [{ text: "Hello world", start: 1, end: 2 }] };
const speaker2 = { segments: [] };

for (const source of [general, speaker1, speaker2]) {
  for (const seg of source.segments) {
    if (seg.text.includes(mdText)) {
      console.log("✅ Found:", seg.text);
    }
  }
}

// Тест 2: Після нормалізації
console.log('\n=== TEST 2: After normalization ===');
const norm = (s) => s.toLowerCase().replace(/[,.!?;]/g, '');
const mdNormalized = norm(mdText);
const segNormalized = norm("Hello, world!");

console.log("MD normalized:", mdNormalized);
console.log("Segment normalized:", segNormalized);
console.log("Match:", segNormalized.includes(mdNormalized) || mdNormalized.includes(segNormalized));

// Тест 3: Повний payload
console.log('\n=== TEST 3: Full payload test ===');
const testPayload = {
  general: {
    segments: [
      { text: "Hello world", start: 1, end: 2 }
    ]
  },
  speaker1: {
    segments: [
      { text: "Hello world", start: 1, end: 2 }
    ]
  },
  speaker2: {
    segments: []
  },
  markdown: `| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | Hello world | 1.0 | 2.0 |`
};

const result = analyzeText(testPayload);
console.log('Result:', {
  Blue: result.Blue.length,
  Green: result.Green.length,
  Red: result.Red.length,
  BlueSamples: result.Blue.slice(0, 2),
  GreenSamples: result.Green.slice(0, 2),
  RedSamples: result.Red.slice(0, 2)
});

// Тест 4: З пунктуацією
console.log('\n=== TEST 4: With punctuation ===');
const testPayload2 = {
  general: {
    segments: [
      { text: "Hello, world!", start: 1, end: 2 }
    ]
  },
  speaker1: {
    segments: [
      { text: "Hello, world!", start: 1, end: 2 }
    ]
  },
  speaker2: {
    segments: []
  },
  markdown: `| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | Hello world | 1.0 | 2.0 |`
};

const result2 = analyzeText(testPayload2);
console.log('Result with punctuation:', {
  Blue: result2.Blue.length,
  Green: result2.Green.length,
  Red: result2.Red.length
});




