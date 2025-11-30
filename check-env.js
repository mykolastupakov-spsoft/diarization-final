#!/usr/bin/env node
// Скрипт для перевірки завантаження .env файлу
require('dotenv').config({ path: require('path').join(__dirname, '.env') });

console.log('🔍 Перевірка .env файлу:');
console.log('');

const keys = [
  'AUDIOSHAKE_API_KEY',
  'OPENROUTER_API_KEY',
  'OPENAI_API_KEY',
  'SPEECHMATICS_API_KEY'
];

keys.forEach(key => {
  const value = process.env[key];
  if (value) {
    const preview = value.substring(0, 30) + '...';
    const isPlaceholder = value.includes('your_') || value.includes('_here');
    const status = isPlaceholder ? '⚠️  PLACEHOLDER' : '✅ OK';
    console.log(`${status} ${key}:`);
    console.log(`   Значення: ${preview}`);
    console.log(`   Довжина: ${value.length} символів`);
    if (isPlaceholder) {
      console.log(`   ⚠️  Це placeholder! Потрібно замінити на реальний ключ.`);
    }
    console.log('');
  } else {
    console.log(`❌ ${key}: NOT SET`);
    console.log('');
  }
});

console.log('📁 Шлях до .env файлу:', require('path').join(__dirname, '.env'));
console.log('📁 Поточна директорія:', __dirname);

