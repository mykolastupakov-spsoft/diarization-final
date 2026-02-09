#!/bin/bash

# Скрипт для очищення кешу LLM та перезапуску сервера

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🗑️  Очищення кешу LLM..."
rm -f cache/llm_responses/*.json
CACHE_COUNT=$(find cache/llm_responses -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "✅ Кеш очищено. Залишилось файлів: $CACHE_COUNT"

echo ""
echo "🛑 Зупинка сервера..."
SERVER_PID=$(ps aux | grep -E "node.*server\.js" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$SERVER_PID" ]; then
  kill $SERVER_PID 2>/dev/null
  sleep 2
  echo "✅ Сервер зупинено (PID: $SERVER_PID)"
else
  echo "⚠️  Сервер не знайдено"
fi

echo ""
echo "🚀 Запуск сервера..."
nohup node server.js > server.log 2>&1 &
sleep 3

if ps aux | grep -E "node.*server\.js" | grep -v grep > /dev/null; then
  echo "✅ Сервер запущено успішно"
  echo ""
  echo "📋 Останні рядки логу:"
  tail -10 server.log
else
  echo "❌ Помилка запуску сервера. Перевірте server.log"
  exit 1
fi

