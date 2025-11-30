#!/bin/bash
# Скрипт для оновлення AUDIOSHAKE_API_KEY в .env файлі

ENV_FILE="/Users/nikolajstupakov/Library/Mobile Documents/com~apple~CloudDocs/SPsoft /diarization-final/.env"

echo "🔧 Оновлення .env файлу"
echo ""
echo "Поточний вміст AUDIOSHAKE_API_KEY:"
grep "^AUDIOSHAKE_API_KEY" "$ENV_FILE" || echo "Не знайдено"
echo ""
echo "Вкажіть ваш реальний AudioShake API ключ:"
read -r API_KEY

if [ -z "$API_KEY" ]; then
    echo "❌ Помилка: API ключ не вказано"
    exit 1
fi

# Перевірка формату (має починатися з ashke_)
if [[ ! "$API_KEY" =~ ^ashke_ ]]; then
    echo "⚠️  Попередження: AudioShake API ключ зазвичай починається з 'ashke_'"
    read -p "Продовжити? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Оновити файл
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|^AUDIOSHAKE_API_KEY=.*|AUDIOSHAKE_API_KEY=$API_KEY|" "$ENV_FILE"
else
    # Linux
    sed -i "s|^AUDIOSHAKE_API_KEY=.*|AUDIOSHAKE_API_KEY=$API_KEY|" "$ENV_FILE"
fi

echo ""
echo "✅ .env файл оновлено!"
echo ""
echo "Перевірка:"
grep "^AUDIOSHAKE_API_KEY" "$ENV_FILE"
echo ""
echo "Тепер перезапустіть сервер."

