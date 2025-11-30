#!/usr/bin/env python3
"""Тестовий скрипт для перевірки завантаження .env файлу"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

print("=" * 60)
print("🔍 ДІАГНОСТИКА ЗАВАНТАЖЕННЯ .env ФАЙЛУ")
print("=" * 60)
print()

# Поточна робоча директорія
cwd = os.getcwd()
print(f"📁 Поточна робоча директорія (CWD): {cwd}")

# Директорія скрипта
script_dir = Path(__file__).parent.absolute()
print(f"📁 Директорія скрипта: {script_dir}")
print()

# Шляхи до .env файлу
env_paths = [
    script_dir / ".env",
    Path(".env"),
    Path.cwd() / ".env",
]

print("🔍 Шукаємо .env файл:")
for env_path in env_paths:
    exists = env_path.exists()
    is_file = env_path.is_file() if exists else False
    status = "✅" if (exists and is_file) else "❌"
    print(f"  {status} {env_path}")
    if exists and is_file:
        size = env_path.stat().st_size
        mtime = env_path.stat().st_mtime
        print(f"     Розмір: {size} байт")
        print(f"     Остання зміна: {mtime}")
print()

# Спробувати знайти через find_dotenv
env_file = find_dotenv()
if env_file:
    print(f"✅ find_dotenv() знайшов: {env_file}")
else:
    print("❌ find_dotenv() не знайшов .env файл")
print()

# Завантажити .env
print("📥 Завантаження .env файлу:")
for env_path in env_paths:
    if env_path.exists() and env_path.is_file():
        print(f"  Спробуємо завантажити: {env_path}")
        result = load_dotenv(dotenv_path=str(env_path), override=True, verbose=True)
        print(f"  Результат: {result}")
        if result:
            break
print()

# Перевірити змінні
print("🔑 Перевірка змінних середовища:")
keys_to_check = [
    "AUDIOSHAKE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
]

for key in keys_to_check:
    value = os.getenv(key)
    if value:
        preview = value[:30] + "..." if len(value) > 30 else value
        is_placeholder = "your_" in value.lower() or "_here" in value.lower()
        status = "⚠️  PLACEHOLDER" if is_placeholder else "✅ OK"
        print(f"  {status} {key}:")
        print(f"     Значення: {preview}")
        print(f"     Довжина: {len(value)} символів")
        if is_placeholder:
            print(f"     ⚠️  Це placeholder! Потрібно замінити на реальний ключ.")
    else:
        print(f"  ❌ {key}: NOT SET")
    print()

# Показати всі змінні що починаються з AUDIOSHAKE
print("🔍 Всі змінні середовища з 'AUDIOSHAKE':")
for key, value in os.environ.items():
    if 'AUDIOSHAKE' in key.upper():
        preview = value[:30] + "..." if len(value) > 30 else value
        print(f"  {key} = {preview}")

print()
print("=" * 60)

