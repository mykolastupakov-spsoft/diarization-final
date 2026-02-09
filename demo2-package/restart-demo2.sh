#!/bin/bash

# Restart Flask + Node for demo2-package

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

stop_pidfile() {
  local name="$1"
  local pidfile="$2"

  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "🛑 Stopping $name (PID: $pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
    else
      echo "⚠️  $name PID file exists, but process not running"
    fi
    rm -f "$pidfile"
  else
    echo "⚠️  $name PID file not found"
  fi
}

if [ ! -x ".venv/bin/python" ]; then
  echo "❌ .venv not found. Create it with: python3.11 -m venv .venv"
  exit 1
fi

stop_pidfile "Flask" "flask_demo2.pid"
stop_pidfile "Node" "node_demo2.pid"

echo "🚀 Starting Flask (port 5005 by default)..."
nohup .venv/bin/python app_demo2.py > flask_demo2.log 2>&1 &
echo $! > flask_demo2.pid
sleep 1

echo "🚀 Starting Node (port 3000 by default)..."
nohup node server.js > node_demo2.log 2>&1 &
echo $! > node_demo2.pid
sleep 1

echo "✅ Restart complete"
