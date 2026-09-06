#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#   Smart Plant Intelligence System : One-Command Startup Script
#   Usage: bash start.sh
#
#   This script starts BOTH the FastAPI server AND the ngrok tunnel
#   in the background, then prints your public URL.
# ═══════════════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

echo ""
echo "🌿 Smart Plant Intelligence System"
echo "══════════════════════════════════"

# ── Kill any previous instances ───────────────────────────────────────
echo "⏹  Stopping any previous server instances..."
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f ngrok 2>/dev/null
sleep 1

# ── Start FastAPI server ──────────────────────────────────────────────
echo "🚀 Starting FastAPI server on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!
sleep 2

# ── Start ngrok tunnel ────────────────────────────────────────────────
echo "🌐 Starting ngrok tunnel on ${NGROK_DOMAIN:-your-subdomain.ngrok-free.dev} ..."
ngrok http 8000 --url=${NGROK_DOMAIN:-your-subdomain.ngrok-free.dev} --log=stdout > /tmp/ngrok_smart_plant.log &
NGROK_PID=$!
sleep 3

# ── Get the public URL from ngrok API ────────────────────────────────
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys,json
tunnels = json.load(sys.stdin).get('tunnels', [])
for t in tunnels:
    if t.get('proto') == 'https':
        print(t['public_url'])
        break
" 2>/dev/null)

echo ""
echo "══════════════════════════════════════════════════════"
echo "✅ System is LIVE!"
echo ""
if [ -n "$NGROK_URL" ]; then
  echo "  🌐 Public URL  :  $NGROK_URL"
  echo "  📡 Dashboard   :  $NGROK_URL"
  echo "  🔌 ESP8266 URL :  $NGROK_URL/predict/compact"
else
  echo "  ⚠️  ngrok URL not detected. Check: http://localhost:4040"
fi
echo "  🖥️  Local URL   :  http://localhost:8000"
echo "══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop everything."
echo ""

# ── Trap Ctrl+C to stop both processes ───────────────────────────────
trap "echo '⏹  Shutting down...'; kill $SERVER_PID $NGROK_PID 2>/dev/null; exit" INT TERM

# ── Keep running ─────────────────────────────────────────────────────
wait
