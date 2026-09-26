#!/usr/bin/env bash
# Deploy pi/lekiwi_ws.py to the robot and restart the bridge.
#
# The file is streamed over the SSH session rather than scp'd separately, so a
# password-authenticated Pi prompts once per step instead of twice.
#
#   ./deploy-pi.sh <user>@192.168.0.119          deploy and restart
#   ./deploy-pi.sh <user>@192.168.0.119 --key    install your SSH key first
#
# The restart reuses the command line the running bridge already has, so a
# virtualenv interpreter, a custom --port or LEKIWI_PORT all survive.
set -euo pipefail

TARGET="${1:?usage: ./deploy-pi.sh user@host [--key]}"
cd "$(dirname "$0")"
[ -f pi/lekiwi_ws.py ] || { echo "pi/lekiwi_ws.py not found"; exit 1; }

if [ "${2:-}" = "--key" ]; then
  echo "==> installing $HOME/.ssh/id_ed25519.pub on $TARGET"
  ssh -o StrictHostKeyChecking=accept-new "$TARGET" \
    'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED' \
    < "$HOME/.ssh/id_ed25519.pub"
fi

echo "==> deploying pi/lekiwi_ws.py to $TARGET"
ssh -o StrictHostKeyChecking=accept-new "$TARGET" 'bash -s' <<'REMOTE'
set -e
PID=$(pgrep -f '[l]ekiwi_ws\.py' | head -1 || true)
if [ -n "$PID" ]; then
  tr '\0' ' ' < /proc/$PID/cmdline > /tmp/lekiwi_cmd
  echo "running bridge: $(cat /tmp/lekiwi_cmd)"
else
  echo "python3 $HOME/lekiwi_ws.py --host 0.0.0.0 --port 8765" > /tmp/lekiwi_cmd
  echo "no bridge running; will start the documented command"
fi
REMOTE

# The heredoc above consumed stdin, so send the file in its own step.
ssh -o StrictHostKeyChecking=accept-new "$TARGET" 'cat > /tmp/lekiwi_ws.py.new && wc -c < /tmp/lekiwi_ws.py.new' < pi/lekiwi_ws.py

echo "==> restarting the bridge"
ssh -o StrictHostKeyChecking=accept-new "$TARGET" 'bash -s' <<'REMOTE'
set -e
python3 -m py_compile /tmp/lekiwi_ws.py.new || { echo "REFUSING: new file does not compile"; exit 1; }
TARGET_PATH=$(awk '{for(i=1;i<=NF;i++) if ($i ~ /lekiwi_ws\.py$/) {print $i; exit}}' /tmp/lekiwi_cmd)
[ -n "$TARGET_PATH" ] || TARGET_PATH="$HOME/lekiwi_ws.py"
[ -f "$TARGET_PATH" ] && cp "$TARGET_PATH" "$TARGET_PATH.bak" || true
cp /tmp/lekiwi_ws.py.new "$TARGET_PATH"
echo "installed to $TARGET_PATH"
pkill -f '[l]ekiwi_ws\.py' || true
sleep 2
cd "$(dirname "$TARGET_PATH")"
setsid nohup $(cat /tmp/lekiwi_cmd) > "$HOME/lekiwi.log" 2>&1 < /dev/null &
sleep 5
if pgrep -f '[l]ekiwi_ws\.py' > /dev/null; then
  echo DEPLOY_OK
else
  echo DEPLOY_FAILED
  tail -30 "$HOME/lekiwi.log"
  exit 1
fi
REMOTE
