#!/usr/bin/env bash
# Wait for the background training run to finish, then render the LinkedIn
# progression video. Keyed on the training log's completion line so we only
# start rendering once the 10 training envs have exited and freed their RAM
# (the machine has 14 GB and no swap — running both at once OOMs the desktop).
set -u
cd /home/balaji/mirte_balance_rl

LOG=runs/ppo_gimbal/train.log
PY=/home/balaji/venvs/mirte_rl/bin/python
TRAIN_PID=${TRAIN_PID:-29127}

echo "[auto-video] waiting for training to end (finish OR process exit; PID $TRAIN_PID)…"
while true; do
    if grep -q "saved model and normalization stats" "$LOG" 2>/dev/null; then
        echo "[auto-video] training finished cleanly."
        break
    fi
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        echo "[auto-video] training process gone (crash/OOM?) — rendering from the"
        echo "             snapshots that exist so the video isn't lost."
        break
    fi
    sleep 30
done
echo "[auto-video] letting memory settle, rendering in 15 s…"
sleep 15

MUJOCO_GL=egl "$PY" record_progress.py \
    --run runs/ppo_gimbal \
    --out mirte_rl_progress.mp4 \
    > runs/ppo_gimbal/video.log 2>&1

echo "[auto-video] finished -> mirte_rl_progress.mp4"
