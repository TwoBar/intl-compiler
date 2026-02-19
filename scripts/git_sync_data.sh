#!/usr/bin/env bash
# git_sync_data.sh — commits and pushes data/ every 2 minutes
# Runs indefinitely; kill with: kill $(cat /tmp/git_sync_data.pid)

set -euo pipefail

INTERVAL=120
LOG=/workspace/scripts/git_sync_data.log
REPO=/workspace

cd "$REPO"

echo "[git-sync] started at $(date) — interval=${INTERVAL}s" | tee "$LOG"
echo $$ > /tmp/git_sync_data.pid

git config user.name  "INTL Agent"
git config user.email "agent@intl-lang.dev"
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/intl-compiler.git"

while true; do
  sleep "$INTERVAL"

  # Count lines in each JSONL to build a meaningful commit message
  PF_TRAIN=$(wc -l < data/python_fastapi/train.jsonl 2>/dev/null || echo 0)
  PF_VAL=$(wc -l   < data/python_fastapi/validation.jsonl 2>/dev/null || echo 0)
  PF_CORR=$(wc -l  < data/python_fastapi/corrections.jsonl 2>/dev/null || echo 0)

  SP_TRAIN=0; SP_VAL=0; SP_CORR=0
  if [ -d data/sql_postgres ]; then
    SP_TRAIN=$(wc -l < data/sql_postgres/train.jsonl 2>/dev/null || echo 0)
    SP_VAL=$(wc -l   < data/sql_postgres/validation.jsonl 2>/dev/null || echo 0)
    SP_CORR=$(wc -l  < data/sql_postgres/corrections.jsonl 2>/dev/null || echo 0)
  fi

  # Only commit if there are actual changes
  if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard data/)" ]; then
    echo "[git-sync] $(date '+%H:%M:%S') — no changes, skipping" | tee -a "$LOG"
    continue
  fi

  MSG="data: python_fastapi ${PF_TRAIN}tr/${PF_VAL}val/${PF_CORR}corr | sql_postgres ${SP_TRAIN}tr/${SP_VAL}val/${SP_CORR}corr"

  git add -A data/
  git commit -m "$MSG" && \
  git push origin main && \
  echo "[git-sync] $(date '+%H:%M:%S') pushed — $MSG" | tee -a "$LOG" || \
  echo "[git-sync] $(date '+%H:%M:%S') push failed — will retry next cycle" | tee -a "$LOG"
done
