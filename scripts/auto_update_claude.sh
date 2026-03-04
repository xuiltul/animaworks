#!/usr/bin/env bash
# auto_update_claude.sh — Claude Code CLI + claude-agent-sdk 自動更新
#
# - PyPI / npm の最新バージョンを定期チェック
# - 更新があれば自動インストール
# - claude-agent-sdk 更新時は Mode S Anima を自動再起動
#
# Usage:
#   scripts/auto_update_claude.sh          # 通常実行
#   scripts/auto_update_claude.sh --dry-run  # チェックのみ（更新しない）
#   scripts/auto_update_claude.sh --force    # バージョン比較せず強制更新

set -uo pipefail

# ── Config ──────────────────────────────────
ANIMAWORKS_API="http://localhost:18500/api"
LOG_TAG="[auto-update-claude]"
PIP="pip3"
PIP_FLAGS="--break-system-packages --quiet"
SDK_PACKAGE="claude-agent-sdk"
ANIMAWORKS_DIR="$HOME/.animaworks"

DRY_RUN=false
FORCE=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --force)   FORCE=true ;;
  esac
done

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $LOG_TAG $*"; }

# ── Helper: Get installed version ───────────
installed_sdk_version() {
  $PIP show "$SDK_PACKAGE" 2>/dev/null | awk '/^Version:/{print $2}'
}

installed_cli_version() {
  claude --version 2>/dev/null | awk '{print $1; exit}'
}

latest_sdk_version() {
  # pip index versions returns non-zero (120) even on success
  local raw
  raw=$($PIP index versions "$SDK_PACKAGE" 2>/dev/null | head -1) || true
  echo "$raw" | grep -oP '\(([^)]+)\)' | tr -d '()' || true
}

latest_cli_version() {
  npm view @anthropic-ai/claude-code version 2>/dev/null || true
}

# ── Helper: Find Mode S Animas ──────────────
mode_s_animas() {
  local animas_dir="$ANIMAWORKS_DIR/animas"
  [ -d "$animas_dir" ] || return 0
  for status_file in "$animas_dir"/*/status.json; do
    [ -f "$status_file" ] || continue
    local name
    name=$(basename "$(dirname "$status_file")")
    python3 -c "
import json, sys
d = json.load(open('$status_file'))
if d.get('enabled') and str(d.get('model','')).startswith('claude-'):
    print('$name')
" 2>/dev/null || true
  done
}

restart_anima() {
  local name="$1"
  local result
  result=$(curl -sf -X POST "$ANIMAWORKS_API/animas/$name/restart" 2>/dev/null) || result='{"error":"unreachable"}'
  log "  Restart $name: $result"
}

# ── Main ────────────────────────────────────
sdk_updated=false
cli_updated=false

# 1) claude-agent-sdk
cur_sdk=$(installed_sdk_version)
new_sdk=$(latest_sdk_version)
log "claude-agent-sdk: installed=$cur_sdk latest=$new_sdk"

if [ -n "$new_sdk" ] && { [ "$FORCE" = true ] || [ "$cur_sdk" != "$new_sdk" ]; }; then
  if [ "$DRY_RUN" = true ]; then
    log "  [DRY-RUN] Would upgrade $SDK_PACKAGE $cur_sdk → $new_sdk"
  else
    log "  Upgrading $SDK_PACKAGE $cur_sdk → $new_sdk ..."
    if $PIP install $PIP_FLAGS --upgrade "$SDK_PACKAGE" 2>&1 | tail -3; then
      sdk_updated=true
      final_sdk=$(installed_sdk_version)
      log "  Done: $SDK_PACKAGE $final_sdk"
    else
      log "  ERROR: Failed to upgrade $SDK_PACKAGE"
    fi
  fi
else
  log "  Up to date."
fi

# 2) Claude Code CLI
cur_cli=$(installed_cli_version)
new_cli=$(latest_cli_version)
log "Claude Code CLI: installed=$cur_cli latest=$new_cli"

if [ -n "$new_cli" ] && { [ "$FORCE" = true ] || [ "$cur_cli" != "$new_cli" ]; }; then
  if [ "$DRY_RUN" = true ]; then
    log "  [DRY-RUN] Would upgrade Claude Code CLI $cur_cli → $new_cli"
  else
    log "  Upgrading Claude Code CLI $cur_cli → $new_cli ..."
    if claude update 2>&1 | tail -3; then
      cli_updated=true
      final_cli=$(installed_cli_version)
      log "  Done: Claude Code CLI $final_cli"
    else
      log "  ERROR: Failed to upgrade Claude Code CLI"
    fi
  fi
else
  log "  Up to date."
fi

# 3) Restart Mode S Animas if SDK was updated
if [ "$sdk_updated" = true ]; then
  log "SDK updated — restarting Mode S Animas..."
  mapfile -t animas < <(mode_s_animas)
  if [ ${#animas[@]} -eq 0 ]; then
    log "  No active Mode S Animas found."
  else
    for name in "${animas[@]}"; do
      restart_anima "$name"
      sleep 2
    done
    log "  Restarted ${#animas[@]} Anima(s): ${animas[*]}"
  fi
fi

# Summary
if [ "$sdk_updated" = true ] || [ "$cli_updated" = true ]; then
  log "Update complete."
else
  log "No updates needed."
fi
