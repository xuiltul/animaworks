#!/usr/bin/env bash

set -euo pipefail

log() {
  printf 'gpu-power-guard: %s\n' "$*" >&2
}

warn() {
  log "warning: $*"
}

power_limit_w="${GPU_POWER_LIMIT_W:-280}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi was not found; skipping GPU power guard"
  exit 0
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  warn "no NVIDIA GPU was detected; skipping GPU power guard"
  exit 0
fi

if ! nvidia-smi -pm 1 >/dev/null 2>&1; then
  warn "failed to enable NVIDIA persistence mode"
  exit 1
fi

if ! nvidia-smi -pl "${power_limit_w}" >/dev/null 2>&1; then
  warn "failed to set NVIDIA GPU power limit to ${power_limit_w}W"
  exit 1
fi

if ! nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits |
  awk -v target="${power_limit_w}" '
    BEGIN { ok = 1; seen = 0 }
    {
      seen = 1
      gsub(/[[:space:]]/, "", $0)
      value = $0 + 0
      if (value < target - 0.5 || value > target + 0.5) {
        ok = 0
      }
    }
    END { exit (seen && ok) ? 0 : 1 }
  '; then
  warn "NVIDIA GPU power limit verification failed; expected ${power_limit_w}W"
  nvidia-smi --query-gpu=index,name,power.limit --format=csv,noheader,nounits >&2 || true
  exit 1
fi

log "enabled persistence mode and verified NVIDIA GPU power limit at ${power_limit_w}W"
