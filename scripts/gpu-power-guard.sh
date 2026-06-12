#!/usr/bin/env bash

set -u

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
  warn "failed to enable NVIDIA persistence mode; skipping GPU power limit"
  exit 0
fi

if ! nvidia-smi -pl "${power_limit_w}" >/dev/null 2>&1; then
  warn "failed to set NVIDIA GPU power limit to ${power_limit_w}W"
  exit 0
fi

log "enabled persistence mode and set NVIDIA GPU power limit to ${power_limit_w}W"
