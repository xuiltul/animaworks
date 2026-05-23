#!/usr/bin/env bash
# LoCoMo Legacy scope_all smoke — 1 conversation regression check.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_NEO4J=0
for arg in "$@"; do
  if [[ "$arg" == "--skip-neo4j" ]]; then
    SKIP_NEO4J=1
  fi
done

if [[ "$SKIP_NEO4J" -eq 0 ]]; then
  if command -v docker >/dev/null 2>&1; then
    if docker compose -f docker-compose.neo4j.yml ps neo4j 2>/dev/null | grep -q healthy; then
      echo "Neo4j: healthy (informational only for Legacy smoke)"
    else
      echo "Neo4j: not healthy — continuing (Legacy smoke does not require Neo4j)"
    fi
  fi
fi

if [[ -z "${OPENAI_API_BASE:-}" && -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "ERROR: OPENAI_API_BASE (or OPENAI_BASE_URL) is required for LoCoMo smoke." >&2
  exit 1
fi

DATA="benchmarks/locomo/data/locomo10.json"
if [[ ! -f "$DATA" ]]; then
  echo "ERROR: dataset missing: $DATA" >&2
  echo "Download: wget -O $DATA https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="/tmp/locomo_smoke_${TS}.json"
PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

echo "Running Legacy scope_all smoke (1 conversation, top-k=10)..."
echo "Output: $OUT"

"$PYTHON" -m benchmarks.locomo.runner \
  --mode scope_all \
  --conversations 1 \
  --top-k 10 \
  --output /tmp/locomo_smoke_run

LATEST="$(ls -t /tmp/locomo_smoke_run/*_scope_all.json 2>/dev/null | head -1 || true)"
if [[ -z "$LATEST" ]]; then
  echo "ERROR: runner did not produce scope_all JSON under /tmp/locomo_smoke_run" >&2
  exit 1
fi

cp "$LATEST" "$OUT"
echo "Smoke complete: $OUT"

BASELINE="benchmarks/locomo/baselines/legacy_scope_all_20260522.json"
"$PYTHON" - <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
run = json.loads(out.read_text(encoding="utf-8"))
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
summary = run.get("summary") or {}
overall = float(summary.get("overall_f1", 0.0))
open_dom = float((summary.get("by_category") or {}).get("open_domain", {}).get("f1", 0.0))
b_overall = float(baseline["overall_f1"])
b_open = float(baseline["by_category"]["open_domain"]["f1"])
d_overall = float(baseline["thresholds"]["overall_f1_min_delta"])
d_open = float(baseline["thresholds"]["open_domain_f1_min_delta"])
print(f"overall_f1={overall:.4f} (baseline {b_overall:.4f}, min {b_overall - d_overall:.4f})")
print(f"open_domain_f1={open_dom:.4f} (baseline {b_open:.4f}, min {b_open - d_open:.4f})")
if overall < b_overall - d_overall:
    raise SystemExit(f"FAIL: overall_f1 regression ({overall:.4f} < {b_overall - d_overall:.4f})")
if open_dom < b_open - d_open:
    raise SystemExit(f"FAIL: open_domain regression ({open_dom:.4f} < {b_open - d_open:.4f})")
print("PASS: within baseline thresholds")
PY
"$OUT" "$BASELINE"

echo "locomo_legacy_smoke: OK"
