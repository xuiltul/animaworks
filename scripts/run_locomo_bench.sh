#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="$ROOT/benchmarks/locomo/results/bench_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "========== STEP 1: LoCoMo dataset stats =========="
python3 -c "
import json
from pathlib import Path
data = json.loads(Path('benchmarks/locomo/data/locomo10.json').read_text())
sample = data[0]
conv = sample.get('conversation', {})
sessions = [k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')]
print(f'Conversation 0: {len(sessions)} sessions')
qa = sample.get('qa', [])
print(f'QA count: {len(qa)}')
"

echo "========== STEP 2: LLM connection test =========="
.venv/bin/python3 -c "
import litellm
r = litellm.completion(
    model='openai/mlx-community/Qwen3.5-397B-A17B-4bit',
    messages=[{'role': 'user', 'content': 'Say hi in one word.'}],
    temperature=0.0,
    max_tokens=10,
    api_base='http://100.72.124.21:8001/v1',
    api_key='dummy',
    extra_body={'chat_template_kwargs': {'enable_thinking': False}},
)
print('Response:', r.choices[0].message.content)
"

echo "========== STEP 3: Run LoCoMo Neo4j benchmark =========="
echo "Using model: openai/mlx-community/Qwen3.5-397B-A17B-4bit"
echo "top_k: 10, conversations: 1"
echo "Start time: $(date)"

.venv/bin/python -m benchmarks.locomo.run_neo4j \
    --conversations 1 \
    --top-k 10 \
    2>&1

echo "End time: $(date)"
echo "========== DONE. Log: $LOG =========="
