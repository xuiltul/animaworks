#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────
PRESET="${PRESET:-en-anime}"
PRESET_DIR="/demo/presets/${PRESET}"
DATA_DIR="${ANIMAWORKS_DATA_DIR:-$HOME/.animaworks}"
CONFIG_JSON="${DATA_DIR}/config.json"

# ── Validation ────────────────────────────────────────────
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "WARNING: ANTHROPIC_API_KEY is not set. Animas will not be able to respond."
    echo "         Set it in demo/.env or pass via -e ANTHROPIC_API_KEY=sk-..."
fi

if [ ! -d "$PRESET_DIR" ]; then
    echo "ERROR: Preset directory not found: ${PRESET_DIR}"
    echo "       Available presets:"
    ls -1 /demo/presets/ 2>/dev/null || echo "       (none)"
    exit 1
fi

# ── First-run initialization ─────────────────────────────
if [ ! -f "$CONFIG_JSON" ]; then
    echo "=== First run detected — initializing AnimaWorks ==="
    echo "Preset: ${PRESET}"

    # 1. Initialize runtime (infrastructure only, no default anima)
    animaworks init --skip-anima
    echo "Runtime directory initialized."

    # 2. Copy company vision if present
    if [ -f "${PRESET_DIR}/vision.md" ]; then
        mkdir -p "${DATA_DIR}/company"
        cp "${PRESET_DIR}/vision.md" "${DATA_DIR}/company/vision.md"
        echo "Company vision installed."
    fi

    # 3. Create animas from character sheets
    for md_file in "${PRESET_DIR}/characters/"*.md; do
        [ -f "$md_file" ] || continue
        name="$(basename "$md_file" .md)"
        role_file="${PRESET_DIR}/roles/${name}.txt"

        create_args=(animaworks anima create --from-md "$md_file")

        if [ -f "$role_file" ]; then
            role="$(sed -n '1p' "$role_file")"
            supervisor="$(sed -n '2p' "$role_file")"

            if [ -n "$role" ]; then
                create_args+=(--role "$role")
            fi
            if [ -n "$supervisor" ]; then
                create_args+=(--supervisor "$supervisor")
            fi
        fi

        echo "Creating anima: ${name}"
        "${create_args[@]}"
    done

    # 4. Apply config overlay (deep-merge into config.json)
    overlay="${PRESET_DIR}/config_overlay.json"
    if [ -f "$overlay" ]; then
        python3 -c "
import json, sys
def deep_merge(base, patch):
    for k, v in patch.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
cfg_path, ovl_path = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = json.load(f)
with open(ovl_path) as f:
    ovl = json.load(f)
deep_merge(cfg, ovl)
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
" "$CONFIG_JSON" "$overlay"
        echo "Config overlay applied."
    fi

    # 5. Copy pre-built character assets
    for char_dir in "${PRESET_DIR}/assets/"*/; do
        [ -d "$char_dir" ] || continue
        char_name="$(basename "$char_dir")"
        target_dir="${DATA_DIR}/animas/${char_name}/assets"
        if [ ! -d "$target_dir" ]; then
            mkdir -p "$target_dir"
        fi
        cp "$char_dir"/* "$target_dir/" 2>/dev/null || true
    done
    echo "Character assets installed."

    # 6. Copy example runtime data (activity logs, state, channels)
    LANG_KEY="${PRESET%%-*}"  # ja or en
    EXAMPLES_DIR="/demo/examples/${LANG_KEY}"
    if [ -d "$EXAMPLES_DIR" ]; then
        # Adjust timestamps to be relative to today (in-place, container is ephemeral)
        if [ -f /demo/adjust_dates.sh ]; then
            /demo/adjust_dates.sh "$EXAMPLES_DIR"
        fi

        for char_dir in "$EXAMPLES_DIR"/*/; do
            char_name="$(basename "$char_dir")"
            [ "$char_name" = "channels" ] && continue
            target_dir="${DATA_DIR}/animas/${char_name}"
            if [ -d "$target_dir" ]; then
                cp -r "$char_dir"/* "$target_dir/" 2>/dev/null || true
            fi
        done
        if [ -d "$EXAMPLES_DIR/channels" ]; then
            mkdir -p "${DATA_DIR}/shared/channels"
            cp "$EXAMPLES_DIR/channels/"* "${DATA_DIR}/shared/channels/" 2>/dev/null || true
        fi
        echo "Example runtime data installed."
    fi

    echo "=== Initialization complete ==="
else
    echo "Existing configuration found — skipping initialization."
fi

# ── Start server ──────────────────────────────────────────
echo "Starting AnimaWorks server on port 18500..."
exec animaworks start --host 0.0.0.0 --port 18500 --foreground
