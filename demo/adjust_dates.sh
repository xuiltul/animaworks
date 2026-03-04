#!/usr/bin/env bash
set -euo pipefail

# Adjust timestamps in demo example data so that the 3-day logs appear
# as "2 days ago → yesterday → today" relative to the current date.
#
# Usage: ./adjust_dates.sh <examples_dir>
#   examples_dir: path containing character dirs + channels/
#                 e.g. /demo/examples/ja

EXAMPLES_DIR="${1:?Usage: adjust_dates.sh <examples_dir>}"

if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "ERROR: Directory not found: $EXAMPLES_DIR"
    exit 1
fi

# Original dates baked into the example files
ORIG_D1="2026-03-01"
ORIG_D2="2026-03-02"
ORIG_D3="2026-03-03"

# Target dates: today and the two preceding days
TODAY=$(date +%Y-%m-%d)
YESTERDAY=$(date -d "$TODAY - 1 day" +%Y-%m-%d 2>/dev/null \
         || date -v-1d +%Y-%m-%d)
DAY_BEFORE=$(date -d "$TODAY - 2 days" +%Y-%m-%d 2>/dev/null \
          || date -v-2d +%Y-%m-%d)

if [ "$ORIG_D1" = "$DAY_BEFORE" ] && [ "$ORIG_D2" = "$YESTERDAY" ] && [ "$ORIG_D3" = "$TODAY" ]; then
    echo "Dates already match today — no adjustment needed."
    exit 0
fi

echo "Adjusting dates: $ORIG_D1→$DAY_BEFORE  $ORIG_D2→$YESTERDAY  $ORIG_D3→$TODAY"

adjust_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return
    fi
    # Two-pass replacement to avoid cascade (e.g. D3→TODAY then TODAY→YESTERDAY)
    sed -i \
        -e "s/${ORIG_D3}/__PLACEHOLDER_D3__/g" \
        -e "s/${ORIG_D2}/__PLACEHOLDER_D2__/g" \
        -e "s/${ORIG_D1}/__PLACEHOLDER_D1__/g" \
        "$file"
    sed -i \
        -e "s/__PLACEHOLDER_D3__/${TODAY}/g" \
        -e "s/__PLACEHOLDER_D2__/${YESTERDAY}/g" \
        -e "s/__PLACEHOLDER_D1__/${DAY_BEFORE}/g" \
        "$file"
}

rename_jsonl() {
    local dir="$1"
    [ -d "$dir" ] || return
    for f in "$dir"/*.jsonl; do
        [ -f "$f" ] || continue
        local base
        base=$(basename "$f")
        local new_name="$base"
        new_name="${new_name//$ORIG_D1/$DAY_BEFORE}"
        new_name="${new_name//$ORIG_D2/$YESTERDAY}"
        new_name="${new_name//$ORIG_D3/$TODAY}"
        if [ "$base" != "$new_name" ]; then
            mv "$dir/$base" "$dir/$new_name"
        fi
        adjust_file "$dir/$new_name"
    done
}

# Process character directories (activity_log JSONL + state files)
for char_dir in "$EXAMPLES_DIR"/*/; do
    char_name="$(basename "$char_dir")"
    [ "$char_name" = "channels" ] && continue

    rename_jsonl "$char_dir/activity_log"

    if [ -f "$char_dir/state/current_task.md" ]; then
        adjust_file "$char_dir/state/current_task.md"
    fi
done

# Process shared channel files
rename_jsonl "$EXAMPLES_DIR/channels"

echo "Date adjustment complete."
