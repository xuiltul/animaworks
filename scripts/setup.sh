#!/usr/bin/env bash
set -euo pipefail

# AnimaWorks Setup Script
# Usage: curl -sSL https://raw.githubusercontent.com/xuiltul/animaworks/main/scripts/setup.sh | bash

REPO_URL="https://github.com/xuiltul/animaworks.git"
INSTALL_DIR="animaworks"

echo "=== AnimaWorks Setup ==="
echo ""

# 1. Install uv if not present
if command -v uv &>/dev/null; then
    echo "[OK] uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
    echo "[*] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "[OK] uv installed"
fi

# 2. Install git if not present
if ! command -v git &>/dev/null; then
    echo "[*] Installing git..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq git
    else
        echo "[ERROR] git not found. Please install git first."
        exit 1
    fi
fi

# 3. Clone repository
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[OK] $INSTALL_DIR already cloned"
    cd "$INSTALL_DIR"
    git pull --ff-only 2>/dev/null || true
else
    if [ -d "$INSTALL_DIR" ]; then
        echo "[ERROR] Directory '$INSTALL_DIR' exists but is not a git repo."
        echo "        Remove it or run from a different directory."
        exit 1
    fi
    echo "[*] Cloning repository..."
    git clone "$REPO_URL"
    cd "$INSTALL_DIR"
fi

# 4. Install dependencies
echo "[*] Installing dependencies..."
uv sync

# 5. Copy .env template if needed
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "[OK] .env created from .env.example"
fi

# 6. Link the CLI into ~/.local/bin so `animaworks` works from any directory.
#    The console scripts in .venv/bin pin this repo's interpreter by absolute
#    path, so a symlink resolves to the exact same environment as `uv run`.
BIN_DIR="$HOME/.local/bin"
REPO_DIR="$PWD"
LINKED=0
mkdir -p "$BIN_DIR"
for cmd in animaworks animaworks-tool; do
    target="$REPO_DIR/.venv/bin/$cmd"
    [ -x "$target" ] || continue
    link="$BIN_DIR/$cmd"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        mv "$link" "$link.bak"
        echo "[!] Existing $link moved to $link.bak"
    fi
    ln -sfn "$target" "$link"
    LINKED=1
done
if [ "$LINKED" = "1" ]; then
    echo "[OK] animaworks linked into $BIN_DIR"
fi

case ":$PATH:" in
    *":$BIN_DIR:"*) PATH_HAS_BIN_DIR=1 ;;
    *) PATH_HAS_BIN_DIR=0 ;;
esac

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
if [ "$LINKED" = "1" ] && [ "$PATH_HAS_BIN_DIR" = "1" ]; then
    echo "  animaworks start          # Start server from any directory; setup wizard opens on first run"
else
    echo "  cd $INSTALL_DIR"
    echo "  uv run animaworks start   # Start server; setup wizard opens on first run"
fi
if [ "$LINKED" = "1" ] && [ "$PATH_HAS_BIN_DIR" = "0" ]; then
    echo ""
    echo "[!] $BIN_DIR is not on your PATH. To run 'animaworks' from any directory, add to your shell rc:"
    echo "      export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
