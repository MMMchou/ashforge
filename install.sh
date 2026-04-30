#!/bin/sh
set -e

REPO="MMMchou/ashforge"
INSTALL_DIR="/usr/local/bin"
BINARY="ashforge"

main() {
    echo "Ashforge Installer"
    echo "=================="
    echo ""

    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux)  PLATFORM="linux" ;;
        darwin) PLATFORM="darwin" ;;
        *)      echo "Error: unsupported OS: $OS"; exit 1 ;;
    esac

    case "$ARCH" in
        x86_64|amd64)  ARCH="amd64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *)             echo "Error: unsupported architecture: $ARCH"; exit 1 ;;
    esac

    echo "Detected: ${PLATFORM}/${ARCH}"

    # Get latest release tag
    echo "Fetching latest release..."
    LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    if [ -z "$LATEST" ]; then
        echo "Error: could not determine latest release."
        echo "Check https://github.com/${REPO}/releases"
        exit 1
    fi
    echo "Latest version: ${LATEST}"

    # Build download URL
    ASSET="${BINARY}-${PLATFORM}-${ARCH}"
    if [ "$PLATFORM" = "linux" ]; then
        ASSET="${BINARY}-linux-${ARCH}"
    elif [ "$PLATFORM" = "darwin" ]; then
        ASSET="${BINARY}-darwin-${ARCH}"
    fi
    URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}.tar.gz"

    # Download
    TMPDIR=$(mktemp -d)
    echo "Downloading ${URL}..."
    if ! curl -fsSL "$URL" -o "${TMPDIR}/ashforge.tar.gz"; then
        # Fallback: try without .tar.gz (raw binary)
        URL="https://github.com/${REPO}/releases/download/${LATEST}/${ASSET}"
        echo "Trying raw binary: ${URL}..."
        if ! curl -fsSL "$URL" -o "${TMPDIR}/ashforge"; then
            echo "Error: download failed."
            echo "Check available assets at: https://github.com/${REPO}/releases/tag/${LATEST}"
            rm -rf "$TMPDIR"
            exit 1
        fi
    else
        # Extract tar.gz
        tar -xzf "${TMPDIR}/ashforge.tar.gz" -C "$TMPDIR"
    fi

    # Install ashforge + bundled llama-server + CUDA libs
    ASHFORGE_DIR="${HOME}/.ashforge/bin"
    mkdir -p "$ASHFORGE_DIR"

    # Move all extracted files to ~/.ashforge/bin/
    for f in "${TMPDIR}"/*; do
        [ -f "$f" ] && cp "$f" "$ASHFORGE_DIR/"
    done
    chmod +x "$ASHFORGE_DIR"/*

    # Symlink ashforge to PATH
    if [ -w "$INSTALL_DIR" ]; then
        ln -sf "$ASHFORGE_DIR/${BINARY}" "${INSTALL_DIR}/${BINARY}"
    else
        echo "Need sudo to install to ${INSTALL_DIR}"
        sudo ln -sf "$ASHFORGE_DIR/${BINARY}" "${INSTALL_DIR}/${BINARY}"
    fi

    rm -rf "$TMPDIR"

    # Verify
    if command -v ashforge >/dev/null 2>&1; then
        echo ""
        echo "Ashforge installed successfully!"
        echo ""
        ashforge version
        echo ""
        echo "Get started:"
        echo "  ashforge run Qwen3-30B-A3B"
    else
        echo ""
        echo "Installed to ${INSTALL_DIR}/${BINARY}"
        echo "Make sure ${INSTALL_DIR} is in your PATH."
    fi
}

main
