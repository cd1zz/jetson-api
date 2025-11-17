#!/bin/bash
# Install systemd services for Jetson LLM API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME=$(whoami)
HOME_DIR="$HOME"
NPROC=$(nproc)

echo "Installing systemd services for Jetson LLM API"
echo "User: $USER_NAME"
echo "Home: $HOME_DIR"
echo "CPU cores: $NPROC"

# Create temporary directory for processed service files
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Process each service file
for service_file in "$SCRIPT_DIR"/*.service; do
    service_name=$(basename "$service_file")
    echo "Processing $service_name..."

    # Replace placeholders
    sed -e "s|%USER%|$USER_NAME|g" \
        -e "s|%HOME%|$HOME_DIR|g" \
        -e "s|%NPROC%|$NPROC|g" \
        "$service_file" > "$TMP_DIR/$service_name"

    # Install to systemd
    sudo cp "$TMP_DIR/$service_name" /etc/systemd/system/
    echo "Installed $service_name"
done

# Reload systemd
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "Services installed successfully!"
echo ""
echo "To enable and start services:"
echo "  sudo systemctl enable llama-deepseek llama-qwen jetson-api"
echo "  sudo systemctl start llama-deepseek llama-qwen jetson-api"
echo ""
echo "To check status:"
echo "  sudo systemctl status llama-deepseek"
echo "  sudo systemctl status llama-qwen"
echo "  sudo systemctl status jetson-api"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u llama-deepseek -f"
echo "  sudo journalctl -u llama-qwen -f"
echo "  sudo journalctl -u jetson-api -f"
