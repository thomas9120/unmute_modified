#!/bin/bash

echo "========================================"
echo "  Unmute Setup Script"
echo "========================================"
echo

# Check if .env already exists
if [ -f .env ]; then
    echo "Found existing .env file."
    read -p "Overwrite it? (y/N): " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file."
        SKIP_TOKEN=true
    fi
fi

if [ "$SKIP_TOKEN" != "true" ]; then
    # Copy template if it exists
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        touch .env
    fi

    echo
    echo "Get your Hugging Face token at: https://huggingface.co/settings/tokens"
    echo "(Required for STT/TTS models)"
    echo

    while true; do
        read -p "Paste your Hugging Face token (hf_...): " HF_TOKEN
        if [ -n "$HF_TOKEN" ]; then
            break
        fi
        echo "Token cannot be empty!"
    done

    # Write to .env file
    cat > .env << EOF
# Unmute Configuration

# Hugging Face token for STT/TTS models
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
EOF
fi

echo
echo "========================================"
echo "  LLM Backend Configuration"
echo "========================================"
echo
echo "Which LLM backend will you use?"
echo "  1. Koboldcpp (default, port 5001)"
echo "  2. Ollama (port 11434)"
echo "  3. OpenAI API"
echo "  4. Custom URL"
echo "  5. Skip (use defaults)"
echo
read -p "Enter choice (1-5) [1]: " LLM_CHOICE
LLM_CHOICE=${LLM_CHOICE:-1}

case $LLM_CHOICE in
    1)
        cat >> .env << EOF

# LLM Configuration - Koboldcpp
KYUTAI_LLM_URL=http://host.docker.internal:5001
EOF
        echo
        echo "Configured for Koboldcpp on port 5001."
        ;;
    2)
        read -p "Enter Ollama model name (e.g., llama3.2): " OLLAMA_MODEL
        cat >> .env << EOF

# LLM Configuration - Ollama
KYUTAI_LLM_URL=http://host.docker.internal:11434
EOF
        if [ -n "$OLLAMA_MODEL" ]; then
            echo "KYUTAI_LLM_MODEL=$OLLAMA_MODEL" >> .env
        fi
        echo
        echo "Configured for Ollama."
        ;;
    3)
        read -p "Enter your OpenAI API key (sk-...): " OPENAI_KEY
        read -p "Enter model name (e.g., gpt-4o) [gpt-4o]: " OPENAI_MODEL
        OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
        cat >> .env << EOF

# LLM Configuration - OpenAI
KYUTAI_LLM_URL=https://api.openai.com
KYUTAI_LLM_MODEL=$OPENAI_MODEL
KYUTAI_LLM_API_KEY=$OPENAI_KEY
EOF
        echo
        echo "Configured for OpenAI API."
        ;;
    4)
        read -p "Enter LLM server URL (e.g., http://host.docker.internal:8080): " CUSTOM_URL
        read -p "Enter model name (optional, press Enter to auto-detect): " CUSTOM_MODEL
        cat >> .env << EOF

# LLM Configuration - Custom
KYUTAI_LLM_URL=$CUSTOM_URL
EOF
        if [ -n "$CUSTOM_MODEL" ]; then
            echo "KYUTAI_LLM_MODEL=$CUSTOM_MODEL" >> .env
        fi
        echo
        echo "Configured for custom LLM server."
        ;;
    5)
        echo
        echo "Using default configuration (Koboldcpp on port 5001)."
        ;;
esac

echo
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo
echo "Your .env file has been created."
echo
echo "Next steps:"
echo "  1. Start your LLM server (Koboldcpp, Ollama, etc.)"
echo "  2. Run: docker compose up --build"
echo "  3. Open http://localhost in your browser"
echo
