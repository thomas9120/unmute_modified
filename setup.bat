@echo off
echo ========================================
echo   Unmute Setup Script
echo ========================================
echo.

REM Check if .env already exists
if exist .env (
    echo Found existing .env file.
    set /p OVERWRITE="Overwrite it? (y/N): "
    if /i not "%OVERWRITE%"=="y" (
        echo Keeping existing .env file.
        goto :llm_setup
    )
)

REM Copy template
copy .env.example .env >nul 2>&1
if errorlevel 1 (
    echo Creating new .env file...
    type nul > .env
)

:token_input
echo.
echo Get your Hugging Face token at: https://huggingface.co/settings/tokens
echo (Required for STT/TTS models)
echo.
set /p HF_TOKEN="Paste your Hugging Face token (hf_...): "

if "%HF_TOKEN%"=="" (
    echo Token cannot be empty!
    goto :token_input
)

REM Write to .env file
echo # Unmute Configuration> .env
echo.>> .env
echo # Hugging Face token for STT/TTS models>> .env
echo HUGGING_FACE_HUB_TOKEN=%HF_TOKEN%>> .env

:llm_setup
echo.
echo ========================================
echo   LLM Backend Configuration
echo ========================================
echo.
echo Which LLM backend will you use?
echo   1. Koboldcpp (default, port 5001)
echo   2. Ollama (port 11434)
echo   3. OpenAI API
echo   4. Custom URL
echo   5. Skip (use defaults)
echo.
set /p LLM_CHOICE="Enter choice (1-5) [1]: "

if "%LLM_CHOICE%"=="" set LLM_CHOICE=1

if "%LLM_CHOICE%"=="1" (
    echo.>> .env
    echo # LLM Configuration - Koboldcpp>> .env
    echo KYUTAI_LLM_URL=http://host.docker.internal:5001>> .env
    echo.
    echo Configured for Koboldcpp on port 5001.
    goto :done
)

if "%LLM_CHOICE%"=="2" (
    set /p OLLAMA_MODEL="Enter Ollama model name (e.g., llama3.2): "
    echo.>> .env
    echo # LLM Configuration - Ollama>> .env
    echo KYUTAI_LLM_URL=http://host.docker.internal:11434>> .env
    if not "%OLLAMA_MODEL%"=="" echo KYUTAI_LLM_MODEL=%OLLAMA_MODEL%>> .env
    echo.
    echo Configured for Ollama.
    goto :done
)

if "%LLM_CHOICE%"=="3" (
    set /p OPENAI_KEY="Enter your OpenAI API key (sk-...): "
    set /p OPENAI_MODEL="Enter model name (e.g., gpt-4o) [gpt-4o]: "
    if "%OPENAI_MODEL%"=="" set OPENAI_MODEL=gpt-4o
    echo.>> .env
    echo # LLM Configuration - OpenAI>> .env
    echo KYUTAI_LLM_URL=https://api.openai.com>> .env
    echo KYUTAI_LLM_MODEL=%OPENAI_MODEL%>> .env
    echo KYUTAI_LLM_API_KEY=%OPENAI_KEY%>> .env
    echo.
    echo Configured for OpenAI API.
    goto :done
)

if "%LLM_CHOICE%"=="4" (
    set /p CUSTOM_URL="Enter LLM server URL (e.g., http://host.docker.internal:8080): "
    set /p CUSTOM_MODEL="Enter model name (optional, press Enter to auto-detect): "
    echo.>> .env
    echo # LLM Configuration - Custom>> .env
    echo KYUTAI_LLM_URL=%CUSTOM_URL%>> .env
    if not "%CUSTOM_MODEL%"=="" echo KYUTAI_LLM_MODEL=%CUSTOM_MODEL%>> .env
    echo.
    echo Configured for custom LLM server.
    goto :done
)

:done
echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Your .env file has been created.
echo.
echo Next steps:
echo   1. Start your LLM server (Koboldcpp, Ollama, etc.)
echo   2. Run: docker compose up --build
echo   3. Open http://localhost in your browser
echo.
pause
