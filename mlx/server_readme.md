# MLX Audio TTS Server

This directory contains a FastAPI-based server implementation that provides an OpenAI-compatible API for Text-to-Speech (TTS) using the `mlx_audio` framework.

## Overview

The `mlx-server.py` script serves as a bridge between the `mlx_audio` model capabilities and OpenAI-compatible clients (like OpenWebUI). It allows for high-quality, low-latency speech generation, including features like voice cloning via reference audio.

## Key Features

- **OpenAI API Compatibility**: Implements the `/v1/audio/speech` endpoint, making it a drop-in replacement for OpenAI's TTS API.
- **Voice Cloning**: Supports using a reference audio file (`--ref-audio`) and its corresponding text (`--ref-text`) to clone a specific voice.
- **Model Management**: Automatically handles model downloading and caching using `modelscope`.
- **Multi-language Support**: Configurable language codes (e.g., `zh` for Chinese) passed to the generation engine.
- **FastAPI Integration**: Built on top of FastAPI for high performance and easy integration.

## Components

### `mlx-server.py`

The core server script. Its main responsibilities include:

1.  **Argument Parsing**: Handles command-line arguments for host, port, reference audio, reference text, and language code.
2.  **Model Loading**: Uses `modelscope` to download the `Qwen3-TTS` model if not present and loads it into memory using `mlx_audio.tts.utils.load_model`.
3.  **API Endpoints**:
    - `GET /health`: Simple health check endpoint.
    - `POST /v1/audio/speech`: The primary endpoint that accepts text input and returns a `.wav` audio stream.
4.  **Audio Generation Logic**:
    - Uses a `threading.Lock` to ensure thread-safe generation (preventing concurrent access to the model).
    - Utilizes `mlx_audio.tts.generate.generate_audio` to perform the actual synthesis.
    - Manages temporary directories and file cleanup for generated audio chunks.
5.  **Fallback Mechanism**: Returns a silent `.wav` file if the text is empty, the model fails to load, or an error occurs during generation, ensuring the client doesn't crash.

## Usage

### Prerequisites

- Python environment with `fastapi`, `uvicorn`, `modelscope`, and `mlx_audio` installed.
- Access to the `mlx-community/Qwen3-TTS-1s-1.7B-Base-8bit` model (automatically downloaded).

### Running the Server

You can start the server from the terminal using the following command:

```bash
python mlx-server.py --host 0.0.0.0 --port 8002 --ref-audio /path/to/your/ref.wav --ref-text "Your reference text here"
```

### API Request Example (cURL)

```bash
curl http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of the MLX audio server.",
    "model": "tts-1"
  }' \
  --output output.wav
```

## Configuration Parameters

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--host` | `0.0.0.0` | The network interface to bind to. |
| `--port` | `8002` | The port to listen on. |
| `--ref-audio` | `/Users/zhenghao/.omlx/al.wav` | Path to the reference audio for cloning. |
| `--ref-text` | `...` | The text that matches the reference audio. |
| `--lang-code` | `zh` | The language code for the generation. |
