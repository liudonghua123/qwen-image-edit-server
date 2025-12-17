# Qwen Image Edit Server

A high-performance, OpenAI-compatible API service for the [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) model.

## Features

- 🚀 **FastAPI**: Built with modern Python web framework.
- 🔄 **OpenAI Compatible**: Drop-in replacement for OpenAI Images API client (extended).
- 🐳 **Docker Ready**: Easy deployment with provided Dockerfile.
- 🔧 **Configurable**: Environment variable based configuration.
- ⚡ **Multi-Image Support**: Extended API to support multi-image editing.

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (Recommended)
- [uv](https://github.com/astral-sh/uv) (Optional, for local dev)

### Local Development

1. Clone the repository
2. Create environment file:
    ```bash
    cp .env.example .env
    ```
3. Install dependencies:
    ```bash
    uv pip install -r pyproject.toml
    ```
    Or with pip:
    ```bash
    pip install -r pyproject.toml
    ```
4. Run the server:
    ```bash
    uvicorn src.main:app --reload
    ```

### Docker Deployment

Build the image:
```bash
docker build -t qwen-image-edit-server .
```

Run the container (with GPU support):
```bash
docker run --gpus all -p 8000:8000 --env-file .env qwen-image-edit-server
```

## API Usage

### Generate/Edit Image

**Endpoint**: `POST /v1/images/generations`

**Headers**:
- `Authorization`: `Bearer <YOUR_API_KEY>`
- `Content-Type`: `application/json`

**Body**:
```json
{
  "prompt": "The magician bear is on the left...",
  "n": 1,
  "size": "1024x1024",
  "response_format": "b64_json",
  "input_images": ["<base64_string_1>", "<base64_string_2>"], 
  "negative_prompt": "blurry, low quality",
  "guidance_scale": 7.5,
  "true_cfg_scale": 4.0,
  "num_inference_steps": 40
}
```

> Note: `input_images` is an extension to the standard OpenAI API. Provide base64 encoded strings of the input images you want to edit.

## License

MIT License
