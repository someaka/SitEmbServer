# SitEmb-v1.5-Qwen3-note Server

This project sets up the [SituatedEmbedding/SitEmb-v1.5-Qwen3-note](https://huggingface.co/SituatedEmbedding/SitEmb-v1.5-Qwen3-note) model as a local server using FastAPI.

## Overview

The SitEmb-v1.5-Qwen3-note model is built on top of Qwen/Qwen3-Embedding-8B and is designed for text embedding and retrieval tasks. This server implementation provides RESTful APIs to encode texts and queries using the model.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (recommended for better performance)

## Setup

1. Clone or download this repository

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

To start the server, run:

```bash
python server.py
```

The server will start on `http://localhost:8000` by default.

On first run, the model will be automatically downloaded from Hugging Face. This may take some time depending on your internet connection.

On first run, the model will be automatically downloaded from Hugging Face. This may take some time depending on your internet connection.

## Server Implementation Details

The server is built using FastAPI and includes several key features:

- **Model Management**: Uses a `ModelManager` class to handle model loading and unloading
- **Automatic Device Detection**: Uses CUDA if available, otherwise falls back to CPU
- **Residual Embeddings**: Supports residual embeddings for improved retrieval performance
- **Multiple Pooling Methods**: Supports different pooling strategies for embeddings
- **Robust Error Handling**: Comprehensive error handling and input validation
- **Automatic Model Loading**: Models are loaded on server startup and unloaded on shutdown

## API Endpoints

### Health Check
- **GET** `/health`
- Returns the health status of the server

### Root
- **GET** `/`
- Returns a welcome message with server information

### Encode Texts
- **POST** `/encode_texts`
- Encodes a list of text passages into embeddings
- Supports residual embeddings when enabled

Example request:
```json
{
  "texts": ["Your text here", "Another text"],
  "batch_size": 8,
  "max_length": 4096,
  "pooling_type": "eos",
  "normalize": true
}
```

Response:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "residual_embeddings": [[0.5, 0.6, ...], [0.7, 0.8, ...]]
}
```

### Encode Queries
- **POST** `/encode_queries`
- Encodes a list of queries into embeddings with task-specific instructions
- Automatically applies appropriate prompting for query encoding

Example request:
```json
{
  "queries": ["Your query here", "Another query"],
  "batch_size": 8,
  "max_length": 8192,
  "pooling_type": "eos",
  "normalize": true
}
```

Response:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "residual_embeddings": [[0.5, 0.6, ...], [0.7, 0.8, ...]]
}
```

## Pooling Methods

The server supports several pooling methods:

- **eos** (default): End-of-sequence token pooling
- **cls**: First token (CLS token) pooling
- **first**: First token pooling
- **mean**/**avg**/**average**: Mean pooling across all tokens
- **ext**: External pooling with custom match index

## Configuration

The server can be configured by editing the parameters at the top of `server.py`:

- `RESIDUAL`: Enable/disable residual embeddings (default: `True`)
- `RESIDUAL_FACTOR`: Factor for residual calculations (default: `0.5`)
- Server host and port can be modified in the `uvicorn.run()` call

## Testing

1. Start the server:
   ```bash
   python server.py
   ```

2. In a new terminal, run the test script:
   ```bash
   python test_server.py
   ```

The test script will verify that all endpoints are working correctly and test:
- Server health check
- Text encoding functionality
- Query encoding functionality
- Response validation

## Model Information

The server loads the model using a two-stage process:

1. **Direct Loading**: Attempts to load the complete model directly with `trust_remote_code=True`
2. **Fallback Loading**: If direct loading fails, loads the base Qwen model and applies the PEFT adapter

Model configuration:
- Base model: `Qwen/Qwen3-Embedding-8B`
- Finetuned model: `SituatedEmbedding/SitEmb-v1.5-Qwen3-note`
- Data type: `torch.bfloat16` (CUDA) or `torch.float32` (CPU)
- Device mapping: Automatic detection (CUDA if available, CPU otherwise)
- Embedding dimensions: 8192 (8B model)
- Residual embeddings: Enabled by default for improved retrieval performance

## Troubleshooting

1. **CUDA Out of Memory**:
   - Reduce `batch_size` parameter in API requests
   - Reduce `max_length` parameter
   - Use smaller batch sizes in the code

2. **Model Download Issues**:
   - Check your internet connection
   - Verify Hugging Face authentication if required
   - Ensure you have enough disk space (model is several GB)

3. **Slow Performance**:
   - Use a CUDA-compatible GPU for better performance
   - Reduce batch size for faster processing
   - Ensure you have sufficient VRAM

4. **Server Startup Errors**:
   - Check console output for specific error messages
   - Verify Python dependencies are installed correctly
   - Ensure port 8000 is available

5. **Tokenization Issues**:
   - Ensure texts are within the `max_length` limit
   - Check for special characters that might cause tokenization problems

6. **Model Loading Failures**:
   - The server includes fallback mechanism for model loading
   - Check if you have sufficient memory for the base model
   - Verify the model repository is accessible

## Implementation Features

The server includes several robust features:

- **Automatic Model Management**: Models are loaded on startup and unloaded on shutdown
- **Device Detection**: Automatically selects the best available device (CUDA/CPU)
- **Error Handling**: Comprehensive error handling for all API endpoints
- **Input Validation**: Validates all input parameters before processing
- **Progress Tracking**: Shows progress bars for batch processing
- **Memory Management**: Proper cleanup of GPU memory when available
- **Residual Support**: Optional residual embeddings for improved performance

## License

This server implementation is provided as-is under the MIT License. The model itself (`SitEmb-v1.5-Qwen3-note`) is subject to its original license from Hugging Face. Please ensure you comply with the terms of use for both the model and any third-party libraries used in this implementation.

## Contributing

If you encounter any issues or have suggestions for improvements, please feel free to submit issues or pull requests. When contributing, please ensure:

1. Code follows the existing style and conventions
2. All new code includes appropriate documentation
3. Changes are tested thoroughly
4. Any new dependencies are added to `requirements.txt`