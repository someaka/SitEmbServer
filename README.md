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

### Option 1: Using Startup Scripts (Recommended)

For Windows, run:
```bash
run_server.bat
```

For Linux/Mac, run:
```bash
./run_server.sh
```

### Option 2: Manual Setup

To start the server manually, run:

```bash
python server.py
```

The server will start on `http://localhost:8000` by default.

On first run, the model will be automatically downloaded from Hugging Face. This may take some time depending on your internet connection.

## API Endpoints

### Health Check
- **GET** `/health`
- Returns the health status of the server

### Root
- **GET** `/`
- Returns a welcome message

### Encode Texts
- **POST** `/encode_texts`
- Encodes a list of texts into embeddings

Example request:
```json
{
  "texts": ["Your text here", "Another text"],
  "batch_size": 8,
  "max_length": 8192,
  "pooling_type": "eos",
  "normalize": true
}
```

### Encode Queries
- **POST** `/encode_queries`
- Encodes a list of queries into embeddings

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

## Testing

1. Start the server using one of the methods described above

2. In a new terminal, run the test script:

```bash
python test_server.py
```

The test script will verify that all endpoints are working correctly.

## Configuration

You can modify the server settings by editing `server.py`:
- Change the host/port in the `uvicorn.run()` call
- Adjust model parameters like `residual` and `residual_factor`

## Model Information

The model is loaded from Hugging Face with the following configuration:
- Base model: `Qwen/Qwen3-Embedding-8B`
- Finetuned model: `SituatedEmbedding/SitEmb-v1.5-Qwen3-note`
- Data type: `torch.bfloat16`
- Device mapping: `{"": 0}` (loads on GPU 0 if available)
- Embedding dimensions: 8192 (8B model)

## Troubleshooting

1. **CUDA Out of Memory**: Reduce `batch_size` or `max_length` parameters
2. **Model Download Issues**: Check your internet connection and Hugging Face authentication
3. **Slow Performance**: Ensure you're running on a CUDA-compatible GPU
4. **Server Startup Errors**: Check the console output for specific error messages

## Improvements Made

1. Added proper error handling for model loading
2. Fixed tensor conversion issues by moving tensors to CPU before converting to lists
3. Added input validation for API endpoints
4. Improved error logging for debugging

## License

This implementation is provided as-is. The model itself is subject to its original license from Hugging Face.