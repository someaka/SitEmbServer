"""Server implementation for SitEmb-v1.5-Qwen3-note model.

This module provides a FastAPI server for encoding texts and queries
using the Situated Embedding model based on Qwen3.
"""

# pylint: disable=too-many-locals

# Standard library imports
from contextlib import asynccontextmanager
from typing import List, Optional, NamedTuple

# Third-party imports
import torch
from transformers import AutoTokenizer, AutoModel, Qwen3Model
from tqdm import tqdm
from more_itertools import chunked
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from peft import PeftModel


# Model configuration
RESIDUAL = True
RESIDUAL_FACTOR = 0.5


class PoolingArgs(NamedTuple):
    """Arguments for pooling function."""

    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor
    pooling: str
    normalize: bool
    input_ids: Optional[torch.Tensor] = None
    match_idx: Optional[int] = None


class ModelManager:
    """Manages the model and tokenizer for the server."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    async def load_model(self):
        """Load the model and tokenizer."""
        print("Loading model...")

        try:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-8B", use_fast=True, padding_side="left"
            )

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            # Try to load the model directly
            # First try with trust_remote_code=True
            try:
                self.model = AutoModel.from_pretrained(
                    "SituatedEmbedding/SitEmb-v1.5-Qwen3-note",
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    device_map={"": 0} if device == "cuda" else None,
                )
                if device == "cpu":
                    self.model = self.model.to(torch.device("cpu"))
            except (OSError, ValueError) as e:
                print(f"Failed to load with trust_remote_code=True: {str(e)}")
                # Fallback to loading the base Qwen model
                # Load the base model and then the adapter

                base_model = Qwen3Model.from_pretrained(
                    "Qwen/Qwen3-Embedding-8B",
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device_map={"": 0} if device == "cuda" else None,
                )
                if device == "cpu":
                    base_model = base_model.to(torch.device("cpu"))
                self.model = PeftModel.from_pretrained(
                    base_model,
                    "SituatedEmbedding/SitEmb-v1.5-Qwen3-note",
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    device_map={"": 0} if device == "cuda" else None,
                )
                if device == "cpu":
                    self.model = self.model.to(torch.device("cpu"))

            print("Model loaded successfully!")
        except (OSError, RuntimeError) as e:
            print(f"Error loading model: {str(e)}")
            raise

    def unload_model(self):
        """Unload the model and tokenizer."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model(self):
        """Get the loaded model."""
        return self.model

    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self.tokenizer


# Initialize model manager
model_manager = ModelManager()


# Lifespan manager for loading and unloading the model
@asynccontextmanager
async def lifespan(app_: FastAPI):  # pylint: disable=unused-argument
    """Lifespan manager for loading and unloading the model.

    Args:
        app_: FastAPI app instance
    """
    await model_manager.load_model()
    yield
    model_manager.unload_model()


# Initialize FastAPI app
app = FastAPI(
    title="SitEmb-v1.5-Qwen3-note Server",
    description="API for Situated Embedding model based on Qwen3",
    version="1.0.0",
    lifespan=lifespan,
)


class TextRequest(BaseModel):
    """Request model for encoding texts.

    Attributes:
        texts: List of text passages to encode
        batch_size: Number of texts to process in each batch (default: 8)
        max_length: Maximum sequence length for tokenization (default: 8192)
        pooling_type: Pooling method to use (default: "eos")
        normalize: Whether to normalize the embeddings (default: True)
    """

    texts: List[str]
    batch_size: Optional[int] = 8
    max_length: Optional[int] = 4096
    pooling_type: Optional[str] = "eos"
    normalize: Optional[bool] = True


class QueryRequest(BaseModel):
    """Request model for encoding queries.

    Attributes:
        queries: List of queries to encode
        batch_size: Number of queries to process in each batch (default: 8)
        max_length: Maximum sequence length for tokenization (default: 8192)
        pooling_type: Pooling method to use (default: "eos")
        normalize: Whether to normalize the embeddings (default: True)
    """

    queries: List[str]
    batch_size: Optional[int] = 8
    max_length: Optional[int] = 8192
    pooling_type: Optional[str] = "eos"
    normalize: Optional[bool] = True


class EmbeddingResponse(BaseModel):
    """Response model for embeddings.

    Attributes:
        embeddings: List of embeddings for the input texts or queries
        residual_embeddings: Optional list of residual embeddings (default: None)
    """

    embeddings: List[List[float]]
    residual_embeddings: Optional[List[List[float]]] = None


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Get detailed instruction for a query.

    Args:
        task_description: Description of the task
        query: The query to process

    Returns:
        Formatted instruction string
    """
    return f"Instruct: {task_description}\nQuery: {query}"


def _pooling(args: PoolingArgs):
    """Pool embeddings based on the specified method.

    Args:
        args: PoolingArgs containing all necessary arguments

    Returns:
        Pooled embeddings
    """
    last_hidden_state = args.last_hidden_state
    attention_mask = args.attention_mask
    pooling = args.pooling
    normalize = args.normalize
    input_ids = args.input_ids
    match_idx = args.match_idx

    if pooling in ["cls", "first"]:
        reps = last_hidden_state[:, 0]
    elif pooling in ["mean", "avg", "average"]:
        masked_hiddens = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling in ["last", "eos"]:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            reps = last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
    elif pooling == "ext":
        if match_idx is None:
            # default mean
            masked_hiddens = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        else:
            # Check if input_ids is valid before using it
            if input_ids is None:
                raise ValueError(
                    "input_ids is required for 'ext' pooling with match_idx"
                )

            for k in range(input_ids.shape[0]):
                sep_index = input_ids[k].tolist().index(match_idx)
                attention_mask[k][sep_index:] = 0
            masked_hiddens = last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    else:
        raise ValueError(f"unknown pooling method: {pooling}")

    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps


def first_eos_token_pooling(last_hidden_states, first_eos_position, normalize):
    """Pool embeddings using the first EOS token position.

    Args:
        last_hidden_states: Last hidden states from the model
        first_eos_position: Position of the first EOS token
        normalize: Whether to normalize the embeddings

    Returns:
        Pooled embeddings
    """
    batch_size = last_hidden_states.shape[0]
    reps = last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), first_eos_position
    ]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps


def encode_query(queries, batch_size, normalize, max_length, pooling):
    """Encode queries into embeddings.

    Args:
        queries: List of queries to encode
        batch_size: Number of queries to process in each batch
        normalize: Whether to normalize the embeddings
        max_length: Maximum sequence length for tokenization
        pooling: Pooling method to use

    Returns:
        Tuple of (query_embeddings, residual_embeddings)
    """
    if RESIDUAL:
        task = "Given a search query, retrieve relevant chunks from fictions that answer the query"
    else:
        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

    sents = []
    for query in queries:
        sents.append(get_detailed_instruct(task, query))

    return encode_passage(sents, batch_size, normalize, max_length, pooling)


def _process_residual_data(sent_b, tokenizer, model, max_length):  # pylint: disable=too-many-locals
    """Process residual data for a batch of sentences.

    Args:
        sent_b: Batch of sentences
        tokenizer: Model tokenizer
        model: Loaded model
        max_length: Maximum sequence length for tokenization

    Returns:
        Tensor of EOS positions
    """
    # Check if tokenizer has pad_token_id
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer does not have pad_token_id")

    batch_dict = tokenizer(
        sent_b, max_length=max_length, padding=True, truncation=True
    )

    input_ids = batch_dict["input_ids"]

    # Check if input_ids is valid
    if input_ids is None:
        raise RuntimeError("Failed to tokenize input")

    # Calculate EOS positions and create tensor directly
    return torch.tensor([
        input_ids[i].index(
            tokenizer.pad_token_id,
            len(input_ids[i]) - sum(batch_dict["attention_mask"][i])
        )
        for i in range(len(input_ids))
    ]).to(
        next(model.parameters()).device
        if hasattr(model, "parameters")
        else torch.device("cpu")
    )


def encode_passage(passages, batch_size, normalize, max_length, pooling):
    """Encode passages into embeddings.

    Args:
        passages: List of text passages to encode
        batch_size: Number of passages to process in each batch
        normalize: Whether to normalize the embeddings
        max_length: Maximum sequence length for tokenization
        pooling: Pooling method to use

    Returns:
        Tuple of (passage_embeddings, residual_embeddings)
    """
    pas_embs = []
    pas_embs_residual = []

    total = len(passages) // batch_size + (1 if len(passages) % batch_size != 0 else 0)

    with tqdm(total=total) as pbar:
        for sent_b in chunked(passages, batch_size):
            tokenizer = model_manager.get_tokenizer()
            if tokenizer is None:
                raise RuntimeError("Tokenizer is not loaded")

            model = model_manager.get_model()
            if model is None:
                raise RuntimeError("Model is not loaded")

            batch_dict = tokenizer(
                sent_b,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)

            eos_pos = None
            if RESIDUAL:
                eos_pos = _process_residual_data(sent_b, tokenizer, model, max_length)

            outputs = model(**batch_dict)
            pemb_ = _pooling(
                PoolingArgs(
                    last_hidden_state=outputs.last_hidden_state,
                    attention_mask=batch_dict["attention_mask"],
                    pooling=pooling,
                    normalize=normalize,
                )
            )

            if RESIDUAL:
                remb_ = first_eos_token_pooling(
                    outputs.last_hidden_state, eos_pos, normalize
                )
                pas_embs_residual.append(remb_)

            pas_embs.append(pemb_)
            pbar.update(1)

    pas_embs = torch.cat(pas_embs, dim=0)

    if pas_embs_residual:
        pas_embs_residual = torch.cat(pas_embs_residual, dim=0)
    else:
        pas_embs_residual = None

    return pas_embs, pas_embs_residual


@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "SitEmb-v1.5-Qwen3-note Server is running!"}


@app.get("/health")
async def health_check():
    """Health check endpoint that returns the server status."""
    return {"status": "healthy"}


@app.post("/encode_texts", response_model=EmbeddingResponse)
async def encode_texts(request: TextRequest):
    """Encode texts into embeddings.

    Args:
        request: TextRequest object containing texts and parameters

    Returns:
        EmbeddingResponse with embeddings
    """
    try:
        # Validate input
        if not request.texts:
            return {"error": "No texts provided"}

        embeddings, residual_embeddings = encode_passage(
            request.texts,
            request.batch_size,
            request.normalize,
            request.max_length,
            request.pooling_type,
        )

        # Move tensors to CPU before converting to list
        embeddings = embeddings.cpu()
        response = EmbeddingResponse(embeddings=embeddings.tolist())

        if residual_embeddings is not None:
            residual_embeddings = residual_embeddings.cpu()
            response.residual_embeddings = residual_embeddings.tolist()

        return response
    except (ValueError, RuntimeError) as exc:
        print(f"Error in encode_texts: {str(exc)}")
        return {"error": str(exc)}


@app.post("/encode_queries", response_model=EmbeddingResponse)
async def encode_queries(request: QueryRequest):
    """Encode queries into embeddings.

    Args:
        request: QueryRequest object containing queries and parameters

    Returns:
        EmbeddingResponse with embeddings
    """
    try:
        # Validate input
        if not request.queries:
            return {"error": "No queries provided"}

        embeddings, residual_embeddings = encode_query(
            request.queries,
            request.batch_size,
            request.normalize,
            request.max_length,
            request.pooling_type,
        )

        # Move tensors to CPU before converting to list
        embeddings = embeddings.cpu()
        response = EmbeddingResponse(embeddings=embeddings.tolist())

        if residual_embeddings is not None:
            residual_embeddings = residual_embeddings.cpu()
            response.residual_embeddings = residual_embeddings.tolist()

        return response
    except (ValueError, RuntimeError) as exc:
        print(f"Error in encode_queries: {str(exc)}")
        return {"error": str(exc)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
