from typing import List, Optional, Union, Dict, Any, Literal
import json
import time
import asyncio
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import logging
import uuid
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="OpenAI-Compatible API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Pydantic models for request/response validation -----

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class CompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class StreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[StreamChoice]

# ----- API routes -----

@app.get("/v1/models")
async def list_models():
    """
    List available models that are compatible with the Chat API
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "custom-ai-provider"
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1677649963,
                "owned_by": "custom-ai-provider"
            }
        ]
    }

@app.post("/v1/chat/completions", response_model=Union[CompletionResponse, StreamResponse])
async def create_chat_completion(request: CompletionRequest):
    """
    Creates a model response for the given chat conversation
    """
    if request.stream:
        return StreamingResponse(stream_chat_completion(request), media_type="text/event-stream")
    else:
        return await generate_chat_completion(request)

async def generate_chat_completion(request: CompletionRequest) -> CompletionResponse:
    """
    Generate a non-streaming chat completion
    """
    # Here you would normally call your LLM
    # For this example, we'll return a simple mock response
    
    # In a real implementation, you would:
    # 1. Convert the request format to your model's format
    # 2. Call your model
    # 3. Convert the model's response to OpenAI format
    
    # Mock response
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    current_time = int(time.time())
    
    # Example mock response content
    response_content = f"This is a mock response from model {request.model}."
    
    # Calculate mock token counts
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(response_content.split())
    total_tokens = prompt_tokens + completion_tokens
    
    return CompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=current_time,
        model=request.model,
        choices=[
            CompletionChoice(
                index=0,
                message=Message(role="assistant", content=response_content),
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )

async def stream_chat_completion(request: CompletionRequest):
    """
    Stream a chat completion response word by word
    """
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    current_time = int(time.time())
    
    # Example response for streaming (you'd replace this with your actual LLM)
    response_content = f"This is a mock streaming response from model {request.model}."
    words = response_content.split()
    
    # First chunk with role
    response = StreamResponse(
        id=completion_id,
        object="chat.completion.chunk",
        created=current_time,
        model=request.model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None
            )
        ]
    )
    yield f"data: {json.dumps(response.model_dump())}\n\n"
    
    # Stream each word individually
    for word in words:
        await asyncio.sleep(0.1)  # Simulate processing time
        
        response = StreamResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=current_time,
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=word + ' '),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {json.dumps(response.model_dump())}\n\n"
    
    # Final chunk
    response = StreamResponse(
        id=completion_id,
        object="chat.completion.chunk",
        created=current_time,
        model=request.model,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaMessage(content=''),
                finish_reason="stop"
            )
        ]
    )
    yield f"data: {json.dumps(response.model_dump())}\n\n"
    
    # End the stream
    yield "data: [DONE]\n\n"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Check if IPv6 is supported
    has_ipv6 = False
    try:
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        has_ipv6 = True
        print("IPv6 is supported")
    except OSError:
        print("IPv6 is not supported")
        
    # Start server with IPv6 dual stack enabled if supported
    uvicorn.run(
        "main:app", 
        host="0.0.0.0" if not has_ipv6 else "::",  # '::' binds to all IPv6 addresses and also IPv4 addresses
        port=8001, 
        reload=True
    ) 