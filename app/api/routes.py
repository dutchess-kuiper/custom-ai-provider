import json
import time
import uuid
from typing import Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.models.chat import CompletionRequest, CompletionResponse, StreamResponse
from app.models.chat import CompletionChoice, CompletionUsage, Message, StreamChoice, DeltaMessage
from app.services.llm import run_pipeline, stream_pipeline
from app.config.settings import AVAILABLE_MODELS

router = APIRouter(prefix="/v1")

@router.get("/models")
async def list_models():
    """
    List available models that are compatible with the Chat API
    """
    return {
        "object": "list",
        "data": AVAILABLE_MODELS
    }

@router.post("/chat/completions", response_model=Union[CompletionResponse, StreamResponse])
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
    Generate a non-streaming chat completion using Haystack
    """
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    current_time = int(time.time())
    
    try:
        # Convert request messages to format expected by Haystack
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Call the Haystack pipeline
        result = await run_pipeline(messages)
        response_content = result["reply"]
        
        # Calculate rough token counts (this is a simplification)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

async def stream_chat_completion(request: CompletionRequest):
    """
    Stream a chat completion response word by word using Haystack
    """
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    current_time = int(time.time())
    
    try:
        # Convert request messages to format expected by Haystack
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
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
        
        # Stream each word using Haystack's streamed response
        async for word in stream_pipeline(messages):
            response = StreamResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=current_time,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=word),
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
    except Exception as e:
        # Send an error in the stream
        error_response = StreamResponse(
            id=completion_id,
            object="chat.completion.chunk",
            created=current_time,
            model=request.model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=f"Error: {str(e)}"),
                    finish_reason="error"
                )
            ]
        )
        yield f"data: {json.dumps(error_response.model_dump())}\n\n"
        yield "data: [DONE]\n\n" 