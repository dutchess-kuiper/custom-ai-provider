import uuid
import logging
import asyncio
import time
import json
from typing import Union, Dict, Any, List
from fastapi import APIRouter, HTTPException, Request, Depends, Header, Security
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader, APIKey

from app.models.chat import CompletionRequest, CompletionResponse, StreamResponse
from app.models.chat import CompletionChoice, CompletionUsage, Message, StreamChoice, DeltaMessage
from app.models.chat import MessageWithToolCalls, ToolCall, FunctionCall
from app.services.llm import run_pipeline, stream_pipeline
from app.config.settings import AVAILABLE_MODELS
import os
# Configure logging
logger = logging.getLogger(__name__)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API keys from environment variable as a comma-separated string
api_keys_str = os.getenv("API_KEYS", "")
# Split the string into a set
API_KEYS = {key.strip() for key in api_keys_str.split(",")}

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header is None:
        raise HTTPException(
            status_code=401,
            detail="API Key missing",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    if api_key_header not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    return api_key_header

router = APIRouter(prefix="/v1")

# Create a separate router for v2 endpoints with no prefix
router_v2 = APIRouter()


@router.get("/models", dependencies=[Depends(get_api_key)])
async def list_models():
    """
    List available models that are compatible with the Chat API
    """
    return {
        "object": "list",
        "data": AVAILABLE_MODELS
    }

@router.post("/chat/completions", response_model=Union[CompletionResponse, StreamResponse], dependencies=[Depends(get_api_key)])
async def create_chat_completion(request: CompletionRequest):
    """
    Creates a model response for the given chat conversation
    """
    if request.stream:
        return StreamingResponse(stream_chat_completion(request), media_type="text/event-stream")
    else:
        return await generate_chat_completion(request)

def generate_chat_completion(request: CompletionRequest) -> CompletionResponse:
    """
    Generate a non-streaming chat completion using Haystack
    """
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    current_time = int(time.time())
    
    try:
        # Convert request messages to format expected by Haystack
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        print(str(request))
        
        # Handle tools/functions if present in the request
        pipeline_kwargs = {}
        if request.tools:
            pipeline_kwargs["tools"] = [
                {
                    "type": tool.type,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters
                    }
                } for tool in request.tools
            ]
        elif request.functions:
            pipeline_kwargs["functions"] = [
                {
                    "name": func.name,
                    "description": func.description,
                    "parameters": func.parameters
                } for func in request.functions
            ]
        
        if request.tool_choice:
            pipeline_kwargs["tool_choice"] = request.tool_choice
        elif request.function_call:
            pipeline_kwargs["function_call"] = request.function_call
        
        # Use a synchronous approach to run_pipeline
        # Use asyncio.run to run the async function in a synchronous context
        result = asyncio.run(run_pipeline(messages, **pipeline_kwargs))
        
        # Check if the response contains tool calls
        if "tool_calls" in result and result["tool_calls"]:
            tool_calls = result["tool_calls"]
            message = MessageWithToolCalls(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call_{i}",
                        type="function",
                        index=i,
                        function=FunctionCall(
                            name=tool_call["function"]["name"],
                            arguments=tool_call["function"]["arguments"]
                        )
                    ) for i, tool_call in enumerate(tool_calls)
                ]
            )
            finish_reason = "tool_calls"
        elif "function_call" in result and result["function_call"]:
            function_call = result["function_call"]
            message = MessageWithToolCalls(
                role="assistant",
                content=None,
                function_call=FunctionCall(     
                    name=function_call["name"],
                    arguments=function_call["arguments"]
                )
            )
            finish_reason = "function_call"
        else:
            response_content = result["reply"]
            message = Message(role="assistant", content=response_content)
            finish_reason = "stop"
        
        # Calculate rough token counts (this is a simplification)
        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
        completion_tokens = len(result.get("reply", "").split()) if "reply" in result else 100  # Estimate for tool calls
        total_tokens = prompt_tokens + completion_tokens
        
        return CompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=current_time,
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=finish_reason
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        )
    except Exception as e:
        logger.error(f"Error in completion: {str(e)}")
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
        
        # Handle tools/functions if present in the request
        pipeline_kwargs = {}
        if request.tools:
            pipeline_kwargs["tools"] = [
                {
                    "type": tool.type,
                    "function": {
                        "name": tool.function.name,
                        "description": tool.function.description,
                        "parameters": tool.function.parameters
                    }
                } for tool in request.tools
            ]
        elif request.functions:
            pipeline_kwargs["functions"] = [
                {
                    "name": func.name,
                    "description": func.description,
                    "parameters": func.parameters
                } for func in request.functions
            ]
        
        if request.tool_choice:
            pipeline_kwargs["tool_choice"] = request.tool_choice
        elif request.function_call:
            pipeline_kwargs["function_call"] = request.function_call
        
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
        try:
            async for chunk in stream_pipeline(messages, **pipeline_kwargs):
                # Check if the chunk is a tool call
                if isinstance(chunk, dict) and "tool_calls" in chunk:
                    for i, tool_call in enumerate(chunk["tool_calls"]):
                        response = StreamResponse(
                            id=completion_id,
                            object="chat.completion.chunk",
                            created=current_time,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=DeltaMessage(
                                        content=None,
                                        tool_calls=[
                                            ToolCall(
                                                id=f"call_{i}",
                                                type="function",
                                                index=i,
                                                function=FunctionCall(
                                                    name=tool_call["function"]["name"],
                                                    arguments=tool_call["function"]["arguments"]
                                                )
                                            )
                                        ]
                                    ),
                                    finish_reason="tool_calls"
                                )
                            ]
                        )
                        yield f"data: {json.dumps(response.model_dump())}\n\n"
                    finish_reason = "tool_calls"
                elif isinstance(chunk, dict) and "function_call" in chunk:
                    function_call = chunk["function_call"]
                    response = StreamResponse(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=current_time,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(
                                    function_call=FunctionCall(
                                        name=function_call["name"],
                                        arguments=function_call["arguments"]
                                    )
                                ),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {json.dumps(response.model_dump())}\n\n"
                    finish_reason = "function_call"
                else:
                    # Regular text content
                    word = chunk
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
                    finish_reason = "stop"
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}")
            # If we encounter an error during streaming, fall back to non-streaming response
            finish_reason = "stop"
            response = StreamResponse(
                id=completion_id,
                object="chat.completion.chunk",
                created=current_time,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=DeltaMessage(content=f"Error during processing: {str(e)}"),
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
                    finish_reason=finish_reason
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

