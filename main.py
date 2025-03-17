from typing import List, Optional, Union, Dict, Any, Literal
import json
import time
import asyncio
import uuid
import socket
import logging
import os
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Haystack imports
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage

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

# ----- Haystack Pipeline Implementation -----

class HaystackPipeline:
    """Haystack pipeline implementation with conversation support."""
    
    def __init__(self):
        """Initialize a Haystack pipeline with OpenAIGenerator for conversation support."""
        try:
            # Configure generator with the API key from environment variable
            self.generator = OpenAIGenerator(
                model="gpt-4o",  # Using GPT-4o instead of GPT-3.5-turbo
                # No system_prompt parameter as we'll incorporate it into our prompt format
            )
            
            # Create pipeline
            self.pipeline = Pipeline()
            
            # Add the generator component to the pipeline
            self.pipeline.add_component("generator", self.generator)
            
            # Initialize conversation history
            self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
            
            # Add a default system message to guide the model's behavior
            self.default_system_message = {
                "role": "system",
                "content": "You are a helpful AI assistant powered by GPT-4. You provide clear, concise, and accurate information. Be friendly and conversational in your responses. When appropriate, provide detailed explanations and examples to help users understand complex topics. You have access to a wide range of knowledge and can help with various tasks."
            }
            
            logger.info("Haystack pipeline initialized successfully with GPT-4")
        except Exception as e:
            logger.error(f"Error initializing Haystack pipeline: {str(e)}")
            raise
    
    async def run(self, messages: List[Dict[str, str]], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the pipeline with the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            conversation_id: Optional ID to maintain conversation history between calls
                            If not provided, a new conversation ID will be generated
        
        Returns:
            Dict containing query results and conversation information
        """
        try:
            # Generate a conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Initialize conversation if it doesn't exist
            if conversation_id not in self.conversation_history:
                self.conversation_history[conversation_id] = []
                
            # Convert messages to Haystack ChatMessage format
            chat_messages = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    chat_messages.append(ChatMessage.from_user(content))
                elif role == "assistant":
                    chat_messages.append(ChatMessage.from_assistant(content))
                elif role == "system":
                    chat_messages.append(ChatMessage.from_system(content))
                elif role == "tool":
                    # Handle tool messages if needed
                    chat_messages.append(ChatMessage.from_tool(content, name=msg.get("name")))
                else:
                    logger.warning(f"Unknown message role: {role}, using as user message")
                    chat_messages.append(ChatMessage.from_user(content))
            
            # Create a prompt that maintains the chat format
            prompt = ""
            system_message = None
            
            # First, find and handle the system message if it exists
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                    break
            
            # Use default system message if none provided
            if not system_message:
                system_message = self.default_system_message["content"]
            
            # Add system message
            prompt += f"System: {system_message}\n\n"
            
            # Add the conversation history
            for msg in messages:
                if msg["role"] != "system":  # Skip system message as it's already handled
                    role = msg["role"].capitalize()
                    content = msg["content"]
                    prompt += f"{role}: {content}\n"
            
            # End with a prompt for the assistant to respond
            prompt += "Assistant: "
            
            # Call the OpenAI generator through the pipeline with the prompt
            result = self.pipeline.run({"generator": {"prompt": prompt}})
            
            # Extract the assistant's response
            assistant_reply = result["generator"]["replies"][0]
            
            # Add all messages to conversation history (if not already there)
            for message in messages:
                if message not in self.conversation_history[conversation_id]:
                    self.conversation_history[conversation_id].append(message)
            
            # Add the assistant's reply to conversation history
            self.conversation_history[conversation_id].append({"role": "assistant", "content": assistant_reply})
            
            return {
                "reply": assistant_reply,
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Error in Haystack pipeline: {str(e)}")
            raise e
    
    async def stream_run(self, messages: List[Dict[str, str]], conversation_id: Optional[str] = None):
        """Stream the pipeline output word by word (simulated with the standard run).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            conversation_id: Optional ID to maintain conversation history between calls
                            If not provided, a new conversation ID will be generated
        
        Yields:
            Words from the generated response
        """
        try:
            # Use the updated run method to get the response
            result = await self.run(messages, conversation_id)
            reply = result["reply"]
            
            # Split the reply into words and yield each one
            words = reply.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Simulated delay
        except Exception as e:
            logger.error(f"Error in Haystack pipeline streaming: {str(e)}")
            yield f"Error: {str(e)}"

# Initialize the Haystack pipeline
logger.info("Initializing Haystack pipeline...")
pipeline = HaystackPipeline()
logger.info("Haystack pipeline initialized successfully")

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
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1677610602,
                "owned_by": "custom-ai-provider"
            },
            {
                "id": "gpt-4o",
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
    Generate a non-streaming chat completion using Haystack
    """
    completion_id = f"chatcmpl-{str(uuid.uuid4())}"
    current_time = int(time.time())
    
    try:
        # Convert request messages to format expected by Haystack
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Call the Haystack pipeline
        result = await pipeline.run(messages)
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
        logger.error(f"Error in generate_chat_completion: {str(e)}")
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
        async for word in pipeline.stream_run(messages):
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
        logger.error(f"Error in stream_chat_completion: {str(e)}")
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