import logging
import uuid
import time
import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Union

from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.tools import ToolInvoker
from haystack.tools import Tool, ComponentTool
from haystack.components.routers import ConditionalRouter
from openai import OpenAI


from app.config.settings import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT

# Configure logging
logger = logging.getLogger(__name__)

# Initialize conversation history dictionary
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# Configure Octagon client
try:
    octagon_api_key = os.environ.get("OCTAGON_API_KEY")
    if not octagon_api_key:
        logger.warning("OCTAGON_API_KEY not found in environment variables")
    
    octagon_client = OpenAI(
        api_key=octagon_api_key,
        base_url="https://api.octagonagents.com/v1",
    )
    logger.info("Octagon client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Octagon client: {str(e)}")
    octagon_client = None

# Create a simple empty function that will be passed to Tool constructor
def empty_tool_function(**kwargs):
    """Empty tool function for Haystack Tool initialization"""
    return kwargs

# Initialize the generator and pipeline
try:    
    # Create a simpler pipeline without complex connections
    pipeline = Pipeline()
    
    # Add generator component with tools
    generator = OpenAIChatGenerator(model=DEFAULT_MODEL)
    pipeline.add_component("generator", generator)
    
    logger.info(f"Haystack pipeline initialized successfully with {DEFAULT_MODEL}")
except Exception as e:
    logger.error(f"Error initializing Haystack pipeline: {str(e)}")
    raise

async def run_pipeline(
    messages: List[Dict[str, str]], 
    conversation_id: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    function_call: Optional[Union[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Run the pipeline with the given messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional ID to maintain conversation history between calls
                        If not provided, a new conversation ID will be generated
        tools: Optional list of tools to pass to the model
        functions: Optional list of functions to pass to the model (legacy)
        tool_choice: Optional tool choice configuration
        function_call: Optional function call configuration (legacy)
    
    Returns:
        Dict containing query results and conversation information
    """
    try:
        # Generate a conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Initialize conversation if it doesn't exist
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []
            
        # Convert messages to ChatMessage format for Haystack
        chat_messages = []
        system_content = DEFAULT_SYSTEM_PROMPT
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                chat_messages.append(ChatMessage.from_user(msg["content"]))
            elif msg["role"] == "assistant":
                chat_messages.append(ChatMessage.from_assistant(msg["content"]))
        
        # Add system message at the beginning
        chat_messages.insert(0, ChatMessage.from_system(system_content))
        
        # Prepare generation kwargs for the OpenAIChatGenerator
        generation_kwargs = {}
        
        # Convert tools/functions to Haystack Tool objects
        haystack_tools = None
        if tools:
            logger.info(f"Setting up generator with {len(tools)} tools")
            haystack_tools = []
            for tool_def in tools:
                if tool_def["type"] == "function":
                    func_def = tool_def["function"]
                    # Create Tool with proper parameters format
                    tool = Tool(
                        name=func_def["name"],
                        description=func_def.get("description", ""),
                        parameters=func_def.get("parameters", {}),
                        function=empty_tool_function
                    )
                    haystack_tools.append(tool)
                    
        elif functions:
            logger.info(f"Setting up generator with {len(functions)} functions")
            haystack_tools = []
            for func in functions:
                # Create Tool with proper parameters format
                tool = Tool(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=func.get("parameters", {}),
                    function=empty_tool_function
                )
                haystack_tools.append(tool)
                
        # Handle tool choice or function call
        if tool_choice and tool_choice != "auto":
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                generation_kwargs["tool_choice"] = {"type": "function", "function": tool_choice.get("function", {})}
        elif function_call and function_call != "auto":
            if isinstance(function_call, dict):
                generation_kwargs["function_call"] = function_call
        
        # Create a new pipeline for this request to avoid concurrency issues
        request_pipeline = Pipeline()
        
        # Configure the generator for this specific request
        request_generator = OpenAIChatGenerator(
            model=DEFAULT_MODEL,
            tools=haystack_tools,
            tools_strict=True,
            api_key=generator.api_key,
            api_base_url=generator.api_base_url
        )
        
        # Add the generator to the pipeline
        request_pipeline.add_component("generator", request_generator)
        
        # Run the pipeline with the configured generator
        pipeline_inputs = {"generator": {"messages": chat_messages}}
        if generation_kwargs:
            pipeline_inputs["generator"].update(generation_kwargs)
            
        result = request_pipeline.run(pipeline_inputs)
        reply = result["generator"]["replies"][0]
        print(str(reply))
        
        # Check if the reply contains tool calls
        assistant_reply = reply.text if reply.text is not None else ""
        result_dict = {"reply": assistant_reply, "conversation_id": conversation_id}
        
        if hasattr(reply, "tool_calls") and reply.tool_calls:
            result_dict["tool_calls"] = [
                {
                    "id": f"call_{str(uuid.uuid4()).replace('-', '')[:12]}",
                    "type": "function",
                    "function": {
                        "name": tool_call.tool_name,
                        "arguments": tool_call.arguments if isinstance(tool_call.arguments, str) else json.dumps(tool_call.arguments)
                    }
                } for tool_call in reply.tool_calls
            ]
            
        elif hasattr(reply, "function_call") and reply.function_call:
            result_dict["function_call"] = {
                "name": reply.function_call.name,
                "arguments": reply.function_call.arguments if isinstance(reply.function_call.arguments, str) else json.dumps(reply.function_call.arguments)
            }
        
        conversation_history[conversation_id].append({"role": "assistant", "content": assistant_reply})
        
        return result_dict
    
    except Exception as e:
        logger.error(f"Error in Haystack pipeline: {str(e)}")
        raise e

async def stream_pipeline(
    messages: List[Dict[str, str]], 
    conversation_id: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    function_call: Optional[Union[str, Dict[str, Any]]] = None
):
    """Stream the pipeline output word by word (simulated with the standard run).
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        conversation_id: Optional ID to maintain conversation history between calls
                        If not provided, a new conversation ID will be generated
        tools: Optional list of tools to pass to the model
        functions: Optional list of functions to pass to the model (legacy)
        tool_choice: Optional tool choice configuration
        function_call: Optional function call configuration (legacy)
    
    Yields:
        Words from the generated response or tool call information
    """
    try:
        # Use the updated run method to get the response with tools
        result = await run_pipeline(
            messages, 
            conversation_id, 
            tools=tools,
            functions=functions,
            tool_choice=tool_choice,
            function_call=function_call
        )
        
        # For simplicity, we'll return the response in a streaming-like way
        # In a real implementation, you would set up streaming with the OpenAI client
        
        # Check if the result contains tool calls or function calls
        if "tool_calls" in result and result["tool_calls"]:
            # Format tool calls to match expected structure
            formatted_tool_calls = []
            for i, tool_call in enumerate(result["tool_calls"]):
                formatted_tool_call = {
                    "id": f"call_{str(uuid.uuid4()).replace('-', '')[:12]}",
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"] if isinstance(tool_call["function"]["arguments"], str) else json.dumps(tool_call["function"]["arguments"])
                    }
                }
                formatted_tool_calls.append(formatted_tool_call)
            
            # Stream tool calls as a single object
            tool_calls_result = {"tool_calls": formatted_tool_calls}
            logger.info(f"Streaming tool calls: {tool_calls_result}")
            yield tool_calls_result
        elif "function_call" in result and result["function_call"]:
            # Stream function call as a single object for now
            function_call_result = {"function_call": result["function_call"]}
            logger.info(f"Streaming function call: {function_call_result}")
            yield function_call_result
        else:
            # For text responses, simulate streaming by yielding the full reply
            # In a real implementation, you would stream words or tokens
            reply = result.get("reply", "")  # Handle cases where reply might be None
            logger.info(f"Streaming text response: {reply}")
            yield reply

    except Exception as e:
        logger.error(f"Error in Haystack pipeline streaming: {str(e)}")
        yield f"Error: {str(e)}"

async def process_streaming_chat_completion(
    model,
    messages,
    tools=None,
    temperature=0.7,
    top_p=1.0,
    n=1,
    max_tokens=None,
    completion_id=None
):
    """Process a streaming chat completion request"""
    logger.info(f"Processing streaming chat completion for model {model}")
    
    # Start with first chunk that includes role: assistant
    current_time = int(time.time())
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": current_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None
            }
        ]
    }
    
    # Stream the response through the pipeline
    result_generator = stream_pipeline(messages, tools=tools)
    
    # Track if we've sent the initial tool_calls chunk (with content: null)
    sent_initial_tool_calls = False
    
    async for result in result_generator:
        # Check if the result contains tool calls or function calls
        if "tool_calls" in result and result["tool_calls"]:
            # Format tool calls to match expected structure
            formatted_tool_calls = []
            for i, tool_call in enumerate(result["tool_calls"]):
                formatted_tool_call = {
                    "index": i,
                    "id": f"call_{str(uuid.uuid4()).replace('-', '')[:12]}",
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"] if isinstance(tool_call["function"]["arguments"], str) else json.dumps(tool_call["function"]["arguments"])
                    }
                }
                formatted_tool_calls.append(formatted_tool_call)
            
            # First send a chunk with content: null and the tool_calls with name/id/type
            if not sent_initial_tool_calls:
                initial_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "index": tc["index"],
                                        "id": tc["id"],
                                        "type": tc["type"],
                                        "function": {
                                            "name": tc["function"]["name"]
                                        }
                                    } for tc in formatted_tool_calls
                                ]
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield initial_chunk
                sent_initial_tool_calls = True
                
                # Then send the arguments in a separate chunk
                arguments_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": current_time,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tc["index"],
                                        "function": {
                                            "arguments": tc["function"]["arguments"]
                                        }
                                    } for tc in formatted_tool_calls
                                ]
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield arguments_chunk
                
        elif isinstance(result, str):
            # Text content chunk
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": current_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": result
                        },
                        "finish_reason": None
                    }
                ]
            }
    
    # Final chunk with finish_reason
    finish_reason = "tool_calls" if sent_initial_tool_calls else "stop"
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": current_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }
        ]
    } 

async def process_chat_completion(
    model,
    messages,
    tools=None,
    temperature=0.7,
    top_p=1.0,
    n=1,
    max_tokens=None,
    completion_id=None
):
    """Process a non-streaming chat completion request"""
    logger.info(f"Processing chat completion for model {model}")
    
    try:
        # Process the request through the pipeline
        result = await run_pipeline(messages, tools=tools)
        
        # Create a proper response format
        current_time = int(time.time())
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": current_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,  # Placeholder
                "completion_tokens": 100,  # Placeholder
                "total_tokens": 200  # Placeholder
            }
        }
        
        # Check if result has tool calls
        if "tool_calls" in result and result["tool_calls"]:
            # Format tool calls
            formatted_tool_calls = []
            for i, tool_call in enumerate(result["tool_calls"]):
                formatted_tool_call = {
                    "id": f"call_{str(uuid.uuid4()).replace('-', '')[:12]}",
                    "type": "function",
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"] if isinstance(tool_call["function"]["arguments"], str) else json.dumps(tool_call["function"]["arguments"])
                    }
                }
                formatted_tool_calls.append(formatted_tool_call)
            
            # Set content to null for tool calls and add the tool calls
            response["choices"][0]["message"]["content"] = None
            response["choices"][0]["message"]["tool_calls"] = formatted_tool_calls
            response["choices"][0]["finish_reason"] = "tool_calls"
        else:
            # Regular text response
            response["choices"][0]["message"]["content"] = result.get("reply", "")
        
        return response
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise e 