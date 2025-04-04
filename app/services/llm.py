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
import random
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo


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


def dummy_weather(location: str):
    return {"temp": f"{random.randint(-10,40)} °C",
            "humidity": f"{random.randint(0,100)}%"}

# Define a function to create the weather tool on demand
def get_octagon_tool():
    """Create and return the weather tool on demand to avoid event loop conflicts"""
    try:
        # In a web server environment, we should completely avoid MCPTool
        # Return a simple synchronous tool instead
        return MCPTool(
            name="octagon",
            description="A Tool to get financial data",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": octagon_api_key}
            )
        )
    except Exception as e:
        logger.error(f"Error creating MCPTool: {str(e)}")
        # Fallback to a very simple tool
        return

routes = [
    {
        "condition": "{{replies[0].tool_calls | length > 0}}",
        "output": "{{replies}}",
        "output_name": "there_are_tool_calls",
        "output_type": List[ChatMessage],  # Use direct type
    },
    {
        "condition": "{{replies[0].tool_calls | length == 0}}",
        "output": "{{replies}}",
        "output_name": "final_replies",
        "output_type": List[ChatMessage],  # Use direct type
    },
]



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
        tools: Optional list of tools to pass to the model (kept for compatibility, not used)
        functions: Optional list of functions to pass to the model (legacy, kept for compatibility)
        tool_choice: Optional tool choice configuration (kept for compatibility, not used)
        function_call: Optional function call configuration (legacy, kept for compatibility)
    
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
        
        # Create a new pipeline for this request to avoid concurrency issues
        request_pipeline = Pipeline()
        
        # Get or create the octagon tool on demand
        octagon_tool = get_octagon_tool()
        
        # Create components - use two generators to avoid input conflicts
        initial_generator = OpenAIChatGenerator(
            model=DEFAULT_MODEL,
            api_key=generator.api_key,
            api_base_url=generator.api_base_url,
            tools=[octagon_tool]
        )
        
        final_generator = OpenAIChatGenerator(
            model=DEFAULT_MODEL,
            api_key=generator.api_key,
            api_base_url=generator.api_base_url
        )
        
        tool_invoker = ToolInvoker(tools=[octagon_tool])
        
        # Define conditional routes
        routes = [
            {
                "condition": "{{replies[0].tool_calls | length > 0}}",
                "output": "{{replies}}",
                "output_name": "tool_call_replies",
                "output_type": List[ChatMessage],
            },
            {
                "condition": "{{replies[0].tool_calls | length == 0}}",
                "output": "{{replies}}",
                "output_name": "final_replies",
                "output_type": List[ChatMessage],
            },
        ]
        
        router = ConditionalRouter(routes, unsafe=True)
        
        # Add components to pipeline
        request_pipeline.add_component("initial_generator", initial_generator)
        request_pipeline.add_component("router", router)
        request_pipeline.add_component("tool_invoker", tool_invoker)
        request_pipeline.add_component("final_generator", final_generator)
        
        # Connect components
        request_pipeline.connect("initial_generator.replies", "router")
        request_pipeline.connect("router.tool_call_replies", "tool_invoker.messages")
        
        # Create a response assembler to combine initial messages + initial reply + tool messages
        from haystack.components.converters import OutputAdapter
        
        assembler = OutputAdapter(
            template="{{initial_messages + initial_replies + tool_messages}}",
            output_type=List[ChatMessage],
            unsafe=True
        )
        request_pipeline.add_component("assembler", assembler)
        
        # Connect to the assembler
        request_pipeline.connect("tool_invoker.tool_messages", "assembler.tool_messages")
        request_pipeline.connect("router.tool_call_replies", "assembler.initial_replies")
        
        # Connect assembler to final generator
        request_pipeline.connect("assembler.output", "final_generator.messages")
        
        # Run the pipeline
        pipeline_inputs = {
            "initial_generator": {"messages": chat_messages},
            "assembler": {"initial_messages": chat_messages}
        }
        result = request_pipeline.run(pipeline_inputs)
        
        # Get the appropriate reply based on whether tools were used
        if "final_generator" in result:
            reply = result["final_generator"]["replies"][0]
        else:
            reply = result["router"]["final_replies"][0]
        
        # Get assistant reply text
        assistant_reply = reply.text if reply.text is not None else ""
        result_dict = {"reply": assistant_reply, "conversation_id": conversation_id}
        
        # Update conversation history
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
        tools: Optional list of tools to pass to the model (kept for compatibility, not used)
        functions: Optional list of functions to pass to the model (legacy, kept for compatibility)
        tool_choice: Optional tool choice configuration (kept for compatibility, not used)
        function_call: Optional function call configuration (legacy, kept for compatibility)
    
    Yields:
        Words from the generated response
    """
    try:
        # Use the updated run method to get the response
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
        
        # Return text response only
        reply = result.get("reply", "")  # Handle cases where reply might be None
        logger.info(f"Streaming text response: {reply}")
        yield reply

    except Exception as e:
        logger.error(f"Error in Haystack pipeline streaming: {str(e)}")
        yield f"Error: {str(e)}"

