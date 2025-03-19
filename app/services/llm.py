import logging
import uuid
import time
import asyncio
import os
from typing import List, Dict, Any, Optional

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

# Create Octagon agent tools
def create_octagon_tools():
    tools = []
    
    if octagon_client:
        # SEC Filings Analysis Tool
        async def analyze_sec_filings(ticker: str, filing_type: str = "10-K", recent_only: bool = True) -> str:
            """
            Analyze SEC filings for a company
            
            Args:
                ticker: Company ticker symbol (e.g., AAPL)
                filing_type: Type of filing to analyze (10-K, 10-Q, 8-K)
                recent_only: Whether to analyze only the most recent filing
                
            Returns:
                Analysis of the SEC filing
            """
            try:
                response = octagon_client.responses.create(
                    model="octagon-sec-agent",
                    stream=True,
                    input=f"Analyze {filing_type} for {ticker}, recent_only={recent_only}",
                    instructions="You analyze SEC filings and extract financial data.",

                )


                response_text = ""

                for chunk in response:
                    # Check what attributes are available in the chunk
                    print(str(chunk))

                    if hasattr(chunk, 'text'):
                        response_text += chunk.text
                    elif hasattr(chunk, 'delta') and chunk.delta:
                        response_text += chunk.delta
                    elif hasattr(chunk, 'content'):
                        response_text += chunk.content
                    elif hasattr(chunk, 'data'):
                        response_text += chunk.data.get('content', '')

                print(str(response))

                return response_text
                
            
            except Exception as e:
                logger.error(f"Error calling Octagon SEC agent: {str(e)}")
                return f"Error analyzing SEC filings: {str(e)}"
        
        sec_filings_tool = Tool(
            name="sec_filings_analyzer",
            description="Analyzes SEC filings (10-K, 10-Q, 8-K) for a company to extract financial data, risks, and other insights",
            function=analyze_sec_filings,
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Company ticker symbol (e.g., AAPL)"},
                    "filing_type": {"type": "string", "description": "Type of filing to analyze (10-K, 10-Q, 8-K)"},
                    "recent_only": {"type": "boolean", "description": "Whether to analyze only the most recent filing"}
                },
                "required": ["ticker", "filing_type"]
            }
        )
        tools.append(sec_filings_tool)
        
        # Companies Database Tool
        async def get_company_info(query: str) -> str:
            """
            Get basic information about a company
            
            Args:
                query: Company name or ticker
                
            Returns:
                Basic company information
            """
            try:
                # In a real implementation, this would call the Octagon API
                # This is a simplified simulation
                response = octagon_client.responses.create(
                    model="octagon-companies-agent",
                    stream=True,
                    input=f"Get company information for {query}",
                    instructions="You retrieve basic company information such as sector, industry, market cap, and business description.",
                )

                response_text = ""
                
                # Handle streaming response
                for chunk in response:
                    # Check what attributes are available in the chunk
                    if hasattr(chunk, 'text'):
                        response_text += chunk.text
                    elif hasattr(chunk, 'delta') and chunk.delta:
                        response_text += chunk.delta
                    elif hasattr(chunk, 'content'):
                        response_text += chunk.content
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        # Standard OpenAI API response format
                        for choice in chunk.choices:
                            if hasattr(choice, 'text'):
                                response_text += choice.text
                            elif hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                                response_text += choice.delta.content or ""
                
                return response_text
            except Exception as e:
                logger.error(f"Error calling Octagon Companies agent: {str(e)}")
                return f"Error retrieving company information: {str(e)}"
        
        companies_tool = Tool(
            name="company_info",
            description="Retrieves basic company information such as sector, industry, market cap, and business description",
            function=get_company_info,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Company name or ticker symbol (e.g., AAPL, Apple)"}
                },
                "required": ["query"]
            }
        )
        tools.append(companies_tool)
    
    return tools

# Initialize the generator and pipeline
try:
    # Get Octagon tools
    octagon_tools = create_octagon_tools()
    
    # Create a simpler pipeline without complex connections
    pipeline = Pipeline()
    
    # Add generator component with tools
    generator = OpenAIChatGenerator(model=DEFAULT_MODEL, tools=octagon_tools)
    pipeline.add_component("generator", generator)
    
    logger.info(f"Haystack pipeline initialized successfully with {DEFAULT_MODEL} and {len(octagon_tools)} Octagon tools")
except Exception as e:
    logger.error(f"Error initializing Haystack pipeline: {str(e)}")
    raise

async def run_pipeline(messages: List[Dict[str, str]], conversation_id: Optional[str] = None) -> Dict[str, Any]:
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
        
        # Initial pipeline run
        result = pipeline.run({"generator": {"messages": chat_messages}})
        reply = result["generator"]["replies"][0]
        
        # Check if the reply contains tool calls
        if hasattr(reply, 'tool_calls') and reply.tool_calls:
            # Process all tool calls and collect their responses
            tool_messages = []
            
            for tool_call in reply.tool_calls:
                tool_name = tool_call.tool_name
                
                # Fix the arguments handling
                if isinstance(tool_call.arguments, dict):
                    arguments = tool_call.arguments
                elif isinstance(tool_call.arguments, str):
                    arguments = eval(tool_call.arguments)  # Parse string to dict
                else:
                    arguments = tool_call.arguments
                
                # Find the requested tool
                selected_tool = None
                for tool in octagon_tools:
                    if tool.name == tool_name:
                        selected_tool = tool
                        break
                
                # Execute the tool
                if selected_tool:
                    if tool_name == "sec_filings_analyzer":
                        ticker = arguments.get("ticker")
                        filing_type = arguments.get("filing_type", "10-K")
                        recent_only = arguments.get("recent_only", True)
                        tool_result = await selected_tool.function(ticker, filing_type, recent_only)
                    elif tool_name == "company_info":
                        query = arguments.get("query")
                        tool_result = await selected_tool.function(query)
                    else:
                        tool_result = "Tool not found"
                else:
                    tool_result = f"Tool '{tool_name}' not available"
                
                # Create a tool response message for this specific tool call
                tool_message = ChatMessage.from_tool(
                    tool_result=tool_result,
                    origin=tool_call,  # This preserves the tool_call_id relationship
                    error=False
                )
                tool_messages.append(tool_message)
            
            # Add the assistant message and all tool response messages
            final_messages = chat_messages + [reply] + tool_messages
            
            # Run the generator again with all messages
            final_result = pipeline.run({"generator": {"messages": final_messages}})
            assistant_reply = final_result["generator"]["replies"][0].text
        else:
            # No tool calls needed, use direct response
            assistant_reply = reply.text
        
        # Add all messages to conversation history
        for message in messages:
            if message not in conversation_history[conversation_id]:
                conversation_history[conversation_id].append(message)
        
        # Add the assistant's reply to conversation history
        conversation_history[conversation_id].append({"role": "assistant", "content": assistant_reply})
        
        return {
            "reply": assistant_reply,
            "conversation_id": conversation_id
        }
    except Exception as e:
        logger.error(f"Error in Haystack pipeline: {str(e)}")
        raise e

async def stream_pipeline(messages: List[Dict[str, str]], conversation_id: Optional[str] = None):
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
        result = await run_pipeline(messages, conversation_id)
        reply = result["reply"]
        
        yield reply


    except Exception as e:
        logger.error(f"Error in Haystack pipeline streaming: {str(e)}")
        yield f"Error: {str(e)}" 