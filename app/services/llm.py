import logging
import uuid
import time
import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Union
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import sys

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
    return {
        "temp": f"{random.randint(-10, 40)} °C",
        "humidity": f"{random.randint(0, 100)}%",
    }


def manual_octagon_invoke(**kwargs):
    """
    Executes either the single-agent MCP script or the multi-agent workflow script.
    
    Args:
        **kwargs: Expected to contain:
            - 'query': The query to pass to the tool
            - 'agent_name': Optional name of the specific agent to use
                           If provided, uses single-agent mode with run_octagon_mcp_pipeline.py
            - 'company_name': Optional company name to focus the research on
                             If provided without agent_name, uses multi-agent workflow
            - 'company_url': Optional company URL for better results
    
    Returns:
        The parsed response from the appropriate Octagon tool or workflow.
    """
    try:
        query = kwargs.get("query")
        if not query:
            logger.error("Missing 'query' argument for manual_octagon_invoke.")
            return {"error": "Tool requires a 'query' argument."}

        # Get agent name and company info if provided
        agent_name = kwargs.get("agent_name", "")
        company_name = kwargs.get("company_name", "")
        company_url = kwargs.get("company_url", "")
        
        # Determine which workflow to use based on parameters
        if agent_name:
            # Single agent mode - use Haystack MCP pipeline
            logger.info(f"Using single Octagon agent: {agent_name}")
            script_path = os.path.join(
                os.path.dirname(__file__), "../../run_octagon_mcp_pipeline.py"
            )
            python_executable = sys.executable
            
            # Command to execute the script
            command = [python_executable, script_path, "--agent", agent_name, query]
            
        elif company_name:
            # Multi-agent research workflow - use the agents library
            logger.info(f"Using multi-agent research workflow for company: {company_name}")
            script_path = os.path.join(
                os.path.dirname(__file__), "../../octagon_agents_workflow.py"
            )
            python_executable = sys.executable
            
            # Command to execute the multi-agent workflow script
            command = [python_executable, script_path, "--company", company_name]
            
            # Add URL if provided
            if company_url:
                command.extend(["--url", company_url])
                
            # Add the query as the last parameter
            if query:
                command.append(query)
                
        else:
            # No agent_name or company_name - default to Haystack with transcripts agent
            logger.info("No specific agent or company provided, using default transcripts agent")
            script_path = os.path.join(
                os.path.dirname(__file__), "../../run_octagon_mcp_pipeline.py"
            )
            python_executable = sys.executable
            command = [python_executable, script_path, query]

        logger.info(f"Running command: {' '.join(command)}")

        # Execute the script using subprocess
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )

        # Send the input and get the output - add a timeout
        try:
            # Add a longer timeout (180 seconds) since multi-agent workflow takes longer
            stdout, stderr = process.communicate(timeout=500)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            logger.error("Octagon script subprocess timed out after 500 seconds.")
            return {
                "error": "Octagon tool script timed out",
                "stderr": stderr.strip(),
            }

        logger.debug(f"Script stdout: {stdout}")
        logger.debug(f"Script stderr: {stderr}")
        logger.debug(f"Script return code: {process.returncode}")

        if process.returncode != 0:
            error_message = f"Error executing Octagon script. Exit code: {process.returncode}. Stderr: {stderr.strip()}"
            logger.error(error_message)
            return {"error": error_message, "raw_stdout": stdout.strip()}

        # If stdout is empty even on success, log warning
        if not stdout.strip():
            logger.warning("Octagon script returned success code but empty stdout.")
            if stderr.strip():
                logger.warning(f"Stderr content on success: {stderr.strip()}")
                return {
                    "result": None,
                    "warning": "Empty output from script",
                    "stderr_info": stderr.strip(),
                }
            else:
                return {"result": None, "warning": "Empty output from script"}

        # Extract just the "Final Response:" part if present
        result = stdout.strip()
        if "Final Response:" in result:
            parts = result.split("Final Response:", 1)
            result = parts[1].strip()

        logger.info(f"Octagon script result obtained")
        logger.info(f"Result: {result}")
        return {"result": result}

    except Exception as e:
        # Catch potential errors during subprocess setup or communication
        logger.exception(f"Error during Octagon script execution: {str(e)}")
        return {"error": f"Internal tool script execution error: {str(e)}"}


# Define a function to create an Octagon tool that uses our manual function

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
    function_call: Optional[Union[str, Dict[str, Any]]] = None,
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
        octagon_tool = Tool(
            name="octagon",
            description="""A Tool to access financial data through Octagon's specialized agents.
                           You can use this in two different ways:
                           
                           1. SINGLE AGENT MODE - For specific financial data needs, specify:
                              - query: Your specific question
                              - agent_name: The specific agent to use (see list below)
                           
                           2. MULTI-AGENT RESEARCH WORKFLOW - For comprehensive company research, specify:
                              - query: Your research question or objective
                              - company_name: The company to research (required)
                              - company_url: The company's website URL (optional, improves results)
                           
                           Available specialized agents for single agent mode:
                           - octagon-sec-agent: For SEC filings, regulatory documents, and company disclosures
                           - octagon-transcripts-agent: For earnings call transcripts and conference calls
                           - octagon-financials-agent: For financial statements, balance sheets, income statements
                           - octagon-stock-data-agent: For stock prices, historical market data, trading volumes
                           - octagon-companies-agent: For general company information, profiles, descriptions
                           - octagon-funding-agent: For funding rounds, capital raises, investment information
                           - octagon-deals-agent: For mergers, acquisitions, partnerships, corporate transactions
                           - octagon-investors-agent: For information about institutional investors, funds, shareholders
                           - octagon-scraper-agent: For web-scraped financial news from online sources
                           - octagon-deep-research-agent: For comprehensive research reports and analyses
                           - octagon-debts-agent: For corporate debt information, bonds, loans, liabilities
                           
                           The multi-agent research workflow automatically orchestrates the following process:
                           1. Retrieves basic company information
                           2. Analyzes funding rounds and investors
                           3. Examines M&A and IPO activities
                           4. Researches key investors
                           5. Reviews debt facilities
                           6. Synthesizes all findings into a comprehensive analysis""",
            function=manual_octagon_invoke,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The specific financial query or research objective",
                    },
                    "agent_name": {
                        "type": "string",
                        "description": "For single agent mode: The specific Octagon agent to use. Don't specify this if using company_name for multi-agent workflow.",
                        "enum": [
                            "octagon-sec-agent",
                            "octagon-transcripts-agent",
                            "octagon-financials-agent",
                            "octagon-stock-data-agent",
                            "octagon-companies-agent",
                            "octagon-funding-agent",
                            "octagon-deals-agent",
                            "octagon-investors-agent",
                            "octagon-scraper-agent",
                            "octagon-deep-research-agent",
                            "octagon-debts-agent",
                        ],
                    },
                    "company_name": {
                        "type": "string",
                        "description": "For multi-agent workflow: The name of the company to research. Providing this without agent_name will trigger the multi-agent research workflow.",
                    },
                    "company_url": {
                        "type": "string", 
                        "description": "For multi-agent workflow: The company's website URL (optional, improves research results for private companies)",
                    }
                },
                "required": ["query"],
            },
        )

        # Create components - use two generators to avoid input conflicts
        initial_generator = OpenAIChatGenerator(
            model=DEFAULT_MODEL,
            api_key=generator.api_key,
            api_base_url=generator.api_base_url,
            tools=[octagon_tool],
        )

        final_generator = OpenAIChatGenerator(
            model=DEFAULT_MODEL,
            api_key=generator.api_key,
            api_base_url=generator.api_base_url,
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
            unsafe=True,
        )
        request_pipeline.add_component("assembler", assembler)

        # Connect to the assembler
        request_pipeline.connect(
            "tool_invoker.tool_messages", "assembler.tool_messages"
        )
        request_pipeline.connect(
            "router.tool_call_replies", "assembler.initial_replies"
        )

        # Connect assembler to final generator
        request_pipeline.connect("assembler.output", "final_generator.messages")

        # Run the pipeline
        pipeline_inputs = {
            "initial_generator": {"messages": chat_messages},
            "assembler": {"initial_messages": chat_messages},
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
        conversation_history[conversation_id].append(
            {"role": "assistant", "content": assistant_reply}
        )

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
    function_call: Optional[Union[str, Dict[str, Any]]] = None,
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
            function_call=function_call,
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
