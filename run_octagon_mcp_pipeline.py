import os
import logging
import argparse
from haystack import Pipeline
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Replace with your preferred OpenAI model for generation
OPENAI_MODEL = "gpt-4o-mini"
# Retrieve the Octagon API key from environment variables
OCTAGON_API_KEY = os.environ.get("OCTAGON_API_KEY")

if not OCTAGON_API_KEY:
    logger.error("Error: OCTAGON_API_KEY environment variable not set.")
    exit(1)

# --- Setup Argument Parser ---
parser = argparse.ArgumentParser(description="Run Haystack pipeline with multiple Octagon MCP Tools.")
parser.add_argument("--agent", dest="agent_name", 
                    help="The name of the specific Octagon agent to use (overrides the multi-agent approach)")
parser.add_argument("query", help="The user query to send to the pipeline.") 
args = parser.parse_args()

# --- Define Multiple MCP Tools for different Octagon agents ---
# If a specific agent is provided, use only that one; otherwise use all specialized agents
if args.agent_name:
    # Use the single agent specified by the user
    octagon_tools = [
        MCPTool(
            name=args.agent_name,
            description="A Tool to get financial data specified by the user.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        )
    ]
else:
    # Use multiple specialized agents
    octagon_tools = [
        MCPTool(
            name="octagon-companies-agent",
            description="Retrieves basic information (like founding date, employee count, description) about a private company.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
        MCPTool(
            name="octagon-funding-agent",
            description="Analyzes funding rounds, investors, valuations, and investment trends for a private company.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
        MCPTool(
            name="octagon-deals-agent",
            description="Analyzes M&A activities and IPO data related to a private company.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
        MCPTool(
            name="octagon-investors-agent",
            description="Provides detailed information about a company's investors and their investment history/criteria.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
        MCPTool(
            name="octagon-debts-agent",
            description="Analyzes private debt facilities, borrowers, and lenders related to a company.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
        MCPTool(
            name="octagon-deep-research-agent",
            description="Conducts in-depth research, aggregates information from multiple sources, and provides comprehensive analysis.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
        # Fallback to transcripts agent for any other financial data needs
        MCPTool(
            name="octagon-transcripts-agent",
            description="Retrieves and analyzes company earnings call transcripts and financial reports.",
            server_info=StdioServerInfo(
                command="npx",
                args=["-y", "octagon-mcp@latest"],
                env={"OCTAGON_API_KEY": OCTAGON_API_KEY},
            ),
        ),
    ]

# --- Build the Haystack Pipeline ---
pipeline = Pipeline()

# 1. Initial LLM: Processes user input and decides if the tool should be called
pipeline.add_component(
    "llm",
    OpenAIChatGenerator(
        model=OPENAI_MODEL, 
        tools=octagon_tools,
        generation_kwargs={
            "temperature": 0.2,
            "timeout": 180,
        }

    )
)

# 2. Tool Invoker: Executes the MCPTool if the LLM requests it
pipeline.add_component("tool_invoker", ToolInvoker(tools=octagon_tools))

# 3. Output Adapter: Assembles messages for the final response generation
# It combines the original user message, the LLM's initial reply (containing the tool call),
# and the message containing the tool's result.
pipeline.add_component(
    "adapter",
    OutputAdapter(
        template="{{ initial_msg + initial_tool_replies + tool_messages }}",
        output_type=list[ChatMessage],
        unsafe=True, # Allow template access to list elements etc.
    ),
)

# 4. Response LLM: Generates the final user-facing response based on the tool's output
pipeline.add_component("response_llm", OpenAIChatGenerator(
    model=OPENAI_MODEL,
    generation_kwargs={
        "temperature": 0.3,
    }
))

# --- Connect Pipeline Components ---
# Send LLM's reply (which might contain a tool call) to the ToolInvoker
pipeline.connect("llm.replies", "tool_invoker.messages")
# Also send the LLM's reply to the adapter (to preserve the conversation flow)
pipeline.connect("llm.replies", "adapter.initial_tool_replies")
# Send the result from the tool back to the adapter
pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
# Send the assembled messages to the final LLM for response generation
pipeline.connect("adapter.output", "response_llm.messages")

# --- Run the Pipeline ---
# Use the query from the command-line argument eg: python run_octagon_mcp_pipeline.py "query" --agent "octagon-companies-agent"
user_input_text = args.query
user_input_msg = ChatMessage.from_user(text=user_input_text)

logger.info(f"Running pipeline with query: '{user_input_text}'")

# Provide the initial message(s) to both the first LLM and the adapter
pipeline_input = {
    "llm": {"messages": [user_input_msg]},
    "adapter": {"initial_msg": [user_input_msg]},
}

try:
    result = pipeline.run(pipeline_input)

    # --- Print the Result ---
    if "response_llm" in result and result["response_llm"]["replies"]:
        final_reply = result["response_llm"]["replies"][0].text
        logger.info("Pipeline finished successfully.")
        print("\nFinal Response:")
        print(final_reply)
    else:
        logger.warning("Pipeline ran but did not produce a final response.")
        print("\nPipeline Result (Partial):")
        print(result) # Print the raw result if the expected output isn't there

except Exception as e:
    logger.exception("An error occurred while running the pipeline.")
    print(f"\nError: {e}") 