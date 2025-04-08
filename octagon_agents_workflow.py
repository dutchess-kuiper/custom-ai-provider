import asyncio
import os
import argparse
import json
from openai import AsyncOpenAI
from agents import Agent, OpenAIResponsesModel, Runner, ItemHelpers, MessageOutputItem, trace

async def run_research_workflow(company_name, company_url=None, query=None):
    """Run the research workflow using the Runner to handle tool calls."""
    print(f"\n🔍 Researching {company_name}...\n")
    print("=" * 60)
    
    # Retrieve API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    octagon_api_key = os.environ.get("OCTAGON_API_KEY")
    
    if not openai_api_key or not octagon_api_key:
        error_msg = "Missing required API keys: "
        if not openai_api_key:
            error_msg += "OPENAI_API_KEY "
        if not octagon_api_key:
            error_msg += "OCTAGON_API_KEY"
        print(error_msg)
        return error_msg
    
    # Configure Octagon client
    octagon_client = AsyncOpenAI(
        api_key=octagon_api_key,
        base_url="https://api-gateway.octagonagents.com/v1",
    )
    
    # Configure OpenAI client
    openai_client = AsyncOpenAI(api_key=openai_api_key)

    # Create Octagon specialized private market agents
    companies_agent = Agent(
        name="Companies Database",
        instructions="You retrieve private company information from Octagon's companies database.",
        model=OpenAIResponsesModel(
            model="octagon-companies-agent",
            openai_client=octagon_client,
        ),
    )

    funding_agent = Agent(
        name="Funding Analysis",
        instructions="You analyze private company funding data, including funding rounds, investors, valuations, and investment trends.",
        model=OpenAIResponsesModel(
            model="octagon-funding-agent",
            openai_client=octagon_client,
        ),
    )

    deals_agent = Agent(
        name="M&A and IPO Analysis",
        instructions="You analyze M&A and IPO transaction data, including acquisitions, mergers, and initial public offerings.",
        model=OpenAIResponsesModel(
            model="octagon-deals-agent",
            openai_client=octagon_client,
        ),
    )

    investors_agent = Agent(
        name="Investors Analysis",
        instructions="You provide information about investors, their investment criteria, and past activities.",
        model=OpenAIResponsesModel(
            model="octagon-investors-agent",
            openai_client=octagon_client,
        ),
    )

    debts_agent = Agent(
        name="Private Debts Analysis",
        instructions="You analyze private debts, borrowers, and lenders, providing insights on debt facilities and terms.",
        model=OpenAIResponsesModel(
            model="octagon-debts-agent",
            openai_client=octagon_client,
        ),
    )

    deep_research_agent = Agent(
        name="Deep Research",
        instructions="You conduct in-depth research by aggregating information from multiple sources to provide comprehensive analysis and insights.",
        model=OpenAIResponsesModel(
            model="octagon-deep-research-agent",
            openai_client=octagon_client,
        ),
    )

    # Create a research workflow coordinator with OpenAI
    coordinator = Agent(
        name="Research Coordinator",
        instructions=(
            "You are a research coordinator tasked with gathering comprehensive information about a private company using specialized agent tools.\n"
            "Follow this exact workflow:\n"
            "1. Use `get_company_info` to retrieve basic company details.\n"
            "2. Use `analyze_funding` to find funding round details and investors.\n"
            "3. Use `analyze_deals` to check for M&A or IPO activities.\n"
            "4. Use `analyze_investors` to learn more about the company's key investors.\n"
            "5. Use `analyze_debts` to identify any debt facilities.\n"
            "6. Finally, use `conduct_deep_research` to synthesize all gathered information and provide a comprehensive analysis or answer specific research questions requiring aggregation.\n\n"
            "**CRITICAL INSTRUCTION:** For each step, you MUST formulate a clear, full-sentence question to ask the tool. This question should explicitly mention the company's name and its URL (if provided)."
            "DO NOT just pass the company name/URL string. You MUST ask a specific question.\n\n"
            "Example Questions:\n"
            "- For `get_company_info`: 'Can you provide the basic company information for Example Corp (example.com)?'\n"
            "- For `analyze_funding`: 'What are the details of recent funding rounds for Example Corp (example.com)?'\n"
            "- For `conduct_deep_research`: 'Provide a deep research analysis on the competitive landscape for Example Corp (example.com), using the information gathered so far.'\n\n"
            "Before calling each tool, briefly explain which step you are on and what information you are seeking.\n"
            "Always follow the numbered steps in order and do not skip any."
        ),
        model=OpenAIResponsesModel(
            model="gpt-4o-mini",  # Using gpt-4o-mini for consistency with the Haystack pipeline
            openai_client=openai_client,
        ),
        tools=[
            companies_agent.as_tool(
                tool_name="get_company_info",
                tool_description="Retrieves basic information (like founding date, employee count, description) about a private company.",
            ),
            funding_agent.as_tool(
                tool_name="analyze_funding",
                tool_description="Analyzes funding rounds, investors, valuations, and investment trends for a private company.",
            ),
            deals_agent.as_tool(
                tool_name="analyze_deals",
                tool_description="Analyzes M&A activities and IPO data related to a private company.",
            ),
            investors_agent.as_tool(
                tool_name="analyze_investors",
                tool_description="Provides detailed information about a company's investors and their investment history/criteria.",
            ),
            debts_agent.as_tool(
                tool_name="analyze_debts",
                tool_description="Analyzes private debt facilities, borrowers, and lenders related to a company.",
            ),
            deep_research_agent.as_tool(
                tool_name="conduct_deep_research",
                tool_description="Conducts in-depth research, aggregates information from multiple sources, and provides comprehensive analysis.",
            ),
        ],
    )
    
    # Create a simple input for our coordinator agent with the company URL if provided
    user_input = ""
    if query:
        # If a specific query is provided, use it
        if company_url:
            user_input = f"{query} for {company_name} ({company_url})"
        else:
            user_input = f"{query} for {company_name}"
    else:
        # Default to comprehensive research report
        if company_url:
            user_input = f"Create a comprehensive investment research report for {company_name} ({company_url})."
        else:
            user_input = f"Create a comprehensive investment research report for {company_name}."
    
    try:
        # Run the entire orchestration in a single trace
        with trace("Private Market Research Orchestrator"):
            result = await Runner.run(coordinator, user_input)
            
            # Print each step's output for logging
            for item in result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        print(f"\n{text}")
        
        print("\n" + "=" * 60)
        print("\n✅ Research complete!")
        
        # Return the final output for the API
        return result.final_output
        
    except Exception as e:
        error_msg = f"\n❌ Error during research: {str(e)}"
        print(error_msg)
        return f"Error: {str(e)}"

async def main():
    # Parse command line arguments eg python octagon_agents_workflow.py --company "Example Corp" --url "example.com" --query "What is the company's revenue?"
    parser = argparse.ArgumentParser(description="Run Octagon multi-agent research workflow")
    parser.add_argument("--company", required=True, help="The company name to research")
    parser.add_argument("--url", help="The company URL (optional)")
    parser.add_argument("query", nargs="?", default=None, help="The specific research query (optional)")
    args = parser.parse_args()
    
    # Run the research workflow
    result = await run_research_workflow(args.company, args.url, args.query)
    
    # Print final report
    print("\nFinal Report:")
    print(result)
    
    # For use with subprocess, we need structured output that can be captured
    print("\nFinal Response:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 