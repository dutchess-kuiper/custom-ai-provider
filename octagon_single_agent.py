import asyncio
import os
import argparse
import json
import sys
from openai import AsyncOpenAI
from agents import Agent, OpenAIResponsesModel, Runner, ItemHelpers, MessageOutputItem, trace

async def run_single_agent(agent_type, company_name, company_url=None, query=None):
    """Run a single Octagon agent for targeted company research.
    
    Args:
        agent_type: Type of agent to run (companies, funding, deals, investors, debts, deep_research)
        company_name: Name of the company to research
        company_url: Optional company URL for better results
        query: Specific query for the agent
        
    Returns:
        The response from the specified agent
    """
    print(f"\n🔍 Researching {company_name} using {agent_type} agent...\n")
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
    
    # Select the appropriate agent based on agent_type
    agent = None
    agent_description = ""
    
    if agent_type == "companies":
        agent_description = "basic company information"
        agent = Agent(
            name="Companies Database",
            instructions="You retrieve private company information from Octagon's companies database.",
            model=OpenAIResponsesModel(
                model="octagon-companies-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "funding":
        agent_description = "funding rounds and investors"
        agent = Agent(
            name="Funding Analysis",
            instructions="You analyze private company funding data, including funding rounds, investors, valuations, and investment trends.",
            model=OpenAIResponsesModel(
                model="octagon-funding-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "deals":
        agent_description = "M&A and IPO activities"
        agent = Agent(
            name="M&A and IPO Analysis",
            instructions="You analyze M&A and IPO transaction data, including acquisitions, mergers, and initial public offerings.",
            model=OpenAIResponsesModel(
                model="octagon-deals-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "investors":
        agent_description = "investor details"
        agent = Agent(
            name="Investors Analysis",
            instructions="You provide information about investors, their investment criteria, and past activities.",
            model=OpenAIResponsesModel(
                model="octagon-investors-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "debts":
        agent_description = "debt facilities"
        agent = Agent(
            name="Private Debts Analysis",
            instructions="You analyze private debts, borrowers, and lenders, providing insights on debt facilities and terms.",
            model=OpenAIResponsesModel(
                model="octagon-debts-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "deep_research":
        agent_description = "comprehensive research"
        agent = Agent(
            name="Deep Research",
            instructions="You conduct in-depth research by aggregating information from multiple sources to provide comprehensive analysis and insights.",
            model=OpenAIResponsesModel(
                model="octagon-deep-research-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "sec":
        agent_description = "SEC filings analysis"
        agent = Agent(
            name="SEC Filings",
            instructions="You analyze SEC filings and disclosures for publicly traded companies.",
            model=OpenAIResponsesModel(
                model="octagon-sec-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "transcripts":
        agent_description = "earnings call transcripts"
        agent = Agent(
            name="Earnings Transcripts",
            instructions="You analyze earnings call transcripts and management commentary.",
            model=OpenAIResponsesModel(
                model="octagon-transcripts-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "financials":
        agent_description = "financial statements"
        agent = Agent(
            name="Financial Statements",
            instructions="You analyze financial statements and calculate financial ratios.",
            model=OpenAIResponsesModel(
                model="octagon-financials-agent",
                openai_client=octagon_client,
            ),
        )
    elif agent_type == "stock":
        agent_description = "stock market data"
        agent = Agent(
            name="Stock Data",
            instructions="You analyze stock price movements, trading volumes, and market trends.",
            model=OpenAIResponsesModel(
                model="octagon-stock-data-agent",
                openai_client=octagon_client,
            ),
        )
    else:
        return f"Error: Invalid agent type '{agent_type}'. Use one of: companies, funding, deals, investors, debts, deep_research, sec, transcripts, financials, stock"
    
    # Format the user input with company name and URL if provided
    user_input = ""
    if query:
        if company_url:
            user_input = f"{query} for {company_name} ({company_url})"
        else:
            user_input = f"{query} for {company_name}"
    else:
        # Default query if none provided
        if company_url:
            user_input = f"Provide {agent_description} for {company_name} ({company_url})"
        else:
            user_input = f"Provide {agent_description} for {company_name}"
    
    try:
        # Run the agent in a trace for better logging
        with trace(f"{agent.name} Research"):
            result = await Runner.run(agent, user_input)
            
            # Process and display result
            output_text = ""
            for item in result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        output_text += f"\n{text}"
                        print(f"\n{text}")
        
        print("\n" + "=" * 60)
        print("\n✅ Research complete!")
        
        # Return the final output for the API
        return output_text if output_text else result.final_output
        
    except Exception as e:
        error_msg = f"\n❌ Error with {agent_type} agent: {str(e)}"
        print(error_msg)
        return f"Error: {str(e)}"

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a single Octagon agent for targeted research")
    parser.add_argument("--agent", required=True, choices=[
        "companies", "funding", "deals", "investors", "debts", "deep_research",
        "sec", "transcripts", "financials", "stock"
    ], help="Type of agent to run")
    parser.add_argument("--company", required=True, help="The company name to research")
    parser.add_argument("--url", help="The company URL (optional, improves results for private companies)")
    parser.add_argument("query", nargs="?", default=None, help="The specific research query (optional)")
    args = parser.parse_args()
    
    # Run the specified agent
    result = await run_single_agent(args.agent, args.company, args.url, args.query)
    
    # Print final result
    print("\nFinal Response:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 