# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""competitor-analysis-agent - AI Competitive Intelligence Agent."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.tools.arxiv import ArxivTools
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.mcp import MultiMCPTools
from agno.tools.mem0 import Mem0Tools
from agno.tools.reasoning import ReasoningTools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global instances
mcp_tools: MultiMCPTools | None = None
agent: Agent | None = None
_initialized = False
_init_lock = asyncio.Lock()


class FirecrawlKeyError(ValueError):
    """Firecrawl API key is missing."""


class AgentNotReadyError(RuntimeError):
    """Agent is not initialized."""


async def initialize_mcp_tools(env: dict[str, str] | None = None) -> None:
    """Initialize and connect to MCP servers.

    Args:
        env: Environment variables dict for MCP servers (e.g., API keys)
    """
    global mcp_tools

    # Initialize MultiMCPTools with all MCP server commands
    mcp_tools = MultiMCPTools(
        commands=[
            # TODO: Add your MCP server commands here
            # Example: "npx -y @modelcontextprotocol/server-google-maps",
        ],
        env=env or dict(os.environ),  # Use provided env or fall back to os.environ
        allow_partial_failure=True,  # Don't fail if one server is unavailable
        timeout_seconds=30,
    )

    # Connect to all MCP servers
    await mcp_tools.connect()
    print("✅ Connected to MCP servers")


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Try multiple possible locations for agent_config.json
    possible_paths = [
        Path(__file__).parent.parent / "agent_config.json",  # Project root
        Path(__file__).parent / "agent_config.json",  # Same directory
        Path.cwd() / "agent_config.json",  # Current working directory
    ]

    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading {config_path}: {e}")
                continue

    # If no config found, create a minimal default
    return {
        "name": "competitor-analysis-agent",
        "description": "AI Competitive Intelligence Agent",
        "version": "1.0.0",
        "deployment": {
            "url": "http://0.0.0.0:3773",
            "expose": True,
            "protocol_version": "1.0.0",
        },
        "environment_variables": [
            {"key": "OPENAI_API_KEY", "description": "OpenAI API key", "required": False},
            {"key": "OPENROUTER_API_KEY", "description": "OpenRouter API key", "required": False},
            {"key": "FIRECRAWL_API_KEY", "description": "Firecrawl API key", "required": True},
            {"key": "MEM0_API_KEY", "description": "Mem0 API key for memory", "required": False},
        ],
    }


async def initialize_agent() -> None:
    """Initialize the competitor analysis agent with proper model and tools."""
    global agent, mcp_tools

    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    mem0_api_key = os.getenv("MEM0_API_KEY")

    if not firecrawl_api_key:
        print("⚠️  FIRECRAWL_API_KEY is required. Get it from: https://firecrawl.dev")
        raise FirecrawlKeyError()

    # Model selection logic
    if openai_api_key:
        model = OpenAIChat(id="gpt-4o", api_key=openai_api_key)
        print("✅ Using OpenAI GPT-4o")
    elif openrouter_api_key:
        model = OpenRouter(id="openai/gpt-4o", api_key=openrouter_api_key)
        print("✅ Using OpenRouter GPT-4o")
    else:
        # Boot without error, fail at runtime when needed
        model = OpenAIChat(id="gpt-4o")
        print("⚠️  No LLM API key provided - agent will fail at runtime")

    # Initialize tools
    firecrawl_tools = FirecrawlTools(
        api_key=firecrawl_api_key,
        enable_search=True,
        enable_crawl=True,
        enable_mapping=True,
        formats=["markdown", "links", "html"],
        search_params={"limit": 2},
        limit=5,
    )

    reasoning_tools = ReasoningTools(add_instructions=True)

    # Collect all available tools
    tools = [firecrawl_tools, reasoning_tools]

    # Add Mem0 tools if API key is available
    if mem0_api_key:
        tools.append(Mem0Tools(api_key=mem0_api_key))
        print("🧠 Mem0 memory enabled")

    # Add Arxiv tools for research paper analysis
    tools.append(ArxivTools(all=True))

    # Add MCP tools if available
    if mcp_tools:
        tools.append(mcp_tools)

    # Create the competitor analysis agent
    agent = Agent(
        name="Competitor Analysis Agent",
        model=model,
        tools=tools,
        instructions=[
            "1. Initial Research & Discovery:",
            "   - Use search tool to find information about the target company",
            "   - Search for '[company name] competitors', 'companies like [company name]'",
            "   - Search for industry reports and market analysis",
            "   - Use the think tool to plan your research approach",
            "2. Competitor Identification:",
            "   - Search for each identified competitor using Firecrawl",
            "   - Find their official websites and key information sources",
            "   - Map out the competitive landscape",
            "3. Website Analysis:",
            "   - Scrape competitor websites using Firecrawl",
            "   - Map their site structure to understand their offerings",
            "   - Extract product information, pricing, and value propositions",
            "   - Look for case studies and customer testimonials",
            "4. Deep Competitive Analysis:",
            "   - Use the analyze tool after gathering information on each competitor",
            "   - Compare features, pricing, and market positioning",
            "   - Identify patterns and competitive dynamics",
            "   - Think through the implications of your findings",
            "5. Strategic Synthesis:",
            "   - Conduct SWOT analysis for each major competitor",
            "   - Use reasoning to identify competitive advantages",
            "   - Analyze market trends and opportunities",
            "   - Develop strategic recommendations",
            "- Always use the think tool before starting major research phases",
            "- Use the analyze tool to process findings and draw insights",
            "- Search for multiple perspectives on each competitor",
            "- Verify information by checking multiple sources",
            "- Be thorough but focused in your analysis",
            "- Provide evidence-based recommendations",
            "- Use Arxiv tools to find relevant research papers when applicable",
            "- Use memory tools to maintain context across analysis sessions",
        ],
        expected_output=dedent("""\
            # Competitive Analysis Report: {Target Company}

            ## Executive Summary
            {High-level overview of competitive landscape and key findings}

            ## Research Methodology
            - Search queries used
            - Websites analyzed
            - Key information sources
            - Research papers reviewed (if applicable)

            ## Market Overview
            ### Industry Context
            - Market size and growth rate
            - Key trends and drivers
            - Regulatory environment
            - Academic research insights

            ### Competitive Landscape
            - Major players identified
            - Market segmentation
            - Competitive dynamics

            ## Competitor Analysis

            ### Competitor 1: {Name}
            #### Company Overview
            - Website: {URL}
            - Founded: {Year}
            - Headquarters: {Location}
            - Company size: {Employees/Revenue if available}

            #### Products & Services
            - Core offerings
            - Key features and capabilities
            - Pricing model and tiers
            - Target market segments

            #### Digital Presence Analysis
            - Website structure and user experience
            - Key messaging and value propositions
            - Content strategy and resources
            - Customer proof points

            #### SWOT Analysis
            **Strengths:**
            - {Evidence-based strengths}

            **Weaknesses:**
            - {Identified weaknesses}

            **Opportunities:**
            - {Market opportunities}

            **Threats:**
            - {Competitive threats}

            ### Competitor 2: {Name}
            {Similar structure as above}

            ### Competitor 3: {Name}
            {Similar structure as above}

            ## Comparative Analysis

            ### Feature Comparison Matrix
            | Feature | {Target} | Competitor 1 | Competitor 2 | Competitor 3 |
            |---------|----------|--------------|--------------|--------------|
            | {Feature 1} | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |
            | {Feature 2} | ✓/✗ | ✓/✗ | ✓/✗ | ✓/✗ |

            ### Pricing Comparison
            | Company | Entry Level | Professional | Enterprise |
            |---------|-------------|--------------|------------|
            | {Pricing details extracted from websites} |

            ### Market Positioning Analysis
            {Analysis of how each competitor positions themselves}

            ## Strategic Insights

            ### Key Findings
            1. {Major insight with evidence}
            2. {Competitive dynamics observed}
            3. {Market gaps identified}
            4. {Research insights from academic papers}

            ### Competitive Advantages
            - {Target company's advantages}
            - {Unique differentiators}

            ### Competitive Risks
            - {Main threats from competitors}
            - {Market challenges}

            ## Strategic Recommendations

            ### Immediate Actions (0-3 months)
            1. {Quick competitive responses}
            2. {Low-hanging fruit opportunities}

            ### Short-term Strategy (3-12 months)
            1. {Product/service enhancements}
            2. {Market positioning adjustments}

            ### Long-term Strategy (12+ months)
            1. {Sustainable differentiation}
            2. {Market expansion opportunities}

            ## Conclusion
            {Summary of competitive position and strategic imperatives}

            ## References
            - Websites analyzed
            - Research papers cited
            - Industry reports referenced
        """),
        add_datetime_to_context=True,
        markdown=True,
        stream_events=True,
    )
    print("✅ Competitor Analysis Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Agent response
    """
    global agent
    if not agent:
        raise AgentNotReadyError()

    # Run the agent and get response
    response = await agent.arun(messages)
    return response


async def cleanup_mcp_tools() -> None:
    """Close all MCP server connections."""
    global mcp_tools

    if mcp_tools:
        try:
            await mcp_tools.close()
            print("🔌 Disconnected from MCP servers")
        except Exception as e:
            print(f"⚠️  Error closing MCP tools: {e}")


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns:
        Agent response
    """
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing MCP tools and agent...")
            # Build environment with API keys
            env = {
                **os.environ,
            }
            await initialize_mcp_tools(env)
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def initialize_all(env: dict[str, str] | None = None):
    """Initialize MCP tools and agent.

    Args:
        env: Environment variables dict for MCP servers
    """
    await initialize_mcp_tools(env)
    await initialize_agent()


def main():
    """Run the main entry point for the Competitor Analysis Agent."""
    global mcp_tools

    parser = argparse.ArgumentParser(description="Bindu Competitor Analysis Agent")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (env: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--firecrawl-api-key",
        type=str,
        default=os.getenv("FIRECRAWL_API_KEY"),
        help="Firecrawl API key (env: FIRECRAWL_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.firecrawl_api_key:
        os.environ["FIRECRAWL_API_KEY"] = args.firecrawl_api_key
    if args.mem0_api_key:
        os.environ["MEM0_API_KEY"] = args.mem0_api_key

    print("🤖 Competitor Analysis Agent - AI Competitive Intelligence")
    print("🔍 Capabilities: Web search, website scraping, competitive analysis, strategic reporting")

    if os.getenv("MEM0_API_KEY"):
        print("🧠 Mem0 memory enabled")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("🚀 Starting Bindu Competitor Analysis Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://0.0.0.0:3773')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup on exit
        print("\n🧹 Cleaning up...")
        asyncio.run(cleanup_mcp_tools())


# Bindufy and start the agent server
if __name__ == "__main__":
    main()
