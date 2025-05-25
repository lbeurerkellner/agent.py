import os
import asyncio
from agentpy.agent import Agent, MCP

async def amain():
    agent = Agent(
        "You are a helpful assistant that can manage emails. You can read the inbox and send emails.",
        model="gpt-4o",
        tools=[MCP("arxiv", command="uvx", args=["arxiv-mcp-server"])]
    )
    await agent.cli(persistent=False)

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nExiting...")