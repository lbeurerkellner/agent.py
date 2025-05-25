#!/usr/bin/env python3
"""
Test script to demonstrate streaming tool calls functionality.
This script shows how the agent streams both content and tool calls in real-time.
"""

import asyncio
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentpy.agent import Agent, tool

@tool
def calculate_fibonacci(n: str) -> str:
    """Calculate the nth Fibonacci number."""
    try:
        n = int(n)
        if n < 0:
            return "Error: n must be a non-negative integer"
        elif n == 0:
            return "0"
        elif n == 1:
            return "1"
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return str(b)
    except ValueError:
        return "Error: n must be a valid integer"

@tool
def get_system_info() -> str:
    """Get basic system information."""
    import platform
    return f"OS: {platform.system()} {platform.release()}, Python: {platform.python_version()}"

async def test_streaming():
    """Test the streaming functionality with tool calls."""
    agent = Agent(
        "You are a helpful math assistant. When asked to calculate something, use the available tools and explain your process.",
        model="gpt-4o",
        tools=[calculate_fibonacci, get_system_info]
    )
    
    print("🤖 Testing Streaming Agent with Tool Calls")
    print("=" * 50)
    print()
    
    # Test case 1: Simple question that should trigger tool calls
    print("Test 1: Asking for Fibonacci calculation")
    print("User: Calculate the 10th Fibonacci number and tell me about the system")
    print()
    
    full_response = ""
    async for chunk in agent.run_stream("Calculate the 10th Fibonacci number and tell me about the system"):
        print(chunk, end="", flush=True)
        full_response += chunk
    
    print("\n")
    print("=" * 50)
    print(f"Complete response: {full_response}")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_streaming())
