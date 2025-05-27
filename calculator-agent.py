import asyncio
from agentpy.agent import Agent, tool, auto

@tool
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtracts second number from the first number."""
    return a - b

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divides first number by the second number. Returns 'None' if division by zero is attempted."""
    return a / b if b != 0 else None

async def amain():
    agent = Agent(
        "I am a Calculator Agent. You can perform basic arithmetic operations like addition, subtraction, multiplication, and division.",
        tools=auto()  # This automatically detects and registers all @tool functions
    )
    await agent.cli(persistent=False)

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("Exiting...")