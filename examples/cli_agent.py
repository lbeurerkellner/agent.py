import os
import asyncio
from agentpy.agent import Agent, context, auto, tool

@tool
def read_file(file_path: str) -> str:
    """Reads the contents of a file."""
    if not os.path.exists(file_path):
        return f"Error: '{file_path}' does not exist."
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a valid file."
    with open(file_path, 'r') as f:
        return f.read()

@tool
def list_directory(path: str = ".") -> str:
    """Lists the contents of a directory."""
    try:
        if not os.path.exists(path):
            return f"Error: '{path}' does not exist."
        if not os.path.isdir(path):
            return f"Error: '{path}' is not a directory."
        
        items = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                items.append(f"📁 {item}/")
            else:
                items.append(f"📄 {item}")
        return "\n".join(items) if items else "Directory is empty"
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"

@context
async def pwd():
    """Get the current working directory."""
    return {"name": os.getcwd(), "type": "directory"}

@context
async def cwd():
    """Get the contents of the current working directory."""
    return [{"name": f, "type": "file" if os.path.isfile(f) else "directory"} for f in os.listdir(os.getcwd())]

async def amain():
    agent = Agent(
        "You are a helpful assistant with access to file system tools. You can read files, write files, and list directories. When using tools, explain what you're doing step by step.",
        model="gpt-4o",
        tools=auto()
    )
    await agent.cli(persistent=True)

if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("\nExiting...")