import os
import asyncio
from agentpy.agent import Agent, context, auto, tool
import tempfile
import sys

def main():
    @tool
    async def spawn_agent(name: str, snippet: str, introduction: str):
        """Creates and runs a new agent with the given name (lowercase, dash-case) and snippet. Set an 'introduction' such that it is a nice greeting message for the user, to see on first startup of the agent. Example snippet: import asyncio\nfrom agentpy.agent import Agent, context, auto, tool # important: don't forget auto()\n\n@tool\ndef read_file(file_path: str) -> str:\n    \"\"\"Reads the contents of a file.\"\"\"\n    # implementation omitted\n\n# tools can be used by the agent to perform actions\n@tool\ndef list_directory(path: str = ".") -> str:\n    \"\"\"Lists the contents of a directory.\"\"\"\n    # implementation omitted\n\n# context functions are automatically called when the agent runs (must not have arguments)\n@context\nasync def pwd():\n    \"\"\"Get the current working directory.\"\"\"\n    # implementation omitted (don't include side-effects like printing here, the function may be called a couple of times)\n\nasync def amain():\n    agent = Agent(\n        "You are a helpful assistant with access to file system tools. You can read files, write files, and list directories. When using tools, explain what you're doing step by step.",\n        model="gpt-4o",\n        tools=auto() # auto detects and registers all @tool and @context functions\n    )\n    await agent.cli(persistent=False)\nif __name__ == "__main__":\n    try:\n        asyncio.run(amain())\n    except KeyboardInterrupt:\n        print("\nExiting...")"""
        with open(name + ".py", "w") as f:
            f.write(snippet)

        print("[agent created at", name + ".py]")
        print("\n", introduction)

        await run(name + ".py")

        sys.exit(0)
            
    async def run(file_path):
            # run the agent with this process's Python interpreter
            process = await asyncio.create_subprocess_exec(
                sys.executable, file_path,
                bufsize=0
            )

            stdout, stderr = await process.communicate()
            
            # return output to caller
            if stdout or stderr:
                return f"Agent ran successfully. Output:\n\n{stdout.decode('utf-8') if stdout else ''}{stderr.decode('utf-8') if stderr else ''}"
            else:
                return "Agent ran successfully, but no output was produced."


    agent = Agent(
        instructions="You are an agent-creator agent. Users ask for agent capabilities and create and spawn_agent them directly. Don't show the code to the user, just return Creating Agent... before you generate code.",
        model="gpt-4o",
        tools=[spawn_agent]
    )

    asyncio.run(agent.cli(persistent=False))