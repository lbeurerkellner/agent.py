from litellm import acompletion
from typing import Callable, List
import inspect
import rich
import os
import asyncio
import sys
import json
import types

from collections import OrderedDict

def to_tool(callable: Callable) -> dict:
    """Convert a Python callable to a tool definition."""
    signature = inspect.signature(callable)
    parameters = {
        "type": "object",
        "properties": {
            param.name: {
                "type": "string" if param.annotation is str else "number",
                "description": param.annotation.__doc__ if hasattr(param.annotation, '__doc__') else ""
            } for param in signature.parameters.values()
        },
        "required": [param.name for param in signature.parameters.values() if param.default is inspect.Parameter.empty]
    }
    return {
        "type": "function",
        "function": {
            "name": callable.__name__,
            "description": callable.__doc__ or "",
            "parameters": parameters
        },
        "callable": callable
    }

class TerminalLogger:
    def __init__(self):
        self.in_content = False

    def reasoning(self, message: str):
        """Logs reasoning messages to the terminal."""
        rich.print(f"[bold blue]Reasoning:[/bold blue] {message}")
    
    def tool_call(self, tool_name: str, args: str):
        """Logs tool call messages to the terminal."""
        if self.in_content:
            # if we are in content, we need to flush the current line
            print("\n")
        
        rich.print("> " + tool_name + "(" + args + ")")
        
        self.in_content = False
    
    def tool_response(self, tool_name: str, response: str):
        """Logs tool response messages to the terminal."""
        response = response if response else "[No content returned from tool]"
        # render at most 4 lines of the response
        response_lines = response.splitlines()
        if len(response_lines) > 4:
            response = "\n".join(response_lines[:4]) + "\n(truncated " + str(len(response_lines) - 4) + " more lines)"
        else:
            response = "\n".join(response_lines)
        # don't render tags
        rich.print(" \[" + tool_name + "] ", end="")
        rich.print(response, end="\n\n", flush=True)
        
        self.in_content = False
    
    def stream_start(self):
        """Called when streaming starts."""
        rich.print("● ", end="")
        self.in_content = True
    
    def stream_content(self, content: str):
        """Logs streaming content to the terminal."""
        print(content, end="", flush=True)
        self.in_content = True
    
    def stream_end(self):
        """Called when streaming ends."""
        print()  # New line after streaming completes

class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o", tools: list[Callable] = None, model_config: dict = None, invariant_project_name: str = None):
        self.instructions = instructions
        self.model = model
        self.model_config = model_config if model_config else {}

        # check whether to use Invariant Gateway
        invariant_project = invariant_project_name or os.getenv('INVARIANT_PROJECT')
        if invariant_project:
            self.invariant_project = invariant_project
            endpoint_type = self.model_config.get('endpoint_type', 'openai')
            self.model_config = self.model_config or {}
            self.model_config['base_url'] = f"https://explorer.invariantlabs.ai/api/v1/gateway/{self.invariant_project}/" + endpoint_type
            self.model_config['headers'] = self.model_config.get('headers', {})
            if not "INVARIANT_API_KEY" in os.environ:
                raise ValueError("INVARIANT_API_KEY environment variable is not set. Please set it to use Invariant.")
            self.model_config['extra_headers'] = {"Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"}

        if tools:
            assert all(callable(tool) for tool in tools), "All tools must be callable functions."
            tools = [to_tool(tool) for tool in tools]
            self.tools = OrderedDict((tool['function']['name'], tool) for tool in tools)
        else:
            self.tools = OrderedDict()

    async def run(self, input: str, logger = TerminalLogger(), history_path: str | None = None):
        """Runs the agent with streaming output. Yields content chunks as they arrive."""
        instance = AgentInstance(self, logger, history_path)
        async for chunk in instance.run(input):
            yield chunk
    
    async def cli(self, persistent: bool = True):
        user_input = sys.argv[1] if len(sys.argv) > 1 else input("> ")
        console = rich.get_console()
        while True:
            try:
                history_path = os.path.join(os.path.expanduser("~"), ".agent-py") if persistent else None
                
                # Use streaming mode
                logger = TerminalLogger()
                logger.stream_start()
                
                full_response = ""
                async for chunk in self.run(user_input, logger=logger, history_path=history_path):
                    logger.stream_content(chunk)
                    full_response += chunk
                
                logger.stream_end()
                
                user_input = input("> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if user_input == "clear":
                    console.clear()
                    user_input = input("> ")
            except KeyboardInterrupt:
                break

def load_history(history_path: str) -> List[dict]:
    """Loads the agent history from a file."""
    if history_path is None:
        return []
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

class AgentInstance:
    def __init__(self, agent: Agent, logger: TerminalLogger = TerminalLogger(), history_path: str | None = None):
        self.agent = agent
        self.logger = logger
        
        self.history = [] if history_path is None else load_history(history_path)
        self.history_path = history_path

    def tools(self):
        """Returns the tool signature, without the callable."""
        tool_signatures = []
        for name, tool in self.agent.tools.items():
            tool_signature = { **tool }
            del tool_signature['callable']  # Remove the callable from the signature
            tool_signatures.append(tool_signature)
        return tool_signatures

    async def append(self, msg: dict):
        """Appends a message to the agent history."""
        self.history.append(msg)
        # Optionally, save the history to a file
        if self.history_path:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

    async def run_startup_tools(self):
        startup_tools = [t for t in self.agent.tools.values() if t['callable'] in context.__all__]
        startup_tool_results = await asyncio.gather(
            *(tool['callable']() for tool in startup_tools)
        )
        # append startup tool results to history
        for tool, result in zip(startup_tools, startup_tool_results):
            await self.append({
                "role": "user",
                "content": "Output of context value '" + tool['function']['name'] + "' if relevant in the user query below.\n\n" + str(result)
            })
    
    async def run(self, input: str):
        """Runs this instance of the agent with streaming output."""
        # check for startup tools, and include 'user' type messages for them
        await self.run_startup_tools()
        
        # Add system message with instructions if not already present
        if not self.history or self.history[0].get("role") != "system":
            await self.append({
                "role": "system",
                "content": self.agent.instructions
            })
        
        await self.append({
            "role": "user",
            "content": input
        })

        # core tool calling loop
        while True:
            # get streaming response from model
            response = await acompletion(
                model=self.agent.model,
                messages=self.history,
                tools=self.tools(),
                stream=True,
                **self.agent.model_config
            )
            
            # collect content and tool calls from stream
            full_content = ""
            tool_calls = []
            
            async for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    full_content += content_chunk
                    yield content_chunk
                
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    # Handle tool calls (they come in chunks too)
                    for tc in chunk.choices[0].delta.tool_calls:
                        # Extend or create tool call entries
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({
                                'id': '',
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            })
                        
                        if tc.id:
                            tool_calls[tc.index]['id'] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[tc.index]['function']['name'] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[tc.index]['function']['arguments'] += tc.function.arguments
            
            # append to history
            await self.append({
                "role": "assistant",
                "content": full_content,
                "tool_calls": tool_calls if tool_calls else None
            })
            
            # handle tool calls (appends tool responses to history)
            tool_responses = await asyncio.gather(
                *(self.handle_and_append_tool_call(tool_call) for tool_call in tool_calls)
            )

            if tool_responses:
                # re-run model with inserted tool responses
                continue
            else:
                # if we don't have any tool calls left, return
                return
    
    async def handle_and_append_tool_call(self, tool_call: dict):
        """Handles a tool call and appends the response to the history."""
        response = await self.handle_tool_call(tool_call)
        
        # append tool response to history
        await self.append({
            "role": "tool",
            "tool_call_id": tool_call['id'],
            "content": response['content']
        })
        
        return response

    async def handle_tool_call(self, tool_call: dict):
        """Internal method that handles tool calls with optional logging."""
        # resolve tool
        tool_name = tool_call['function']['name']

        # log the tool call
        self.logger.tool_call(tool_name, tool_call['function']['arguments'])

        async def make_response():
            """Creates the response/result for the tool call, i.e. tries to call the tool."""
            if tool_name not in self.agent.tools:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": f"Tool '{tool_name}' not found."
                }
            
            # get callable
            tool_callable = self.agent.tools[tool_name]['callable']
            
            # get and parse arguments
            args = tool_call['function']['arguments']
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": f"Error: Failed to parse arguments for tool '{tool_name}', try again and ensure it is valid JSON."
                }
            
            # call the tool and append tool response
            try:
                result = await tool_callable(**args) if inspect.iscoroutinefunction(tool_callable) else tool_callable(**args)
                
                return {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": str(result) if result is not None else "[No content returned from tool.]"
                }
            except Exception as e:
                return {
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": f"Error: Failed to call tool '{tool_name}': {str(e)}"
                }
        
        # make the tool response
        response = await make_response()
        
        # log the tool response
        self.logger.tool_response(tool_name, response['content'])

        return response

def context(func):
    """Decorator to mark a function as a startup tool, to be included in the agent's context on startup."""
    import inspect
    # ensure the function can be called without arguments
    if not inspect.isfunction(func) or len(inspect.signature(func).parameters) != 0:
        raise ValueError("context functions must be callable without arguments.")
    # ensure the function is async
    if not inspect.iscoroutinefunction(func):
        raise ValueError("context functions must be async functions.")

    context.__all__.append(func)
    return func
context.__all__ = []

def tool(func):
    """Decorator to mark a function as a tool, to be included in the agent's tools."""
    tool.__all__.append(func)
    return func
tool.__all__ = []

def auto():
    # Get the frame of the caller
    frame = inspect.currentframe()
    caller_frame = frame.f_back

    # Get the module where the caller is located
    module = inspect.getmodule(caller_frame)
    if module is None:
        return []

    # Traverse module attributes and collect locally defined functions
    functions = []
    for name, obj in vars(module).items():
        if isinstance(obj, types.FunctionType) and obj.__module__ == module.__name__:
            if obj in tool.__all__ or obj in context.__all__:
                # If the function is decorated with @tool or @context, include it
                functions.append(obj)

    return functions