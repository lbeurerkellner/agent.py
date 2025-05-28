from typing import Callable, List
import inspect
import rich
import tempfile
import os
import asyncio
import argparse
import sys
from functools import partial
import json
import types
import textwrap
import aioconsole

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from collections import OrderedDict

def callable_to_tool(callable: Callable) -> dict:
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

def mcp_to_tool(mcp: 'MCP') -> dict:
    """Convert an MCP instance to a tool definition."""
    return {
        "type": "mcp-server",
        "function": {
            "name": mcp.name,
        },
        "mcp": mcp
    }

def to_tool(tool: Callable | 'MCP') -> dict:
    """Convert a callable or MCP instance to a tool definition."""
    if isinstance(tool, MCP):
        return mcp_to_tool(tool)
    elif callable(tool):
        return callable_to_tool(tool)
    else:
        raise ValueError("Tool must be a callable or an instance of MCP.")

class MCP:
    def __init__(self, name: str, command: str, args: list[str], env: dict | None = None, timeout: int = 4, transport: str = "stdio"):
        if transport != "stdio":
            raise ValueError("Only 'stdio' transport is currently supported in this example.")
        # replace everything except [A-z0-9_] with underscores
        self.name = "".join(c if c.isalnum() else '' for c in name)
        self.server_params = StdioServerParameters(command=command, args=args, env=env)
        self.transport = transport
        self.timeout = timeout

        self._client_context = None
        self._client = None
        self.session = None
        self.tools = None  # will be initialized after the client context is created

        self.ready = asyncio.Event()
        self._terminate_event = asyncio.Event()
        self._background_task = None

    async def call_tool(self, tool_name: str, args: str):
        """Calls a tool on the MCP server."""
        if not self.session:
            raise RuntimeError("MCP server is not initialized. Call 'initialize' first.")
        try:
            response = await self.session.call_tool(tool_name, args)
            return "\n".join([content.model_dump_json() for content in response.content])
        except Exception as e:
            print(f"Error calling tool '{tool_name}' on MCP server '{self.name}': {str(e)}", flush=True)
            return None

    async def initialize(self):
        """Starts the background task to manage the client/session lifecycle."""
        if self._background_task is None or self._background_task.done():
            self._terminate_event.clear()
            self._background_task = asyncio.create_task(self._background_lifecycle())
        await self.ready.wait()

    async def terminate(self):
        """Signals the background task to close the client/session."""
        self._terminate_event.set()
        if self._background_task:
            await self._background_task

    async def _background_lifecycle(self):
        with tempfile.NamedTemporaryFile(delete=True) as error_log:
            try:
                self._client_context = stdio_client(self.server_params, errlog=error_log)
                self._client = self._client_context.__aenter__()
                read, write = await self._client
                self.session = ClientSession(read, write, sampling_callback=None)
                await self.session.__aenter__()

                # initialize the session
                await asyncio.wait_for(self.session.initialize(), timeout=self.timeout)
                # query tools
                tools = await self.session.list_tools()

                # convert to model format            
                self.tools = OrderedDict()
                for tool in tools.tools:
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": "mcp_" + self.name + "_" + tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        },
                        "mcp": self
                    }
                    self.tools[tool_def['function']['name']] = tool_def

                self.ready.set()

                # Wait for terminate signal
                await self._terminate_event.wait()

            except Exception as e:
                with open(error_log.name, 'r') as f:
                    error_content = f.read()
                print(f"Failed to initialize MCP server '{self.name}': {str(e)}\nServer Output:\n{error_content}", flush=True)
                import traceback
                traceback.print_exc()
                self.ready.set()
            finally:
                await self._cleanup()

    async def _cleanup(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)
            self._client_context = None
        self.tools = None
        self.ready.clear()

    async def close(self):
        """Alias for terminate, for compatibility."""
        await self.terminate()


class TerminalLogger:
    def __init__(self):
        self.in_content = False
        self.num_content_chunks = 0

    async def aprint(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, partial(rich.print, *args, **kwargs))

    async def tool_call(self, tool_name: str, args: str):
        """Logs tool call messages to the terminal."""
        if self.in_content:
            self.in_content = False
            if self.num_content_chunks == 0:
                await self.aprint("[no reasoning output]", end="", flush=True)
            # if we are in content, we need to flush the current line
            await self.aprint("\n")
        
        await self.aprint("> " + tool_name + "(" + args + ")")
        
        self.num_content_chunks = 0
    
    async def tool_response(self, tool_name: str, response: str):
        """Logs tool response messages to the terminal."""
        response = response if response else "[No content returned from tool]"
        # render at most 4 lines of the response
        response_lines = response.splitlines()
        if len(response_lines) > 4:
            response = "\n".join(response_lines[:4]) + "\n\n(truncated " + str(len(response_lines) - 4) + " more lines)"
        else:
            response = "\n".join(response_lines)
        # make sure total response is still not longer than 512
        if len(response) > 512:
            response = response[:512] + "\n\n(truncated to 512 characters)"
        # don't render tags
        response = textwrap.indent(response, " " * 2)
        # render it
        rendered = rich.text.Text.from_markup("[green]" + response + "[green]")
        for i in range(0, len(rendered), 128):
            chunk = rendered[i:i+128]
            await self.aprint(chunk, end="", flush=True)
        
        await self.aprint("\n")

        self.in_content = False
        self.num_content_chunks = 0
    
    async def stream_start(self):
        """Called when streaming starts."""
        await self.aprint("● ", end="")
        self.in_content = True
    
    async def stream_content(self, content: str):
        """Logs streaming content to the terminal."""
        await self.aprint(content, end="", flush=True)
        self.in_content = True
        self.num_content_chunks += 1
    
    async def stream_end(self):
        """Called when streaming ends."""
        await self.aprint()  # New line after streaming completes

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
            if "INVARIANT_API_KEY" not in os.environ:
                raise ValueError("INVARIANT_API_KEY environment variable is not set. Please set it to use Invariant.")
            self.model_config['extra_headers'] = {"Invariant-Authorization": f"Bearer {os.getenv('INVARIANT_API_KEY')}"}

        # create map of all tools
        if tools:
            assert all(callable(tool) or isinstance(tool, MCP) for tool in tools), "All tools must be callable or instances of MCP."
            tools = [to_tool(tool) for tool in tools]
            self.tools = OrderedDict((tool['function']['name'], tool) for tool in tools)
        else:
            self.tools = OrderedDict()

    async def run(self, input: str, logger = TerminalLogger(), history_path: str | None = None):
        """Runs the agent with streaming output. Yields content chunks as they arrive."""
        async with AgentInstance(self, logger, history_path) as instance:
            async for chunk in instance.run(input):
                yield chunk
    
    @staticmethod
    def tools_from_uris(args: List[str], result: OrderedDict | None = None) -> List[dict]:
        if result is None: 
            result = OrderedDict()
        
        for tool_uri in args:
            if isinstance(tool_uri, str):
                if tool_uri.startswith("npx:"): # assume it's an npx-based MCP server
                    name = tool_uri[4:]
                    result[name] = MCP(name, "npx", [name])
                elif tool_uri.endswith(".json"):
                    with open(tool_uri, 'r') as f:
                        tool_data = json.load(f)
                        servers = tool_data.get("mcpServers", {})
                        assert isinstance(servers, dict), f"mcpServers in MCP configuration file '{tool_uri}' must be a dictionary."
                        Agent.tools_from_uris([{"name": name, **kwargs} for (name, kwargs) in servers.items()], result)
                else: # assume it's a local, uvx-based MCP server
                    name = os.path.basename(tool_uri)
                    result[name] = MCP(name, "uvx", [tool_uri])
            elif isinstance(tool_uri, dict):
                name = tool_uri.get("name")
                result[name] = MCP(name, tool_uri.get("command", "uvx"), tool_uri.get("args", []), env=tool_uri.get("env"), timeout=tool_uri.get("timeout", 4), transport=tool_uri.get("type", "stdio"))
            else:
                assert False, f"Invalid tool URI: {tool_uri}. Must be a string or a dictionary."
        
        return [to_tool(tool) for name, tool in result.items()]

    async def cli(self, persistent: bool = True):
        # print examples usage
        parser = argparse.ArgumentParser(description="A CLI for tool-calling LLM agents.", formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
                agentpy --with arxiv-mcp-server
                agentpy --with npx:@playwright/mcp@latest
                agentpy --with ~/.cursor/mcp.json
            """))
        parser.add_argument("initial_input", nargs='?', default=None, help="Initial user input to start the agent with.")
        parser.add_argument("--with", action="append", default=[], help="Add a tool to the agent (e.g. --with your-mcp-server, --with npx:your-mcp-server --with mcp.json)", type=str, dest="with_")
        parser.add_argument("--persistent", default=persistent, help="Enable persistent mode, which saves the agent history to a file in the user's home directory.", type=bool)
        args = parser.parse_args()

        # load history if persistent mode is enabled
        history_path = os.path.join(os.path.expanduser("~"), ".agent-py") if args.persistent else None

        # use terminal logger by default
        logger = TerminalLogger()
        console = rich.get_console()

        # parse tools
        for tool in Agent.tools_from_uris(args.with_):
            self.tools[tool['function']['name']] = tool
        
        # create the agent instance
        async with AgentInstance(self, logger=logger, history_path=history_path) as instance:
            initial_user_input = args.initial_input if args.initial_input else None
            next_input_extra_context = ""
            # user interaction loop
            while True:
                try:        
                    # get user input
                    user_input = next_input_extra_context + (initial_user_input or await aioconsole.ainput("> "))
                    initial_user_input = None  # reset after first input
                    next_input_extra_context = ""  # reset extra context

                    # check for special commands
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    elif user_input == "clear":
                        console.clear()
                        continue
                    elif user_input == "help":
                        helptext = parser.format_help()
                        console.print(f"[bold blue]Agent CLI Help:[/bold blue]\n{helptext}")
                        next_input_extra_context = helptext + "\nThis was shown to the user. Now they are asking:\n\n"
                        continue
                    elif user_input == "tools":
                        # list available tools
                        tool_signatures = await instance.tools()
                        if not tool_signatures:
                            console.print("[bold red]No tools available. Run with --with, to enable tools.[/bold red]")
                        else:
                            console.print("[bold green]Available Tools:[/bold green]")
                            for tool in tool_signatures:
                                description = tool['function']['description'] or "No description available."
                                # replace NLs 
                                description = description.replace("\n", " ")
                                # truncate long descriptions
                                if len(description) > 100:
                                    description = description[:100] + "..."
                                console.print(f"- [bold]{tool['function']['name']}[/bold]: {description}")
                        continue

                    await logger.stream_start()

                    # run and stream agent response
                    full_response = ""
                    async for chunk in instance.run(user_input):
                        await logger.stream_content(chunk)
                        full_response += chunk
                    
                    # end the stream
                    await logger.stream_end()
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

        self.ready = asyncio.Event()

    async def tools(self):
        """Returns the tool signature, without the callable."""
        tool_signatures = []
        for name, tool in self.agent.tools.items():
            if tool['type'] == 'function':
                tool_signature = { **tool }
                del tool_signature['callable']  # Remove the callable from the signature
                tool_signatures.append(tool_signature)
            elif tool['type'] == 'mcp-server':
                await tool['mcp'].ready.wait()  # ensure MCP connection is ready
                
                for tool in tool['mcp'].tools.values():
                    tool_signature = { **tool }
                    del tool_signature['mcp']
                    tool_signatures.append(tool_signature)

        return tool_signatures

    async def __aenter__(self):
        # if already initialized, do nothing
        if self.ready.is_set():
            return
        
        # initialize MCP servers, if any
        async def init_mcp(tool):
            await asyncio.gather(
                *(tool['mcp'].initialize() for tool in self.agent.tools.values() if tool['type'] == 'mcp-server')
            )
            self.ready.set()
        
        asyncio.create_task(init_mcp(self.agent))

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # close MCP servers, if any
        await asyncio.gather(
            *(tool['mcp'].close() for tool in self.agent.tools.values() if tool['type'] == 'mcp-server')
        )
        self.ready.clear()

    async def append(self, msg: dict):
        """Appends a message to the agent history."""
        self.history.append(msg)
        # Optionally, save the history to a file
        if self.history_path:
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

    async def run_startup_tools(self):
        startup_tools = [t for t in self.agent.tools.values() if t['type'] == 'function' and t['callable'] in context.__all__]
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
        from litellm import acompletion

        # check for startup tools, and include 'user' type messages for them
        await self.run_startup_tools()

        # ensure we are ready
        await self.ready.wait()
        
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
                tools=await self.tools(),
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
        await self.logger.tool_call(tool_name, tool_call['function']['arguments'])

        async def make_response():
            """Creates the response/result for the tool call, i.e. tries to call the tool."""
            nonlocal tool_name

            # check for MCP match
            if tool_name.startswith("mcp_"):
                server_name = tool_name.split("_")[1]
                server = self.agent.tools.get(server_name)
                # check if the server has this tool
                if server is None or server['type'] != 'mcp-server' or tool_name not in server['mcp'].tools:
                    return {
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": f"Tool '{tool_name}' not found in MCP server '{server_name}'."
                    }
                # calling it goes via the MCP server
                async def tool_callable(**kwargs):
                    """Calls the MCP tool."""
                    nonlocal tool_name
                    # call the MCP tool
                    tool_name = tool_name[len("mcp_") + len(server_name) + 1:]
                    return await server['mcp'].call_tool(tool_name, args)
            else:
                # check for regular tool match
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
        await self.logger.tool_response(tool_name, response['content'])

        return response

def context(func):
    """Decorator to mark a function as a context component,     to be included in the agent's context on startup."""
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

def main():
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use the agent.")
    agent = Agent(
        instructions="You are a helpful assistant.",
        model="gpt-4o"
    )

    asyncio.run(agent.cli(persistent=False))