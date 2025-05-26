# `agent.py`: minimal agent tool use scaffold

`agent.py` is a one-file agent library designed to quickly build and experiment with agents, while keeping code complexity to a minimum.

**Features:**

- **CLI first**: Call and operate agents from the CLI for quick turn-around.
- **Simple Implementation**: Implemented in <600 LOC in `src/agent.py`
- **MCP Support**: Quickly add new tools by integration MCP servers.

**Quickstart:**

To use `agent.py`, ensure you have [`uv`](https://docs.astral.sh/uv/) installed, then you can create and run your first agent like this:

```bash
uvx --from "git+https://github.com/lbeurerkellner/agent.py@main" \
    agentpy "agent with a calculator tool"
```

*Requires an `OPENAI_API_KEY`

This command will generate a new agent with calculator tools, put it in an appropriately-named file in your current directory and allows you to interact with it.

## Interacting with the agent

An interaction trace with the agent will look like this:

```bash
[agent created at calculator-agent.py]

"Welcome! I'm your calculator agent. I can perform basic arithmetic operations like addition, subtraction, multiplication, and division.(type 'tools' for a list of tools, 'exit' to termiante the agent)"

# use 'tools' to see the agent's generated toolset
> tools 

Available Tools:
- add: Add two numbers.
- subtract: Subtract two numbers.
- multiply: Multiply two numbers.
- divide: Divide two numbers.
- welcome_message: Welcome message when the agent starts.

# interact by asking questions
> What is 1234123*341?
● [no reasoning output]

# tool calls are shown
> multiply({"a":1234123,"b":341})
  420835943

The result of \(1,234,123 \times 341\) is 420,835,943.
```

You can also inspect the generated file (here `calculator-agent.py`) to review, update or extend the agent's implementation.

## Examples

Explore different capabilities and usages through example scripts available in the `examples` directory:

- **CLI Agent**: Demonstrates command-line interface integration.
- **Paper Agent**: Shows how to handle document-related tasks by integrating an arxiv MCP server.
- **Email Demo**: Illustrates basic email processing features with dummy data.
