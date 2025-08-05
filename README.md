# `agent.py`: minimal agent tool use scaffold

`agent.py` is a one-file agent library designed to quickly build and experiment with agents, while keeping code complexity to a minimum.

**Features:**

- **CLI first**: Call and operate agents from the CLI for quick turn-around.
- **Simple Implementation**: Implemented in <600 LOC in `src/agent.py`
- **MCP Support**: Quickly add new tools by integrating MCP servers (`--with arxiv-mcp-server`)

**Quickstart:**

To use `agent.py`, ensure you have [`uv`](https://docs.astral.sh/uv/) installed, then you can create and run your first agent like this:

```bash
uvx git+https://github.com/lbeurerkellner/agent.py@main
```

*Requires an `OPENAI_API_KEY`

This command will generate a new agent with calculator tools, put it in an appropriately-named file in your current directory and allows you to interact with it.

## Interacting with the agent

An interaction trace with the agent will look like this:

```markdown
> uvx git+https://github.com/lbeurerkellner/agent.py@main --with arxiv-mcp-server

> tools

Available Tools:

- mcp_arxivmcpserver_search_papers: Search for papers on arXiv with advanced filtering
- mcp_arxivmcpserver_download_paper: Download a paper and create a resource for it
- mcp_arxivmcpserver_list_papers: List all existing papers available as resources
- mcp_arxivmcpserver_read_paper: Read the full content of a stored paper in markdown format

> what can you do?

● I can assist you with various tasks related to searching, downloading, and reading academic papers from arXiv. Here are some specific things I can do:

1. **Search for Papers**: I can search for academic papers on arXiv based on specific queries, dates, and categories.
2. **Download Papers**: I can download papers from arXiv given their paper ID and provide them for you to read.
3. **List Available Papers**: I can list all papers that are available as resources, which have been downloaded previously.
4. **Read Papers**: I can display the full content of a stored paper in markdown format for you to review its contents.

If you have any specific task in mind, feel free to ask, and I'll do my best to assist you!

> research papers about MCP

● > mcp_arxivmcpserver_search_papers({"query":"MCP","max_results":5})
  {"type":"text","text":"{\n  \"total_results\": 5,\n  \"papers\": [\n    {\n      \"id\": \"2508.01805v1\",\n      \"title\": \"M3LLM: Model Context Protocol-aided Mixture of Vision Experts For Multimodal LLMs in Networks\",\n      \"authors\": [\n        \"Yongjie Zeng\",\n        \"Hongyang Du\"\n      ],\n      \"abstract\": \"Current Multimodal Large Language Models (MLLMs) rely on centralized\\narchitectures and often suffer from poor alignment between the input task and\\ntheir fixed visual encoding mo

  (truncated to 512 characters)

Here are some research papers related to MCP found on arXiv:

1. **Title**: [M3LLM: Model Context Protocol-aided Mixture of Vision Experts For Multimodal LLMs in Networks](http://arxiv.org/pdf/2508.01805v1)
   - **Authors**: Yongjie Zeng, Hongyang Du
   - **Abstract**: Proposes M3LLM, where the Model Context Protocol (MCP) coordinates a mixture of vision experts for distributed MLLMs, enhancing multimodal large language models (MLLMs). This improves task accuracy and reduces communication costs in dynamic wireless networks.
   - **Published**: August 3, 2025

2. **Title**: [LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools?](http://arxiv.org/pdf/2508.01780v1)
   - **Authors**: Guozhao Mo, Wenliang Zhong, Jiawei Chen, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun
   - **Abstract**: Introduces LiveMCPBench to evaluate LLM agents' capabilities at scale across diverse servers within the MCP ecosystem. Includes LiveMCPEval for automated evaluation and proposes the MCP Copilot Agent for dynamic tool routing.
   - **Published**: August 3, 2025
...

If you need more details or want to download any of these papers, let me know!
```

You can also inspect the generated file (here `calculator-agent.py`) to review, update or extend the agent's implementation.

## Use With Invariant

To use agent.py with Invariant tools like Guardrails and Explorer, you can just run it as follows:

```
export INVARIANT_API_KEY=... # get from explorer.invariantlabs.ai
INVARIANT_PROJECT=my-arxiv-agent uvx git+https://github.com/lbeurerkellner/agent.py@main --with arxiv-mcp-server
```

This will automatically trace all logs to a new Explorer project. If you configure guardrailing rules in the Explorer project, they will also be enforced.

## Examples

Explore different capabilities and usages through example scripts available in the `examples` directory:

- **CLI Agent**: Demonstrates command-line interface integration.
- **Paper Agent**: Shows how to handle document-related tasks by integrating an arxiv MCP server.
- **Email Demo**: Illustrates basic email processing features with dummy data.
