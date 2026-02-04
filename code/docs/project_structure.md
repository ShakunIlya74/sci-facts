# Project Structure

## Overview

The MCP Client project is organized to support literature review automation with LLMs. It connects to the Scholar Inbox MCP Server for data access and uses a vLLM server for inference.

## Directory Layout

```
sci-facts/
├── code/
│   ├── docs/                      # Documentation
│   │   ├── README.md
│   │   ├── project_structure.md   # This file
│   │   ├── api_reference.md       # Client API docs
│   │   ├── testing_guide.md       # Testing documentation
│   │   └── configuration.md       # Config options
│   │
│   ├── llm_utils/                 # vLLM server utilities
│   │   ├── __init__.py
│   │   ├── vLLM_server.py         # Server management
│   │   └── test_cuda.py           # CUDA verification
│   │
│   ├── mcp_client/                # Main MCP client package
│   │   ├── __init__.py            # Package exports
│   │   ├── client.py              # MCPClient class
│   │   ├── config.py              # Configuration settings
│   │   ├── logging_utils.py       # Logging helpers
│   │   └── tests/                 # Test suite
│   │       ├── __init__.py
│   │       ├── conftest.py        # Pytest fixtures
│   │       ├── test_discovery.py  # Discovery endpoint tests
│   │       ├── test_tools.py      # Tool endpoint tests
│   │       ├── test_resources.py  # Resource endpoint tests
│   │       └── test_prompts.py    # Prompt endpoint tests
│   │
│   └── main_pipeline/            # Main workflow scripts
│       └── sandbox.py
│
├── logs/                         # Test and runtime logs
│   └── tests/                    # Test session logs
│
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

## Module Descriptions

### `mcp_client/`

The core MCP client package providing:

- **`client.py`**: `MCPClient` class for communicating with Scholar Inbox MCP Server
  - Async HTTP client using `httpx`
  - JSON-RPC 2.0 protocol implementation
  - Methods for all 14 tools, 3 resources, and 5 prompts

- **`config.py`**: Configuration management
  - `Settings` dataclass with sensible defaults
  - Environment variable support
  - Cached settings singleton

- **`logging_utils.py`**: Test logging utilities
  - Response trimming (max 1000 chars by default)
  - Structured test result logging
  - Session-based log files

### `mcp_client/tests/`

Comprehensive pytest test suite:

- **`test_discovery.py`**: Tests for discovery endpoints
  - `initialize` - Server handshake
  - `tools/list` - Available tools
  - `resources/list` - Available resources
  - `prompts/list` - Available prompts

- **`test_tools.py`**: Tests for all 14 tools
  - Paper search tools (4)
  - Topic tools (4)  
  - Collection tools (6)

- **`test_resources.py`**: Tests for resource URIs
  - `paper://{id}`
  - `collection://{id}`
  - `topic://{name}`

- **`test_prompts.py`**: Tests for prompt templates
  - `litreview_plan`
  - `litreview_screening_criteria`
  - `litreview_search_clarification`
  - `synthesis_summary`
  - `synthesis_extract_claims`

### `llm_utils/`

vLLM server management utilities:

- **`vLLM_server.py`**: 
  - `start_vllm_server()` - Start vLLM subprocess
  - `wait_for_vllm()` - Wait for server readiness
  - `get_local_qwen_llm()` - Get LangChain client
  - `local_qwen_llm()` - Context manager for server lifecycle

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          MCP Client Application                         │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │   Tests         │    │   MCPClient     │    │   LLM Utils         │ │
│  │   (pytest)      │───▶│   (client.py)   │    │   (vLLM_server.py)  │ │
│  │                 │    │                 │    │                     │ │
│  └─────────────────┘    └────────┬────────┘    └──────────┬──────────┘ │
│                                  │                        │            │
│                                  ▼                        ▼            │
│                         ┌─────────────────┐     ┌─────────────────┐    │
│                         │  Scholar Inbox  │     │  vLLM Server    │    │
│                         │  MCP Server     │     │  (Inference)    │    │
│                         │  /mcp           │     │  /v1/chat       │    │
│                         └─────────────────┘     └─────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Async-first**: All MCP communication is async for better concurrency
2. **Typed configuration**: Dataclass-based settings with env var support
3. **Trimmed logging**: Responses capped at 1000 chars to avoid log bloat
4. **Graceful degradation**: Tests skip cleanly when resources don't exist
5. **Context manager support**: `MCPClient` supports `async with` for cleanup
