# sci-facts

A fact-checking and literature review system using LLM pipelines with MCP (Model Context Protocol) integration.

## Features

- **Synthesis Pipeline**: Direct LLM processing for paper synthesis with configurable goals and tones
- **Agent Chat Pipeline**: Multi-step AI agent for literature review with automatic paper discovery
- **MCP Integration**: Connects to Scholar Inbox MCP server for paper search and management
- **vLLM Support**: Uses local vLLM server for efficient LLM inference

## Setup

1. Create and activate a conda environment:
   ```bash
   conda create -n sci-facts python=3.11
   conda activate sci-facts
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional, copy from .env.example):
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Quick Start

### Starting the Pipeline

The easiest way to start is using the startup script which launches both the vLLM server and task listener:

```bash
./start_pipeline.sh
```

This will:
1. Start the vLLM server with the configured model
2. Wait for the server to be ready
3. Start the task listener to process incoming tasks

### Manual Startup

If you prefer to start components separately:

1. **Start vLLM Server**:
   ```bash
   python -m llm_utils.start_server --model Qwen/Qwen3-VL-4B-Instruct --port 8000
   ```

2. **Start Task Listener** (in another terminal):
   ```bash
   python -m main_pipeline.task_listener --poll-interval 2.0
   ```

## Architecture

### Pipelines

#### Synthesis Pipeline (`main_pipeline/synthesis_pipeline.py`)

Processes synthesis tasks by:
1. Fetching paper details from MCP server
2. Building context from paper abstracts
3. Generating synthesis using appropriate goal and tone prompts

**Supported Goals:**
- `summary`: Comprehensive overview of findings
- `fact_extraction`: Extract key facts and claims
- `topic_extraction`: Identify themes and topics
- `keywords_extraction`: Extract relevant keywords

**Supported Tones:**
- `precise`: Academic, thorough language
- `fun_facts`: Engaging, accessible presentation
- `eli5`: Simple explanations for non-experts
- `brutal`: Direct, critical analysis

#### Agent Chat Pipeline (`main_pipeline/agent_pipeline.py`)

A multi-step literature review agent using LangGraph:

1. **Keyword Extraction**: Extract search keywords from user query
2. **Keyword Search**: Execute search and select relevant papers
3. **Query Augmentation**: Expand and refine the search query
4. **Semantic Search**: Find semantically similar papers
5. **Advanced Search**: Generate custom search query based on findings

The agent sends intermediate results to the MCP server during processing and collects papers into a session.

### MCP Client (`code/mcp_client/`)

Provides async methods for communicating with the Scholar Inbox MCP server:

- Tool calls (paper search, topic management, collections)
- Resource reading (papers, collections, topics)
- Prompt retrieval (synthesis, agent prompts)
- Task stream (SSE-based task queue listening)

### Task Listener (`main_pipeline/task_listener.py`)

Connects to the MCP server task queue and routes tasks to appropriate pipelines:

- `synthesis` tasks → Synthesis Pipeline
- `agent_chat` tasks → Agent Chat Pipeline

## Configuration

Environment variables (can be set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_URL` | `http://localhost:8001/mcp` | MCP server endpoint |
| `MCP_API_KEY` | `test-api-key` | MCP server API key |
| `VLLM_HOST` | `127.0.0.1` | vLLM server host |
| `VLLM_PORT` | `8000` | vLLM server port |
| `VLLM_API_KEY` | `token-abc123` | vLLM server API key |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | Model to serve |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_MAX_TOKENS` | `256` | Max response tokens |

## Project Structure

```
sci-facts/
├── code/
│   ├── llm_utils/           # vLLM server utilities
│   │   ├── start_server.py  # vLLM startup script
│   │   └── vLLM_server.py   # vLLM management
│   ├── main_pipeline/       # LLM processing pipelines
│   │   ├── synthesis_pipeline.py   # Synthesis processing
│   │   ├── agent_pipeline.py       # Agent chat pipeline
│   │   └── task_listener.py        # Task queue listener
│   └── mcp_client/          # MCP client library
│       ├── client.py        # Async MCP client
│       ├── config.py        # Configuration
│       └── tests/           # Test suite
├── logs/                    # Log files
├── start_pipeline.sh        # Combined startup script
├── requirements.txt         # Python dependencies
└── README.md
```

## Testing

Run the test suite:

```bash
pytest code/mcp_client/tests/ -v
```

Note: Tests require a running MCP server.

## Development

### Adding New Task Types

1. Add a task handler method in `TaskListener`
2. Update the `handle_task` method to route the new type
3. Create any required prompts in the MCP server's `prompts.py`

### Adding New Pipeline Steps

For the agent pipeline, add nodes to the LangGraph state machine in `agent_pipeline.py` and update the edge definitions.

## License

See [LICENSE](LICENSE)