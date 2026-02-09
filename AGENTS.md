# Agents Notes

## Architecture

### MCP Server (External)
- **Location**: `~/Projects/scholar_inbox/code/mcp_server`
- **Port**: 8001
- **Endpoints**:
  - `POST /mcp` — JSON-RPC 2.0 (tools, prompts, resources)
  - `GET /mcp/tasks/stream` — SSE task stream
- **Note**: The MCP server lives in a separate repository (`scholar_inbox`), not in this repo.
- **Routing**: The server uses Starlette `Mount("/mcp", ...)` which strips the `/mcp` prefix before routing to the inner ASGI app. The FastMCP `streamable_http_path` must be set to `"/"` (not the default `"/mcp"`) to avoid double-nesting. The task stream wrapper must also use prefix-stripped paths (e.g., `"/tasks/stream"` not `"/mcp/tasks/stream"`).

### This Repo (sci-facts)
- **MCP Client**: `code/mcp_client/` — async client for communicating with the MCP server
- **Task Listener**: `code/main_pipeline/task_listener.py` — SSE-based listener that processes tasks (synthesis, agent_chat)
- **Pipelines**: `code/main_pipeline/` — synthesis and agent chat pipelines
- **vLLM**: Local inference server on port 8000
