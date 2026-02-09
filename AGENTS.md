# Agents Notes

## Architecture

### MCP Server (External)
- **Location**: `~/Projects/scholar_inbox/code/mcp_server`
- **Port**: 8001
- **Endpoints**:
  - `POST /mcp` — JSON-RPC 2.0 (tools, prompts, resources)
  - `GET /mcp/tasks/stream` — SSE task stream
- **Note**: The MCP server lives in a separate repository (`scholar_inbox`), not in this repo.
- **Routing**: The server uses `mcp.streamable_http_app()` (Starlette with `Route("/mcp")`) as its base, wrapped in a thin ASGI layer for custom routes. Do NOT use Starlette `Mount` — it strips path prefixes and doesn't propagate lifespan events.

### This Repo (sci-facts)
- **MCP Client**: `code/mcp_client/` — async client for communicating with the MCP server
- **Task Listener**: `code/main_pipeline/task_listener.py` — SSE-based listener that processes tasks (synthesis, agent_chat)
- **Pipelines**: `code/main_pipeline/` — synthesis and agent chat pipelines
- **vLLM**: Local inference server on port 8000
