"""
MCP Client for communicating with Scholar Inbox MCP Server.

Provides async methods for calling tools, reading resources, and getting prompts.
"""

import json
import logging
from typing import Any, Optional

import httpx

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP client errors."""
    
    def __init__(self, message: str, code: Optional[int] = None, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class MCPClient:
    """
    Async MCP client for Scholar Inbox server.
    
    Uses JSON-RPC 2.0 protocol over Streamable HTTP transport.
    Manages session state for stateful MCP connections.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize MCP client with settings."""
        self.settings = settings or get_settings()
        self._request_id = 0
        self._session: Optional[httpx.AsyncClient] = None
        self._mcp_session_id: Optional[str] = None
    
    @property
    def headers(self) -> dict[str, str]:
        """Get HTTP headers for MCP requests."""
        headers = {
            "Authorization": f"Bearer {self.settings.mcp_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._mcp_session_id:
            headers["mcp-session-id"] = self._mcp_session_id
        return headers
    
    def _next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                timeout=self.settings.http_timeout,
            )
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session and clear session ID."""
        if self._session is not None:
            await self._session.aclose()
            self._session = None
        self._mcp_session_id = None
    
    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def _request(self, method: str, params: Optional[dict] = None) -> dict:
        """
        Send JSON-RPC request to MCP server.
        
        Args:
            method: JSON-RPC method name
            params: Method parameters
        
        Returns:
            Result from server response
        
        Raises:
            MCPError: If server returns an error
        """
        session = await self._get_session()
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params is not None:
            payload["params"] = params
        
        logger.debug(f"MCP request: {method} - {params}")
        
        response = await session.post(
            self.settings.mcp_server_url, 
            json=payload,
            headers=self.headers,
        )
        response.raise_for_status()

        # Extract session ID from response headers (set during initialize)
        if "mcp-session-id" in response.headers:
            self._mcp_session_id = response.headers["mcp-session-id"]
            logger.debug(f"MCP session ID: {self._mcp_session_id}")

        # Parse SSE (Server-Sent Events) response
        # Streamable HTTP transport returns events in format:
        # event: message\ndata: {...}\n
        response_text = response.text
        data = None

        if response_text.startswith("event:"):
            # Parse SSE format
            for line in response_text.split("\n"):
                if line.startswith("data:"):
                    json_str = line[5:].strip()  # Remove "data:" prefix
                    data = json.loads(json_str)
                    break
        else:
            # Fallback to plain JSON
            data = response.json()

        if data is None:
            raise MCPError("Failed to parse server response")

        if "error" in data:
            error = data["error"]
            raise MCPError(
                message=error.get("message", "Unknown error"),
                code=error.get("code"),
                data=error.get("data"),
            )

        logger.debug(f"MCP response: {data.get('result')}")
        return data.get("result", {})
    
    # === Discovery Methods ===
    
    async def initialize(
        self,
        client_name: str = "sci-facts-mcp-client",
        client_version: str = "0.1.0",
    ) -> dict:
        """
        Initialize connection with MCP server.
        
        Returns server capabilities and info.
        """
        return await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": client_name,
                "version": client_version,
            }
        })
    
    async def list_tools(self) -> dict:
        """List all available tools from MCP server."""
        return await self._request("tools/list")
    
    async def list_resources(self) -> dict:
        """List all available resources from MCP server."""
        return await self._request("resources/list")
    
    async def list_prompts(self) -> dict:
        """List all available prompts from MCP server."""
        return await self._request("prompts/list")
    
    # === Tool Methods ===
    
    async def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        """
        Call an MCP tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return await self._request("tools/call", params)
    
    # Paper Search Tools
    
    async def papers_search_keyword(
        self,
        query: str,
        n_results: int = 20,
        sources: Optional[list[str]] = None,
        min_citations: Optional[int] = None,
    ) -> dict:
        """Search papers using keyword matching."""
        args = {"query": query, "n_results": n_results}
        if sources:
            args["sources"] = sources
        if min_citations is not None:
            args["min_citations"] = min_citations
        return await self.call_tool("papers_search_keyword", args)
    
    async def papers_search_semantic(
        self,
        seed_paper_ids: list[int],
        n_neighbors: int = 20,
    ) -> dict:
        """Find semantically similar papers by paper IDs."""
        return await self.call_tool("papers_search_semantic", {
            "seed_paper_ids": seed_paper_ids,
            "n_neighbors": n_neighbors,
        })
    
    async def papers_search_by_text(
        self,
        query: str,
        n_results: int = 20,
    ) -> dict:
        """Find similar papers using text query embedding."""
        return await self.call_tool("papers_search_by_text", {
            "query": query,
            "n_results": n_results,
        })
    
    async def papers_get_details(
        self,
        paper_ids: list[int],
        include_abstracts: bool = True,
        include_summaries: bool = False,
    ) -> dict:
        """Get detailed information for specific papers."""
        return await self.call_tool("papers_get_details", {
            "paper_ids": paper_ids,
            "include_abstracts": include_abstracts,
            "include_summaries": include_summaries,
        })
    
    # Topic Tools
    
    async def topics_search(
        self,
        query: str,
        level: Optional[str] = None,
        limit: int = 10,
    ) -> dict:
        """Search topics by name."""
        args = {"query": query, "limit": limit}
        if level:
            args["level"] = level
        return await self.call_tool("topics_search", args)
    
    async def topics_list(
        self,
        level: Optional[str] = None,
        min_papers: Optional[int] = None,
        limit: int = 100,
    ) -> dict:
        """List all topics with optional filters."""
        args = {"limit": limit}
        if level:
            args["level"] = level
        if min_papers is not None:
            args["min_papers"] = min_papers
        return await self.call_tool("topics_list", args)
    
    async def topics_get_impactful_papers(
        self,
        topic_name: str,
        level: Optional[str] = None,
        page: int = 1,
        papers_per_page: int = 10,
    ) -> dict:
        """Get top papers in a topic by citations/year."""
        args = {
            "topic_name": topic_name,
            "page": page,
            "papers_per_page": papers_per_page,
        }
        if level:
            args["level"] = level
        return await self.call_tool("topics_get_impactful_papers", args)
    
    async def topics_get_hierarchy(self, topic_name: str) -> dict:
        """Get topic with ancestor chain."""
        return await self.call_tool("topics_get_hierarchy", {
            "topic_name": topic_name,
        })
    
    # Collection Tools
    
    async def collections_create(
        self,
        user_id: int,
        name: str,
        paper_ids: Optional[list[int]] = None,
    ) -> dict:
        """Create a new collection."""
        args = {"user_id": user_id, "name": name}
        if paper_ids:
            args["paper_ids"] = paper_ids
        return await self.call_tool("collections_create", args)
    
    async def collections_list(self, user_id: int) -> dict:
        """List user's collections."""
        return await self.call_tool("collections_list", {"user_id": user_id})
    
    async def collections_get(self, collection_id: int) -> dict:
        """Get collection details and papers."""
        return await self.call_tool("collections_get", {"collection_id": collection_id})
    
    async def collections_add_items(
        self,
        collection_id: int,
        paper_ids: list[int],
    ) -> dict:
        """Add papers to a collection."""
        return await self.call_tool("collections_add_items", {
            "collection_id": collection_id,
            "paper_ids": paper_ids,
        })
    
    async def collections_remove_item(
        self,
        collection_id: int,
        paper_id: int,
    ) -> dict:
        """Remove a paper from a collection."""
        return await self.call_tool("collections_remove_item", {
            "collection_id": collection_id,
            "paper_id": paper_id,
        })
    
    async def collections_delete(self, collection_id: int) -> dict:
        """Delete a collection."""
        return await self.call_tool("collections_delete", {"collection_id": collection_id})
    
    # === Resource Methods ===
    
    async def read_resource(self, uri: str) -> dict:
        """
        Read a resource by URI.
        
        Supported URI patterns:
        - paper://{paper_id}
        - collection://{collection_id}
        - topic://{topic_name}
        """
        return await self._request("resources/read", {"uri": uri})
    
    async def read_paper(self, paper_id: int) -> dict:
        """Read paper resource."""
        return await self.read_resource(f"paper://{paper_id}")
    
    async def read_collection(self, collection_id: int) -> dict:
        """Read collection resource."""
        return await self.read_resource(f"collection://{collection_id}")
    
    async def read_topic(self, topic_name: str) -> dict:
        """Read topic resource."""
        return await self.read_resource(f"topic://{topic_name}")
    
    # === Prompt Methods ===
    
    async def get_prompt(self, name: str, arguments: Optional[dict] = None) -> dict:
        """
        Get a prompt template with arguments.
        
        Available prompts:
        - litreview_plan
        - litreview_screening_criteria
        - litreview_search_clarification
        - synthesis_summary
        - synthesis_extract_claims
        """
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return await self._request("prompts/get", params)
    
    async def get_litreview_plan(
        self,
        topic: str,
        scope: str = "comprehensive",
        time_range: Optional[str] = None,
    ) -> dict:
        """Get literature review plan prompt."""
        args = {"topic": topic, "scope": scope}
        if time_range:
            args["time_range"] = time_range
        return await self.get_prompt("litreview_plan", args)
    
    async def get_litreview_screening_criteria(
        self,
        topic: str,
        inclusion_rules: Optional[str] = None,
        exclusion_rules: Optional[str] = None,
    ) -> dict:
        """Get literature review screening criteria prompt."""
        args = {"topic": topic}
        if inclusion_rules:
            args["inclusion_rules"] = inclusion_rules
        if exclusion_rules:
            args["exclusion_rules"] = exclusion_rules
        return await self.get_prompt("litreview_screening_criteria", args)
    
    async def get_litreview_search_clarification(
        self,
        search_mode: str,
        initial_search_string: str,
    ) -> dict:
        """Get search clarification prompt."""
        return await self.get_prompt("litreview_search_clarification", {
            "search_mode": search_mode,
            "initial_search_string": initial_search_string,
        })
    
    async def get_synthesis_summary(
        self,
        paper_abstracts: str,
        goal: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> dict:
        """Get synthesis summary prompt."""
        args = {"paper_abstracts": paper_abstracts}
        if goal:
            args["goal"] = goal
        if tone:
            args["tone"] = tone
        return await self.get_prompt("synthesis_summary", args)
    
    async def get_synthesis_extract_claims(
        self,
        paper_id: int,
        goal: Optional[str] = None,
        detail_level: Optional[str] = None,
    ) -> dict:
        """Get claim extraction prompt."""
        args = {"paper_id": paper_id}
        if goal:
            args["goal"] = goal
        if detail_level:
            args["detail_level"] = detail_level
        return await self.get_prompt("synthesis_extract_claims", args)
