"""
MCP Client for communicating with Scholar Inbox MCP Server.

Provides async methods for calling tools, reading resources, and getting prompts.
Also provides SSE-based task stream for receiving pending tasks from the server.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Callable, Optional
from urllib.parse import quote, urljoin

import httpx

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


def _normalize_prompt_arguments(args: dict) -> dict:
    """
    Normalize prompt arguments to have string values only.

    MCP protocol requires all prompt argument values to be strings.
    This function converts:
    - Lists/arrays to JSON strings
    - Integers/floats to strings
    - None values are omitted

    Args:
        args: Dictionary of arguments with any value types

    Returns:
        Dictionary with all values converted to strings
    """
    normalized = {}
    for key, value in args.items():
        if value is None:
            continue
        elif isinstance(value, str):
            normalized[key] = value
        elif isinstance(value, (list, dict)):
            # Convert complex types to JSON strings
            normalized[key] = json.dumps(value)
        else:
            # Convert numbers and other types to strings
            normalized[key] = str(value)
    return normalized


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
                follow_redirects=True,
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
    
    async def _request(self, method: str, params: Optional[dict] = None, _retry: bool = True) -> dict:
        """
        Send JSON-RPC request to MCP server.

        Args:
            method: JSON-RPC method name
            params: Method parameters
            _retry: Internal flag for retry on session expiry

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

        # Handle 404 (session expired/invalid) by re-initializing
        if response.status_code == 404 and _retry and method != "initialize":
            logger.warning("MCP session invalid (404), re-initializing...")
            self._mcp_session_id = None
            await self.initialize()
            return await self._request(method, params, _retry=False)

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
            Tool result (parsed from content)
        """
        params = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        result = await self._request("tools/call", params)

        # MCP tool results are wrapped in content array
        # Format: {"content": [{"type": "text", "text": "{...}"}]}
        content = result.get("content", [])
        if content and isinstance(content, list):
            first_item = content[0]
            if first_item.get("type") == "text":
                text = first_item.get("text", "{}")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool result as JSON: {text[:100]}")
                    return {"text": text}

        # Fallback: return as-is if not in expected format
        return result
    
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
        return await self.read_resource(f"topic://{quote(topic_name)}")
    
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
        inclusion_rules: str,
        exclusion_rules: Optional[str] = None,
    ) -> dict:
        """Get literature review screening criteria prompt.

        Args:
            topic: Research topic being reviewed
            inclusion_rules: Comma-separated criteria for including papers
            exclusion_rules: Optional comma-separated criteria for excluding papers
        """
        args = {"topic": topic, "inclusion_rules": inclusion_rules}
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
        paper_id: int | str,
        goal: Optional[str] = None,
        detail_level: Optional[str] = None,
    ) -> dict:
        """Get claim extraction prompt.

        Args:
            paper_id: ID of the paper to analyze (will be converted to string)
            goal: What to extract - 'key findings', 'methodology', 'all claims'
            detail_level: How detailed - 'brief', 'moderate', 'comprehensive'
        """
        args = {"paper_id": str(paper_id)}
        if goal:
            args["goal"] = goal
        if detail_level:
            args["detail_level"] = detail_level
        return await self.get_prompt("synthesis_extract_claims", args)

    async def get_synthesis_build_prompt(
        self,
        goal: str,
        tone: str,
        papers_context: str,
    ) -> str:
        """Get the complete LLM-ready synthesis prompt from the MCP server.

        Calls the `synthesis_build_prompt` MCP prompt, which is the single
        source of truth for synthesis prompt construction.

        Args:
            goal: Synthesis goal (summary, fact_extraction, topic_extraction, keywords_extraction)
            tone: Writing style (precise, fun_facts, eli5, brutal)
            papers_context: Pre-formatted concatenated paper text from task payload

        Returns:
            Complete prompt text ready to send to the LLM

        Raises:
            MCPError: If the MCP server returns an error
            ValueError: If the response contains no prompt text
        """
        result = await self.get_prompt("synthesis_build_prompt", {
            "goal": goal,
            "tone": tone,
            "papers_context": papers_context,
        })

        # Extract prompt text from MCP prompt response
        # Response format: {"messages": [{"role": "user", "content": {"type": "text", "text": "..."}}]}
        messages = result.get("messages", [])
        if messages:
            content = messages[0].get("content", {})
            if isinstance(content, dict):
                text = content.get("text", "")
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            if text:
                return text

        raise ValueError("No prompt text returned from synthesis_build_prompt MCP prompt")

    # === Task Stream Methods ===

    def _get_task_stream_url(self) -> str:
        """Get the task stream SSE endpoint URL."""
        return self.settings.mcp_task_stream_url

    async def task_stream(
        self,
        poll_interval: float = 2.0,
        task_types: Optional[list[str]] = None,
        mark_processing: bool = True,
    ) -> AsyncGenerator[dict, None]:
        """
        Connect to the task stream SSE endpoint and yield incoming tasks.

        This method opens a persistent HTTP connection to the server's task stream
        endpoint and yields task dictionaries as they arrive via Server-Sent Events.

        Args:
            poll_interval: Seconds between server-side polls (default: 2.0)
            task_types: Optional list of task types to filter (e.g., ['synthesis'])
            mark_processing: If True, server marks tasks as 'processing' when sent

        Yields:
            Task dictionaries with keys:
                - task_id: Unique task identifier
                - task_type: Type of task (e.g., 'synthesis', 'agent_chat')
                - synthesis_id: Associated synthesis ID (if applicable)
                - payload: Task-specific data
                - status: Current status ('pending' or 'processing')
                - created_at: ISO timestamp of creation

        Example:
            async for task in client.task_stream(task_types=['synthesis']):
                print(f"Received task: {task['task_id']}")
                # Process the task...
                await client.complete_task(task['task_id'], result={'status': 'done'})
        """
        url = self._get_task_stream_url()

        # Build query parameters
        params = {'poll_interval': str(poll_interval)}
        if task_types:
            params['task_types'] = ','.join(task_types)
        if not mark_processing:
            params['mark_processing'] = 'false'

        headers = {
            "Authorization": f"Bearer {self.settings.mcp_api_key}",
            "Accept": "text/event-stream",
        }

        logger.info(f"Connecting to task stream: {url}")

        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream('GET', url, params=params, headers=headers) as response:
                response.raise_for_status()

                current_event = None
                current_data = []

                async for line in response.aiter_lines():
                    line = line.strip()

                    if not line:
                        # Empty line marks end of event
                        if current_event and current_data:
                            data_str = '\n'.join(current_data)
                            try:
                                data = json.loads(data_str)
                                if current_event == 'task':
                                    logger.debug(f"Received task: {data.get('task_id')}")
                                    yield data
                                elif current_event == 'heartbeat':
                                    logger.debug(f"Heartbeat: {data}")
                                elif current_event == 'error':
                                    logger.error(f"Task stream error: {data}")
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse event data: {e}")

                        current_event = None
                        current_data = []
                        continue

                    if line.startswith('event:'):
                        current_event = line[6:].strip()
                    elif line.startswith('data:'):
                        current_data.append(line[5:].strip())

    async def listen_for_tasks(
        self,
        handler: Callable[[dict], Any],
        poll_interval: float = 2.0,
        task_types: Optional[list[str]] = None,
        mark_processing: bool = True,
        stop_on_error: bool = False,
    ) -> None:
        """
        Listen for tasks and process them with the provided handler.

        This is a convenience method that wraps task_stream() with error handling
        and automatic reconnection.

        Args:
            handler: Async or sync callable that receives task dictionaries
            poll_interval: Seconds between server-side polls
            task_types: Optional list of task types to filter
            mark_processing: If True, server marks tasks as 'processing'
            stop_on_error: If True, stop listening on first handler error

        Example:
            async def process_task(task: dict):
                print(f"Processing: {task['task_id']}")
                # Do work...
                return {'result': 'success'}

            await client.listen_for_tasks(process_task, task_types=['synthesis'])
        """
        logger.info("Starting task listener...")

        while True:
            try:
                async for task in self.task_stream(
                    poll_interval=poll_interval,
                    task_types=task_types,
                    mark_processing=mark_processing,
                ):
                    try:
                        # Call handler (support both sync and async)
                        if asyncio.iscoroutinefunction(handler):
                            await handler(task)
                        else:
                            handler(task)
                    except Exception as e:
                        logger.error(f"Task handler error for {task.get('task_id')}: {e}")
                        if stop_on_error:
                            raise

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error in task stream: {e}")
                if e.response.status_code >= 400 and e.response.status_code < 500:
                    # Client error - don't retry
                    raise
                # Server error - retry after delay
                await asyncio.sleep(5.0)

            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                logger.warning(f"Connection error in task stream: {e}")
                # Reconnect after delay
                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                logger.info("Task listener cancelled")
                raise

            except Exception as e:
                logger.error(f"Unexpected error in task stream: {e}")
                await asyncio.sleep(5.0)

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> dict:
        """
        Update a task's status via MCP tool call.

        Note: This requires a corresponding tool on the server side.
        For now, tasks are updated directly by the server based on synthesis completion.

        Args:
            task_id: The task ID to update
            status: New status ('completed', 'failed', 'cancelled')
            error_message: Optional error message for failed tasks

        Returns:
            Result from the server
        """
        args = {"task_id": task_id, "status": status}
        if error_message:
            args["error_message"] = error_message
        return await self.call_tool("task_update_status", args)

    async def complete_task(
        self,
        task_id: str,
        result: Optional[dict] = None,
    ) -> dict:
        """
        Mark a task as completed with optional result.

        Note: This requires a corresponding tool on the server side.

        Args:
            task_id: The task ID to complete
            result: Optional result data to store

        Returns:
            Result from the server
        """
        args = {"task_id": task_id, "status": "completed"}
        if result:
            args["result"] = result
        return await self.call_tool("task_update_status", args)

    # === Agent Session Methods ===

    async def agent_session_create(
        self,
        user_id: int,
        query: str,
        task_id: Optional[str] = None,
    ) -> dict:
        """
        Create a new agent session for literature review.
        
        Args:
            user_id: ID of the user initiating the session
            query: The user's research query
            task_id: Optional task ID (generated if not provided)
            
        Returns:
            Result with session_id and status
        """
        args = {"user_id": user_id, "query": query}
        if task_id:
            args["task_id"] = task_id
        return await self.call_tool("agent_session_create", args)

    async def agent_session_add_papers(
        self,
        session_id: str,
        paper_ids: list[int],
        step: str,
        selection_reason: Optional[str] = None,
    ) -> dict:
        """
        Add papers to an agent session.
        
        Args:
            session_id: The session identifier
            paper_ids: List of paper IDs to add
            step: Pipeline step that found these papers
            selection_reason: Optional reason for selection
            
        Returns:
            Result with papers_added and total_papers
        """
        args = {
            "session_id": session_id,
            "paper_ids": paper_ids,
            "step": step,
        }
        if selection_reason:
            args["selection_reason"] = selection_reason
        return await self.call_tool("agent_session_add_papers", args)

    async def agent_session_get_papers(
        self,
        session_id: str,
        include_details: bool = False,
    ) -> dict:
        """
        Get all papers collected in an agent session.
        
        Args:
            session_id: The session identifier
            include_details: If True, include full paper metadata
            
        Returns:
            Result with papers list and total count
        """
        return await self.call_tool("agent_session_get_papers", {
            "session_id": session_id,
            "include_details": include_details,
        })

    async def agent_session_complete(
        self,
        session_id: str,
        summary: Optional[str] = None,
        final_paper_ids: Optional[list[int]] = None,
    ) -> dict:
        """
        Mark an agent session as complete.
        
        Args:
            session_id: The session identifier
            summary: Optional summary of findings
            final_paper_ids: Optional final curated paper list
            
        Returns:
            Result with status and total_papers
        """
        args = {"session_id": session_id}
        if summary:
            args["summary"] = summary
        if final_paper_ids:
            args["final_paper_ids"] = final_paper_ids
        return await self.call_tool("agent_session_complete", args)

    async def agent_session_update_progress(
        self,
        session_id: str,
        step: str,
        step_data: Optional[dict] = None,
    ) -> dict:
        """
        Update the progress of an agent session.
        
        Args:
            session_id: The session identifier
            step: Current pipeline step name
            step_data: Optional data for the current step
            
        Returns:
            Result with success status
        """
        args = {"session_id": session_id, "step": step}
        if step_data:
            args["step_data"] = step_data
        return await self.call_tool("agent_session_update_progress", args)

    # === Agent Prompt Methods ===

    async def get_agent_keyword_extraction(self, user_query: str) -> dict:
        """Get keyword extraction prompt for agent pipeline."""
        return await self.get_prompt("agent_keyword_extraction", {
            "user_query": user_query,
        })

    async def get_agent_query_augmentation(
        self,
        user_query: str,
        keywords: list[str],
        selected_papers: list[str],
    ) -> dict:
        """Get query augmentation prompt for agent pipeline."""
        return await self.get_prompt("agent_query_augmentation", {
            "user_query": user_query,
            "keywords": ", ".join(keywords),
            "selected_papers": "\n".join(f"- {p}" for p in selected_papers),
        })

    async def get_agent_paper_selection(
        self,
        user_query: str,
        papers_json: str,
        already_selected: Optional[list[str]] = None,
    ) -> dict:
        """Get paper selection prompt for agent pipeline."""
        args = {
            "user_query": user_query,
            "papers_json": papers_json,
        }
        if already_selected:
            args["already_selected"] = "\n".join(f"- {p}" for p in already_selected)
        return await self.get_prompt("agent_paper_selection", args)

    async def get_agent_custom_search_design(
        self,
        user_query: str,
        collected_papers: list[str],
        gaps_identified: Optional[str] = None,
    ) -> dict:
        """Get custom search design prompt for agent pipeline."""
        args = {
            "user_query": user_query,
            "collected_papers": "\n".join(f"- {p}" for p in collected_papers),
        }
        if gaps_identified:
            args["gaps_identified"] = gaps_identified
        return await self.get_prompt("agent_custom_search_design", args)
