"""
Task Listener - Main entry point for processing MCP tasks.

Connects to the MCP server's SSE task stream endpoint (GET /mcp/tasks/stream)
and processes tasks using the appropriate pipeline.

MCP-NATIVE SYNTHESIS WORKFLOW:
1. Receive task from SSE stream (payload has papers_context, goal, tone — NO prompt)
2. Call `synthesis_build_prompt` MCP prompt to get the complete LLM-ready prompt
3. Send prompt to vLLM for inference
4. Return result via `task_update_status` MCP tool

IMPORTANT: Tasks are streamed from GET /mcp/tasks/stream (SSE endpoint),
NOT from POST /mcp (JSON-RPC endpoint). The JSON-RPC endpoint is only used
for tool calls (task_update_status), prompt retrieval (synthesis_build_prompt) and maybe resources usage.

Usage:
    python -m main_pipeline.task_listener

Environment Variables:
    MCP_SERVER_URL: MCP JSON-RPC endpoint (default: http://localhost:8001/mcp)
    VLLM_HOST: vLLM server host
    VLLM_PORT: vLLM server port
    VLLM_API_KEY: vLLM server API key
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from mcp_client.client import MCPClient
from mcp_client.config import Settings, get_settings
from main_pipeline.agent_pipeline import AgentChatPipeline, create_agent_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TaskListener:
    """
    Listens for MCP tasks via SSE stream and dispatches them to appropriate pipelines.

    Uses GET /mcp/tasks/stream for receiving tasks (SSE),
    and POST /mcp for JSON-RPC calls (tool calls, prompt retrieval).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        agent_pipeline: Optional[AgentChatPipeline] = None,
    ):
        """
        Initialize task listener.

        Args:
            settings: Configuration settings
            agent_pipeline: Pre-configured agent pipeline
        """
        self.settings = settings or get_settings()
        self.mcp_client = MCPClient(settings=self.settings)

        # Initialize pipelines lazily
        self._agent_pipeline = agent_pipeline
        self._llm: Optional[ChatOpenAI] = None

        # Control flags
        self._running = False
        self._shutdown_event = asyncio.Event()

    def _get_llm(self) -> ChatOpenAI:
        """Get or create LLM instance."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.vllm_model,
                openai_api_key=self.settings.vllm_api_key,
                openai_api_base=self.settings.vllm_base_url,
                temperature=self.settings.temperature,
                max_tokens=2048,  # Higher for synthesis tasks
            )
        return self._llm

    def get_agent_pipeline(self) -> AgentChatPipeline:
        """Get or create agent pipeline."""
        if self._agent_pipeline is None:
            self._agent_pipeline = AgentChatPipeline(
                llm=self._get_llm(),
                mcp_client=self.mcp_client,
                top_k_per_search=5,
            )
        return self._agent_pipeline

    # =========================================================================
    # Synthesis Task Processing (MCP-Native)
    # =========================================================================

    async def process_synthesis_task(self, task: dict) -> dict:
        """
        Process a synthesis task using the MCP-native workflow.

        The task payload contains papers_context (pre-formatted by the server),
        goal, and tone — but NO prompt. We retrieve the prompt by calling
        the `synthesis_build_prompt` MCP prompt, then send it to the LLM.

        Args:
            task: Task dictionary from SSE stream with payload containing
                  papers_context, goal, tone, paper_count

        Returns:
            Result dictionary with synthesis content or error
        """
        task_id = task.get("task_id")
        payload = task.get("payload", {})

        logger.info(f"Processing synthesis task {task_id}")

        # Extract parameters from payload
        goal = payload.get("goal", "summary")
        tone = payload.get("tone", "precise")
        papers_context = payload.get("papers_context", "")
        paper_count = payload.get("paper_count", 0)

        if not papers_context:
            return {
                "success": False,
                "error": "No papers_context in task payload",
            }

        logger.info(
            f"Synthesis params: goal={goal}, tone={tone}, "
            f"paper_count={paper_count}, context_length={len(papers_context)}"
        )

        try:
            # Step 1: Retrieve the complete prompt from MCP server
            logger.info(f"Retrieving synthesis prompt from MCP server (goal={goal}, tone={tone})")
            prompt_text = await self.mcp_client.get_synthesis_build_prompt(
                goal=goal,
                tone=tone,
                papers_context=papers_context,
            )
            logger.info(f"Retrieved prompt ({len(prompt_text)} chars)")

            # Step 2: Send prompt to vLLM for inference
            logger.info("Sending prompt to LLM for synthesis generation")
            llm = self._get_llm()
            response = await llm.ainvoke([HumanMessage(content=prompt_text)])

            content = response.content
            if not content:
                return {
                    "success": False,
                    "error": "LLM returned empty response",
                }

            logger.info(f"Synthesis generated: {len(content)} characters")

            return {
                "success": True,
                "content": content,
                "paper_count": paper_count,
                "goal": goal,
                "tone": tone,
            }

        except Exception as e:
            logger.error(f"Synthesis task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Agent Chat Task Processing
    # =========================================================================

    async def process_agent_task(self, task: dict) -> dict:
        """
        Process an agent_chat task.

        Args:
            task: Task dictionary with payload containing user query

        Returns:
            Result dictionary with collected papers or error
        """
        task_id = task.get("task_id")
        payload = task.get("payload", {})

        logger.info(f"Processing agent_chat task {task_id}")

        # Extract user query
        user_query = payload.get("query") or payload.get("user_query") or payload.get("message")
        if not user_query:
            return {
                "success": False,
                "error": "No query provided in task payload",
            }

        try:
            # Create agent session
            session_result = await self.mcp_client.call_tool(
                "agent_session_create",
                {
                    "user_id": payload.get("user_id", 1),
                    "query": user_query,
                    "task_id": task_id,
                }
            )

            session_id = session_result.get("session_id")
            if not session_id:
                logger.warning("Could not create agent session, proceeding without session tracking")
                session_id = task_id  # Use task_id as fallback

            # Run agent pipeline
            pipeline = self.get_agent_pipeline()
            result = await pipeline.run(
                user_query=user_query,
                task_id=task_id,
            )

            # Complete the agent session
            if session_result.get("success"):
                paper_ids = [p.get("paper_id") for p in result.all_selected_papers]
                await self.mcp_client.call_tool(
                    "agent_session_complete",
                    {
                        "session_id": session_id,
                        "summary": f"Found {len(paper_ids)} papers",
                        "final_paper_ids": paper_ids,
                    }
                )

            return result.to_dict()

        except Exception as e:
            logger.error(f"Agent task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    # =========================================================================
    # Task Routing & Status Updates
    # =========================================================================

    async def handle_task(self, task: dict) -> None:
        """
        Handle a single task from the SSE stream.

        Routes to the appropriate handler, then updates the task status
        on the MCP server via the task_update_status tool.

        Args:
            task: Task dictionary from the SSE stream
        """
        task_id = task.get("task_id")
        task_type = task.get("task_type")

        logger.info(f"Handling task {task_id} of type {task_type}")

        try:
            # Route to appropriate handler
            if task_type == "synthesis":
                result = await self.process_synthesis_task(task)
            elif task_type == "agent_chat":
                result = await self.process_agent_task(task)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                result = {
                    "success": False,
                    "error": f"Unknown task type: {task_type}",
                }

            # Log the result before sending
            result_preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
            logger.info(f"Task {task_id} result preview: {result_preview}")

            # Update task status on MCP server via JSON-RPC tool call
            if result.get("success"):
                logger.info(f"Sending completed status for task {task_id}")
                try:
                    await self.mcp_client.call_tool(
                        "task_update_status",
                        {
                            "task_id": task_id,
                            "status": "completed",
                            "result": result,
                        }
                    )
                    logger.info(f"Task {task_id} completed successfully")
                except Exception as update_err:
                    logger.error(f"Failed to send task {task_id} completion: {update_err}")
            else:
                logger.warning(f"Task {task_id} failed, sending failed status")
                try:
                    await self.mcp_client.call_tool(
                        "task_update_status",
                        {
                            "task_id": task_id,
                            "status": "failed",
                            "error_message": result.get("error", "Unknown error"),
                        }
                    )
                except Exception as update_err:
                    logger.error(f"Failed to send task {task_id} failure: {update_err}")
                logger.error(f"Task {task_id} failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error handling task {task_id}: {e}")
            try:
                await self.mcp_client.call_tool(
                    "task_update_status",
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "error_message": str(e),
                    }
                )
            except Exception as update_error:
                logger.error(f"Failed to update task status: {update_error}")

    # =========================================================================
    # Main Run Loop (SSE Stream)
    # =========================================================================

    async def run(
        self,
        poll_interval: float = 2.0,
        task_types: Optional[list[str]] = None,
    ) -> None:
        """
        Run the task listener using SSE task stream.

        Connects to GET /mcp/tasks/stream and processes tasks as they arrive.
        Automatically reconnects on connection errors.

        Args:
            poll_interval: Seconds between server-side task queue polls
            task_types: List of task types to listen for (default: all)
        """
        self._running = True

        logger.info("=" * 60)
        logger.info("Starting Task Listener (SSE Stream)")
        logger.info("=" * 60)
        logger.info(f"MCP JSON-RPC: {self.settings.mcp_server_url}")
        logger.info(f"MCP Task Stream: {self.settings.mcp_task_stream_url}")
        logger.info(f"vLLM Server: {self.settings.vllm_base_url}")
        logger.info(f"Poll Interval: {poll_interval}s")
        logger.info(f"Task Types: {task_types or 'all'}")
        logger.info("=" * 60)

        try:
            # Initialize MCP client (establishes JSON-RPC session for tool/prompt calls)
            await self.mcp_client.initialize()
            logger.info("MCP client initialized (JSON-RPC session ready)")

            # Use the MCPClient's built-in listen_for_tasks with auto-reconnection.
            # This connects to GET /mcp/tasks/stream (SSE), NOT POST /mcp.
            async def _task_handler(task: dict):
                if self._shutdown_event.is_set():
                    raise asyncio.CancelledError("Shutdown requested")
                await self.handle_task(task)

            await self.mcp_client.listen_for_tasks(
                handler=_task_handler,
                poll_interval=poll_interval,
                task_types=task_types,
                mark_processing=True,
            )

        except asyncio.CancelledError:
            logger.info("Task listener cancelled")
        except Exception as e:
            logger.error(f"Task listener error: {e}")
            raise
        finally:
            self._running = False
            await self.mcp_client.close()
            logger.info("Task listener stopped")

    def request_shutdown(self):
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Task Listener (SSE Stream)")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between server-side task queue polls (default: 2.0)",
    )
    parser.add_argument(
        "--task-types",
        type=str,
        default=None,
        help="Comma-separated task types to listen for (default: all)",
    )
    args = parser.parse_args()

    # Parse task types
    task_types = None
    if args.task_types:
        task_types = [t.strip() for t in args.task_types.split(",")]

    # Create and run listener
    listener = TaskListener()

    # Handle signals
    def signal_handler(signum, frame):
        listener.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run listener
    await listener.run(
        poll_interval=args.poll_interval,
        task_types=task_types,
    )


if __name__ == "__main__":
    asyncio.run(main())
