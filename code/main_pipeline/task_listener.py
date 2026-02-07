"""
Task Listener - Main entry point for processing MCP tasks.

This module listens for tasks from the MCP server task queue and processes them
using the appropriate pipeline:
- synthesis: Direct LLM synthesis of paper content
- agent_chat: Multi-step agent pipeline for literature review

Usage:
    python -m main_pipeline.task_listener
    
Environment Variables:
    MCP_SERVER_URL: URL of the MCP server
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

from mcp_client.client import MCPClient
from mcp_client.config import Settings, get_settings
from main_pipeline.synthesis_pipeline import SynthesisPipeline, create_synthesis_pipeline
from main_pipeline.agent_pipeline import AgentChatPipeline, create_agent_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TaskListener:
    """
    Listens for MCP tasks and dispatches them to appropriate pipelines.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        synthesis_pipeline: Optional[SynthesisPipeline] = None,
        agent_pipeline: Optional[AgentChatPipeline] = None,
    ):
        """
        Initialize task listener.
        
        Args:
            settings: Configuration settings
            synthesis_pipeline: Pre-configured synthesis pipeline
            agent_pipeline: Pre-configured agent pipeline
        """
        self.settings = settings or get_settings()
        self.mcp_client = MCPClient(settings=self.settings)
        
        # Initialize pipelines lazily
        self._synthesis_pipeline = synthesis_pipeline
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
    
    def get_synthesis_pipeline(self) -> SynthesisPipeline:
        """Get or create synthesis pipeline."""
        if self._synthesis_pipeline is None:
            self._synthesis_pipeline = SynthesisPipeline(
                llm=self._get_llm(),
                max_papers=20,
                max_abstract_length=500,
            )
        return self._synthesis_pipeline
    
    def get_agent_pipeline(self) -> AgentChatPipeline:
        """Get or create agent pipeline."""
        if self._agent_pipeline is None:
            self._agent_pipeline = AgentChatPipeline(
                llm=self._get_llm(),
                mcp_client=self.mcp_client,
                top_k_per_search=5,
            )
        return self._agent_pipeline
    
    async def process_synthesis_task(self, task: dict) -> dict:
        """
        Process a synthesis task.
        
        Args:
            task: Task dictionary with payload containing paper_ids, goal, tone
            
        Returns:
            Result dictionary with synthesis content or error
        """
        task_id = task.get("task_id")
        payload = task.get("payload", {})
        
        logger.info(f"Processing synthesis task {task_id}")
        
        # Extract parameters
        paper_ids = payload.get("paper_ids", [])
        goal = payload.get("goal", "summary")
        tone = payload.get("tone", "precise")
        
        if not paper_ids:
            return {
                "success": False,
                "error": "No paper_ids provided in task payload",
            }
        
        try:
            # Fetch paper details from MCP server
            paper_result = await self.mcp_client.papers_get_details(
                paper_ids=paper_ids,
                include_abstracts=True,
                include_summaries=False,
            )
            
            papers = paper_result.get("papers", [])
            if not papers:
                return {
                    "success": False,
                    "error": f"Could not fetch papers for IDs: {paper_ids}",
                }
            
            # Run synthesis pipeline
            pipeline = self.get_synthesis_pipeline()
            result = await pipeline.process_async(
                papers=papers,
                goal=goal,
                tone=tone,
            )
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Synthesis task failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
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
        user_query = payload.get("query") or payload.get("user_query")
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
            logger.error(f"Agent task failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def handle_task(self, task: dict) -> None:
        """
        Handle a single task from the queue.
        
        Args:
            task: Task dictionary from the MCP server
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
            
            # Update task status
            if result.get("success"):
                logger.info(f"Sending completed status for task {task_id} to MCP server")
                try:
                    update_response = await self.mcp_client.call_tool(
                        "task_update_status",
                        {
                            "task_id": task_id,
                            "status": "completed",
                            "result": result,
                        }
                    )
                    logger.info(f"Task {task_id} update response: {update_response}")
                    logger.info(f"Task {task_id} completed successfully")
                except Exception as update_err:
                    logger.error(f"Failed to send task {task_id} completion to server: {update_err}")
            else:
                logger.warning(f"Sending failed status for task {task_id} to MCP server")
                try:
                    update_response = await self.mcp_client.call_tool(
                        "task_update_status",
                        {
                            "task_id": task_id,
                            "status": "failed",
                            "error_message": result.get("error", "Unknown error"),
                        }
                    )
                    logger.info(f"Task {task_id} failed update response: {update_response}")
                except Exception as update_err:
                    logger.error(f"Failed to send task {task_id} failure to server: {update_err}")
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
    
    async def run(
        self,
        poll_interval: float = 2.0,
        task_types: Optional[list[str]] = None,
    ) -> None:
        """
        Run the task listener.
        
        Args:
            poll_interval: Seconds between task queue polls
            task_types: List of task types to listen for (default: all)
        """
        self._running = True
        
        logger.info("=" * 60)
        logger.info("Starting Task Listener")
        logger.info("=" * 60)
        logger.info(f"MCP Server: {self.settings.mcp_server_url}")
        logger.info(f"vLLM Server: {self.settings.vllm_base_url}")
        logger.info(f"Poll Interval: {poll_interval}s")
        logger.info(f"Task Types: {task_types or 'all'}")
        logger.info("=" * 60)
        
        # Track processed tasks to avoid duplicates
        processed_task_ids = set()
        
        try:
            # Initialize MCP client
            await self.mcp_client.initialize()
            logger.info("Connected to MCP server")
            
            # Poll for tasks using the task_list_pending tool
            while not self._shutdown_event.is_set():
                try:
                    # Get pending tasks via MCP tool
                    task_types_str = ",".join(task_types) if task_types else None
                    result = await self.mcp_client.call_tool(
                        "task_list_pending",
                        {
                            "limit": 10,
                            "task_types": task_types_str,
                        }
                    )
                    
                    tasks = result.get("tasks", [])
                    
                    for task in tasks:
                        task_id = task.get("task_id")
                        
                        # Skip already processed tasks
                        if task_id in processed_task_ids:
                            continue
                        
                        # Mark as processing
                        if task.get("status") == "pending":
                            await self.mcp_client.call_tool(
                                "task_update_status",
                                {"task_id": task_id, "status": "processing"}
                            )
                        
                        # Process the task
                        processed_task_ids.add(task_id)
                        await self.handle_task(task)
                        
                        # Clean up old task IDs to prevent memory growth
                        if len(processed_task_ids) > 1000:
                            # Keep only the most recent 500
                            processed_task_ids = set(list(processed_task_ids)[-500:])
                    
                except Exception as e:
                    logger.error(f"Error polling for tasks: {e}")
                
                # Wait before next poll
                await asyncio.sleep(poll_interval)
                
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
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Task Listener")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between task queue polls (default: 2.0)",
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
