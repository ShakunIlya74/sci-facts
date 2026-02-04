"""
MCP Client for Scholar Inbox Literature Review Automation.

This package provides an MCP client that connects to the Scholar Inbox MCP Server
and integrates with a vLLM server for agentic literature review workflows.
"""

from .client import MCPClient
from .config import Settings, get_settings

__version__ = "0.1.0"
__all__ = ["MCPClient", "Settings", "get_settings"]
