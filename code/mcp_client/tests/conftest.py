"""
Pytest configuration and fixtures for MCP Client tests.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio

from mcp_client.client import MCPClient
from mcp_client.config import Settings
from mcp_client.logging_utils import ResultLogger, setup_test_logging


# Configure event loop
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """
    Test settings configuration.
    
    Loads from environment variables and .env file.
    """
    return Settings.from_env()


@pytest.fixture(scope="session")
def test_logger(test_settings: Settings) -> logging.Logger:
    """Set up test session logger."""
    return setup_test_logging(
        log_dir=test_settings.log_dir,
        log_level="DEBUG",
        session_name="mcp_endpoints_test",
    )


@pytest.fixture(scope="session")
def result_logger(test_logger: logging.Logger, test_settings: Settings) -> ResultLogger:
    """Create result logger for trimmed response logging."""
    return ResultLogger(
        logger=test_logger,
        max_chars=test_settings.log_response_max_chars,
    )


def check_mcp_server_available(settings: Settings) -> bool:
    """Check if MCP server is reachable."""
    try:
        response = httpx.post(
            settings.mcp_server_url,
            headers={
                "Authorization": f"Bearer {settings.mcp_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "connection-test", "version": "0.1.0"}
                },
                "id": 0
            },
            timeout=10.0,
        )
        # Server returns SSE format on success (status 200 with "event: message")
        return response.status_code == 200 and "event:" in response.text
    except Exception as e:
        print(f"MCP server check failed: {e}")
        return False


@pytest.fixture(scope="session")
def mcp_server_available(test_settings: Settings, test_logger: logging.Logger) -> bool:
    """Check if MCP server is available for testing."""
    available = check_mcp_server_available(test_settings)
    if not available:
        test_logger.warning(
            f"MCP server not available at {test_settings.mcp_server_url}. "
            "Some tests will be skipped."
        )
    else:
        test_logger.info(f"MCP server available at {test_settings.mcp_server_url}")
    return available


@pytest_asyncio.fixture
async def mcp_client(test_settings: Settings, mcp_server_available: bool) -> AsyncGenerator[MCPClient, None]:
    """
    Create MCP client for tests.
    
    Yields an initialized client instance and ensures cleanup after tests.
    Skips if MCP server is not available.
    """
    if not mcp_server_available:
        pytest.skip("MCP server not available")
    
    client = MCPClient(settings=test_settings)
    # Initialize session to get session ID
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture(scope="session")
def sample_paper_query() -> str:
    """Sample query for paper search tests."""
    return "transformer attention mechanism"


@pytest.fixture(scope="session")
def sample_topic_query() -> str:
    """Sample query for topic search tests."""
    return "machine learning"


@pytest.fixture(scope="session")
def sample_user_id() -> int:
    """Sample user ID for collection tests."""
    return 1
