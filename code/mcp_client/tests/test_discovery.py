"""
Tests for MCP Server discovery endpoints.

Tests initialization, tools/list, resources/list, and prompts/list.
"""

import pytest
from mcp_client.client import MCPClient, MCPError
from mcp_client.logging_utils import ResultLogger


@pytest.mark.asyncio
class TestDiscoveryEndpoints:
    """Test MCP server discovery/listing endpoints."""
    
    async def test_initialize(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test MCP server initialization handshake."""
        result_logger.log_request("initialize")
        
        try:
            response = await mcp_client.initialize()
            
            result_logger.log_response("initialize", response, success=True)
            
            # Verify non-empty response
            assert response is not None, "Response should not be None"
            assert isinstance(response, dict), "Response should be a dictionary"
            
            result_logger.log_test_result("test_initialize", passed=True, response=response)
            
        except MCPError as e:
            result_logger.log_test_result("test_initialize", passed=False, error=e)
            raise
    
    async def test_list_tools(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test listing all available tools."""
        result_logger.log_request("tools/list")
        
        try:
            response = await mcp_client.list_tools()
            
            result_logger.log_response("tools/list", response, success=True)
            
            # Verify non-empty response
            assert response is not None, "Response should not be None"
            assert isinstance(response, dict), "Response should be a dictionary"
            
            # Check for tools array
            if "tools" in response:
                tools = response["tools"]
                assert isinstance(tools, list), "Tools should be a list"
                assert len(tools) > 0, "Should have at least one tool"
                
                # Log tool names
                tool_names = [t.get("name", "unknown") for t in tools]
                result_logger.logger.info(f"Available tools ({len(tools)}): {tool_names}")
            
            result_logger.log_test_result("test_list_tools", passed=True, response=response)
            
        except MCPError as e:
            result_logger.log_test_result("test_list_tools", passed=False, error=e)
            raise
    
    async def test_list_resources(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test listing all available resources."""
        result_logger.log_request("resources/list")
        
        try:
            response = await mcp_client.list_resources()
            
            result_logger.log_response("resources/list", response, success=True)
            
            # Verify non-empty response
            assert response is not None, "Response should not be None"
            assert isinstance(response, dict), "Response should be a dictionary"
            
            # Check for resources array
            if "resources" in response:
                resources = response["resources"]
                assert isinstance(resources, list), "Resources should be a list"
                
                # Log resource URIs
                resource_uris = [r.get("uri", "unknown") for r in resources]
                result_logger.logger.info(f"Available resources ({len(resources)}): {resource_uris}")
            
            result_logger.log_test_result("test_list_resources", passed=True, response=response)
            
        except MCPError as e:
            result_logger.log_test_result("test_list_resources", passed=False, error=e)
            raise
    
    async def test_list_prompts(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test listing all available prompts."""
        result_logger.log_request("prompts/list")
        
        try:
            response = await mcp_client.list_prompts()
            
            result_logger.log_response("prompts/list", response, success=True)
            
            # Verify non-empty response
            assert response is not None, "Response should not be None"
            assert isinstance(response, dict), "Response should be a dictionary"
            
            # Check for prompts array
            if "prompts" in response:
                prompts = response["prompts"]
                assert isinstance(prompts, list), "Prompts should be a list"
                
                # Log prompt names
                prompt_names = [p.get("name", "unknown") for p in prompts]
                result_logger.logger.info(f"Available prompts ({len(prompts)}): {prompt_names}")
            
            result_logger.log_test_result("test_list_prompts", passed=True, response=response)
            
        except MCPError as e:
            result_logger.log_test_result("test_list_prompts", passed=False, error=e)
            raise
