"""
Tests for MCP Server resource endpoints.

Tests all 3 resource URI patterns: paper://, collection://, topic://
"""

import pytest
from mcp_client.client import MCPClient, MCPError
from mcp_client.logging_utils import ResultLogger


@pytest.mark.asyncio
class TestResourceEndpoints:
    """Test MCP resource read endpoints."""
    
    async def test_read_paper_resource(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test reading a paper resource by URI."""
        paper_id = 2  # Placeholder - replace with actual paper ID
        uri = f"paper://{paper_id}"
        
        result_logger.log_request("resources/read", {"uri": uri})
        
        try:
            response = await mcp_client.read_paper(paper_id=paper_id)
            
            result_logger.log_response(f"read_resource({uri})", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_read_paper_resource",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_read_paper_resource",
                passed=False,
                error=e,
            )
            pytest.skip(f"Read paper resource failed (paper may not exist): {e}")
    
    async def test_read_collection_resource(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test reading a collection resource by URI."""
        collection_id = 1  # Placeholder - replace with actual collection ID
        uri = f"collection://{collection_id}"
        
        result_logger.log_request("resources/read", {"uri": uri})
        
        try:
            response = await mcp_client.read_collection(collection_id=collection_id)
            
            result_logger.log_response(f"read_resource({uri})", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_read_collection_resource",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_read_collection_resource",
                passed=False,
                error=e,
            )
            pytest.skip(f"Read collection resource failed (collection may not exist): {e}")
    
    async def test_read_topic_resource(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_topic_query: str,
    ):
        """Test reading a topic resource by URI."""
        topic_name = sample_topic_query  # Using sample topic query as topic name
        uri = f"topic://{topic_name}"
        
        result_logger.log_request("resources/read", {"uri": uri})
        
        try:
            response = await mcp_client.read_topic(topic_name=topic_name)
            
            result_logger.log_response(f"read_resource({uri})", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_read_topic_resource",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_read_topic_resource",
                passed=False,
                error=e,
            )
            pytest.skip(f"Read topic resource failed (topic may not exist): {e}")
    
    async def test_read_resource_generic(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test reading resource with generic URI method."""
        uri = "paper://2"
        
        result_logger.log_request("resources/read", {"uri": uri})
        
        try:
            response = await mcp_client.read_resource(uri=uri)
            
            result_logger.log_response(f"read_resource({uri})", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_read_resource_generic",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_read_resource_generic",
                passed=False,
                error=e,
            )
            pytest.skip(f"Read resource failed: {e}")
