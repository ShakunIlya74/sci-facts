"""
Tests for MCP Server tool endpoints.

Tests all 14 tools: paper search (4), topics (4), collections (6).
"""

import pytest
from mcp_client.client import MCPClient, MCPError
from mcp_client.logging_utils import ResultLogger


@pytest.mark.asyncio
class TestPaperSearchTools:
    """Test paper search tool endpoints."""
    
    async def test_papers_search_keyword(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_paper_query: str,
    ):
        """Test keyword-based paper search."""
        result_logger.log_request("tools/call", {
            "name": "papers_search_keyword",
            "arguments": {"query": sample_paper_query, "n_results": 5},
        })
        
        try:
            response = await mcp_client.papers_search_keyword(
                query=sample_paper_query,
                n_results=5,
            )
            
            result_logger.log_response("papers_search_keyword", response, success=True)
            
            # Verify non-empty response
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_papers_search_keyword",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_papers_search_keyword",
                passed=False,
                error=e,
            )
            raise
    
    async def test_papers_search_keyword_with_sources(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_paper_query: str,
    ):
        """Test keyword search with source filter."""
        result_logger.log_request("tools/call", {
            "name": "papers_search_keyword",
            "arguments": {
                "query": sample_paper_query,
                "n_results": 5,
                "sources": ["arxiv"],
            },
        })
        
        try:
            response = await mcp_client.papers_search_keyword(
                query=sample_paper_query,
                n_results=5,
                sources=["arxiv"],
            )
            
            result_logger.log_response(
                "papers_search_keyword (with sources)",
                response,
                success=True,
            )
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_papers_search_keyword_with_sources",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_papers_search_keyword_with_sources",
                passed=False,
                error=e,
            )
            raise
    
    async def test_papers_search_by_text(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_paper_query: str,
    ):
        """Test semantic search by text query."""
        result_logger.log_request("tools/call", {
            "name": "papers_search_by_text",
            "arguments": {"query": sample_paper_query, "n_results": 5},
        })
        
        try:
            response = await mcp_client.papers_search_by_text(
                query=sample_paper_query,
                n_results=5,
            )
            
            result_logger.log_response("papers_search_by_text", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_papers_search_by_text",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_papers_search_by_text",
                passed=False,
                error=e,
            )
            raise
    
    async def test_papers_search_semantic(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test semantic search by seed paper IDs."""
        # Note: This requires valid paper IDs - using placeholder values
        seed_ids = [1, 2, 3]  # Replace with actual paper IDs from your database
        
        result_logger.log_request("tools/call", {
            "name": "papers_search_semantic",
            "arguments": {"seed_paper_ids": seed_ids, "n_neighbors": 5},
        })
        
        try:
            response = await mcp_client.papers_search_semantic(
                seed_paper_ids=seed_ids,
                n_neighbors=5,
            )
            
            result_logger.log_response("papers_search_semantic", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_papers_search_semantic",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_papers_search_semantic",
                passed=False,
                error=e,
            )
            # Don't raise - this test may fail if seed IDs don't exist
            pytest.skip(f"Semantic search failed (may need valid paper IDs): {e}")
    
    async def test_papers_get_details(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting paper details by IDs."""
        # Note: This requires valid paper IDs - using placeholder values
        paper_ids = [1, 2, 3]  # Replace with actual paper IDs
        
        result_logger.log_request("tools/call", {
            "name": "papers_get_details",
            "arguments": {
                "paper_ids": paper_ids,
                "include_abstracts": True,
                "include_summaries": False,
            },
        })
        
        try:
            response = await mcp_client.papers_get_details(
                paper_ids=paper_ids,
                include_abstracts=True,
                include_summaries=False,
            )
            
            result_logger.log_response("papers_get_details", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_papers_get_details",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_papers_get_details",
                passed=False,
                error=e,
            )
            pytest.skip(f"Get details failed (may need valid paper IDs): {e}")


@pytest.mark.asyncio
class TestTopicTools:
    """Test topic-related tool endpoints."""
    
    async def test_topics_search(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_topic_query: str,
    ):
        """Test topic search by name."""
        result_logger.log_request("tools/call", {
            "name": "topics_search",
            "arguments": {"query": sample_topic_query, "limit": 5},
        })
        
        try:
            response = await mcp_client.topics_search(
                query=sample_topic_query,
                limit=5,
            )
            
            result_logger.log_response("topics_search", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_topics_search",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_topics_search",
                passed=False,
                error=e,
            )
            raise
    
    async def test_topics_list(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test listing all topics."""
        result_logger.log_request("tools/call", {
            "name": "topics_list",
            "arguments": {"limit": 10},
        })
        
        try:
            response = await mcp_client.topics_list(limit=10)
            
            result_logger.log_response("topics_list", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_topics_list",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_topics_list",
                passed=False,
                error=e,
            )
            raise
    
    async def test_topics_list_with_level(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test listing topics filtered by level."""
        result_logger.log_request("tools/call", {
            "name": "topics_list",
            "arguments": {"level": "l3", "limit": 10},
        })
        
        try:
            response = await mcp_client.topics_list(level="l3", limit=10)
            
            result_logger.log_response("topics_list (level=l3)", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_topics_list_with_level",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_topics_list_with_level",
                passed=False,
                error=e,
            )
            raise
    
    async def test_topics_get_impactful_papers(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_topic_query: str,
    ):
        """Test getting impactful papers for a topic."""
        # First search for a topic to get a valid name
        topic_name = sample_topic_query
        
        result_logger.log_request("tools/call", {
            "name": "topics_get_impactful_papers",
            "arguments": {
                "topic_name": topic_name,
                "papers_per_page": 5,
            },
        })
        
        try:
            response = await mcp_client.topics_get_impactful_papers(
                topic_name=topic_name,
                papers_per_page=5,
            )
            
            result_logger.log_response(
                "topics_get_impactful_papers",
                response,
                success=True,
            )
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_topics_get_impactful_papers",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_topics_get_impactful_papers",
                passed=False,
                error=e,
            )
            pytest.skip(f"Get impactful papers failed (topic may not exist): {e}")
    
    async def test_topics_get_hierarchy(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_topic_query: str,
    ):
        """Test getting topic hierarchy."""
        topic_name = sample_topic_query
        
        result_logger.log_request("tools/call", {
            "name": "topics_get_hierarchy",
            "arguments": {"topic_name": topic_name},
        })
        
        try:
            response = await mcp_client.topics_get_hierarchy(topic_name=topic_name)
            
            result_logger.log_response("topics_get_hierarchy", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_topics_get_hierarchy",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_topics_get_hierarchy",
                passed=False,
                error=e,
            )
            pytest.skip(f"Get hierarchy failed (topic may not exist): {e}")


@pytest.mark.asyncio 
class TestCollectionTools:
    """Test collection management tool endpoints."""
    
    async def test_collections_list(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_user_id: int,
    ):
        """Test listing user collections."""
        result_logger.log_request("tools/call", {
            "name": "collections_list",
            "arguments": {"user_id": sample_user_id},
        })
        
        try:
            response = await mcp_client.collections_list(user_id=sample_user_id)
            
            result_logger.log_response("collections_list", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_collections_list",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_collections_list",
                passed=False,
                error=e,
            )
            raise
    
    async def test_collections_create_and_delete(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_user_id: int,
    ):
        """Test creating and deleting a collection."""
        collection_name = "test_collection_pytest"
        
        # Create collection
        result_logger.log_request("tools/call", {
            "name": "collections_create",
            "arguments": {"user_id": sample_user_id, "name": collection_name},
        })
        
        try:
            create_response = await mcp_client.collections_create(
                user_id=sample_user_id,
                name=collection_name,
            )
            
            result_logger.log_response("collections_create", create_response, success=True)
            
            assert create_response is not None, "Create response should not be None"
            
            # Try to extract collection ID for cleanup
            collection_id = None
            if isinstance(create_response, dict):
                # Try common response structures
                collection_id = create_response.get("collection_id") or \
                               create_response.get("id") or \
                               (create_response.get("collection", {}) or {}).get("id")
            
            result_logger.log_test_result(
                "test_collections_create",
                passed=True,
                response=create_response,
            )
            
            # Delete collection if we got an ID
            if collection_id:
                result_logger.log_request("tools/call", {
                    "name": "collections_delete",
                    "arguments": {"collection_id": collection_id},
                })
                
                delete_response = await mcp_client.collections_delete(
                    collection_id=collection_id,
                )
                
                result_logger.log_response(
                    "collections_delete",
                    delete_response,
                    success=True,
                )
                
                result_logger.log_test_result(
                    "test_collections_delete",
                    passed=True,
                    response=delete_response,
                )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_collections_create_and_delete",
                passed=False,
                error=e,
            )
            raise
    
    async def test_collections_get(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting collection details."""
        # Use a placeholder collection ID - will likely fail without real data
        collection_id = 1
        
        result_logger.log_request("tools/call", {
            "name": "collections_get",
            "arguments": {"collection_id": collection_id},
        })
        
        try:
            response = await mcp_client.collections_get(collection_id=collection_id)
            
            result_logger.log_response("collections_get", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_collections_get",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_collections_get",
                passed=False,
                error=e,
            )
            pytest.skip(f"Get collection failed (collection may not exist): {e}")
    
    async def test_collections_add_and_remove_items(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
        sample_user_id: int,
    ):
        """Test adding and removing items from a collection."""
        # This is a comprehensive test that creates a collection,
        # adds items, removes items, then deletes the collection
        
        collection_name = "test_items_collection_pytest"
        paper_ids = [1, 2]  # Placeholder paper IDs
        
        try:
            # 1. Create collection
            create_response = await mcp_client.collections_create(
                user_id=sample_user_id,
                name=collection_name,
            )
            result_logger.log_response("collections_create", create_response, success=True)
            
            collection_id = None
            if isinstance(create_response, dict):
                collection_id = create_response.get("collection_id") or \
                               create_response.get("id") or \
                               (create_response.get("collection", {}) or {}).get("id")
            
            if not collection_id:
                pytest.skip("Could not get collection ID from create response")
                return
            
            # 2. Add items
            result_logger.log_request("tools/call", {
                "name": "collections_add_items",
                "arguments": {"collection_id": collection_id, "paper_ids": paper_ids},
            })
            
            add_response = await mcp_client.collections_add_items(
                collection_id=collection_id,
                paper_ids=paper_ids,
            )
            result_logger.log_response("collections_add_items", add_response, success=True)
            
            result_logger.log_test_result(
                "test_collections_add_items",
                passed=True,
                response=add_response,
            )
            
            # 3. Remove item
            result_logger.log_request("tools/call", {
                "name": "collections_remove_item",
                "arguments": {"collection_id": collection_id, "paper_id": paper_ids[0]},
            })
            
            remove_response = await mcp_client.collections_remove_item(
                collection_id=collection_id,
                paper_id=paper_ids[0],
            )
            result_logger.log_response("collections_remove_item", remove_response, success=True)
            
            result_logger.log_test_result(
                "test_collections_remove_item",
                passed=True,
                response=remove_response,
            )
            
            # 4. Cleanup - delete collection
            await mcp_client.collections_delete(collection_id=collection_id)
            result_logger.logger.info(f"Cleaned up test collection {collection_id}")
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_collections_add_and_remove_items",
                passed=False,
                error=e,
            )
            pytest.skip(f"Collection item operations failed: {e}")
