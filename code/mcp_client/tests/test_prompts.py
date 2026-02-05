"""
Tests for MCP Server prompt endpoints.

Tests all 5 prompt templates:
- litreview_plan
- litreview_screening_criteria
- litreview_search_clarification
- synthesis_summary
- synthesis_extract_claims
"""

import pytest
from mcp_client.client import MCPClient, MCPError
from mcp_client.logging_utils import ResultLogger


@pytest.mark.asyncio
class TestPromptEndpoints:
    """Test MCP prompt retrieval endpoints."""
    
    async def test_get_litreview_plan_prompt(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting literature review plan prompt."""
        args = {
            "topic": "attention mechanisms in transformers",
            "scope": "comprehensive",
        }
        
        result_logger.log_request("prompts/get", {
            "name": "litreview_plan",
            "arguments": args,
        })
        
        try:
            response = await mcp_client.get_litreview_plan(
                topic=args["topic"],
                scope=args["scope"],
            )
            
            result_logger.log_response("litreview_plan", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            # Check for messages structure typically returned by prompts
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                assert isinstance(messages, list), "Messages should be a list"
                assert len(messages) > 0, "Should have at least one message"
            
            result_logger.log_test_result(
                "test_get_litreview_plan_prompt",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_litreview_plan_prompt",
                passed=False,
                error=e,
            )
            raise
    
    async def test_get_litreview_plan_with_time_range(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test literature review plan with time range."""
        args = {
            "topic": "deep learning",
            "scope": "focused",
            "time_range": "2020-2024",
        }
        
        result_logger.log_request("prompts/get", {
            "name": "litreview_plan",
            "arguments": args,
        })
        
        try:
            response = await mcp_client.get_litreview_plan(
                topic=args["topic"],
                scope=args["scope"],
                time_range=args["time_range"],
            )
            
            result_logger.log_response(
                "litreview_plan (with time_range)",
                response,
                success=True,
            )
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_get_litreview_plan_with_time_range",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_litreview_plan_with_time_range",
                passed=False,
                error=e,
            )
            raise
    
    async def test_get_litreview_screening_criteria_prompt(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting screening criteria prompt."""
        args = {
            "topic": "neural network pruning",
            "inclusion_rules": "peer-reviewed, published after 2018",
            "exclusion_rules": "preprints, non-English",
        }

        result_logger.log_request("prompts/get", {
            "name": "litreview_screening_criteria",
            "arguments": args,
        })

        try:
            response = await mcp_client.get_litreview_screening_criteria(
                topic=args["topic"],
                inclusion_rules=args["inclusion_rules"],
                exclusion_rules=args["exclusion_rules"],
            )
            
            result_logger.log_response(
                "litreview_screening_criteria",
                response,
                success=True,
            )
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_get_litreview_screening_criteria_prompt",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_litreview_screening_criteria_prompt",
                passed=False,
                error=e,
            )
            raise
    
    async def test_get_litreview_search_clarification_prompt(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting search clarification prompt."""
        args = {
            "search_mode": "keyword",
            "initial_search_string": "transformer models NLP",
        }
        
        result_logger.log_request("prompts/get", {
            "name": "litreview_search_clarification",
            "arguments": args,
        })
        
        try:
            response = await mcp_client.get_litreview_search_clarification(
                search_mode=args["search_mode"],
                initial_search_string=args["initial_search_string"],
            )
            
            result_logger.log_response(
                "litreview_search_clarification",
                response,
                success=True,
            )
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_get_litreview_search_clarification_prompt",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_litreview_search_clarification_prompt",
                passed=False,
                error=e,
            )
            raise
    
    async def test_get_synthesis_summary_prompt(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting synthesis summary prompt."""
        sample_abstracts = """
        Paper 1: We present a novel attention mechanism...
        Paper 2: This work explores transformer architectures...
        Paper 3: Our approach combines CNN and attention layers...
        """
        
        args = {
            "paper_abstracts": sample_abstracts,
            "goal": "identify common themes and methods",
            "tone": "academic",
        }
        
        result_logger.log_request("prompts/get", {
            "name": "synthesis_summary",
            "arguments": args,
        })
        
        try:
            response = await mcp_client.get_synthesis_summary(
                paper_abstracts=args["paper_abstracts"],
                goal=args["goal"],
                tone=args["tone"],
            )
            
            result_logger.log_response("synthesis_summary", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_get_synthesis_summary_prompt",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_synthesis_summary_prompt",
                passed=False,
                error=e,
            )
            raise
    
    async def test_get_synthesis_extract_claims_prompt(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test getting claim extraction prompt."""
        args = {
            "paper_id": 1,  # Placeholder paper ID
            "goal": "extract main contributions",
            "detail_level": "detailed",
        }
        
        result_logger.log_request("prompts/get", {
            "name": "synthesis_extract_claims",
            "arguments": args,
        })
        
        try:
            response = await mcp_client.get_synthesis_extract_claims(
                paper_id=args["paper_id"],
                goal=args["goal"],
                detail_level=args["detail_level"],
            )
            
            result_logger.log_response("synthesis_extract_claims", response, success=True)
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_get_synthesis_extract_claims_prompt",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_synthesis_extract_claims_prompt",
                passed=False,
                error=e,
            )
            raise
    
    async def test_get_prompt_generic(
        self,
        mcp_client: MCPClient,
        result_logger: ResultLogger,
    ):
        """Test generic prompt retrieval method."""
        prompt_name = "litreview_plan"
        args = {"topic": "machine learning", "scope": "brief"}
        
        result_logger.log_request("prompts/get", {
            "name": prompt_name,
            "arguments": args,
        })
        
        try:
            response = await mcp_client.get_prompt(name=prompt_name, arguments=args)
            
            result_logger.log_response(
                f"get_prompt({prompt_name})",
                response,
                success=True,
            )
            
            assert response is not None, "Response should not be None"
            
            result_logger.log_test_result(
                "test_get_prompt_generic",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_get_prompt_generic",
                passed=False,
                error=e,
            )
            raise
