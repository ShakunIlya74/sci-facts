# Testing Guide

## Overview

The MCP Client test suite verifies connectivity and functionality with the Scholar Inbox MCP Server. Tests are organized by endpoint type and include comprehensive logging for debugging.

## Running Tests

### Prerequisites

1. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio httpx
   ```

2. Ensure MCP server is running at `http://localhost:8001/mcp`

3. Set environment variables (optional):
   ```bash
   export MCP_SERVER_URL="http://localhost:8001/mcp"
   export MCP_API_KEY="your-api-key"
   ```

### Run All Tests

```bash
# From project root
cd /home/scholar/Projects/sci-facts
pytest
```

### Run Specific Test Files

```bash
# Discovery endpoints only
pytest code/mcp_client/tests/test_discovery.py

# Tool endpoints only
pytest code/mcp_client/tests/test_tools.py

# Resource endpoints only
pytest code/mcp_client/tests/test_resources.py

# Prompt endpoints only
pytest code/mcp_client/tests/test_prompts.py
```

### Run Specific Test Classes

```bash
# Paper search tools
pytest code/mcp_client/tests/test_tools.py::TestPaperSearchTools

# Topic tools
pytest code/mcp_client/tests/test_tools.py::TestTopicTools

# Collection tools
pytest code/mcp_client/tests/test_tools.py::TestCollectionTools
```

### Run with Verbose Output

```bash
pytest -v --tb=long
```

### Run with Log Capture

```bash
pytest --log-cli-level=DEBUG
```

---

## Test Organization

### Test Files

| File | Coverage |
|------|----------|
| `test_discovery.py` | `initialize`, `tools/list`, `resources/list`, `prompts/list` |
| `test_tools.py` | All 14 tools (papers, topics, collections) |
| `test_resources.py` | All 3 resource URI patterns |
| `test_prompts.py` | All 5 prompt templates |

### Test Markers

```python
@pytest.mark.asyncio      # Async test
@pytest.mark.skip         # Skip test
@pytest.mark.skipif       # Conditional skip
```

---

## Logging

### Log Files

Test logs are saved to `logs/tests/` with timestamps:

```
logs/tests/mcp_endpoints_test_20260204_143022.log
```

### Log Format

```
2026-02-04 14:30:22 | INFO     | mcp_client_test | REQUEST: tools/call
2026-02-04 14:30:22 | DEBUG    | mcp_client_test |   Params: {"name": "papers_search_keyword", ...}
2026-02-04 14:30:23 | INFO     | mcp_client_test | RESPONSE: papers_search_keyword - SUCCESS (original: 2543 chars)
2026-02-04 14:30:23 | DEBUG    | mcp_client_test |   Content:
{"papers": [...]}
... [TRIMMED: 1543 more chars]
```

### Response Trimming

Responses are trimmed to 1000 characters by default to prevent log bloat. Configure with:

```bash
export LOG_RESPONSE_MAX_CHARS=2000
```

Or in test settings:

```python
settings = Settings(log_response_max_chars=2000)
```

---

## Fixtures

### Available Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_settings` | session | Configuration settings |
| `test_logger` | session | Logging instance |
| `result_logger` | session | Test result logger with trimming |
| `mcp_client` | function | MCPClient instance (auto-cleanup) |
| `sample_paper_query` | session | Default query: "transformer attention mechanism" |
| `sample_topic_query` | session | Default query: "machine learning" |
| `sample_user_id` | session | Default user ID: 1 |

### Custom Fixtures

Add to `conftest.py`:

```python
@pytest.fixture(scope="session")
def my_custom_fixture():
    return {"custom": "data"}
```

---

## Writing Tests

### Basic Test Structure

```python
import pytest
from mcp_client.client import MCPClient, MCPError
from mcp_client.logging_utils import TestResultLogger

@pytest.mark.asyncio
class TestMyFeature:
    
    async def test_my_endpoint(
        self,
        mcp_client: MCPClient,
        result_logger: TestResultLogger,
    ):
        """Test description."""
        # Log request
        result_logger.log_request("method_name", {"params": "here"})
        
        try:
            # Make request
            response = await mcp_client.some_method()
            
            # Log response
            result_logger.log_response("method_name", response, success=True)
            
            # Assertions
            assert response is not None
            
            # Log test result
            result_logger.log_test_result(
                "test_my_endpoint",
                passed=True,
                response=response,
            )
            
        except MCPError as e:
            result_logger.log_test_result(
                "test_my_endpoint",
                passed=False,
                error=e,
            )
            raise
```

### Graceful Skipping

For tests that may fail due to missing data:

```python
async def test_requires_data(self, mcp_client, result_logger):
    try:
        response = await mcp_client.get_paper_details([12345])
        assert response is not None
    except MCPError as e:
        pytest.skip(f"Paper not found: {e}")
```

---

## Debugging Tips

### 1. Check Server Connectivity

```python
import httpx

async def check_server():
    async with httpx.AsyncClient() as client:
        r = await client.get("http://localhost:8000/health")
        print(r.json())
```

### 2. View Full Responses

Temporarily increase max chars:

```python
@pytest.fixture(scope="session")
def test_settings():
    return Settings(log_response_max_chars=10000)
```

### 3. Run Single Test with Full Debug

```bash
pytest -v -s --log-cli-level=DEBUG \
    code/mcp_client/tests/test_tools.py::TestPaperSearchTools::test_papers_search_keyword
```

### 4. Check Log Files

```bash
tail -f logs/tests/mcp_endpoints_test_*.log
```

---

## Test Coverage Summary

### Discovery Endpoints (4)
- ✅ `initialize`
- ✅ `tools/list`
- ✅ `resources/list`
- ✅ `prompts/list`

### Paper Search Tools (4)
- ✅ `papers_search_keyword`
- ✅ `papers_search_keyword` (with sources)
- ✅ `papers_search_semantic`
- ✅ `papers_search_by_text`
- ✅ `papers_get_details`

### Topic Tools (4)
- ✅ `topics_search`
- ✅ `topics_list`
- ✅ `topics_list` (with level filter)
- ✅ `topics_get_impactful_papers`
- ✅ `topics_get_hierarchy`

### Collection Tools (6)
- ✅ `collections_create`
- ✅ `collections_list`
- ✅ `collections_get`
- ✅ `collections_add_items`
- ✅ `collections_remove_item`
- ✅ `collections_delete`

### Resources (3)
- ✅ `paper://{id}`
- ✅ `collection://{id}`
- ✅ `topic://{name}`

### Prompts (5)
- ✅ `litreview_plan`
- ✅ `litreview_plan` (with time_range)
- ✅ `litreview_screening_criteria`
- ✅ `litreview_search_clarification`
- ✅ `synthesis_summary`
- ✅ `synthesis_extract_claims`
