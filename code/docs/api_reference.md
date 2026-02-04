# MCP Client API Reference

## MCPClient Class

The main client class for communicating with Scholar Inbox MCP Server.

### Initialization

```python
from mcp_client import MCPClient, Settings

# Using default settings
client = MCPClient()

# With custom settings
settings = Settings(
    mcp_server_url="http://localhost:8000/mcp",
    mcp_api_key="your-api-key",
    http_timeout=30.0,
)
client = MCPClient(settings=settings)
```

### Context Manager Usage

```python
async with MCPClient() as client:
    result = await client.papers_search_keyword("transformer")
    # Client automatically closes on exit
```

---

## Discovery Methods

### `initialize()`

Initialize connection with MCP server.

```python
response = await client.initialize(
    client_name="my-client",      # Optional, default: "sci-facts-mcp-client"
    client_version="1.0.0",       # Optional, default: "0.1.0"
)
```

**Returns**: Server capabilities and info

### `list_tools()`

List all available tools.

```python
response = await client.list_tools()
# response["tools"] -> list of tool definitions
```

### `list_resources()`

List all available resources.

```python
response = await client.list_resources()
# response["resources"] -> list of resource templates
```

### `list_prompts()`

List all available prompts.

```python
response = await client.list_prompts()
# response["prompts"] -> list of prompt definitions
```

---

## Paper Search Tools

### `papers_search_keyword()`

Search papers using keyword matching.

```python
response = await client.papers_search_keyword(
    query="attention mechanism",  # Required
    n_results=20,                 # Optional, default: 20, max: 100
    sources=["arxiv"],            # Optional, filter by sources
    min_citations=10,             # Optional, minimum citations
)
```

### `papers_search_semantic()`

Find similar papers by seed paper IDs.

```python
response = await client.papers_search_semantic(
    seed_paper_ids=[12345, 67890],  # Required, max: 50
    n_neighbors=20,                  # Optional, default: 20, max: 100
)
```

### `papers_search_by_text()`

Semantic search using text query embedding.

```python
response = await client.papers_search_by_text(
    query="neural network optimization",  # Required
    n_results=20,                          # Optional, default: 20
)
```

### `papers_get_details()`

Get detailed paper information.

```python
response = await client.papers_get_details(
    paper_ids=[12345, 67890],    # Required, max: 50
    include_abstracts=True,      # Optional, default: True
    include_summaries=False,     # Optional, default: False
)
```

---

## Topic Tools

### `topics_search()`

Search topics by name.

```python
response = await client.topics_search(
    query="machine learning",  # Required
    level="l3",                # Optional, topic level
    limit=10,                  # Optional, default: 10
)
```

### `topics_list()`

List all topics with filters.

```python
response = await client.topics_list(
    level="l3",          # Optional, filter by level
    min_papers=100,      # Optional, minimum paper count
    limit=100,           # Optional, default: 100, max: 200
)
```

### `topics_get_impactful_papers()`

Get top papers in a topic.

```python
response = await client.topics_get_impactful_papers(
    topic_name="deep learning",  # Required
    level="l3",                  # Optional
    page=1,                      # Optional, default: 1
    papers_per_page=10,          # Optional, default: 10
)
```

### `topics_get_hierarchy()`

Get topic with ancestor chain.

```python
response = await client.topics_get_hierarchy(
    topic_name="convolutional neural networks"  # Required
)
```

---

## Collection Tools

### `collections_create()`

Create a new collection.

```python
response = await client.collections_create(
    user_id=1,                    # Required
    name="My Research Papers",    # Required
    paper_ids=[12345, 67890],     # Optional, initial papers
)
```

### `collections_list()`

List user's collections.

```python
response = await client.collections_list(
    user_id=1  # Required
)
```

### `collections_get()`

Get collection details.

```python
response = await client.collections_get(
    collection_id=123  # Required
)
```

### `collections_add_items()`

Add papers to a collection.

```python
response = await client.collections_add_items(
    collection_id=123,            # Required
    paper_ids=[12345, 67890],     # Required, max: 100
)
```

### `collections_remove_item()`

Remove a paper from a collection.

```python
response = await client.collections_remove_item(
    collection_id=123,  # Required
    paper_id=12345,     # Required
)
```

### `collections_delete()`

Delete a collection.

```python
response = await client.collections_delete(
    collection_id=123  # Required
)
```

---

## Resource Methods

### `read_resource()`

Read any resource by URI.

```python
response = await client.read_resource(uri="paper://12345")
```

### `read_paper()`

Read paper resource.

```python
response = await client.read_paper(paper_id=12345)
```

### `read_collection()`

Read collection resource.

```python
response = await client.read_collection(collection_id=123)
```

### `read_topic()`

Read topic resource.

```python
response = await client.read_topic(topic_name="machine learning")
```

---

## Prompt Methods

### `get_prompt()`

Get any prompt by name.

```python
response = await client.get_prompt(
    name="litreview_plan",
    arguments={"topic": "deep learning", "scope": "comprehensive"},
)
```

### `get_litreview_plan()`

Get literature review plan prompt.

```python
response = await client.get_litreview_plan(
    topic="attention mechanisms",     # Required
    scope="comprehensive",            # Optional, default: "comprehensive"
    time_range="2020-2024",           # Optional
)
```

### `get_litreview_screening_criteria()`

Get screening criteria prompt.

```python
response = await client.get_litreview_screening_criteria(
    topic="neural networks",          # Required
    inclusion_rules="peer-reviewed",  # Optional
    exclusion_rules="preprints",      # Optional
)
```

### `get_litreview_search_clarification()`

Get search clarification prompt.

```python
response = await client.get_litreview_search_clarification(
    search_mode="keyword",            # Required
    initial_search_string="NLP",      # Required
)
```

### `get_synthesis_summary()`

Get synthesis summary prompt.

```python
response = await client.get_synthesis_summary(
    paper_abstracts="Paper 1: ...",   # Required
    goal="identify themes",           # Optional
    tone="academic",                  # Optional
)
```

### `get_synthesis_extract_claims()`

Get claim extraction prompt.

```python
response = await client.get_synthesis_extract_claims(
    paper_id=12345,                   # Required
    goal="main contributions",        # Optional
    detail_level="detailed",          # Optional
)
```

---

## Exception Handling

### MCPError

Error from MCP server.

```python
from mcp_client.client import MCPError

try:
    result = await client.papers_search_keyword("query")
except MCPError as e:
    print(f"Error: {e}")
    print(f"Code: {e.code}")
    print(f"Data: {e.data}")
```

### HTTP Errors

Network and HTTP errors raise `httpx` exceptions.

```python
import httpx

try:
    result = await client.papers_search_keyword("query")
except httpx.HTTPStatusError as e:
    print(f"HTTP {e.response.status_code}: {e.response.text}")
except httpx.ConnectError as e:
    print(f"Connection failed: {e}")
```
