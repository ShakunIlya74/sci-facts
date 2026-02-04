# Configuration Guide

## Overview

The MCP Client uses a `Settings` dataclass for configuration. Settings can be set via:
1. Direct instantiation
2. Environment variables
3. Defaults

## Settings Class

```python
from mcp_client.config import Settings, get_settings

# Get cached settings (uses env vars + defaults)
settings = get_settings()

# Create custom settings
settings = Settings(
    mcp_server_url="http://localhost:8000/mcp",
    mcp_api_key="your-api-key",
)
```

---

## All Configuration Options

### MCP Server Settings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `mcp_server_url` | `MCP_SERVER_URL` | `http://localhost:8000/mcp` | MCP server endpoint |
| `mcp_api_key` | `MCP_API_KEY` | `test-api-key` | Bearer token for auth |

### vLLM Server Settings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `vllm_host` | `VLLM_HOST` | `127.0.0.1` | vLLM server host |
| `vllm_port` | `VLLM_PORT` | `8001` | vLLM server port |
| `vllm_api_key` | `VLLM_API_KEY` | `token-abc123` | vLLM API key |
| `vllm_model` | `VLLM_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | Model name |

### LLM Inference Settings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `temperature` | `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `max_tokens` | `LLM_MAX_TOKENS` | `256` | Max output tokens |

### HTTP Client Settings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `http_timeout` | `HTTP_TIMEOUT` | `30.0` | Request timeout (seconds) |

### Logging Settings

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `log_level` | `LOG_LEVEL` | `INFO` | Logging level |
| `log_dir` | `LOG_DIR` | `logs` | Log file directory |
| `log_response_max_chars` | `LOG_RESPONSE_MAX_CHARS` | `1000` | Max chars for logged responses |

---

## Environment Variables

### Using .env File

Create a `.env` file in the project root:

```bash
# .env
MCP_SERVER_URL=http://scholarinbox.example.com/mcp
MCP_API_KEY=your-production-key
VLLM_HOST=gpu-server.local
VLLM_PORT=8001
LOG_LEVEL=DEBUG
```

Load with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()

from mcp_client.config import get_settings
settings = get_settings()
```

### Export Environment Variables

```bash
export MCP_SERVER_URL="http://localhost:8000/mcp"
export MCP_API_KEY="your-api-key"
export VLLM_HOST="127.0.0.1"
export VLLM_PORT="8001"
export LOG_LEVEL="DEBUG"
```

---

## Usage Examples

### Development Configuration

```python
dev_settings = Settings(
    mcp_server_url="http://localhost:8000/mcp",
    mcp_api_key="dev-key",
    http_timeout=60.0,  # Longer timeout for debugging
    log_level="DEBUG",
)
```

### Production Configuration

```python
prod_settings = Settings(
    mcp_server_url="https://api.scholarinbox.com/mcp",
    mcp_api_key=os.environ["MCP_API_KEY"],
    http_timeout=30.0,
    log_level="WARNING",
    log_response_max_chars=500,  # Less verbose logs
)
```

### Test Configuration

```python
test_settings = Settings(
    mcp_server_url="http://localhost:8000/mcp",
    mcp_api_key="test-api-key",
    http_timeout=10.0,  # Faster timeout for tests
    log_dir=Path("logs/tests"),
    log_level="DEBUG",
    log_response_max_chars=1000,
)
```

---

## Computed Properties

### `vllm_base_url`

Returns the full vLLM API URL:

```python
settings = Settings(vllm_host="127.0.0.1", vllm_port=8001)
print(settings.vllm_base_url)  # http://127.0.0.1:8001/v1
```

---

## Settings Caching

The `get_settings()` function is cached using `@lru_cache()`:

```python
from mcp_client.config import get_settings

# First call creates settings from env vars
settings1 = get_settings()

# Subsequent calls return same instance
settings2 = get_settings()
assert settings1 is settings2  # True
```

To clear cache (e.g., for testing):

```python
from mcp_client.config import get_settings

get_settings.cache_clear()
```

---

## Validation

Settings are validated at instantiation. No values are required, but you should verify critical settings:

```python
settings = get_settings()

# Check MCP server URL
assert settings.mcp_server_url.startswith("http"), "Invalid MCP URL"

# Check API key is set (not default)
if settings.mcp_api_key == "test-api-key":
    print("Warning: Using default API key")
```

---

## Best Practices

1. **Never commit secrets**: Use environment variables for API keys
2. **Use get_settings()**: Prefer the cached singleton for consistency
3. **Override for tests**: Create custom Settings for test isolation
4. **Log appropriately**: Use DEBUG for development, INFO/WARNING for production
5. **Tune timeouts**: Increase for slow networks, decrease for quick failure detection
