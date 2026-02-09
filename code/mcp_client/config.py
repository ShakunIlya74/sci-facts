"""
Configuration settings for MCP Client.

Supports environment variables and .env files for configuration.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Load .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in project root (parent of code/)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly


@dataclass
class Settings:
    """MCP Client configuration settings."""
    
    # MCP Server settings
    mcp_server_url: str = "http://localhost:8001/mcp"
    mcp_api_key: str = "test-api-key"
    
    # vLLM Server settings
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_api_key: str = "token-abc123"
    vllm_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    
    # LLM inference settings
    temperature: float = 0.2
    max_tokens: int = 256
    
    # HTTP client settings
    http_timeout: float = 30.0
    
    # Logging settings
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    log_response_max_chars: int = 1000
    
    @property
    def vllm_base_url(self) -> str:
        """Get vLLM server base URL."""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"
    
    @property
    def mcp_server_base_url(self) -> str:
        """Get MCP server base URL (without /mcp path)."""
        url = self.mcp_server_url
        if url.endswith('/mcp'):
            return url[:-4]
        return url.rstrip('/')
    
    @property
    def mcp_task_stream_url(self) -> str:
        """Get the SSE task stream endpoint URL."""
        return f"{self.mcp_server_base_url}/mcp/tasks/stream"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            mcp_server_url=os.getenv("MCP_SERVER_URL", cls.mcp_server_url),
            mcp_api_key=os.getenv("MCP_API_KEY", cls.mcp_api_key),
            vllm_host=os.getenv("VLLM_HOST", cls.vllm_host),
            vllm_port=int(os.getenv("VLLM_PORT", str(cls.vllm_port))),
            vllm_api_key=os.getenv("VLLM_API_KEY", cls.vllm_api_key),
            vllm_model=os.getenv("VLLM_MODEL", cls.vllm_model),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.temperature))),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(cls.max_tokens))),
            http_timeout=float(os.getenv("HTTP_TIMEOUT", str(cls.http_timeout))),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_dir=Path(os.getenv("LOG_DIR", "logs")),
            log_response_max_chars=int(os.getenv("LOG_RESPONSE_MAX_CHARS", "1000")),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
