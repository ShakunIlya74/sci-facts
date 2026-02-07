#!/usr/bin/env python3
"""
vLLM Server Startup Script.

Starts the vLLM OpenAI-compatible server and keeps it running for LLM inference.
Does NOT manage the server lifecycle - just starts it and keeps it running.

Usage:
    python -m llm_utils.start_server
    
    Or directly:
    python code/llm_utils/start_server.py
    
Options via environment variables:
    VLLM_MODEL: Model to serve (default: Qwen/Qwen3-VL-4B-Instruct)
    VLLM_HOST: Host to bind to (default: 0.0.0.0)
    VLLM_PORT: Port to serve on (default: 8000)
    VLLM_API_KEY: API key for authentication (default: token-abc123)
    VLLM_DTYPE: Data type for model (default: auto)
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Optional

from llm_utils.vLLM_server import start_vllm_server, wait_for_vllm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start vLLM server and keep it running")
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
        help="Model to serve (default: Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("VLLM_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VLLM_PORT", "8000")),
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "token-abc123"),
        help="API key for authentication (default: token-abc123)",
    )
    parser.add_argument(
        "--dtype",
        default=os.getenv("VLLM_DTYPE", "auto"),
        help="Data type for model (default: auto)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout for server startup in seconds (default: 600)",
    )
    parser.add_argument(
        "--extra-args",
        nargs="*",
        default=["--max-model-len", "50000"],
        help="Additional arguments to pass to vLLM server (default: --max-model-len 50000)",
    )
    return parser.parse_args()


def main():
    """Main entry point for vLLM server startup."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting vLLM Server")
    logger.info("=" * 60)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Dtype: {args.dtype}")
    logger.info("=" * 60)
    
    # Start the server
    proc = start_vllm_server(
        model=args.model,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        dtype=args.dtype,
        extra_args=args.extra_args,
    )
    
    # Handle graceful shutdown
    shutdown_requested = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Wait for server to become ready
        # Use localhost for health check if binding to 0.0.0.0
        check_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        base_url = f"http://{check_host}:{args.port}"
        logger.info(f"Waiting for server to be ready at {base_url}...")
        
        wait_for_vllm(
            base_url=base_url,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        
        logger.info("=" * 60)
        logger.info("vLLM Server is ready!")
        logger.info(f"API endpoint: {base_url}/v1")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        
        # Keep the script running and monitor the server process
        while not shutdown_requested:
            # Check if process is still running
            poll_result = proc.poll()
            if poll_result is not None:
                logger.error(f"vLLM server exited with code {poll_result}")
                sys.exit(poll_result)
            
            time.sleep(1)
            
    except TimeoutError as e:
        logger.error(f"Server startup timeout: {e}")
        proc.terminate()
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        
    finally:
        logger.info("Shutting down vLLM server...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
            logger.info("Server shut down gracefully")
        except Exception:
            logger.warning("Force killing server...")
            proc.kill()


if __name__ == "__main__":
    main()
