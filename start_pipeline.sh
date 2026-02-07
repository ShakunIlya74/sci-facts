#!/bin/bash
# Start both vLLM server and task listener
# Usage: ./start_pipeline.sh [--model MODEL] [--port PORT]

set -e

# Default values
MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-token-abc123}"
POLL_INTERVAL="${POLL_INTERVAL:-2.0}"
TASK_TYPES="${TASK_TYPES:-synthesis,agent_chat}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --task-types)
            TASK_TYPES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL           vLLM model to serve (default: Qwen/Qwen3-VL-4B-Instruct)"
            echo "  --host HOST             Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT             Port for vLLM server (default: 8000)"
            echo "  --api-key KEY           API key (default: token-abc123)"
            echo "  --poll-interval SECS    Task queue poll interval (default: 2.0)"
            echo "  --task-types TYPES      Comma-separated task types (default: synthesis,agent_chat)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Export environment variables
export VLLM_MODEL="$MODEL"
export VLLM_HOST="$HOST"
export VLLM_PORT="$PORT"
export VLLM_API_KEY="$API_KEY"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"

echo "============================================================"
echo "Starting LLM Pipeline"
echo "============================================================"
echo "Model: $MODEL"
echo "vLLM Server: $HOST:$PORT"
echo "Poll Interval: $POLL_INTERVAL seconds"
echo "Task Types: $TASK_TYPES"
echo "============================================================"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down..."
    
    # Kill the vLLM server process group
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    
    # Kill the task listener process group
    if [ -n "$LISTENER_PID" ]; then
        echo "Stopping task listener (PID: $LISTENER_PID)..."
        kill -TERM "$LISTENER_PID" 2>/dev/null || true
        wait "$LISTENER_PID" 2>/dev/null || true
    fi
    
    echo "Shutdown complete"
}

trap cleanup EXIT INT TERM

# Start vLLM server in background
echo "Starting vLLM server..."
cd "$CODE_DIR"
python -m llm_utils.start_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" &
VLLM_PID=$!

echo "vLLM server starting (PID: $VLLM_PID)"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600
WAIT_INTERVAL=5
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    
    # Check if process is still running
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    
    echo "Still waiting... ($WAITED/$MAX_WAIT seconds)"
    sleep $WAIT_INTERVAL
    WAITED=$((WAITED + WAIT_INTERVAL))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    exit 1
fi

# Start task listener
echo ""
echo "Starting task listener..."
python -m main_pipeline.task_listener \
    --poll-interval "$POLL_INTERVAL" \
    --task-types "$TASK_TYPES" &
LISTENER_PID=$!

echo "Task listener started (PID: $LISTENER_PID)"
echo ""
echo "============================================================"
echo "Pipeline is running!"
echo "Press Ctrl+C to stop"
echo "============================================================"

# Wait for either process to exit
wait -n "$VLLM_PID" "$LISTENER_PID"
EXIT_CODE=$?

echo "A process exited with code $EXIT_CODE"
exit $EXIT_CODE
