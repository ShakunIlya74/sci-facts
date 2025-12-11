from llm_utils.vLLM_server import local_qwen_llm
from langchain_core.messages import HumanMessage
import torch


if __name__ == "__main__":
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\n" + "=" * 60)
        print("WARNING: CUDA is not available!")
        print("=" * 60)
        print("This appears to be a system-level NVIDIA driver issue.")
        print("The driver is loaded but returning error 999 (CUDA_ERROR_UNKNOWN).")
        print("\nPossible solutions:")
        print("1. Reboot your system to reset the NVIDIA driver state")
        print("2. Check 'sudo dmesg | grep nvidia' for kernel errors")
        print("3. Try reloading NVIDIA modules (requires stopping X server)")
        print("\nCannot run vLLM without CUDA. Exiting...")
        print("=" * 60)
        exit(1)
    
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("Starting vLLM server...\n")
    
    with local_qwen_llm(
        # model="Qwen/Qwen2.5-3B-Instruct",
        model="Qwen/Qwen3-VL-4B-Instruct",
        host="127.0.0.1",
        port=8000,
        api_key="token-abc123",
        temperature=0.2,
        max_tokens=256,
        extra_args=["--max-model-len", "50000"],
    ) as llm:
        resp = llm.invoke([HumanMessage(content="Hello from Qwen via vLLM!")])
        print(resp.content)