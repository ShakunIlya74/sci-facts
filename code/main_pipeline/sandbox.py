from llm_utils.vLLM_server import local_qwen_llm
from langchain_core.messages import HumanMessage


if __name__ == "__main__":
    with local_qwen_llm(
        # model="Qwen/Qwen2.5-3B-Instruct",
        model="Qwen/Qwen3-VL-4B-Instruct",
        host="127.0.0.1",
        port=8000,
        api_key="token-abc123",
        temperature=0.2,
        max_tokens=256,
    ) as llm:
        resp = llm.invoke([HumanMessage(content="Hello from Qwen via vLLM!")])
        print(resp.content)