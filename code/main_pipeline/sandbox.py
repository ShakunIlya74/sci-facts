from llm_utils.vLLM_server import get_local_qwen_llm
from langchain_core.messages import HumanMessage

llm = get_local_qwen_llm(
    model="Qwen/Qwen3-4B-Instruct",
    host="localhost",
    port=8000,
    api_key="token-abc123",
)

resp = llm.invoke([HumanMessage(content="Hello from Qwen via vLLM!")])
print(resp.content)
