import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

load_dotenv()

AVAILABLE_MODELS = [
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
]

app = FastAPI()


def load_prompt() -> str:
    """从 prompt.txt 加载系统提示词"""
    prompt_path = Path(__file__).parent / "prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def create_llm(model_name: str) -> ChatTongyi:
    """创建千问 LLM 实例"""
    return ChatTongyi(
        model=model_name,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]


@app.get("/models")
def get_models():
    return {"models": AVAILABLE_MODELS}


@app.post("/chat")
def chat(req: ChatRequest):
    llm = create_llm(req.model)
    system_prompt = load_prompt()

    langchain_messages = [SystemMessage(content=system_prompt)]
    for msg in req.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))

    # 计算输入 token（估算）
    prompt_text = system_prompt + "\n" + "\n".join([m.content for m in req.messages])
    prompt_tokens = len(prompt_text) // 4  # 粗略估算

    completion_text = ""

    def generate():
        nonlocal completion_text
        for chunk in llm.stream(langchain_messages):
            if chunk.content:
                completion_text += chunk.content
                data = json.dumps({"content": chunk.content}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        # 计算输出 token
        completion_tokens = len(completion_text) // 4  # 粗略估算
        total_tokens = prompt_tokens + completion_tokens

        # 发送 token 统计信息
        usage_data = json.dumps({
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }, ensure_ascii=False)
        yield f"data: {usage_data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
