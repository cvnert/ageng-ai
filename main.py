import json
import os
from pathlib import Path

import dashscope
from dashscope import Generation
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

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
    system_prompt = load_prompt()

    # 构建消息列表
    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.messages:
        if msg.role == "user":
            messages.append({"role": "user", "content": msg.content})

    def generate():
        completion_text = ""
        usage_info = None

        # 使用 DashScope 流式调用
        response = Generation.call(
            model=req.model,
            messages=messages,
            stream=True,
            result_format='message'
        )

        for chunk in response:
            if chunk.status_code == 200:
                # 获取输出内容
                output = chunk.output
                if output and output.choices:
                    choice = output.choices[0]
                    if choice.message and choice.message.content:
                        content = choice.message.content
                        completion_text += content
                        data = json.dumps({"content": content}, ensure_ascii=False)
                        yield f"data: {data}\n\n"

                # 获取 token 使用情况（在最后一个 chunk 中）
                if chunk.usage:
                    usage_info = {
                        "prompt_tokens": chunk.usage.input_tokens,
                        "completion_tokens": chunk.usage.output_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }
            else:
                # 错误处理
                error_msg = f"Error: {chunk.code}, {chunk.message}"
                data = json.dumps({"content": error_msg}, ensure_ascii=False)
                yield f"data: {data}\n\n"

        # 发送 token 统计信息
        if usage_info:
            usage_data = json.dumps({"usage": usage_info}, ensure_ascii=False)
            yield f"data: {usage_data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
