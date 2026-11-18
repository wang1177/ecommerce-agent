"""统一入口：Gradio UI + HTTP API 共用一个服务。

启动：python main.py
访问：
    http://127.0.0.1:8001         → Gradio 界面（左侧对话 / 右侧 trace）
    http://127.0.0.1:8001/chat    → HTTP API（POST {"content": "..."}）
    http://127.0.0.1:8001/health  → 健康检查
"""

import os

import gradio as gr
from fastapi import FastAPI
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel

from agent import run_agent
from tools import TOOLS
from ui import build_ui

# ─────────────── 模型（全局只建一次）───────────────
_llm = ChatTongyi(
    model="qwen-max",
    temperature=0.2,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
AGENT_LLM = _llm.bind_tools(TOOLS)

# ─────────────── FastAPI（只放 API，页面交给 Gradio）───────────────
api = FastAPI(title="电商客服 Agent", docs_url="/docs")


class ChatRequest(BaseModel):
    content: str               #必填，用户的话
    # 可选：带上之前的对话，让模型能理解"它""那单"这类指代
    history: list[dict] | None = None
    max_turns: int = 3         #可选，最大轮数，默认3


class ChatResponse(BaseModel):
    answer: str              #必填，模型的回答
    trace: list              #必填，每步工具调用记录
    steps: int               #必填，模型的轮数
    reason: str | None = None  #可选，模型的拒绝理由
    # 是否需要人工跟进
    #  agent.py 的 finish() 已经返回这个字段，但【必须在这里声明】
    #  否则 FastAPI 按 response_model 序列化时会【静默丢弃】
    needs_human: bool = False
    # 请求 ID（一次请求的所有日志用它串起来）
    request_id: str | None = None
    history_turns_used: int | None = None  #可选，历史对话轮数
    # 估算 token（len//2，已知低估 3.5~4.5 倍）
    est_tokens: int | None = None
    # 真实 token：API 返回的 usage（input/output/cached/total）
    usage: dict | None = None
    # 实际耗时（秒）
    elapsed_sec: float | None = None
    tool_names:list[str]=[]


@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Agent 主接口。用 def 而非 async def —— 内部是同步阻塞调用，
    交给 FastAPI 线程池执行，避免堵事件循环。"""
    return run_agent(req.content, AGENT_LLM, history=req.history,
                     max_turns=req.max_turns)


@api.get("/health")
def health():
    return {"status": "ok", "tools": [t.name for t in TOOLS]}


# ─────────────── 把 Gradio 挂到 "/" ───────────────
demo = build_ui(AGENT_LLM)
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
