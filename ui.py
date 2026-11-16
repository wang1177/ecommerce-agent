"""Gradio 界面：左边聊天（含多轮历史），右边展示本次执行的 trace。

用法（在 main.py 里）：
    from ui import build_ui
    demo = build_ui(AGENT_LLM)
    app = gr.mount_gradio_app(app, demo, path="/")

设计要点
────────
· 不在这里 import main，改为把 llm 作为参数传进来 —— 避免循环导入。
· 用 gr.Blocks 手动布局（而不是 gr.ChatInterface），因为需要在右侧固定
  展示 trace；ChatInterface 只有单栏。
· 清空输入框靠 .then(lambda: "", None, msg) —— Gradio 的 Textbox
  提交后不会自动清空，必须显式置空。
"""

import json

import gradio as gr

from agent import run_agent

REASON_CN = {
    "loop_detected": "检测到重复调用，已收口",
    "token_budget": "超出 token 预算，已收口",
    "timeout": "超出时间上限，已收口",
    "max_steps": "达到最大步数，已收口",
    "escalated": "已转人工",
    "llm_error": "模型调用失败",
}

EXAMPLES = [
    "订单 ORD202407150001 到哪了？",
    "ORD202407150005 怎么还没发货？",
    "我上周三买的耳机还没发货，订单号 ORD2023101 开头",
    "买的东西有质量问题，能退吗？",
    "再不退款我就去投诉你们！",
    "我的订单到哪了",
]

EMPTY_TRACE = (
    '<div style="color:#999;font-size:13px;padding:24px 8px;text-align:center">'
    '发送消息后，这里会显示 Agent 的执行轨迹<br>'
    '<span style="font-size:12px">（每步调了什么工具 / 参数 / 返回什么）</span></div>'
)

CUSTOM_CSS = """
.gradio-container { max-width: 1280px !important; }
#tracebox { background:#fafafa; border:1px solid #eee; border-radius:8px;
            padding:12px; min-height:520px; overflow-y:auto; }
.tstep { background:#fff; border:1px solid #e5e5e5; border-left:3px solid #1677ff;
         border-radius:6px; padding:10px 12px; margin-bottom:10px; font-size:13px; }
.tstep .h { color:#1677ff; font-weight:600; margin-bottom:4px; }
.tstep .tool { color:#c41d7f; font-weight:600; }
.tstep .kv { color:#666; margin-top:4px; word-break:break-all; }
.tstep pre { background:#f7f7f7; border-radius:4px; padding:6px 8px; margin:4px 0 0;
             font-size:12px; white-space:pre-wrap; max-height:120px; overflow:auto; }
.tmeta { background:#f6ffed; border:1px solid #b7eb8f; border-radius:6px;
         padding:8px 10px; font-size:12px; color:#389e0d; margin-bottom:10px; }
.tmeta.warn { background:#fff1f0; border-color:#ffccc7; color:#cf1322; }
"""


def _esc(s):
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_trace(out: dict, history_turns: int | None = None) -> str:#视图函数，不计算任何东西，只做“数据-视图”的转换
    """把 run_agent 的返回渲染成 HTML 卡片。"""
    steps = out.get("steps", 0)
    trace = out.get("trace") or []
    reason = out.get("reason")
    used = out.get("history_turns_used")

    bits = [f'共 <b>{steps}</b> 步', f'工具调用 <b>{len(trace)}</b> 次']
    if used is not None:
        bits.append(f'带入历史 <b>{used}</b> 轮')
    if history_turns is not None:
        bits.append(f'当前会话共 <b>{history_turns}</b> 轮')

    # ⭐ 真实 token（API 报的数）；拿不到就退回估算值
    usage = out.get("usage")
    if usage and usage.get("total"):
        bits.append(
            f'真实 token <b>{usage["total"]}</b>'
            f'（in {usage.get("input", 0)} / out {usage.get("output", 0)}'
            + (f' / 缓存 {usage["cached"]}' if usage.get("cached") else "")
            + "）")
        est = out.get("est_tokens")
        if est:
            bits.append(f'估算 <b>{est}</b>')
    elif out.get("est_tokens"):
        bits.append(f'估算 token <b>{out["est_tokens"]}</b>')

    # ⭐ 实际耗时（秒）
    if out.get("elapsed_sec") is not None:
        bits.append(f'耗时 <b>{out["elapsed_sec"]}s</b>')

    if reason:
        bits.append(f'<b>{REASON_CN.get(reason, reason)}</b>')

    head = (f'<div class="tmeta{"" if reason in (None, "escalated") else " warn"}">'
            + " · ".join(bits) + "</div>")

    if not trace:
        return head + ('<div style="color:#999;font-size:13px;padding:12px 4px">'
                       '本次未调用任何工具（直接回答）</div>')

    cards = []
    for t in trace:
        args = json.dumps(t.get("args") or {}, ensure_ascii=False)
        result = str(t.get("result") or "")
        if len(result) > 320:
            result = result[:320] + "…"
        cards.append(
            f'<div class="tstep">'
            f'<div class="h">第 {t.get("step", 0) + 1} 步</div>'
            f'<div class="tool">→ {_esc(t.get("name"))}</div>'
            f'<div class="kv">参数：{_esc(args)}</div>'
            f'<div class="kv">返回：<pre>{_esc(result)}</pre></div>'
            f'</div>')
    return head + "".join(cards)


def build_ui(agent_llm, max_turns: int = 3):
    def respond(message, history):
        """发送一条消息：返回 (更新后的 history, trace HTML)。

        ⭐ history 是 Gradio 传进来的【之前】的对话（不含当前这条），
           我们把它一并交给 run_agent，这样模型就能理解"它""那单"这类指代。
        """
        history = history or []
        if not (message or "").strip():
            return history, render_trace({"steps": 0, "trace": [], "reason": None})

        try:
            out = run_agent(message, agent_llm, history=history,
                            max_turns=max_turns)
            answer = out.get("answer") or "（未生成回复）"
        except Exception as e:                       # 兜底：UI 不能白屏
            out = {"steps": 0, "trace": [], "reason": "llm_error"}
            answer = f"抱歉，系统异常：{type(e).__name__}: {e}"

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return history, render_trace(out, history_turns=len(history) // 2)

    def use_example(text):
        return text

    def reset():
        return [], EMPTY_TRACE

    with gr.Blocks(title="电商客服 Agent", css=CUSTOM_CSS,
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 电商客服 Agent\n"
            "ReAct 循环 · 4 个工具（查订单 / 查政策 / 查物流 / 转人工） · 四道防护 · "
            "**左侧对话，右侧执行轨迹**")

        with gr.Row():
            # ── 左：聊天区 ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话", height=520,
                    buttons=["copy"],          # Gradio 6：用 buttons 取代 show_copy_button
                )
                msg = gr.Textbox(
                    placeholder="输入用户反馈，回车发送 / Shift+回车换行…",
                    show_label=False,
                    # ⭐ lines=1：单行输入框，【回车】一定触发提交；
                    #    如果写成 lines=2（多行 textarea），部分版本里回车会被
                    #    当成换行吃掉。max_lines 让它随内容自动长高。
                    lines=1, max_lines=6, autoscroll=True)
                with gr.Row():
                    submit = gr.Button("发送", variant="primary", scale=2)
                    clear = gr.Button("清空", scale=1)

            # ── 右：轨迹区 ──
            with gr.Column(scale=2):
                trace_box = gr.HTML(EMPTY_TRACE, elem_id="tracebox")
                gr.Examples(examples=EXAMPLES, inputs=msg, label="点一下试这些场景")

        # 发送（按钮 + 回车），随后清空输入框
        for trigger in (submit.click, msg.submit):
            trigger(respond, inputs=[msg, chatbot], outputs=[chatbot, trace_box]) \
                .then(lambda: "", outputs=msg)
        clear.click(reset, outputs=[chatbot, trace_box])

    return demo


if __name__ == "__main__":
    # 单独调试 UI：python ui.py
    import os
    from langchain_community.chat_models import ChatTongyi
    from tools import TOOLS

    llm = ChatTongyi(model="qwen-max", temperature=0.2,
                     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
    build_ui(llm.bind_tools(TOOLS)).launch(server_port=7860)
