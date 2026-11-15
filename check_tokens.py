"""查看一次 Agent 对话的真实 token 消耗（API 报的数，不是估算）。

用法：
    cd "D:\\Dev\\PycharmProjects\\e-commerce agent"
    .\\.venv\\Scripts\\python.exe check_tokens.py
    .\\.venv\\Scripts\\python.exe check_tokens.py "ORD202407150005 怎么还没发货？"

不传参数时，跑默认的三个问题做对比。
"""
import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import agent as A
import tools as T
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

LLM = ChatTongyi(model="qwen-max", temperature=0.2,
                 dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")).bind_tools(T.TOOLS)


def read_usage(resp):
    """从 API 响应里读真实 token（这是官方报的数，不是我算的）。"""
    tu = (resp.response_metadata or {}).get("token_usage") or {}
    return (tu.get("input_tokens", 0),
            tu.get("output_tokens", 0),
            tu.get("total_tokens", 0),
            (tu.get("prompt_tokens_details") or {}).get("cached_tokens", 0))


def trace_one(question: str):
    """完整跑一次对话，打印每一圈的真实 token。"""
    messages = [SystemMessage(content=A.SYSTEM_PROMPT), HumanMessage(content=question)]
    my_est = sum(A._count_tokens(m.content) for m in messages)
    tin = tout = 0
    rounds = 0

    print(f"\n{'=' * 74}")
    print(f"问题: {question}")
    print(f"{'=' * 74}")
    print(f"  {'圈':<4}{'真实input':>10}{'真实output':>12}{'缓存命中':>10}"
          f"{'我的估算累计':>14}")

    for rnd in range(1, A.MAX_STEPS + 1):
        resp = LLM.invoke(messages)
        pin, pout, _ptot, pcache = read_usage(resp)
        tin += pin
        tout += pout
        my_est += A._count_tokens(resp.content)
        rounds = rnd

        print(f"  第{rnd}圈 {pin:>10}{pout:>12}{pcache:>10}{my_est:>14}")

        messages.append(resp)
        if not getattr(resp, "tool_calls", None):
            break
        for tc in resp.tool_calls:
            r = A._truncate(T.TOOL_MAP[tc["name"]].invoke(tc["args"]))
            my_est += A._count_tokens(r)
            messages.append(ToolMessage(content=r, tool_call_id=tc["id"]))

    real = tin + tout
    print(f"  {'─' * 66}")
    print(f"  合计   {tin:>10}{tout:>12}{'':>10}{my_est:>14}")
    print(f"\n  ⭐ 真实账单 = {real} token   （转了 {rounds} 圈）")
    print(f"  ⭐ 我的估算 = {my_est} token")
    print(f"  ⭐ 真实 / 估算 = {real / my_est:.2f} 倍" if my_est else "")
    print(f"  ⭐ TOKEN_BUDGET = {A.TOKEN_BUDGET} → "
          f"{'真实已超出！' if real > A.TOKEN_BUDGET else '真实没超'}"
          f" / {'估算触发' if my_est > A.TOKEN_BUDGET else '估算不触发'}")
    return real, my_est


def main():
    print("=" * 74)
    print("真实 token 消耗查看器")
    print("=" * 74)
    print("\n⭐ 数据来源：每次 API 响应里的 response_metadata['token_usage']")
    print("   （阿里云自己报的数，不是 len//2 估的）")

    # 单次请求的固定开销（不花太多钱，只调 2 次）
    print(f"\n{'=' * 74}")
    print("先看固定开销（一次请求里有什么）")
    print(f"{'=' * 74}")
    plain = ChatTongyi(model="qwen-max", temperature=0.2,
                       dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
    i1, _, _, _ = read_usage(plain.invoke([HumanMessage(content="你好")]))
    i2, _, _, _ = read_usage(plain.invoke([SystemMessage(content=A.SYSTEM_PROMPT),
                                           HumanMessage(content="你好")]))
    i3, _, _, _ = read_usage(LLM.invoke([SystemMessage(content=A.SYSTEM_PROMPT),
                                         HumanMessage(content="你好")]))
    print(f"  只发 '你好'                        input = {i1:>5}")
    print(f"  + 系统提示词（{len(A.SYSTEM_PROMPT)} 字）        input = {i2:>5}"
          f"   ⭐ 系统提示词 = {i2 - i1}")
    print(f"  + 3 个工具 schema                  input = {i3:>5}"
          f"   ⭐ 工具 schema = {i3 - i2}")

    # 跑真实问题
    qs = sys.argv[1:] if len(sys.argv) > 1 else [
        "订单 ORD202407150001 到哪了？",
        "ORD202407150005 怎么还没发货？",
        "你们这群骗子",
    ]
    results = []
    for q in qs:
        results.append((q,) + trace_one(q))

    # 汇总
    print(f"\n{'=' * 74}")
    print("汇总")
    print(f"{'=' * 74}")
    print(f"  {'问题':<28}{'真实':>8}{'估算':>8}{'倍数':>8}")
    tot_real = tot_est = 0
    for q, real, est in results:
        tot_real += real
        tot_est += est
        print(f"  {q[:26]:<28}{real:>8}{est:>8}{real / est:>7.2f}x")
    print(f"  {'─' * 52}")
    print(f"  {'合计':<28}{tot_real:>8}{tot_est:>8}{tot_real / tot_est:>7.2f}x")
    print(f"\n  ⭐ 平均每条真实消耗: {tot_real / len(results):.0f} token")


if __name__ == "__main__":
    main()
