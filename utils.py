"""项目二的核心工具函数（Week 2 手打/默写产出）。

四个函数分别对应四道防护里的三道 + token 统计：
  _truncate    截断，防上下文膨胀
  _safe_call   异常降级，工具崩了不抛
  _is_looping  死循环检测
  _read_usage  读 API 真实 token 用量
"""

import json


from agent import SYSTEM_PROMPT as AGENT_PROMPT   # ⭐ 别名，避免后面被覆盖
from tools import TOOL_MAP


def _truncate(text, limit=800):
    """超长就截到 limit 字，并在末尾标注原文总字数。"""
    if len(text) <= limit:
        return text
    else:
        return text[:limit] + f"内容已截断，原文共{len(text)}字"


def _safe_call(name, args):
    """执行工具调用；任何情况都返回字符串，绝不抛异常。"""
    if name not in TOOL_MAP:
        return json.dumps({"error": f"没有这个工具，可用的是{list(TOOL_MAP)}"},
                          ensure_ascii=False)
    try:
        return TOOL_MAP[name].invoke(args or {})
    except Exception as e:
        return json.dumps({"error": f"工具调用失败：{type(e).__name__}: {e}"},
                          ensure_ascii=False)


def _is_looping(calls, window=1):
    """最近 window 次调用与再往前 window 次完全相同 → 判定死循环。"""
    if len(calls) < window * 2:
        return False
    sign = lambda c: (c["name"], json.dumps(c["args"], sort_keys=True, ensure_ascii=False))
    recent = [sign(c) for c in calls[-window:]]
    earlier = [sign(c) for c in calls[-2 * window:-window]]
    return recent == earlier


def _read_usage(resp):
    """读 API 返回的真实 token 用量，拿不到返回 None。"""
    meta = resp.response_metadata or {}
    tu = meta.get("token_usage") or {}
    if not tu:
        return None
    details = tu.get("prompt_tokens_details") or {}
    return (tu.get("input_tokens", 0),
            tu.get("output_tokens", 0),
            details.get("cached_tokens", 0))


from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from tools import TOOL_MAP

# ⭐ 练习用的简化 prompt（改名，不覆盖 AGENT_PROMPT）
PRACTICE_PROMPT=("你是一个电商客服助手，通过工具协助用户解决问题。你有三个工具：帮用户查订单，查政策，必要时转人工。")

def run_agent(question,llm,max_turns=5,system_prompt=None):
    # ⭐ 不传 system_prompt 就用项目的完整 prompt（1613 字，含 9 条规则）
    prompt = system_prompt or AGENT_PROMPT
    messages=[SystemMessage(content=prompt),HumanMessage(content=question)]
    calls=[]
    for step in range(max_turns):
        resp=llm.invoke(messages)
        messages.append(resp)
        if not resp.tool_calls:
            return resp.content
        for tc in resp.tool_calls:
            name=tc["name"]
            args=tc["args"]
            result=TOOL_MAP[name].invoke(args or {})
            messages.append(ToolMessage(content=result,tool_call_id=tc["id"]))
    return("超出步数上限，未能完成")
















# ─────────────── 自测 ───────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  _truncate")
    print("=" * 60)
    for txt, lim in [("短", 800), ("a" * 3000, 800), ("a" * 3000, 100),
                     ("a" * 800, 800), ("a" * 801, 800)]:
        r = _truncate(txt, lim)
        print(f"  输入 {len(txt):>5} 字 limit={lim:>4} → 输出 {len(r):>5} 字")

    print()
    print("=" * 60)
    print("  _safe_call")
    print("=" * 60)
    print(f"  [正常] {_safe_call('lookup_order', {'order_id': 'ORD202407150005'})[:70]}")
    print(f"  [不存在] {_safe_call('不存在的工具', {})}")
    print(f"  [异常] {_safe_call('lookup_order', {'order_id': '坏号'})[:70]}")
    print(f"  [args=None] 没崩 → {_safe_call('lookup_order', None)[:50]}")

    print()
    print("=" * 60)
    print("  _is_looping")
    print("=" * 60)
    t = lambda n, a: {"name": n, "args": a}
    cases = [
        ("连续两次相同", [t("lookup_order", {"order_id": "ORD001"}),
                     t("lookup_order", {"order_id": "ORD001"})], 1),
        ("参数不同", [t("lookup_order", {"order_id": "ORD001"}),
                  t("lookup_order", {"order_id": "ORD002"})], 1),
        ("只有1条", [t("lookup_order", {"order_id": "ORD001"})], 1),
        ("字典顺序反了", [t("lookup_order", {"order_id": "ORD001", "n": 1}),
                    t("lookup_order", {"n": 1, "order_id": "ORD001"})], 1),
        ("window=2 的ABAB", [t("a", {"x": 1}), t("b", {"y": 2}),
                          t("a", {"x": 1}), t("b", {"y": 2})], 2),
    ]
    for desc, calls, w in cases:
        print(f"  {desc:<18} window={w} → {_is_looping(calls, w)}")

    print()
    print("=" * 60)
    print("  _read_usage")
    print("=" * 60)

    class M:
        def __init__(self, meta):
            self.response_metadata = meta

    full = M({"token_usage": {"input_tokens": 1694, "output_tokens": 33,
                              "prompt_tokens_details": {"cached_tokens": 1024}}})
    no_detail = M({"token_usage": {"input_tokens": 100, "output_tokens": 5}})
    empty = M({"token_usage": {}})
    none_meta = M(None)
    print(f"  [完整]      {_read_usage(full)}")
    print(f"  [缺details] {_read_usage(no_detail)}")
    print(f"  [空usage]   {_read_usage(empty)}")
    print(f"  [meta=None] {_read_usage(none_meta)}")

    print()
    print("✅ 四个函数全部跑通")
