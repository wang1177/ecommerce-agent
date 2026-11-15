"""四道防护的单元测试（不调用真实 API，用 mock 模型）。

验证：
  ① 死循环检测  —— 连续两轮相同调用会被收口
  ② 结果截断    —— 超长工具返回被截断并标注
  ③ 异常兜底    —— 不存在的工具 / 工具抛异常都返回 error 而不崩
  ④ token 预算  —— 超预算提前收口
"""
import io
import sys
import types

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import agent as A


# ────────── mock 模型 ──────────
class FakeMsg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """按预设脚本依次返回；脚本用完后返回最终答复。"""

    def __init__(self, script):
        self.script = list(script)
        self.n = 0

    def invoke(self, messages):
        self.n += 1
        if self.script:
            return self.script.pop(0)
        return FakeMsg(content="（脚本用尽）")


def tc(name, args, tid=None):
    return {"name": name, "args": args, "id": tid or f"call_{name}_{id(args)}"}


PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {msg}")
    else:
        FAIL += 1
        print(f"  ❌ {msg}")


# ═══════════ ① 死循环检测 ═══════════
print("=" * 70)
print("① 死循环检测")
print("=" * 70)

# 构造：连续 4 次用完全相同的参数调 lookup_order
same = [tc("lookup_order", {"order_id": "ORD202407150001"}) for _ in range(10)]
fake = FakeLLM([FakeMsg(tool_calls=[c]) for c in same])
out = A.run_agent("死循环测试", fake)
print(f"  实际步数: {out['steps']}  reason: {out['reason']}")
print(f"  答案: {out['answer']}")
check(out["reason"] == "loop_detected", "被判定为死循环并收口")
check(out["steps"] <= 3, f"在 3 步内收口（实际 {out['steps']} 步）——window=1 比 window=2 早 2 步")

# 反例：参数不同，不应该被误判
diff = [FakeMsg(tool_calls=[tc("lookup_order", {"order_id": f"ORD20240715000{i}"})])
        for i in range(1, 4)]
diff.append(FakeMsg(content="好的，已处理完毕。"))
fake2 = FakeLLM(diff)
out2 = A.run_agent("参数不同测试", fake2)
print(f"  反例（参数不同）: 步数 {out2['steps']}  reason: {out2['reason']}")
check(out2["reason"] is None, "参数不同时不被误判为死循环")

# ═══════════ ② 截断 ═══════════
print()
print("=" * 70)
print("② 结果截断")
print("=" * 70)
long_text = "啊" * 3000
truncated = A._truncate(long_text, limit=800)
print(f"  原文 {len(long_text)} 字 → 截断后 {len(truncated)} 字")
print(f"  尾部标注: ...{truncated[-30:]}")
check(len(truncated) < 900, "长度被截到 800 附近")
check("已截断" in truncated and "3000" in truncated, "标注了'已截断'和原长")
check(A._truncate("短文本") == "短文本", "短文本不被截断")

# search_kb 的真实返回值（政策全文）是否会被截断？
from tools import search_kb
policy = search_kb.invoke({"query": "全部政策"})
print(f"\n  政策全文长度: {len(policy)} 字 → 截断后 {len(A._truncate(policy))} 字")
check(len(policy) <= 800, "当前政策全文在 800 字以内，不会被截断（说明截断阈值合适）")

# ═══════════ ③ 异常兜底 ═══════════
print()
print("=" * 70)
print("③ 异常兜底")
print("=" * 70)

r1 = A._safe_call("不存在的工具", {})
print(f"  不存在的工具 → {r1}")
check("error" in r1 and "不存在" in r1, "不存在的工具名返回 error 而不是抛异常")

# 构造一个会抛异常的工具
import tools as T
orig = T.TOOL_MAP.get("search_kb")


def boom(query):
    raise RuntimeError("模拟数据库连接失败")


T.TOOL_MAP["search_kb"] = types.SimpleNamespace(invoke=boom)
r2 = A._safe_call("search_kb", {"query": "x"})
print(f"  工具内部抛异常 → {r2}")
check("error" in r2 and "RuntimeError" in r2, "工具抛异常时被兜住并返回 error")

# 端到端：工具抛异常后 Agent 仍能给出回复
fake3 = FakeLLM([
    FakeMsg(tool_calls=[tc("search_kb", {"query": "发货时效"})]),
    FakeMsg(content="抱歉，暂时查不到政策，我帮您转人工。"),
])
out3 = A.run_agent("异常场景", fake3)
print(f"  端到端: 步数 {out3['steps']}  reason: {out3['reason']}")
print(f"  答案: {out3['answer']}")
check(out3["reason"] is None and out3["steps"] == 2, "工具异常后 Agent 没有崩，正常走完第 2 步")

T.TOOL_MAP["search_kb"] = orig  # 还原

# ═══════════ ④ token 预算 ═══════════
print()
print("=" * 70)
print("④ token 预算")
print("=" * 70)
old_budget = A.TOKEN_BUDGET
A.TOKEN_BUDGET = 50          # 临时调小，方便触发
fake4 = FakeLLM([FakeMsg(tool_calls=[tc("search_kb", {"query": "政策"})])
                 for _ in range(10)])
out4 = A.run_agent("token 预算测试", fake4)
print(f"  预算 50 时: 步数 {out4['steps']}  reason: {out4['reason']}")
check(out4["reason"] == "token_budget", "超预算时以 token_budget 收口")
A.TOKEN_BUDGET = old_budget

# ═══════════ ⑤ 最大步数兜底 ═══════════
print()
print("=" * 70)
print("⑤ 最大步数兜底（第五道，顺带验证）")
print("=" * 70)
# 每步参数都不同，不会被死循环检测拦住，但会跑满 5 步
never_end = [FakeMsg(tool_calls=[tc("lookup_order", {"order_id": f"ORD2024071500{i:02d}"})])
             for i in range(1, 20)]
fake5 = FakeLLM(never_end)
out5 = A.run_agent("最大步数测试", fake5)
print(f"  步数 {out5['steps']}  reason: {out5['reason']}")
print(f"  答案: {out5['answer']}")
check(out5["reason"] == "max_steps" and out5["steps"] == A.MAX_STEPS,
      f"跑满 {A.MAX_STEPS} 步后以 max_steps 收口")

# ═══════════ ⑥ 时间超时 ═══════════
print()
print("=" * 70)
print("⑥ 时间超时兜底（和 token 正交的维度）")
print("=" * 70)


class SlowLLM:
    """每次 invoke 睡 0.15 秒的假模型，用来触发时间闸。"""

    def __init__(self):
        self.i = 0

    def invoke(self, messages):
        import time as _t
        _t.sleep(0.15)
        self.i += 1
        if self.i <= 10:
            return FakeMsg(tool_calls=[tc("lookup_order",
                                          {"order_id": f"ORD2024071500{self.i:02d}"})])
        return FakeMsg(content="答完了")


old_time = A.TIME_BUDGET
A.TIME_BUDGET = 0.3            # 临时调小，0.15s × 2 次就会超

out6 = A.run_agent("时间超时测试", SlowLLM())
print(f"  TIME_BUDGET=0.3 时: 步数 {out6['steps']}  reason: {out6['reason']}  "
      f"耗时 {out6['elapsed_sec']}s")
check(out6["reason"] == "timeout", "超过时间预算时以 timeout 收口")
check(0 < out6["steps"] < A.MAX_STEPS, "在跑满步数之前就收口了")

A.TIME_BUDGET = old_time
out6b = A.run_agent("时间超时测试", SlowLLM())
print(f"  TIME_BUDGET={old_time} 时: 步数 {out6b['steps']}  reason: {out6b['reason']}")
check(out6b["reason"] != "timeout", "预算正常时不误判为超时")

check(out6.get("elapsed_sec") is not None, "返回值带实际耗时 elapsed_sec")

# ═══════════ 汇总 ═══════════
print()
print("=" * 70)
print(f"结果：{PASS} 通过 / {FAIL} 失败")
print("=" * 70)
sys.exit(1 if FAIL else 0)
