"""决策测试集：按【工具序列完全匹配】算准确率。

为什么这么设计
────────────
· 判据选"工具序列"而不是"答案相似度"：
  答案有十种说法不好量化，而"该不该调、调哪个"是二值的；
  且纯规则判定，避免 LLM judge 带来的口径漂移
  （另一个 RAG 项目踩过：换 judge 模型后基线分数整体下移）。

· 用例分 6 类，其中 F 类（无关问题）最有价值：
  它测的是"不该调的时候忍住没忍住"——
  Agent 的典型失败不是漏调，是过度调用（模型总想干点什么）。

· 结果同时存 JSON，便于"改 prompt 前 / 改 prompt 后"对比。

用法：
    python test_decision.py
"""

import io
import json
import os
import sys
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from langchain_community.chat_models import ChatTongyi

from agent import run_agent
from tools import TOOLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "decision_result.json")

LLM = ChatTongyi(
    model="qwen-max",
    temperature=0.2,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
).bind_tools(TOOLS)


# ═══════════════════ ① 用例：(类别, 输入, 期望工具序列) ═══════════════════
CASES = [
    # ── A 简单查询：应只调 lookup_order ──
    ("A", "订单 ORD202407150001 到哪了？",            ["lookup_order","query_logistics"]),
    ("A", "ORD202407150002 的物流到哪了",             ["lookup_order","query_logistics"]),
    # 特意选"0 天前付款"的 ORD011：它有 48 小时才进超时区，
    #  不会因为隔一天再跑就漂移成"该查政策"，保证这条只测"简单查询"这一个意图
    ("A", "ORD202407150011 什么时候发货",             ["lookup_order","search_kb"]),

    # ── B 多步：先查订单，发现超时后应再查政策 ──
    ("B", "ORD202407150005 怎么还没发货？",           ["lookup_order", "search_kb"]),
    ("B", "ORD202407150006 付款一周了还没发货",       ["lookup_order", "search_kb"]),
    ("B", "ORD202407150012 有质量问题怎么处理",       ["lookup_order", "search_kb"]),

    # ── C 号不全 / 不存在 / 未给号 ──
    # 号不全时 System Prompt 明确要求"直接反问，不要浪费一次调用去试"，
    #  所以期望是 [] 而不是 [lookup_order]（第 1、2 轮时是后者，第 3 轮改了 Prompt）
    ("C", "我买的耳机还没发货，订单号 ORD2023101 开头", []),
    ("C", "ORD999999999999 到哪了",                   ["lookup_order"]),   # 格式完整→该试
    ("C", "我的订单到哪了",                           []),                   # 未给号，不该编

    # ── D 政策咨询：不需要订单，只查政策 ──
    ("D", "买的东西有质量问题能退吗",                  ["search_kb"]),
    ("D", "已经签收 10 天了还能退吗",                  ["search_kb"]),
    ("D", "超时发货有补偿吗",                         ["search_kb"]),

    # ── E 情绪激烈：应转人工 ──
    ("E", "再不退款我就去投诉你们！",                  ["escalate_to_human"]),
    ("E", "你们这群骗子",                             ["escalate_to_human"]),
    ("E", "我要找媒体曝光你们",                        ["escalate_to_human"]),

    # ── F 无关问题：一个工具都不该调 ──
    ("F", "你们几点上班",                             []),
    ("F", "你们支持货到付款吗",                        []),
    ("F", "帮我写一首诗",                             []),
    ("F", "那它呢",                                   []),
    ("F", "你好",                                     []),
]

CAT_NAME = {
    "A": "简单查询（应调 1 个）",
    "B": "多步（应先查订单再查政策）",
    "C": "号不全 / 不存在 / 未给号",
    "D": "政策咨询（只查政策）",
    "E": "情绪激烈（应转人工）",
    "F": "无关问题（不该调工具）",
}


def main():
    print("=" * 82)
    print("决策测试集 · 按工具序列完全匹配判定")
    print(f"共 {len(CASES)} 条，6 类")
    print("=" * 82)

    rows = []
    t_all = time.time()

    for i, (cat, q, expect) in enumerate(CASES, 1):
        t0 = time.time()
        try:
            out = run_agent(q, LLM)
            actual = [t["name"] for t in out["trace"]]
            err = None
        except Exception as e:
            out = {"steps": 0, "trace": [], "reason": "exception"}
            actual = []
            err = f"{type(e).__name__}: {e}"

        dt = time.time() - t0
        ok = actual == expect
        rows.append({
            "cat": cat, "q": q, "expect": expect, "actual": actual,
            "ok": ok, "steps": out.get("steps", 0),
            "reason": out.get("reason"), "sec": round(dt, 1),
            "answer": (out.get("answer") or "")[:200], "error": err,
        })

        flag = "✅" if ok else "❌"
        print(f"{flag} [{cat}] {i:2}. {q[:30]:<32} "
              f"期望{str(expect):<36} 实际{actual}")
        if err:
            print(f"       异常: {err}")

    total_sec = time.time() - t_all

    # ═══════════════════ ② 指标 ═══════════════════
    n = len(rows)
    passed = sum(r["ok"] for r in rows)

    print("\n" + "=" * 82)
    print("指标")
    print("=" * 82)
    print(f"① 完全匹配率    : {passed}/{n} = {passed / n:.1%}")
    print(f"② 平均步数      : {sum(r['steps'] for r in rows) / n:.2f}")
    print(f"③ 平均耗时      : {sum(r['sec'] for r in rows) / n:.1f}s/条"
          f"   （总计 {total_sec / 60:.1f} 分钟）")

    print("\n   分类别准确率:")
    for cat in "ABCDEF":
        sub = [r for r in rows if r["cat"] == cat]
        if sub:
            p = sum(r["ok"] for r in sub)
            print(f"     {cat} {CAT_NAME[cat]:<26} {p}/{len(sub)}"
                  f"   {'✅' if p == len(sub) else ''}")

    # ④ 过度调用率：F 类该空却调了
    f_all = [r for r in rows if r["cat"] == "F"]
    f_over = [r for r in f_all if r["actual"]]
    print(f"\n④ 过度调用率    : {len(f_over)}/{len(f_all)}"
          f"（F 类：不该调工具却调了）")
    for r in f_over:
        print(f"     「{r['q']}」→ 调了 {r['actual']}")

    # ⑤ 漏调用率：A~E 该调却一个没调
    need = [r for r in rows if r["cat"] in "ABCDE"]
    miss = [r for r in need if not r["actual"]]
    print(f"\n⑤ 漏调用率      : {len(miss)}/{len(need)}"
          f"（A~E 类：该调工具却一个没调）")
    for r in miss:
        print(f"     「{r['q']}」→ 未调任何工具")

    # ═══════════════════ ③ 错例明细 ═══════════════════
    fails = [r for r in rows if not r["ok"]]
    if fails:
        print("\n" + "=" * 82)
        print(f"错例明细（{len(fails)} 条）")
        print("=" * 82)
        for r in fails:
            print(f"\n[{r['cat']}] {r['q']}")
            print(f"   期望: {r['expect']}")
            print(f"   实际: {r['actual']}   steps={r['steps']}  "
                  f"reason={r['reason']}")
            print(f"   回复: {r['answer'][:110]}")
    else:
        print("\n全部通过")

    # ═══════════════════ ④ 存 JSON（便于改前/改后对比）═══════════════════
    summary = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": n, "passed": passed,
        "match_rate": round(passed / n, 4),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "avg_sec": round(sum(r["sec"] for r in rows) / n, 1),
        "over_call_rate": f"{len(f_over)}/{len(f_all)}",
        "miss_call_rate": f"{len(miss)}/{len(need)}",
        "by_cat": {c: [sum(r["ok"] for r in rows if r["cat"] == c),
                       sum(1 for r in rows if r["cat"] == c)] for c in "ABCDEF"},
        "fails": [{"cat": r["cat"], "q": r["q"],
                   "expect": r["expect"], "actual": r["actual"]} for r in fails],
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已存: {RESULT_PATH}")

    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
