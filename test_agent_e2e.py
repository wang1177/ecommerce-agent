"""冒烟检查（smoke test）· 改完代码后右键跑一次，30 秒知道有没有坏。

用法：PyCharm 里右键 → Run 'test_agent_e2e'
      （或 Shift+F10）

它会检查 6 件事：
  ① 能不能 import（语法/依赖有没有坏）
  ② 提示词长度（改提示词后看有没有生效）
  ③ 工具列表（加/删工具后看有没有注册上）
  ④ 数据库能不能连（订单数）
  ⑤ 真实跑一次，看工具序列和回答
  ⑥ 真实 token 消耗

注意：第 ⑤ 步会调真实 API（约 0.1 元/次）。
"""

import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

PASS, FAIL = [], []


def check(ok, msg):
    (PASS if ok else FAIL).append(msg)
    print(("  ✅ " if ok else "  ❌ ") + msg)


def main():
    print("=" * 74)
    print("  冒烟检查 · check_agent")
    print("=" * 74)

    # ── ① 能不能 import ──
    print("")
    print("【① import 检查】")
    try:
        import agent as A
        import tools as T
        check(True, "agent / tools 都能 import")
    except Exception as e:
        check(False, "import 失败：" + str(e))
        summary()
        return

    # ── ② 提示词长度 ──
    print("")
    print("【② 提示词】")
    sp_len = len(A.SYSTEM_PROMPT)
    print("    长度：" + str(sp_len) + " 字"
          "（原始 1611 字；删规则实验后是 1428 字）")
    check(sp_len > 800, "提示词长度正常（>800 字）")

    # ── ③ 工具列表 ──
    print("")
    print("【③ 工具注册】")
    names = [t.name for t in T.TOOLS]
    print("    已注册：" + str(names))
    check(len(names) >= 3, "至少注册了 3 个工具")
    check("lookup_order" in names, "lookup_order 已注册")
    check("search_kb" in names, "search_kb 已注册")

    # ── ④ 数据库 ──
    print("")
    print("【④ 数据库】")
    try:
        import sqlite3
        conn = sqlite3.connect(T.DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        conn.close()
        print("    订单数：" + str(n) + " 条")
        check(n > 0, "订单表可读，有 " + str(n) + " 条数据")
    except Exception as e:
        check(False, "数据库读不了：" + str(e))

    # ── ⑤ 真实跑一次 ──
    print("")
    print("【⑤ 真实跑一次】（约 5~15 秒，花钱约 0.1 元）")
    try:
        from langchain_community.chat_models import ChatTongyi
        llm = ChatTongyi(
            model="qwen-max", temperature=0.2,
            dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
        ).bind_tools(T.TOOLS)
        q = "ORD202407150005 怎么还没发货？"
        print("    问题：" + q)
        out = A.run_agent(q, llm)
        tool_seq = [t["name"] for t in out["trace"]]
        print("    工具序列：" + str(tool_seq))
        print("    圈数：" + str(out["steps"]) + "  收口：" + str(out["reason"]))
        print("    回答：" + (out["answer"] or "")[:100].replace("\n", " ") + "…")
        check(out["answer"] is not None and len(out["answer"]) > 10,
              "模型给出了回答")
        check(out["reason"] is None, "正常收口（reason=None）")

        # ── ⑥ token ──
        print("")
        print("【⑥ token】")
        u = out.get("usage") or {}
        if u:
            print("    真实 token：" + str(u.get("total"))
                  + "（in " + str(u.get("input"))
                  + " / out " + str(u.get("output")) + "）")
            check(u.get("total", 0) > 0, "token 采集正常")
        else:
            check(False, "没拿到 usage（mock 场景下正常）")
        print("    耗时：" + str(out.get("elapsed_sec")) + " 秒")
    except Exception as e:
        check(False, "真实调用失败：" + type(e).__name__ + ": " + str(e))

    summary()


def summary():
    print("")
    print("=" * 74)
    print("  结果：" + str(len(PASS)) + " 通过 / " + str(len(FAIL)) + " 失败")
    print("=" * 74)
    if FAIL:
        print("  ❌ 失败项：")
        for f in FAIL:
            print("     · " + f)
        print("")
        print("  有失败项 → 说明刚才的改动可能坏了，先修再继续。")
    else:
        print("  全绿 → 改动没破坏东西，可以继续。")


if __name__ == "__main__":
    main()
