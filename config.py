"""项目二的所有可调参数（集中管理，避免魔法数字散落在代码里）。

⭐ 为什么单独拆一个文件：
   1. 原来 6 个数字（5/8000/800/1/30）散在 agent.py 里，改一个要翻文件
   2. 这些数字都是"拍的"或"实测调的"，集中放便于标注来源
   3. test_guardrails.py 通过 agent.TOKEN_BUDGET = 50 这种模块属性修改，
      所以 agent.py 用【显式 import】把这些名字带进自己的命名空间，
      测试仍然能 patch（已验证）。
"""

import os

# ─────────────── ReAct 循环 ───────────────
MAX_STEPS = int(os.getenv("MAX_STEPS", "5"))
"""最多几圈。⭐ 这是唯一不依赖判断的硬上限（计数器），所以最可靠。"""

HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "3"))
"""多轮对话只保留最近几轮（1 轮 = 一问一答）。"""

# ─────────────── 防护参数 ───────────────
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "8000"))
"""token 预算。⚠️ 现在用【最后一圈的 input_tokens】判断（不是累加），
因为：
  · 累加值会虚高 2.6 倍（每圈都重发完整 messages）
  · 而且累加值和 MAX_STEPS 重叠 —— 步数已经锁死总量
  · 最后圈 input 反映的是"上下文膨胀程度"，这才需要单独防护
⚠️ 在 MAX_STEPS=5 + 截断 800 下，实际触发概率仍低；
   若将来放宽 MAX_STEPS，它就是主要防线。"""

TRUNCATE_LIMIT = int(os.getenv("TRUNCATE_LIMIT", "800"))
"""工具返回的长度上限（字符）。⭐ 现在按 JSON 字段截断，保证结构合法。"""

LOOP_WINDOW = int(os.getenv("LOOP_WINDOW", "1"))
"""死循环检测窗口。本项目三个工具都是确定性的，所以 window=1：
同样参数必然同样结果，重试毫无价值，应当立刻收口。
⚠️ 如果工具变成非确定性（查实时库存），应改成 2。"""

TIME_BUDGET = float(os.getenv("TIME_BUDGET", "30"))
"""一次对话的总时长上限（秒）。实测 2 圈 4.2s / 3 圈 10.6s / 5 圈约 13s，
所以 30 秒在 MAX_STEPS=5 下永不触发 —— 它只防"多圈累积超时"。"""

SINGLE_CALL_BUDGET = float(os.getenv("SINGLE_CALL_BUDGET", "20"))
"""⭐ 单次 llm.invoke 的耗时上限（秒）。

⚠️ 这道检查【不能打断】正在等待的调用 —— 它是在 invoke 返回【之后】才判断。
   所以如果某次 invoke 卡了 60 秒，用户还是得等 60 秒。
   它的价值是：① 记录慢调用（可观测）② 立刻止损，不再跑下一圈。
⭐ 真正能"打断"的做法是给 invoke 套线程池：
     ex.submit(llm.invoke, ...) → ex.result(timeout=30) → ex.shutdown(wait=False)
   ⚠️ 不能用 with ThreadPoolExecutor(...) —— __exit__ 会等所有任务结束，
      超时形同虚设（实测 3 秒超时实际等了 10 秒）。
"""

# ─────────────── 新增的两道上限 ───────────────
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "10"))
"""⭐ 工具调用【总次数】上限。

为什么需要它：MAX_STEPS 限的是"圈数"，但一圈可能调多个工具
（模型一次返回 3 个 tool_calls）。5 圈 × 3 = 15 次调用，
每次结果都塞进 messages → 上下文会爆。
所以需要一个独立的"总次数"上限。"""

MAX_TOOL_FAILS = int(os.getenv("MAX_TOOL_FAILS", "3"))
"""⭐ 工具【连续】失败次数上限。成功一次就清零。

为什么需要它：工具失败会返回 {"error": ...}，模型看到后会换策略重试。
如果一直失败（比如工具坏了），会一直重试到死循环检测才停 —— 白烧 token。
连续失败 3 次就收口，直接转人工。"""

# ─────────────── LLM 调用重试 ───────────────
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
"""⭐ LLM 调用最多重试几次（含首次）。原来一次失败就整个收口。"""

LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.0"))
"""重试的指数退避基数（秒）：第 1 次退避 1s，第 2 次 2s，第 3 次 4s。"""

# ─────────────── 工具返回值契约 ───────────────
REQUIRED_FIELDS = {
    # ⭐ 每个工具的返回值【必须】包含哪些字段 —— 缺字段就报错，不许静默返回半截数据
    "lookup_order": {"order_id", "status", "amount", "hours_since_pay", "overdue"},
    "search_kb": None,          # 纯文本，不校验
    "escalate_to_human": {"escalated", "reason"},
}

# ─────────────── 收口原因分类 ───────────────
# ⭐ 哪些 reason 需要人工跟进 —— 返回给上层，由 UI/API 决定建工单还是发告警
NEEDS_HUMAN_REASONS = frozenset({
    "escalated",        # 模型判断：用户需要人工（业务需求）
    "loop_detected",    # 技术异常：模型卡住
    "max_steps",        # 技术异常：步数用尽
    "timeout",          # 技术异常：总时长超限
    "slow_invoke",      # 技术异常：单次调用过慢
    "token_budget",     # 技术异常：上下文膨胀
    "tool_failed",      # 技术异常：工具连续失败
})
"""⭐ 技术异常 ≠ 业务需求，上层要分开处理：
· 业务需求（escalated）→ 建人工工单
· 技术异常 → 先进监控告警，人工处理不了 loop_detected 这种事"""