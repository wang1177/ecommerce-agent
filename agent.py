"""ReAct 循环 + 六道防护。

六道防护及其位置（2026-11-15 重构后）
──────────────────────────────────
【闸】每圈开头检查：超了就收口，不再继续
  ① 死循环检测      _is_looping()     轮间 + ⭐ 同圈内（一圈多个相同调用）
  ④ token 预算      TOKEN_BUDGET      ⭐ 改用【最后一圈 input】（不再累加）
  ⑤ 时间守卫        TIME_BUDGET       轮间总时长
  ⑤b ⭐ 单次耗时    SINGLE_CALL_BUDGET 一次 invoke 超过 20s → 止损

【闸】计数上限
  ⑥ MAX_STEPS       for 循环范围       唯一不依赖判断的硬上限
  ⑦ ⭐ MAX_TOOL_CALLS  工具总次数（步行不代表调用少）
  ⑧ ⭐ MAX_TOOL_FAILS  工具连续失败（失败会一直重试，白烧 token）

【降级】不终止，继续跑
  ② 工具结果截断    _truncate()       ⭐ 按 JSON 字段截，保证结构合法
  ③ 工具异常兜底    _safe_call()      ⭐ 内含 schema 校验（缺字段就报错）

设计说明
────────
· 为什么 window=1：本项目三个工具都是【确定性】的（SQLite 查询、读文件、
  返回固定标记），同样参数必然得到同样结果 —— 所以模型用相同参数重试一次
  毫无价值，应当立刻收口。window=1 能在第 3 步就拦下，比 window=2 省 2 次
  模型调用。
  ⚠️ 如果工具是非确定性的（查实时库存、调外部 API 有波动），这里应改成
  window=2 或给不同工具设不同权重 —— 因为那时"重试一次"确实可能有意义。
· ⭐ 为什么截断要按 JSON 字段截：工具返回的是 JSON 字符串，按字符数硬切会把它
  切残（变成非法 JSON）→ 模型看到残破结构，可能输出变量名而不是值。
  这是本项目最危险的"静默错误"之一。
· 为什么工具异常要返回 error 而不是抛出：把异常翻译成模型能理解的反馈，
  它就能自己换策略（比如查订单失败→转去查政策），而不是整个 Agent 崩掉。
· ⭐ 为什么闸触发时不调 escalate_to_human 工具：闸是最后的安全网，不能依赖
  另一个可能失败的系统（工具要写数据库）；而且技术异常会制造无意义工单。
  正确做法是返回 needs_human 标记，由上层分层处理（告警 vs 工单）。
"""

import json
import re
import time
import uuid

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)

from config import (LLM_MAX_RETRIES, LLM_RETRY_BACKOFF, LOOP_WINDOW,
                    MAX_STEPS, MAX_TOOL_CALLS, MAX_TOOL_FAILS,
                    NEEDS_HUMAN_REASONS, REQUIRED_FIELDS,
                    SINGLE_CALL_BUDGET, TIME_BUDGET, TOKEN_BUDGET,
                    TRUNCATE_LIMIT)
from tools import TOOL_MAP

# ─────────────── 参数 ───────────────
# ⭐ 所有参数已移到 config.py（避免魔法数字散落）。
#    这里用【显式 import】把名字带进本模块命名空间，
#    所以 test_guardrails.py 里的 `A.TOKEN_BUDGET = 50` 仍然能生效。
# MAX_STEPS / TOKEN_BUDGET / TRUNCATE_LIMIT / LOOP_WINDOW / TIME_BUDGET
# SINGLE_CALL_BUDGET / MAX_TOOL_CALLS / MAX_TOOL_FAILS 都来自 config

# ─────────────── 系统提示词 ───────────────
SYSTEM_PROMPT = """你是电商平台的客服助手，通过工具5协助用户解决问题。你有四个工具：

1. lookup_order(order_id)      查订单状态。order_id 必须是完整订单号：
                               ORD 开头 + 12 位数字，共 15 位（如 ORD202407150001）。
2. search_kb(query)            查客服政策条款原文。
3. query_logistics(order_id)   查物流进度。order_id 必须是完整订单号：
                               ORD 开头 + 12 位数字，共 15 位（如 ORD202407150001）。
4. escalate_to_human(reason)   转人工客服（调用后对话立即结束）。

工作规则：
1. 回答任何订单状态之前，必须先调用 lookup_order，不允许猜测。
2.回答任何订单状态之前，必须先调用 lookup_order，不允许猜测。
   如果用户问的是物流（"快递到哪了""什么时候到""物流到哪了"）：
   · 查完订单状态，若是【已发货】或【已签收】，必须再调 query_logistics 查轨迹；
   · 若是【待发货】或【已退款】，直接用订单状态回答，不要再调物流工具；
   · 绝不允许编造轨迹、"正在派送"这类没有依据的话；
   · 不要只把运单号丢给用户让他自己去查。
3. 用户给的订单号如果【格式明显不完整】（不是 ORD + 12 位数字），
   直接礼貌地请用户提供完整订单号，**不要浪费一次工具调用去试** ——
   格式不对必然查不到。也绝对不要补全或编造订单号 ——
   用户会照着错误的号去核对，反而更麻烦。
4. 凡涉及政策（多久发货、能不能退、有没有补偿、质量问题时效），
   必须先调用 search_kb，依据政策原文回答，不要凭记忆。
   ⚠️ 特别是：当 lookup_order 返回 overdue=true（未发货且超时）时，
   你必须【再调一次 search_kb】查到具体补偿比例后再回复，
   不允许笼统地说"可以申请补偿"——用户需要知道确切数字。  
   ⚠️ 金额一律以 lookup_order 返回的 compensation_5pct / compensation_cap 为准，
   **严禁自己用订单金额做乘法**（你算出来的数不可靠，会给用户错误承诺）。
   如果工具没返回这两个字段，就说"建议联系人工客服核算"，不要自己算。
5. 判断用户情绪时不要只匹配关键词，而要判断【意图】。
   【硬性要求】只要用户的话里出现下列任何一种情况，
   你【必须】立刻调用 escalate_to_human，不要再追问订单号、不要再解释：
   · 骂人或质疑诚信（"骗子""坑人""垃圾""黑店""是不是在骗我"）
   · 威胁维权（投诉、曝光、找媒体、12315、消协、律师、报警、起诉）
   · 明显愤怒的感叹（带"！"的指责、要求立刻退款/赔偿）
   调用时在 reason 里写清你是根据哪句话判断的。
   此外：涉及金额纠纷、需超出政策额度审批、或已连续两次查不到信息，
   也应调用 escalate_to_human。
6. 每次只调用一个工具，看到结果后再决定下一步。
7. 最多 5 步。最终回复要简洁、具体、有温度，不要罗列一堆套话。
8. ⚠️ 订单号是 12 位精确字符串（形如 ORD202407150001），极易记错。规则：
   · 需要订单号时，只能从【用户当前这句话】或【对话历史】里【原样复制】；
   · 不允许改写、补全、推断或"回忆"任何一位数字；
   · 如果你不能 100% 确定完整订单号，就直接问用户，不要先猜一个再查 ——
     查错单号比不查更糟，用户会以为你在瞎编。
9. ⚠️ 不要引用任何你并不具备的界面元素或功能 —— 例如"点击页面上的按钮"
   "在订单详情页查看"。你只能通过文字告知信息，或调用工具。
   需要用户操作时，用文字描述操作步骤（如"请提供订单号""请联系人工客服申请"）。
10. ⚠️ 如果你掌握的资料（政策原文、工具返回）里**没有**用户问的信息
   —— 例如营业时间、支付方式、配送范围、会员权益、发票类型 ——
   必须回答"这个我不太确定，建议您联系人工客服确认"，
   **绝不能用常识或行业惯例推测后当作事实回答**。
   宁可说不知道，也不要给一个听起来合理但可能是错的答案。
"""


# ─────────────── 四道防护 ───────────────
def _count_tokens(text) -> int:
    """粗估 token：中文约 2 字 1 token。

    ⚠️ 这只是【粗估】，实测真实值是它的 3.5~4.5 倍，原因：
      ① 它只算 content 长度，看不到 tool_calls 的 JSON
      ② 它看不到 bind_tools 注入的工具 schema（每次请求 620 token）
      ③ 它把每段内容只算 1 次，而 LLM 无状态、每圈都要重发全部历史
    所以它现在只用于 fail-safe 兜底；成本统计请看 _read_usage()。
    """
    return len(str(text)) // 2


def _read_usage(resp):
    """读 API 返回的真实 token 用量（阿里云自己报的数）。

    返回 (input_tokens, output_tokens, cached_tokens)；拿不到返回 None。
    """ 
    meta = getattr(resp, "response_metadata", None) or {}
    tu = meta.get("token_usage") or {}
    if not tu:
        return None
    details = tu.get("prompt_tokens_details") or {}
    return (tu.get("input_tokens", 0),
            tu.get("output_tokens", 0),
            details.get("cached_tokens", 0))


def _is_looping(calls, window: int = LOOP_WINDOW) -> bool:   #call是目前所有工具调用记录
    """最近 window 次调用 与 再往前 window 次完全相同 → 判定死循环。"""
    if len(calls) < window * 2:     #记录少于2条就不判断
        return False
    signature = lambda c: (c["name"],               #定义一个签名，把一个调用压缩成工具名，参数字符串
                           json.dumps(c["args"], ensure_ascii=False, sort_keys=True))  #ensure_ascii保留中文，sort_keys排序
    recent = [signature(c) for c in calls[-window:]]  #取最近一条的签名
    earlier = [signature(c) for c in calls[-2 * window:-window]]  #取再往前一条的签名
    return recent == earlier   #两签名一样，判定死循环


def _truncate(text, limit: int = TRUNCATE_LIMIT) -> str:
    """按【JSON 字段】截断：保证截断后仍是合法 JSON。

    ⭐ 为什么不用 s[:limit] 硬切：
       工具返回的是 JSON 字符串，从中间切断会变成非法 JSON ——
       模型看到残破结构后，可能输出变量名而不是实际值（本项目真实踩过）。
       这是最危险的"静默错误"：不报错，但答案是垃圾。

    策略：
      ① 能解析成 dict → 逐字段累加，超限就停，末尾打 _truncated 标记
      ② 解析不了（纯文本）→ 只能按字符截，末尾标注原文长度
    """
    s = str(text)
    if len(s) <= limit:
        return s

    # ① 尝试按 JSON 字段截
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        obj = None

    if isinstance(obj, dict):
        kept = {}
        for k, v in obj.items():
            trial = dict(kept)
            trial[k] = v
            # ⭐ 留 60 字符余量给 "_truncated" 标记和括号
            if len(json.dumps(trial, ensure_ascii=False)) > limit - 60:
                break
            kept = trial
        if kept:                                  # 至少保留了一个字段
            kept["_truncated"] = True
            kept["_original_chars"] = len(s)
            return json.dumps(kept, ensure_ascii=False)

    # ② 纯文本：按字符截
    return s[:limit] + f"…（内容已截断，原文共 {len(s)} 字）"


def _safe_call(name: str, args: dict) -> str:
    """包住工具调用：异常不抛出，转成模型能读的 error。

    ⭐ 2026-11-15 新增：返回值 schema 校验。
       为什么：如果 DB 字段改名（如 compensation_5pct → comp_5pct），
       工具仍会返回 JSON，但缺了关键字段 —— 不报错，只是模型看不到，
       于是回答"建议联系人工"，看起来像正常行为。这是第二个静默错误。
       现在缺字段直接在 error 里说出来，能立刻被发现。
    """
    if name not in TOOL_MAP:
        return json.dumps({"error": f"不存在名为 {name} 的工具，"
                                    f"可用工具：{list(TOOL_MAP)}"}, ensure_ascii=False)
    try:
        raw = TOOL_MAP[name].invoke(args or {})
    except Exception as e:
        return json.dumps({"error": f"工具执行失败：{type(e).__name__}: {e}"},
                          ensure_ascii=False)

    # ⭐ schema 校验：该工具的返回值必须包含约定字段
    required = REQUIRED_FIELDS.get(name)
    if required:
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return json.dumps({"error": f"{name} 返回值不是合法 JSON：{str(raw)[:80]}"},
                              ensure_ascii=False)
        if isinstance(parsed, dict) and "error" not in parsed:
            missing = required - set(parsed)
            if missing:
                return json.dumps({
                    "error": f"{name} 返回数据缺字段 {sorted(missing)}",
                    "got_fields": sorted(parsed),
                }, ensure_ascii=False)

    return raw


# ─────────────── 订单号锚定（防多轮幻觉）───────────────
# 订单号格式：ORD + 12 位数字 = 15 个字符，例如 ORD202407150001
_ORDER_RE = re.compile(r"ORD\d{12}", re.IGNORECASE)


def _collect_order_ids(*texts) -> list:
    """从给定文本里机械提取【完整】订单号（ORD + 12 位数字，共 15 字符），去重排序。

    ⭐ 为什么要做这件事：订单号是精确字符串，模型在多轮对话里很容易
       "记混一位"（实测把 ORD202407150001 记成 ORD202307150001）。
       与其指望提示词约束，不如用代码把【确实出现过】的订单号列出来注入
       system prompt —— 把"回忆"变成"复制"。
    """
    found = set()   #准备一个不允许重复的容器
    for t in texts:   #把传进来的每段文本都看一遍
        if t:
            found.update(m.upper() for m in _ORDER_RE.findall(str(t)))  #_ORDER_RE.findall用正则找订单号，upper统一转成大写，
    return sorted(found)   #sorted(found)按字母序排序后返回


# ─────────────── 主循环 ───────────────
def run_agent(question: str, llm, history: list | None = None,
              max_turns: int = 3) -> dict:
    """执行一次 ReAct。

    history: 之前的对话，格式 [{"role": "user"/"assistant", "content": str}, ...]
             ⭐ 只取最近 max_turns 轮塞进上下文 —— 这是有意的上下文工程：
                Agent 每轮都要把完整 messages 再发一次，历史越长成本越高，
                而且早期对话对当前问题通常没用，所以截断。
    max_turns: 保留最近几轮（1 轮 = 一问一答）。设 0 表示不带历史。

    返回 {"answer": str, "trace": list, "steps": int, "reason": str|None,
          "history_turns_used": int}
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]   #message是模型的完整记忆

    # ⭐ 订单号锚定：把历史和当前问题里出现过的订单号列给模型，
    #    逼它"复制"而不是"回忆"（实测模型会记错一位数字）。
    history = history or []
    known_ids = _collect_order_ids(
        *[h.get("content") for h in history], question)
    if known_ids:
        messages[0] = SystemMessage(
            content=SYSTEM_PROMPT
            + "\n\n【本轮可用订单号（只能从这里原样复制，不得改写任何一位）】\n"
            + "\n".join(f"- {i}" for i in known_ids))

    # ⭐ 把历史对话塞进上下文（只取最近 max_turns 轮）
    used_turns = 0
    if max_turns > 0 and history:
        keep = history[-max_turns * 2:]          # 每轮 2 条（问 + 答）
        for h in keep:
            role = h.get("role")
            content = h.get("content") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        used_turns = len(keep) // 2

    messages.append(HumanMessage(content=question))

    # ⭐ request_id：一次请求的所有日志都能用同一个 ID 串起来
    #    （不然后台跑 50 条日志混在一起，分不清哪条属于哪个请求）
    rid = uuid.uuid4().hex[:8]

    calls: list = []
    used_tokens = 0
    # 历史也计入 token 预算，否则长对话会悄悄超预算
    used_tokens += sum(_count_tokens(m.content) for m in messages)

    # ⭐ 真实 token 统计（API 自己报的数，和上面的估算分开记）
    #    mock 模型（测试用）没有 response_metadata，所以 has_real 会是 False
    real_in = real_out = real_cached = 0
    last_input = 0            # ⭐ 最后一圈的 input_tokens（token 预算用它判断）
    has_real = False

    # ⭐ 2026-11-15 新增的两个计数器
    total_tool_calls = 0      # 工具调用【总次数】（一口圈可能调多个）
    tool_fail_count = 0       # 工具【连续】失败次数（成功清零）

    started = time.time()          # ⭐ 计时起点，用于 ⑤ 时间超时

    def finish(answer, reason, steps):
        """统一构造返回值，避免 8 个 return 各写一遍。

        ⭐ 2026-11-15 新增 needs_human：
           闸触发时不调 escalate_to_human 工具（闸是最后安全网，不能依赖
           另一个可能失败的系统），而是把"需不需要人工"这件事交给上层。
           上层可以分层处理：技术异常 → 监控告警；业务需求 → 建工单。
        """
        return {
            "answer": answer,
            "trace": calls,
            "steps": steps,
            "reason": reason,
            "needs_human": reason in NEEDS_HUMAN_REASONS,   # ⭐ 新增
            "request_id": rid,                              # ⭐ 新增
            "history_turns_used": used_turns,
            # 估算值（保留兼容；已知低估 3.5~4.5 倍）
            "est_tokens": used_tokens,
            # ⭐ 真实值：API 报的 input/output token；mock 场景下为 None
            "usage": ({"input": real_in, "output": real_out,
                       "cached": real_cached,
                       "total": real_in + real_out} if has_real else None),
            # ⭐ 实际耗时（秒），时间维度的可观测性
            "elapsed_sec": round(time.time() - started, 2),
        }

    for step in range(MAX_STEPS):
        round_started = time.time()      # ⭐ 每圈开始时间（用于 trace 的 round_sec）

        # ① 死循环检测（轮间）
        if _is_looping(calls):
            return finish("这个问题比较复杂，我帮您转接人工客服处理。",
                          "loop_detected", step + 1)

        # ⑤ 时间超时（和 token 正交：token 管"多少内容"，时间管"等多久"）
        # ⚠️ 这是轮间检查，不是请求级超时 —— 单次 invoke 卡住时它拦不住
        if time.time() - started > TIME_BUDGET:
            return finish("这个话题比较复杂，我帮您转接人工客服。",
                          "timeout", step + 1)

        # ④ token 预算
        # ⭐ 2026-11-15 改：用【最后一圈的 input】而不是累加值
        #    · 累加值虚高 2.6 倍（每圈都重发完整 messages）
        #    · 而且累加值和 MAX_STEPS 重叠（步数已锁死总量）
        #    · 最后圈 input 反映"上下文膨胀程度"，这才需要单独防护
        #    mock 模型没有 usage → 回退到字符估算（沿用旧逻辑）
        spent = last_input if has_real else used_tokens
        if spent > TOKEN_BUDGET:
            return finish("这个需要人工核实，我帮您转接。",
                          "token_budget", step + 1)

        # ── 调模型（⭐ 带重试 + 单次耗时检查）──
        t_invoke = time.time()
        resp = None
        last_err = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                resp = llm.invoke(messages)
                break
            except Exception as e:
                last_err = e
                if attempt < LLM_MAX_RETRIES - 1:
                    # ⭐ 指数退避：1s / 2s / 4s，避免连续打同一个故障点
                    time.sleep(LLM_RETRY_BACKOFF * (2 ** attempt))
        if resp is None:
            return finish(f"抱歉，系统繁忙，请稍后重试。（{type(last_err).__name__}）",
                          "llm_error", step + 1)

        # ⭐ ⑤b 单次调用耗时检查
        #    ⚠️ 注意：这【不能打断】等待，只在返回后判断。
        #       要真正打断得用线程池 result(timeout=)（见 config.py 说明）
        invoke_sec = time.time() - t_invoke
        if invoke_sec > SINGLE_CALL_BUDGET:
            return finish(f"这次查询响应较慢（{invoke_sec:.1f} 秒），帮您转接人工客服。",
                          "slow_invoke", step + 1)

        # ⭐ 采集真实 token（这是本次请求的 input + output，不等同于累计）
        u = _read_usage(resp)
        if u:
            has_real = True
            real_in += u[0]
            real_out += u[1]
            real_cached += u[2]
            last_input = u[0]        # ⭐ token 预算看这个

        messages.append(resp)
        used_tokens += _count_tokens(resp.content)

        # 没有工具调用 → 已可给出最终回复
        if not getattr(resp, "tool_calls", None):
            return finish(resp.content or "（未生成回复）", None, step + 1)

        # ── 执行工具 ──
        for tc in resp.tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}

            # ⭐ ⑦ 工具调用总次数上限（MAX_STEPS 只限圈数，一圈可能调多个）
            total_tool_calls += 1
            if total_tool_calls > MAX_TOOL_CALLS:
                return finish(f"这个问题需要调用的工具太多（已 {total_tool_calls} 次），"
                              f"帮您转接人工客服。", "too_many_tool_calls", step + 1)

            raw_result = _safe_call(name, args)      # ③ 异常兜底 + schema 校验
            result = _truncate(raw_result)           # ② 按 JSON 字段截断
            used_tokens += _count_tokens(result)

            # ⭐ ⑧ 工具连续失败计数（成功一次就清零）
            if '"error"' in result[:40]:
                tool_fail_count += 1
                if tool_fail_count >= MAX_TOOL_FAILS:
                    return finish(f"这个信息我连续 {tool_fail_count} 次都查不到，"
                                  f"帮您转接人工客服核实。", "tool_failed", step + 1)
            else:
                tool_fail_count = 0

            calls.append({
                "step": step,
                "name": name,
                "args": args,
                "result": result,
                "tokens": _count_tokens(result),
                "round_sec": round(time.time() - round_started, 2),   # ⭐ 每圈耗时
            })
            messages.append(ToolMessage(content=result, tool_call_id=tc.get("id")))

            # ⭐ ①b 死循环检测（同圈内：一圈里连续出现相同调用）
            #    原来只在轮间检查，一圈内模型返回 3 个相同 tool_calls 检测不到
            if _is_looping(calls):
                return finish("这个问题比较复杂，我帮您转接人工客服处理。",
                              "loop_detected", step + 1)

            # 转人工是终止动作
            if name == "escalate_to_human":
                return finish("已为您转接人工客服，稍后会有专人与您联系。",
                              "escalated", step + 1)

    return finish("抱歉，这个问题我转人工处理，请稍等。", "max_steps", MAX_STEPS)
