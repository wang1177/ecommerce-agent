"""三个工具：查订单 / 查政策 / 转人工。

设计要点
─────────
1. docstring 就是工具的 description —— 它直接决定模型"什么时候想到调它"，
   所以写得越具体越好（尤其是"什么时候该调我"）。
2. lookup_order 明确声明"需要 12 位完整订单号"，模型才知道号不全时该追问，
   而不是瞎猜一个号。
3. lookup_order 的返回值里带 overdue 字段 —— 把"是否超时"算在代码里，
   不让模型自己算日期（模型算日期很容易出错）。
"""

import json
import os
import re
import sqlite3
from datetime import datetime

from langchain_core.tools import tool

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#__file__是当前文件路径，os.path.abspath(__file__)转成绝对路径，os.path.dirname(...)取它所在的文件夹

DB_PATH = os.path.join(BASE_DIR, "orders.db")
POLICY_PATH = os.path.join(BASE_DIR, "policies.md")

# 工单数据单独一个库。为什么不放 orders.db：
#  orders.db 是 init_db.py 生成的【可重建】初始数据，
#  而工单是【运行时累积】的。放一起的话，重跑 init_db.py 会把工单一起清掉。
TICKETS_DB_PATH = os.path.join(BASE_DIR, "tickets.db")

# 订单号格式：ORD + 12 位数字 = 15 个字符，例如 ORD202407150001。
#  用正则校验而不是 len()，因为 len 只能挡"太短"，挡不住"格式不对"。
ORDER_ID_RE = re.compile(r"^ORD\d{12}$", re.IGNORECASE)  #re.IGNORECASE忽略大小写

# 物流轨迹：直接写 dict，不建 SQLite 表。
#  因为这是"演示用轨迹"，没有增量写入需求 —— 只有转人工要往库里写。
#  key = 订单号，value = 该订单的物流状态。
_LOGISTICS = {
    "ORD202407150001": {"status": "运输中",
                        "latest": "快件已到达 杭州分拨中心",
                        "updated_at": "21:40"},
    "ORD202407150002": {"status": "运输中",
                        "latest": "快件已到达 宁波分拨中心，正在发往网点",
                        "updated_at": "07:15"},
    "ORD202407150010": {"status": "派送中",
                        "latest": "宁波鄞州网点快递员派送中，电话 138****6621",
                        "updated_at": "08:52"},
    "ORD202407150007": {"status": "已签收",
                        "latest": "快件已签收，签收人：本人",
                        "updated_at": "15:08"},
    "ORD202407150012": {"status": "已签收",
                        "latest": "快件已签收，签收人：门卫代收",
                        "updated_at": "11:31"},
}

@tool
def lookup_order(order_id: str) -> str:
    """查询订单状态。order_id 必须是完整订单号：ORD 开头 + 12 位数字，共 15 位
    （例如 ORD202407150001）。

    当 order_id 格式不对或查不到时会返回 error 字段，此时应当礼貌地请用户提供
    完整订单号，绝不要自行补全或编造订单号。

    返回字段含义：status 订单状态；hours_since_pay 付款至今多少小时；
    overdue 是否属于"未发货且已超过 48 小时"（补偿判定的依据）；
    carrier/tracking 承运商与物流单号；amount 订单金额；
    compensation_5pct 订单金额 5% 的超时补偿金额（已算好）；
    compensation_cap 单笔累计补偿上限 = 订单金额 10%（已算好）。
    涉及金额时只能引用 compensation_* 字段，不要自己做乘法。
    """
    oid = (order_id or "").strip().upper()      # .strip()：去掉首尾空格；统一大写，兼容用户输入小写
    if not ORDER_ID_RE.match(oid):
        return json.dumps(                            #校验格式
            {"error": f"订单号 '{oid}' 格式不正确，应为 ORD + 12 位数字"
                      f"（共 15 位，例如 ORD202407150001）"},
            ensure_ascii=False)

    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT status, pay_time, carrier, tracking, amount, category "
            "FROM orders WHERE order_id = ?", (oid,)).fetchone()
        conn.close()
    except sqlite3.Error as e:
        return json.dumps({"error": f"订单库查询失败：{e}"}, ensure_ascii=False)

    if not row:
        return json.dumps({"error": f"订单 {oid} 不存在，请核对订单号"},
                          ensure_ascii=False)

    status, pay_time, carrier, tracking, amount, category = row

    # 以「小时」为单位判定超时，而不是「天」：
    #  政策是"付款后 48 小时内发货"，用整数天数会把边界订单判错
    #  （付款 2 天整 = 48 小时，可能刚好卡在线上）。
    try:
        paid_at = datetime.strptime(pay_time, "%Y-%m-%d %H:%M")  #把pay_time字符串转成时间
    except ValueError:
        try:
            paid_at = datetime.strptime(pay_time, "%Y-%m-%d")   # 兼容只存日期的旧数据
        except ValueError:
            paid_at = None

    if paid_at:
        delta = datetime.now() - paid_at
        hours = int(delta.total_seconds() // 3600)
        days = delta.days
    else:
        hours, days = -1, -1

    overdue = (status == "待发货") and hours > 48

    # 金额一律由代码算，模型只引用 —— 不把乘法留给模型。
    #  实测模型会"四舍五入凭感觉"（1299×5% 报"约 65 元"，实为 64.95），
    #  而补偿金额是承诺给用户的数字，必须精确。
    #  这两个字段永远返回（不管有没有超时），由模型按政策决定是否引用。
    compensation_5pct = round(amount * 0.05, 2) if amount else None
    compensation_cap = round(amount * 0.10, 2) if amount else None

    return json.dumps({
        "order_id": oid,
        "status": status,
        "pay_time": pay_time,
        "days_since_pay": days,
        "hours_since_pay": hours,
        "overdue": overdue,
        "overdue_rule": "待发货且付款超过 48 小时",
        "carrier": carrier,
        "tracking": tracking,
        "amount": amount,
        # 预计算好的补偿金额（模型只能引用，不许自己乘）
        "compensation_5pct": compensation_5pct,
        "compensation_cap": compensation_cap,
        "category": category,
    }, ensure_ascii=False)


@tool
def search_kb(query: str) -> str:
    """查询客服政策条款原文。

    涵盖：发货时效、退款规则、质量问题处理、物流异常补偿、补偿上限、发票售后。
    凡是涉及"多久发货 / 能不能退 / 有没有补偿 / 质量问题的时效"这类问题，
    必须先调用本工具，依据政策原文回答，不能凭记忆回答。
    """
    try:
        with open(POLICY_PATH, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return json.dumps({"error": f"政策库读取失败：{e}"}, ensure_ascii=False)

    return content


@tool
def query_logistics(order_id:str)->str:
    """
    这个工具用于查询物流轨迹，输入订单号，返回物流状态。
    适用场景：当用户问 快递到哪/发货了没/物流信息时调用
    你需要先调用lookup_order确认订单状态，别自己编正在派送等错误信息，一切消息都要通过订单号查出来
    """
    oid=(order_id or "").strip().upper()  #.strip()：去掉首尾空格；.upper():转大写
    #格式不对
    if not ORDER_ID_RE.match(oid):
        return json.dumps({"error":f"订单格式不正确，应为 ORD + 12 位数字，例如 ORD202407150001"},ensure_ascii=False)
    #格式对，但是没有物流记录
    row=_LOGISTICS.get(oid)
    if not row:
        return json.dumps({"error":f"订单 {oid} 不存在，请先用lookup_order确认订单状态"},ensure_ascii=False)
    #查到订单
    return json.dumps({
        "order_id": oid,
        "status": row["status"],
        "latest": row["latest"],
        "updated_at": row["updated_at"]
    },ensure_ascii=False)


@tool
def escalate_to_human(reason: str) -> str:
    """转人工客服。调用后本次对话立即结束并交给人工处理。

    适用场景：用户情绪激烈（提到投诉、曝光、律师、报警）、涉及金额纠纷、
    需要超出政策额度审批、或你已连续两次查不到所需信息。调用时请填写具体原因。
    """
    # 原来只返回 {"escalated": True}，没有任何记录 ——
    #  "假转人工"（用户以为有人接手，其实什么都没有）。
    #  现在真的写一条工单记录到 handoffs 表，人工可以直接查。
    #  工单写【tickets.db】而不是 orders.db —— 前者是运行时累积数据，
    #     后者是 init_db.py 可重建的初始数据，混在一起会被一起清掉。
    ticket_id = None
    try:
        conn = sqlite3.connect(TICKETS_DB_PATH, timeout=3)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS handoffs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                reason     TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        cur = conn.execute(
            "INSERT INTO handoffs (reason, created_at) VALUES (?, ?)",
            (str(reason)[:500], datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        ticket_id = cur.lastrowid
        conn.commit()
        conn.close()
    except Exception as e:
        # 写库失败不能让转人工本身失败 —— 至少要把 reason 返回给模型
        return json.dumps({"escalated": True, "reason": reason,
                           "warning": f"工单落库失败：{type(e).__name__}"},
                          ensure_ascii=False)

    return json.dumps({"escalated": True, "ticket_id": ticket_id, "reason": reason},
                      ensure_ascii=False)


# 供 agent / main 使用
TOOLS = [lookup_order, search_kb, query_logistics, escalate_to_human]  #把三个工具都放到列表里面
TOOL_MAP = {t.name: t for t in TOOLS}
#用字典推导式将列表转成字典t.name：键=名字， t for t in TOOLS遍历TOOLS值=对象


if __name__ == "__main__":
    # 简单自测：不经过模型，直接调工具
    print("=== 工具自测 ===")
    print("① 查已发货订单:")
    print("  ", lookup_order.invoke({"order_id": "ORD202407150001"})[:200])
    print("\n② 查超时未发货订单:")
    print("  ", lookup_order.invoke({"order_id": "ORD202407150005"})[:200])
    print("\n③ 订单号不完整:")
    print("  ", lookup_order.invoke({"order_id": "ORD2023101"}))
    print("\n④ 订单不存在:")
    print("  ", lookup_order.invoke({"order_id": "ORD999999999999"}))
    print("\n⑤ 查政策（前 120 字）:")
    print("  ", search_kb.invoke({"query": "发货时效"})[:120].replace("\n", " "))
    print("\n⑥ 转人工:")
    print("  ", escalate_to_human.invoke({"reason": "用户要求投诉"}))
    print("\n⑦ 查物流（运输中）:")
    print("  ", query_logistics.invoke({"order_id": "ORD202407150001"}))
    print("\n⑧ 查物流（已签收）:")
    print("  ", query_logistics.invoke({"order_id": "ORD202407150007"}))
    print("\n⑨ 查物流（待发货，故意没轨迹）:")
    print("  ", query_logistics.invoke({"order_id": "ORD202407150003"}))
