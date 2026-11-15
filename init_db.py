"""建库脚本：生成 orders.db 并写入 12 条模拟订单。

覆盖 4 种状态，其中特意包含：
  · 待发货且已超时  → 用于演示"查出来确实超时，可申请 5% 补偿"
  · 已发货有物流单号 → 用于演示"报出真实物流信息"
运行：python init_db.py
"""

import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.db")

# (order_id, status, 距今几天前付款, carrier, tracking, amount, category)
SEEDS = [
    # ⭐ 已发货，有物流单号 —— 简单查询场景
    ("ORD202407150001", "已发货", 3, "顺丰", "SF1234567890", 299.00, "蓝牙耳机"),
    ("ORD202407150002", "已发货", 5, "中通", "ZT9988776655", 129.00, "手机壳"),
    # 待发货，未超时（1 天）—— 正常安抚
    ("ORD202407150003", "待发货", 1, None, None, 499.00, "蓝牙音箱"),
    ("ORD202407150004", "待发货", 2, None, None, 89.00, "数据线"),
    # ⭐ 待发货且已超时（5 天 / 7 天）—— 补偿场景
    ("ORD202407150005", "待发货", 5, None, None, 358.00, "无线鼠标"),
    ("ORD202407150006", "待发货", 7, None, None, 1299.00, "机械键盘"),
    # 已签收 —— 退货咨询场景
    ("ORD202407150007", "已签收", 4, "顺丰", "SF5566778899", 219.00, "耳机"),
    ("ORD202407150008", "已签收", 12, "圆通", "YT1122334455", 45.00, "贴膜"),
    # 已退款 —— 边界场景
    ("ORD202407150009", "已退款", 20, None, None, 168.00, "充电宝"),
    ("ORD202407150010", "已发货", 6, "韵达", "YD6677889900", 79.00, "支架"),
    ("ORD202407150011", "待发货", 0, None, None, 999.00, "显示器挂灯"),
    ("ORD202407150012", "已签收", 9, "顺丰", "SF3344556677", 599.00, "降噪耳机"),
]


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"已删除旧库: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE orders (
            order_id  TEXT PRIMARY KEY,
            status    TEXT NOT NULL,
            pay_time  TEXT NOT NULL,
            carrier   TEXT,
            tracking  TEXT,
            amount    REAL,
            category  TEXT
        )
    """)

    today = datetime.now()
    rows = []
    for oid, status, days_ago, carrier, tracking, amount, category in SEEDS:
        # ⭐ 存到小时精度：这样"超过 48 小时"才能精确判定，
        #    否则只存日期会丢掉小时信息，导致边界订单（付款 2 天整）判不准。
        pay_time = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M")
        rows.append((oid, status, pay_time, carrier, tracking, amount, category))

    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()

    n = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    stat = conn.execute(
        "SELECT status, COUNT(*) FROM orders GROUP BY status").fetchall()
    conn.close()

    print(f"✅ 建库完成: {DB_PATH}")
    print(f"   共 {n} 条订单")
    for s, c in stat:
        print(f"   {s}: {c} 条")
    print("\n⭐ 超时未发货的订单（用于演示补偿场景）:")
    for oid, status, days_ago, *_ in SEEDS:
        if status == "待发货" and days_ago > 2:
            print(f"   {oid}  付款 {days_ago} 天前")


if __name__ == "__main__":
    main()
