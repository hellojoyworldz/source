import sqlite3
import os

DB_PATH = "db/history.db"

"""
sqlite3.Row -> row['id']로 접근 가능
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    # 조회 결과 반환 설정
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # 디렉토리 생성
    # os.mkdir("db")
    os.makedirs("db", exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS transactions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      date TEXT,
      category TEXT,
      merchant TEXT,
      amount INTEGER
    )
    """)

    # cursor.execute("PRAGMA table_info(transactions)")
    # columns = [row["name"] for row in cursor.fetchall()]

    # if "date" not in columns and "data" in columns:
    #     cursor.execute("ALTER TABLE transactions RENAME COLUMN data TO date")
    #     columns[columns.index("data")] = "date"

    # if "date" not in columns:
    #     cursor.execute("ALTER TABLE transactions ADD COLUMN date TEXT")

    # if "amount" not in columns:
    #     cursor.execute("ALTER TABLE transactions ADD COLUMN amount INTEGER")

    conn.commit()
    conn.close()

    print("db 초기화")


def get_table_columns(table_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info({})".format(table_name))
    columns = cursor.fetchall()
    conn.close()
    return [column[1] for column in columns]


def get_dashboard():
    conn = get_connection()
    cursor = conn.cursor()

    # 카테고리 별 사용금액
    cursor.execute("""
        SELECT category, SUM(amount) AS amount
        FROM transactions
        GROUP BY category
        ORDER BY SUM(amount) DESC
        """)
    category_rows = cursor.fetchall()

    # 월별 사용금액
    cursor.execute("""
        SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS amount
        FROM transactions
        GROUP BY strftime('%Y-%m', date)
        ORDER BY strftime('%Y-%m', date)
        """)
    monthly_rows = cursor.fetchall()
    conn.close()

    return {
        "category": [
            {"category": row["category"], "amount": row["amount"]}
            for row in category_rows
        ],
        "monthly": [
            {"month": row["month"], "amount": row["amount"]} for row in monthly_rows
        ],
    }
