import csv
import sqlite3
from fastapi import UploadFile
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.ai.llm import watson_llm
from backend.service.db_service import get_connection, get_table_columns


async def upload_csv(file: UploadFile):
    """
    csv 파일을 테이블에 저장
    """
    conn = get_connection()
    cursor = conn.cursor()

    # file.read(): upload 파일 비동기 함수
    contents = await file.read()
    csv_text = contents.decode("utf-8")
    reader = csv.DictReader(csv_text.splitlines())
    count = 0
    for row in reader:
        cursor.execute(
            "INSERT INTO transactions(date, category, merchant, amount) VALUES(?,?,?,?)",
            (row["date"], row["category"], row["merchant"], row["amount"]),
        )
        count += 1

    conn.commit()
    conn.close()
    return {"message": f"{count} 건 저장 완료"}


def card_history():
    """
    db에서 카드 정보 조회
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM transactions ORDER BY date DESC")
        rows = cursor.fetchall()
        query_result = [dict(row) for row in rows]
        conn.close()
    except Exception as e:
        query_result = [f"SQL 실행 오류:{e}"]

    return query_result


def sql_generate_llm(question):
    """
    자연어 -> SQL 변경
    """

    sql_prompt = ChatPromptTemplate.from_template("""
        당신은 SQLite 전문가입니다.
        
        테이블명: transactions
        
        컬럼: {column}
        
        질문: {question}
        
        SQL만 출력하세요                                 
    """)

    columns = get_table_columns("transactions")
    columns_text = ", ".join(columns)
    sql_chain = sql_prompt | watson_llm | StrOutputParser()
    sql = sql_chain.invoke({"column": columns_text, "question": question})
    print(f"sql: {sql}")

    sql = sql.replace("```sql", "").replace("```", "").strip()

    # sql문 실제 실행 결과 받기
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        query_result = cursor.fetchall()
    except sqlite3.Error as e:
        conn.close()
        return {"message": f"SQL 실행 오류: {e}"}
    finally:
        if conn:
            conn.close()

    return query_result


def card_analysis(question: str):
    query_result = sql_generate_llm(question)
    analysis_prompt = ChatPromptTemplate.from_template(""" 
        사용자질문: {question}
        
        SQL결과: {result}
        
        결과를 설명하고 소비 습관을 분석하여 절약 팁을 제시해주세요                            
        """)
    analysis_chain = analysis_prompt | watson_llm | StrOutputParser()
    answer = analysis_chain.invoke(
        {"question": question, "result": query_result}
    )
    return {"message": answer}
