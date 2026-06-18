import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import DeclarativeBase

Path("db").mkdir(parents=True, exist_ok=True)
# os.makedirs("db", exist_ok=True)

# 엔진
engine = create_engine("sqlite:///db/callcenter.db", echo=True)

# 세션 팩토리 생성
# 직접 세션을 만드는 방식
# FastAPI 요청 lifecycle과 분리
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# 부모 역할 / v1 - Base = declarative_base()
class Base(DeclarativeBase):
    pass


# SessionLocal을 다른 모듈에서 사용할 수 있도록 제공
# FastAPI 라우터/의존성 함수에서 사용 - arguments로 넘겨줄 때 Depends로 감싸서 사용
# 요청이 끝나면 db.close() 자동 실행, 라이터가 새션 생성/종료 책임 안 짐
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
