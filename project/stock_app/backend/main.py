from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.repository.db_init import Base, engine
from backend.routers.stock_router import router as stock_router


# 앱 시작 시 자동 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버시작")

    # Base에 등록된 모든 모델의 테이블 자동 생성
    Base.metadata.create_all(bind=engine)
    print("[DB] 테이블 생성 완료 (또는 이미 존재)")

    yield
    print("서버 종료")


app = FastAPI(
    lifespan=lifespan,
    title="📈 Stock AI",
    version="1.0",
    description="""## 주식 AI 분석 
    FastAPI + LangGraph + yfinance + ChromaDB
    
    ### 주요 기능
    |번호|기능|엔드포인드|
    |---|---|---|
    |1|기업 종합 분석|`POST /api/stock/analyze`|
    |2|투자 의견 조회|`GET /api/stock/opinion/:analysis_id|`
    """,
)

# static 폴더 지정
app.mount("/static/", StaticFiles(directory="backend/static"), name="static")

app.include_router(stock_router)
