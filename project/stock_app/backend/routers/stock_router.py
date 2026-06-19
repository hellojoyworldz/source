from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.repository.db_init import get_db
from backend.schemas.stock_schema import RecommendStock, StockAnalyzeReq
from backend.services.recommend_service import (
    generate_portfolio_report,
    portfolio_service,
)
from backend.services.stock_service import get_stock_service

router = APIRouter(prefix="/api/stock", tags=["Stock"])


# 이렇게 요청이 들어올 때 만 DB연결
@router.post(
    path="/analyze",
    summary="기업 종합 분석",
    description="자연어로 해당기업을 요청하면 뉴스, 재무, 기술적, 경쟁사 분석을 실행합니다",
)
async def stock_analyze(
    req: StockAnalyzeReq,
    db: Session = Depends(get_db),
):
    service = service = get_stock_service()
    return await service.analyze(req.query, db=db)


@router.post(
    path="/opinion/{analysis_id}",
    summary="투자 의견 조회",
    description="",
)
async def stock_analyze_opinion(analysis_id: int, db: Session = Depends(get_db)):
    service = get_stock_service()
    return await service.opinion_service(analysis_id, db)


@router.post(
    path="/recommend",
    summary="투자 추천",
    description="",
)
async def stock_recommend(req: RecommendStock, db: Session = Depends(get_db)):
    result = portfolio_service(req, db)
    report = await generate_portfolio_report(result["portfolio"], req.risk_type)

    return {
        "risk_type": req.risk_type,
        "portfolio": result["portfolio"],
        "report": report,
    }
