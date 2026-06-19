import math

from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.ai.llm import hugging_llm
from backend.schemas.stock_schema import RecommendStock, RiskType
from backend.repository.models import StockAnalysis


def portfolio_service(req, db):
    """
    사용자가 입력한 종목에 대해 분석정보 찾기
    risk_type:
    """

    analyses = (
        db.query(StockAnalysis).filter(StockAnalysis.ticker.in_(req.tickers)).all()
    )
    portfolio_items = []
    for analysis in analyses:
        if not analysis.opinion:
            continue

        portfolio_items.append(
            {
                "ticker": analysis.ticker,
                "score": analysis.opinion.score,
                "rating": analysis.opinion.rating,
            }
        )
    if not portfolio_items:
        raise HTTPException(status_code=404, detail="분석된 종목을 찾을 수 없습니다.")

    # 투자성향
    risk_type = req.risk_type

    for item in portfolio_items:
        score = item["score"]

        if risk_type == RiskType.AGGRESSIVE:
            item["adjusted_score"] = score**1.5
        elif risk_type == RiskType.CONSERVATIVE:
            item["adjusted_score"] = math.sqrt(score)
        else:
            item["adjusted_score"] = score

    total_adjusted = sum(item["adjusted_score"] for item in portfolio_items)

    for item in portfolio_items:
        item["weight"] = round(item["adjusted_score"] / total_adjusted * 100, 2)

    return {"risk_type": req.risk_type, "portfolio": portfolio_items}


async def generate_portfolio_report(portfolio_items, risk_type):

    portfolio_text = "\n".join(
        [f"{item['ticker']}" f"{item['weight']}%" for item in portfolio_items]
    )
    prompt = f"""
    당신은 월가의 포트폴리오 전략가입니다.
    
    투자성향:s
    {risk_type.value}
    
    포트폴리오:
    {portfolio_text}
    
    다음을 작성하세요.
    1. 포트폴리오 요약
    2. 강점
    3. 약점
    4. 주요 리스크
    5. 추천 투자기간
    6. 리밸런싱 전략
    """

    result = await hugging_llm.ainvoke(prompt)
    return result.content
