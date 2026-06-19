import yahooquery as yq
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from typing import Optional

from backend.ai.llm import hugging_llm
from backend.repository.models import StockAnalysis, InvestmentOpinion
from backend.schemas.stock_schema import TickerInfo
from backend.services.score_service import evaluation, get_rating
from backend.graph.stock_graph import build_stock_graph

COMPANY_MAP = {
    "NVDA": {
        "company_name": "NVIDIA",
        "aliases": ["엔비디아", "nvidia", "nvda"],
    },
    "AAPL": {
        "company_name": "Apple",
        "aliases": ["애플", "apple", "aapl"],
    },
    "MSFT": {
        "company_name": "Microsoft",
        "aliases": ["마이크로소프트", "microsoft", "msft"],
    },
    "TSLA": {
        "company_name": "Tesla",
        "aliases": ["테슬라", "tesla", "tsla"],
    },
    "GOOGL": {
        "company_name": "Alphabet (Google)",
        "aliases": ["구글", "google", "알파벳", "alphabet", "googl"],
    },
    "AMZN": {
        "company_name": "Amazon",
        "aliases": ["아마존", "amazon", "amzn"],
    },
    "META": {
        "company_name": "Meta Platforms",
        "aliases": ["메타", "meta", "페이스북", "facebook", "fb"],
    },
    "AMD": {
        "company_name": "Advanced Micro Devices",
        "aliases": ["amd"],
    },
    "INTC": {
        "company_name": "Intel",
        "aliases": ["인텔", "intel", "intc"],
    },
    "TSM": {
        "company_name": "TSMC",
        "aliases": ["tsmc", "tsm", "대만반도체"],
    },
    "QCOM": {
        "company_name": "Qualcomm",
        "aliases": ["퀄컴", "qualcomm", "qcom"],
    },
    "AVGO": {
        "company_name": "Broadcom",
        "aliases": ["브로드컴", "broadcom", "avgo"],
    },
    "MU": {
        "company_name": "Micron Technology",
        "aliases": ["마이크론", "micron", "mu"],
    },
    "V": {
        "company_name": "Visa",
        "aliases": ["비자", "visa", "v"],
    },
    "MA": {
        "company_name": "Mastercard",
        "aliases": ["마스터카드", "mastercard", "ma"],
    },
    "JPM": {
        "company_name": "JPMorgan Chase",
        "aliases": ["jp모건", "jpmorgan", "jpm"],
    },
    "JNJ": {
        "company_name": "Johnson & Johnson",
        "aliases": ["존슨앤존슨", "johnson", "jnj"],
    },
    "MSFT": {
        "company_name": "Microsoft",
        "aliases": ["마이크로소프트", "microsoft", "msft"],
    },
}


class StockService:
    """주식 분석"""

    async def _extract(self, query: str):
        """
        사용자 쿼리에서 티커와 회사명 추출
        1차: Map 이용
        2차: yahooquery 이용

        query: {엔비디아} 분석해줘
        """

        keyword = self._extract_company_keyword(query)
        result = self._extract_ticker_from_query(keyword)

        # 1차 map
        if result:
            return result

        # 2차 yahooquery
        result = await self._search_yahoo_symbol(keyword)
        if result:
            return result

        raise HTTPException(status_code=400, detail="종목을 찾을 수 없습니다.")

    def _extract_ticker_from_query(self, query: str):
        """
        COMPANY_MAP 안에서 일치하는 회사 찾기
        """
        if not query:
            return None

        query = query.lower()

        for ticker, info in COMPANY_MAP.items():
            for alias in info["aliases"]:
                if alias.lower() in query:
                    return TickerInfo(ticker=ticker, company_name=info["company_name"])
        return None

    def _extract_company_keyword(self, query: str):
        """
        query: 엔비디아 분석해줘
        => 필요없는 문장 제거한 후 키워드 추출
        """
        stop_words = ["분석해줘", "분석", "전망", "재무", "알려줘", "어때", "어떻게"]

        keyword = query

        for word in stop_words:
            keyword = keyword.replace(word, "")

        return keyword.strip()

    async def _search_yahoo_symbol(self, keyword):
        """
        yahooquery 이용
        """
        if not keyword:
            return None

        try:
            result = yq.search(keyword, first_quote=True)
        except Exception:
            return None

        if not result:
            return None

        return TickerInfo(ticker=result["symbol"], company_name=result["longname"])

    async def analyze(self, query: str, db):
        ticker_info = await self._extract(query)

        # 그래프 실행
        stock_graph = build_stock_graph()
        result = await stock_graph.ainvoke(
            {
                "query": query,
                "ticker": ticker_info.ticker,
                "company_name": ticker_info.company_name,
            }
        )

        # 데이터베이스 저장
        entity = StockAnalysis(
            ticker=result["ticker"],
            company_name=result["company_name"],
            analysis_json=jsonable_encoder(result),
            report=result["report"],
        )
        db.add(entity)
        db.commit()
        db.refresh(entity)

        return {"analysis_id": entity.analysis_id, **result}

    async def opinion_service(self, analysis_id, db):
        analysis = db.get(StockAnalysis, analysis_id)
        score_result = await evaluation(analysis)

        total_score = score_result["total_score"]
        rating = get_rating(total_score)

        # ai 투자 의견서
        opinion = await self.generate_opinion(analysis.report, total_score, rating)
        entity = InvestmentOpinion(
            analysis_id=analysis_id, opinion=opinion, rating=rating, score=total_score
        )
        db.add(entity)
        db.commit()
        db.refresh(entity)

        return {
            "opinion_id": entity.opinion_id,
            "analysis_id": analysis_id,
            "opinion": opinion,
            "rating": rating,
            **score_result,
        }

    async def generate_opinion(self, report, total_score, rating):
        prompt = f"""
        당신은 월가이ㅡ 수석 애널리스트입니다.
        
        종합 점수: {total_score} / 100
        투자 등급: {rating}
        
        다음 기업 분석 보고서를 기반으로 투자 의견서를 작성하세요.
        {report}
        
        반드시 아래 형식으로 작성하세요.
        1. 투자 등급
        2. 투자 근거
        3. 핵심 리스크
        4. 단기 전망
        5. 장기 전망
        6. 최종 의견
        """

        result = await hugging_llm.ainvoke(prompt)
        return result.content


# 싱글톤(객체 하나만 생성) 서비스 인스턴스
_service: Optional[StockService] = None


def get_stock_service():
    global _service
    if _service is None:
        _service = StockService()
    return _service
