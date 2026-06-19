from backend.services.competitor_service import get_com_info
from backend.services.financial_service import get_financial_info
from backend.services.news_service import get_news
from backend.services.technical_service import get_technical_info

from backend.ai.llm import hugging_llm
from backend.prompts.all_prompt import REPORT_PROMPT
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
report_chain = prompt | hugging_llm


async def news_node(state):
    news_list = await get_news(state["company_name"])
    return {"news": news_list}


async def financial_node(state):
    financials = await get_financial_info(state["ticker"])
    return {"financials": financials}


async def technical_node(state):
    technicals = await get_technical_info(state["ticker"])
    return {"technicals": technicals}


async def competitor_node(state):
    competitors = await get_com_info(state["ticker"])
    return {"competitors": competitors}


async def report_node(state):
    """
    수집된 정보를 LLM에게 넘기고 분석 요청
    1. 뉴스기사
    2. 재무정보(시가총액, 매출액, 순이익)
    3. 기술적분석(현재 주가, 20일선, 60일선, rsi)
    4. 경쟁회사 재무정보
    """

    news_text = "\n".join(
        [f"제목: {n.title}\n내용: {n.snippet}" for n in state["news"]]
    )

    financials = state["financials"]
    financial_text = f"""
    시가총액: {financials['market_cap']}
    매출: {financials['revenue']}
    총자산: {financials['total_assets']}
    순이익: {financials['net_income']}
    PER: {financials['pe_ratio']}
    PBR: {financials['pb_ratio']}
    """

    technicals = state["technicals"]
    technical_text = f"""
    현재주가: {technicals["current_price"]}
    SMA20: {technicals["sma20"]}
    SMA60: {technicals["sma60"]}
    RSI: {technicals["rsi"]}
    MACD: {technicals["macd"]}
    MACD_SIGNAL: {technicals["macd_signal"]}
    추세: {technicals["trend"]}
    """

    competitors = state["competitors"]
    competitor_text = "\n".join([f"""
        회사이름: {c["company_name"]}
        티커: {c["ticker"]}
        시가총액: {c["market_cap"]}
        PER: {c["pe_ratio"]}
        매출성장률: {c["revenue_growth"]}
        순이익: {c["profit_margin"]}
        """ for c in competitors])

    report = await report_chain.ainvoke(
        {
            "company_name": state["company_name"],
            "news": news_text,
            "financials": financial_text,
            "technicals": technical_text,
            "competitors": competitor_text,
        }
    )

    return {"report": report.content}
