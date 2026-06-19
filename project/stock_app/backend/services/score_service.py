from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from backend.ai.llm import hugging_llm
from backend.prompts.all_prompt import INVESTOR_SENTIMENT_PROMPT


def get_rating(score):
    """최종판단"""
    score = int(score)
    if score >= 85:
        return "STRONG BUY"
    elif score >= 70:
        return "BUY"
    elif score >= 50:
        return "HOLD"
    elif score >= 30:
        return "SELL"
    return "STRONG SELL"


# 재무 정보 1.PER (20점)
def get_per_score(per):
    if per is None:
        return 0
    if per < 20:
        return 20
    elif per < 35:
        return 15
    return 10


# 2. 순이익률 (20점)
def get_profit_margin(financials):
    net_income = financials.get("net_income")
    revenue = financials.get("revenue")

    if not revenue:
        return 0, 0
    profit_margin = net_income / revenue

    profit_score = 0
    if profit_margin > 0.3:
        profit_score = 20
    elif profit_margin > 0.15:
        profit_score = 15
    else:
        profit_score = 10

    return profit_score, profit_margin


# 3. 기술 분석 점수
def get_tech_score(technicals):
    technicals_score = 0
    trend = technicals["trend"]
    if trend == "bullish":
        technicals_score += 10

    macd = technicals["macd"]
    macd_signal = technicals["macd_signal"]

    if macd > macd_signal:
        technicals_score += 10
    return technicals_score


# 4. 경쟁사 분석 (20점)
def get_com_score(competitors, per, profit_margin):
    # 경쟁사 per 평균 비교
    competitors_score = 0
    pe_list = [c["pe_ratio"] for c in competitors if c["pe_ratio"] is not None]

    if not pe_list:
        avg_pe = None
    else:
        avg_pe = sum(pe_list) / len(pe_list)

    if avg_pe is not None and per < avg_pe:
        competitors_score += 10

    # 경쟁사 profit_margin 평균 비교
    profit_margin_list = [
        c["profit_margin"] for c in competitors if c["profit_margin"] is not None
    ]
    avg_profit_margin = sum(profit_margin_list) / len(profit_margin_list)
    if profit_margin > avg_profit_margin:
        competitors_score += 10

    return competitors_score


# 5. 뉴스분석(20점)
async def get_news_score(news_list):
    # 뉴스 투자 심리
    news_text = "\n".join(
        [f"제목: {n["title"]}\n내용: {n["snippet"]}" for n in news_list]
    )
    prompt = ChatPromptTemplate.from_template(INVESTOR_SENTIMENT_PROMPT)
    chain = prompt | hugging_llm | JsonOutputParser()
    sentiment = await chain.ainvoke({"news_text": news_text})

    total = sentiment["positive"] + sentiment["neutral"] + sentiment["negative"]

    if total == 0:
        return 0

    positive_ratio = sentiment["positive"] / total
    return round(positive_ratio * 20)


async def evaluation(analysis):
    score = 0
    data = analysis.analysis_json

    # 1. per점수
    pe_ratio = data["financials"].get("pe_ratio")
    per_score = get_per_score(pe_ratio)

    score += per_score

    # 2.순이익율
    profit_score, profit_margin = get_profit_margin(data["financials"])
    score += profit_score

    # 3. 기술적 점수
    technicals_score = get_tech_score(data["technicals"])
    score += technicals_score

    # 4. 경쟁사 점수
    competitor_score = get_com_score(data["competitors"], pe_ratio, profit_margin)
    score += competitor_score

    # 5. 뉴스 점수
    news_score = await get_news_score(data["news"])
    score += news_score

    return {
        "total_score": score,
        "per_score": per_score,
        "profit_score": profit_score,
        "technicals_score": technicals_score,
        "competitor_score": competitor_score,
        "news_score": news_score,
    }
