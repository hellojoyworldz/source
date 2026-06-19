import yfinance as yf

COMPETITOR_MAP = {
    "NVDA": ["AMD", "INTC", "TSM", "AVGO"],
    "AMD": ["NVDA", "INTC", "TSM"],
    "TSLA": ["RIVN", "GM", "F"],
    "AAPL": ["MSFT", "GOOGL", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN"],
    "GOOGL": ["MSFT", "META", "AMZN"],
}


async def get_com_info(ticker: str):
    """
    경쟁 관계 회사에 대한 핵심 정보 추출
    """
    competitor_tickers = COMPETITOR_MAP.get(ticker, [])
    results = []

    for comp in competitor_tickers:
        company = yf.Ticker(comp)
        info = company.info

        results.append(
            {
                # 티커정보
                "ticker": comp,
                # 회사이름
                "company_name": info.get("longName"),
                # 시가총액
                "market_cap": info.get("marketCap"),
                # per
                "pe_ratio": info.get("trailingPE"),
                # 매출 성장률
                "revenue_growth": info.get("revenueGrowth"),
                # 순이익
                "profit_margin": info.get("profitMargins"),
            }
        )

    return results
