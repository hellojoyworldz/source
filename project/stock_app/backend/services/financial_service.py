import yfinance as yf


async def get_financial_info(ticker: str):
    """
    yfinance api 이용
    """
    company = yf.Ticker(ticker)
    info = company.info
    income = company.income_stmt
    balance = company.balance_sheet
    cashflow = company.cash_flow

    def last_value(frame, row_name):
        if row_name not in frame.index or frame.empty:
            return None
        value = frame.loc[row_name].iloc[0]
        return None if value is None else value

    return {
        # 시가총액
        "market_cap": info.get("marketCap"),
        # 현재주가
        "current_price": info.get("currentPrice"),
        # 매출
        "revenue": last_value(income, "Total Revenue"),
        # 손익계산
        "operating_income": last_value(income, "Operating Income"),
        # 순이익
        "net_income": last_value(income, "Net Income"),
        # 총자산
        "total_assets": last_value(balance, "Total Assets"),
        # 총부채
        "total_debt": last_value(balance, "Total Debt"),
        # 현금흐름
        "free_cash_flow": last_value(cashflow, "Free Cash Flow"),
        # per
        "pe_ratio": info.get("trailingPE"),
        # pbr
        "pb_ratio": info.get("priceToBook"),
    }
