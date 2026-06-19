from typing import TypedDict


class StockAnalysisState(TypedDict):
    query: str
    ticker: str
    company_name: str
    news: list
    financials: dict
    technicals: dict
    competitors: list
    report: str
