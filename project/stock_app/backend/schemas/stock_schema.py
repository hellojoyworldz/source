from enum import Enum

from pydantic import BaseModel


class StockAnalyzeReq(BaseModel):
    query: str


class TickerInfo(BaseModel):
    ticker: str
    company_name: str


# enum : 상수
class RiskType(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


class RecommendStock(BaseModel):
    tickers: list[str]
    risk_type: RiskType
