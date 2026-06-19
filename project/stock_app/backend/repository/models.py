from datetime import datetime

from backend.repository.db_init import Base
from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship


class StockAnalysis(Base):
    __tablename__ = "stock_analysis"

    analysis_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str]
    company_name: Mapped[str]
    analysis_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    report: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)
    opinion = relationship(
        "InvestmentOpinion", back_populates="analysis", uselist=False
    )


# 컬럼명 수정하면 반영이 안 됨
# 수정 할 때 마다 반영하고 싶으면 라이브러리 설치해야함 - 나중에 찾아서 해보세요


class InvestmentOpinion(Base):
    __tablename__ = "investment_opinion"

    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("stock_analysis.analysis_id"), unique=True
    )
    opinion_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    opinion: Mapped[str]
    # 투자의견: buy, sell
    rating: Mapped[str] = mapped_column(String(20), nullable=True)
    score: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)
    analysis = relationship("StockAnalysis", back_populates="opinion")
