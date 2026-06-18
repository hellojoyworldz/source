from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.repository.db_init import Base


# class이름은 복수 X
# table은 복수
class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)

    def __str__(self):
        return f"<Customer id={self.id} name={self.name} phone={self.phone}>"


# 상담 기록 저장
class CallHistory(Base):
    __tablename__ = "call_history"

    call_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    transcript: Mapped[str]
    summary: Mapped[str]
    category: Mapped[str] = mapped_column(String[50])
    sentiment: Mapped[str] = mapped_column(String[20])
    customer_issue: Mapped[str]
    resolution: Mapped[str]
    # created_at: Mapped[datetime] = mapped_column(
    #     server_default=func.now(), nullable=False
    # )
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)


# 상담 평가 저장 - 1:1관계
class CallEvaluation(Base):
    __tablename__ = "call_evaluation"

    evaluation_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("call_history.call_id"))

    # 평가 기준
    # 고객 신원 확인(25)
    identity_verification: Mapped[bool] = mapped_column(default=False)
    identity_verification_reason: Mapped[str]
    # 공감 표현(25)
    empathy: Mapped[bool] = mapped_column(default=False)
    empathy_reason: Mapped[str]
    # 문제 해결(25)
    issue_resolution: Mapped[bool] = mapped_column(default=False)
    issue_resolution_reason: Mapped[str]
    # 설문 조사 안내(25)
    survey_guidance: Mapped[bool] = mapped_column(default=False)
    survey_guidance_reason: Mapped[str]
    score: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)
