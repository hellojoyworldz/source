from datetime import datetime

from pydantic import BaseModel


class SummaryRequest(BaseModel):
    transcript: str


class CallSummary(BaseModel):
    summary: str
    keywords: list[str]
    category: str
    sentiment: str
    action_items: list[str]
    customer_issue: str
    resolution: str


# 상담 요청 시 사용할 타입
class CallRequest(BaseModel):
    customer_id: int
    transcript: str


# 상담 저장
class CallCreate(BaseModel):
    customer_id: int
    transcript: str
    summary: str
    category: str
    sentiment: str
    customer_issue: str
    resolution: str


# 저장
class CallSaveResponse(BaseModel):
    pass


# 평가 스키마
class CallEvaluationResponse(BaseModel):
    identity_verification: bool
    identity_verification_reason: str
    empathy: bool
    empathy_reason: str
    issue_resolution: bool
    issue_resolution_reason: str
    survey_guidance: bool
    survey_guidance_reason: str
