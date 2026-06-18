from sqlalchemy.orm import Session
from langchain_core.prompts import ChatPromptTemplate

from backend.ai.llm import hugging_llm
from backend.prompts.all_prompt import CALL_EVALUATION_PROMPT, SUMMARY_SYSTEM_PROMPT
from backend.repository.models import CallEvaluation, CallHistory
from backend.schemas.summary_schema import (
    CallEvaluationResponse,
    CallRequest,
    CallSummary,
    CallCreate,
)


# LLM transcript 요약 시키기
def summary_call(transcript: str):
    summary_prompt = ChatPromptTemplate.from_messages(
        [("system", SUMMARY_SYSTEM_PROMPT), ("human", "상담내용\n{transcript}")]
    )

    # pydantic parser
    # response 모델을 지정하고 그 형태로 응답할 수 있게
    # 그것보다 간단하게 사용해보기
    structured_llm = hugging_llm.with_structured_output(CallSummary)
    summary_chain = summary_prompt | structured_llm

    result = summary_chain.invoke({"transcript": transcript})
    return result


def save_call_history(db, data):
    """상담 저장"""
    history = CallHistory(
        customer_id=data.customer_id,
        transcript=data.transcript,
        summary=data.summary,
        category=data.category,
        sentiment=data.sentiment,
        customer_issue=data.customer_issue,
        resolution=data.resolution,
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    return history


def evaluate_call(transcript: str):
    """상담 평가"""
    prompt = ChatPromptTemplate.from_template(CALL_EVALUATION_PROMPT)
    structured_llm = hugging_llm.with_structured_output(CallEvaluationResponse)
    chain = prompt | structured_llm

    return chain.invoke({"transcript": transcript})


def calculate_score(evaluation: CallEvaluationResponse):
    score = 0

    if evaluation.identity_verification:
        score += 25

    if evaluation.empathy:
        score += 25

    if evaluation.issue_resolution:
        score += 25

    if evaluation.survey_guidance:
        score += 25

    return score


def save_call_evaluation(db: Session, call_id: int, evaluation: CallEvaluationResponse):
    """상담 평가 내용 저장"""
    score = calculate_score(evaluation)
    entity = CallEvaluation(
        call_id=call_id,
        identity_verification=evaluation.identity_verification,
        identity_verification_reason=evaluation.identity_verification_reason,
        empathy=evaluation.empathy,
        empathy_reason=evaluation.empathy_reason,
        issue_resolution=evaluation.issue_resolution,
        issue_resolution_reason=evaluation.issue_resolution_reason,
        survey_guidance=evaluation.survey_guidance,
        survey_guidance_reason=evaluation.survey_guidance_reason,
        score=score,
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)

    return entity


def create_call_history(req: CallRequest, db: Session):
    # 요약
    summary = summary_call(req.transcript)

    # 데이터베이스 저장
    call_data = CallCreate(
        customer_id=req.customer_id,
        transcript=req.transcript,
        summary=summary.summary,
        category=summary.category,
        sentiment=summary.sentiment,
        customer_issue=summary.customer_issue,
        resolution=summary.resolution,
    )

    history = save_call_history(db=db, data=call_data)

    # 상담 평가
    evaluation = evaluate_call(req.transcript)

    # 평가 저장
    save_call_evaluation(db=db, call_id=history.call_id, evaluation=evaluation)

    return CallSummary(
        summary=summary.summary,
        keywords=summary.keywords,
        category=summary.category,
        sentiment=summary.sentiment,
        action_items=summary.action_items,
        customer_issue=summary.customer_issue,
        resolution=summary.resolution,
    )
