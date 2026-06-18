from sqlalchemy.orm import Session

from backend.repository.models import CallEvaluation
from backend.schemas.evaluation_schema import EvaluationResponse


def get_call_evaluation(call_id: int, db: Session):

    # db에서 데이터 가져오기
    if call_id:
        evaluations = (
            db.query(CallEvaluation).filter(CallEvaluation.call_id == call_id).first()
        )

        return EvaluationResponse(
            evaluation_id=evaluations.evaluation_id,
            call_id=evaluations.call_id,
            identity_verification=evaluations.identity_verification,
            identity_verification_reason=evaluations.identity_verification_reason,
            empathy=evaluations.empathy,
            empathy_reason=evaluations.empathy_reason,
            issue_resolution=evaluations.issue_resolution,
            issue_resolution_reason=evaluations.issue_resolution_reason,
            survey_guidance=evaluations.survey_guidance,
            survey_guidance_reason=evaluations.survey_guidance_reason,
            score=evaluations.score,
            created_at=evaluations.created_at,
        )
