from datetime import datetime

from pydantic import BaseModel


class EvaluationResponse(BaseModel):
    evaluation_id: int
    call_id: int
    identity_verification: bool
    identity_verification_reason: str
    empathy: bool
    empathy_reason: str
    issue_resolution: bool
    issue_resolution_reason: str
    survey_guidance: bool
    survey_guidance_reason: str
    score: int
    created_at: datetime
