from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from backend.repository.db_init import get_db
from backend.schemas.evaluation_schema import EvaluationResponse
from backend.services.evalu_service import get_call_evaluation

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


# 요약 라우터
@router.post("/{call_id}", response_model=EvaluationResponse)
def read_evaluation(call_id: int, db: Session = Depends(get_db)):
    return get_call_evaluation(call_id=call_id, db=db)
