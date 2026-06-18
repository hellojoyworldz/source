from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.repository.db_init import get_db
from backend.schemas.assistant_schema import AssistantResponse, AssistantRequest
from backend.services.assistant_service import answer_assistant_question

router = APIRouter(prefix="/api/assistant", tags=["Assistant"])


# 사용자 질문 받는 라우터
# Depends: 라우터를 통해서 들어왔을 때 만 세션을 연결하겠다
@router.post("", response_model=AssistantResponse)
def ask_assistant(req: AssistantRequest, db: Session = Depends(get_db)):
    return answer_assistant_question(
        customer_id=req.customer_id, question=req.transcript, db=db
    )
