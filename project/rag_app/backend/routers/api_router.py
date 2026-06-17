from fastapi import APIRouter, UploadFile
from fastapi.responses import StreamingResponse

from backend.schemas.basic_schema import QuestionRequest
from backend.services.llm_service import question_and_answer
from backend.services.rag_service import rag_chat_stream, upload_document, rag_chat

router = APIRouter(prefix="/api")


@router.post("/question")
async def question(payload: QuestionRequest):
    answer = question_and_answer(payload.question)
    return {"message": answer}


@router.post("/rag/upload")
async def file_upload(file: UploadFile):
    return upload_document(file)


@router.post("/rag/question")
async def question(payload: QuestionRequest):
    answer = rag_chat(payload.question)
    return {"message": answer}


@router.post("/rag/question/stream")
async def question(payload: QuestionRequest):
    answer = rag_chat_stream(payload.question)
    return StreamingResponse(answer)
