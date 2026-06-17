from fastapi import APIRouter, UploadFile
from backend.schemas.card_schema import AnalysisRequest
from backend.service.card_service import upload_csv, card_analysis
from backend.service.db_service import get_dashboard

router = APIRouter(prefix="/api/card")


@router.post("/upload")
async def upload_file(file: UploadFile):
    return await upload_csv(file)


@router.get("/dashboard")
async def dashboard():
    return get_dashboard()


@router.post("/analysis")
async def sql_llm_analysis(request: AnalysisRequest):
    return card_analysis(request.question)
