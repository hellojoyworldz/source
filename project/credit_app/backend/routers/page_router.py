from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from backend.service.card_service import card_history

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/card/upload")
async def upload(request: Request):
    return templates.TemplateResponse(request=request, name="card.html")


@router.get("/card/history")
async def history(request: Request):
    card_infos = card_history()
    return templates.TemplateResponse(
        request=request, name="history.html", context={"history": card_infos}
    )


@router.get("/card/dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@router.get("/card/analysis")
async def analysis(request: Request):
    return templates.TemplateResponse(request=request, name="analysis.html")
