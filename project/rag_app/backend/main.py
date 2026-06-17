from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.services.llm_service import question_and_answer
from backend.routers.page_router import router as page_router
from backend.routers.api_router import router as api_router

app = FastAPI()

app.include_router(page_router)
app.include_router(api_router)

# static 폴더 지정
app.mount("/static/", StaticFiles(directory="backend/static"), name="static")
