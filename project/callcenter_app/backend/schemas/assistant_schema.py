from pydantic import BaseModel


class AssistantRequest(BaseModel):
    customer_id: int
    transcript: str


class AssistantResponse(BaseModel):
    answer: str
