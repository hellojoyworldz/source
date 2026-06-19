from pydantic import BaseModel


class NewsItem(BaseModel):
    title: str
    snippet: str
    url: str
    source: str
    date: str
