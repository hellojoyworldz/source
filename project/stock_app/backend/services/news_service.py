import aiohttp

from langchain_community.utilities import GoogleSerperAPIWrapper
from backend.config.settings import settings
from backend.schemas.news_schema import NewsItem


async def get_news(company_name: str):
    """
    구글 뉴스 검색 후
    title, snippet, url, source, date 추출
    serper
    """

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        search = GoogleSerperAPIWrapper(
            type="news",
            serper_api_key=settings.serper_api_key,
            aiosession=session,
        )
        results = await search.aresults(f"{company_name} stock news")

    news_list = []
    for item in results.get("news", [])[:10]:
        news_list.append(
            NewsItem(
                title=item.get("title") or "",
                snippet=item.get("snippet") or "",
                url=item.get("link") or "",
                source=item.get("source") or "",
                date=item.get("date") or "",
            )
        )

    return news_list
