from langchain_ibm import WatsonxEmbeddings

from backend.config.settings import settings

watson_embedding = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url=settings.watsonx_url,
    api_key=settings.watsonx_api_key,
    project_id=settings.watsonx_project_id,
)
