from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from langchain_ollama import ChatOllama
from backend.config.settings import settings

watson_llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url=settings.watsonx_url,
    api_key=settings.watsonx_api_key,
    project_id=settings.watsonx_project_id,
    max_tokens=2000,
    params={"temperature": 0},
)

hugging_llm = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.hf_token,
    model="Qwen/Qwen3-8B:nscale",
)

vision_llm = ChatOllama(model="minimax-m3:cloud", temperature=0)
