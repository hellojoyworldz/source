from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    # model_config = SettingsConfigDict(
    #     env_file="backend/.env", extra="ignore"
    # )  # extra ignore env에서 안가져 오는 애들 무시

    watsonx_api_key: str = Field(alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(alias="WATSONX_URL")


settings = Settings()
