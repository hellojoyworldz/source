from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    watsonx_api_key: str = Field(alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(alias="WATSONX_URL")
    hf_token: str = Field(alias="HF_TOKEN")
    # serper_api_key: str = Field(
    #     validation_alias=AliasChoices("SERPER_API_KEY", "SUPPER_API_KEY")
    # )
    serper_api_key: str = Field(alias="SERPER_API_KEY")


settings = Settings()
