from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    firebase_credentials: str
    firebase_storage_bucket: str | None = None
    upload_dir: str = "uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()