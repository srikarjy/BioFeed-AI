from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://biofeed:biofeed@localhost:5432/biofeed"

    # "sentence-transformers" (real PubMedBERT embeddings, needs
    # requirements-ml.txt) or "hash" (dependency-free fallback for tests and
    # local dev). Docker sets sentence-transformers explicitly.
    embedding_backend: str = "hash"
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"


settings = Settings()
