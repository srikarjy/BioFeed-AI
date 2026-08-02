from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Config for the anomaly-explanation LLM feature, isolated from the
    main app Settings so this whole module can be deleted without touching
    app/config.py. All fields are env-driven (LLM_* prefix) since the vLLM
    host is a rented GPU instance that gets torn down and re-provisioned.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="LLM_")

    enabled: bool = True
    # OpenAI-compatible base URL vLLM serves, e.g. http://<gpu-host>:8000/v1
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    max_tokens: int = 256
    temperature: float = 0.2
    request_timeout_seconds: float = 60.0


llm_settings = LLMSettings()
