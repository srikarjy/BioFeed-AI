from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    """Config for v0.3 auth, isolated from app.config.Settings the same way
    app/llm/config.py is -- this whole module should be deletable without
    touching unrelated code.

    ``provider``: "fake" (default) accepts identity tokens the app itself
    signs, for tests and local dev without a real Apple/Google app
    registration. "real" verifies against Apple/Google's actual JWKS
    endpoints -- requires ``apple_client_id``/``google_client_id`` to be set
    to your app's real Services ID / OAuth client ID. Docker/prod should set
    AUTH_PROVIDER=real explicitly; nothing here defaults to real so a
    misconfigured deployment fails closed into "obviously fake tokens
    rejected" rather than silently trusting anything.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_prefix="AUTH_")

    provider: str = "fake"

    jwt_secret: str = "dev-insecure-secret-change-me-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 30
    refresh_token_ttl_days: int = 30

    apple_client_id: str = ""
    google_client_id: str = ""


auth_settings = AuthSettings()
