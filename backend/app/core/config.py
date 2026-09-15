"""Application settings.

All configuration is read from environment variables (or a local ``.env`` file)
so that the same code can run against SQLite in development and a managed
PostgreSQL instance in production without code changes.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    project_name: str = "PhysioAI API"
    api_version: str = "0.2.0"

    # Database -----------------------------------------------------------
    # Default targets a clean, SQLAlchemy-managed database file.
    database_url: str = "sqlite:///./physioai.db"
    # The previous architecture's database. Never deleted, only read from.
    legacy_database_path: str = "../aiphysio.db"

    # Authentication -----------------------------------------------------
    secret_key: str = "dev-insecure-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    # CORS ---------------------------------------------------------------
    # Comma separated origins. Never use "*" in production.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Login throttling ---------------------------------------------------
    # Number of failed sign-ins allowed inside the window before the account
    # e-mail (or unknown address) is temporarily locked out.
    login_max_failed_attempts: int = 5
    login_attempt_window_minutes: int = 15
    # After this many failures the UI is told to offer "Forgot password?".
    login_forgot_password_threshold: int = 3

    # Password reset -----------------------------------------------------
    password_reset_code_ttl_minutes: int = 10
    password_reset_max_attempts: int = 5
    password_reset_resend_cooldown_seconds: int = 60
    password_reset_max_requests_per_hour: int = 5
    # Lifetime of the short-lived token returned after a code is verified.
    password_reset_token_ttl_minutes: int = 10
    password_reset_code_length: int = 6

    # Email --------------------------------------------------------------
    # "smtp" sends real mail, "console" logs the message server-side (used for
    # local development and tests), "auto" picks smtp when SMTP_HOST is set.
    email_backend: str = "auto"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from_email: str = "no-reply@physioai.local"
    smtp_from_name: str = "PhysioAI"
    smtp_use_tls: bool = True
    email_timeout_seconds: int = 20

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def resolved_email_backend(self) -> str:
        """Return the effective mail backend ('smtp' or 'console')."""
        backend = (self.email_backend or "auto").strip().lower()
        if backend == "auto":
            return "smtp" if self.smtp_host else "console"
        return backend


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
