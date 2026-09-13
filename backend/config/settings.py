"""
Application configuration — all settings from environment variables.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI Physio"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "aiphysio.db"
    )

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # MediaPipe
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5

    # Session
    ALARM_THRESHOLD: int = 3  # consecutive bad-form frames before alarm
    ALARM_TIMEOUT: int = 60   # seconds before guardian alert

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
