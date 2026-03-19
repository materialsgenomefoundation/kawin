"""
Configuration for Al-Mg-Si Precipitation Simulation API.

Phase 1: Minimal configuration for basic deployment.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App settings
    APP_NAME: str = "Al-Mg-Si-Cu Precipitation Simulation API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Simulation constraints
    MAX_SIMULATION_TIME_HOURS: int = 100

    # Paths
    TDB_FILE: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "AlMgSi.tdb",
    )
    ALCU_TDB_FILE: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "AlCu.tdb",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
