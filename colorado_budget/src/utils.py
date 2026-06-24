import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from loguru import logger

_ANTHROPIC_KEY_ALIASES = ["ANTHROPIC_API_KEY", "OPENWEBUI_ANTHROPIC_API_KEY", "OPENWEBUI_ANTHROPIC_API_KEY_1"]


def load_api_keys() -> dict:
    """Load API keys from .env (colorado_budget/ then repo root) and normalize key names."""
    api_keys: dict = {}
    candidates = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]
    for dotenv_path in candidates:
        if dotenv_path.exists():
            api_keys = dotenv_values(dotenv_path)
            load_dotenv(dotenv_path)
            logger.info(f"Loaded {len(api_keys)} API keys from {dotenv_path}")
            break
    else:
        logger.info("No .env file found — relying on environment variables")

    # Ensure ANTHROPIC_API_KEY is set for Strands/Anthropic SDK
    if not os.environ.get("ANTHROPIC_API_KEY"):
        for alias in _ANTHROPIC_KEY_ALIASES[1:]:
            value = os.environ.get(alias) or api_keys.get(alias)
            if value:
                os.environ["ANTHROPIC_API_KEY"] = value
                logger.info(f"Mapped {alias} -> ANTHROPIC_API_KEY")
                break

    return api_keys
