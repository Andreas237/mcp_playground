from os import getenv
from pathlib import Path
from loguru import logger
from dotenv import dotenv_values, load_dotenv


def load_api_keys():
    dotenv_path = Path(__file__).parent / ".env"
    api_keys = dotenv_values(dotenv_path)  # some things want the key passed as a value
    load_dotenv(dotenv_path)  # some things want keys in the environment
    logger.info(f"found api keys ({api_keys})")
    return api_keys


if __name__ == "__main__":
    load_api_keys()
