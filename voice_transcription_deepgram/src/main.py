from os import getenv

from loguru import logger
from dotenv import load_dotenv

from agents.utils import load_api_keys

def main():
    logger.debug("Hello from mcp-playground!")
    logger.debug(load_api_keys())



if __name__ == "__main__":
    main()
