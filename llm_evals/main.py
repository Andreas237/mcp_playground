from os import getenv

from loguru import logger
from dotenv import load_dotenv

from utils import load_api_keys
from eval.eval import DemoMistral


def main():
    logger.debug("Hello from llm_evals!")
    logger.debug(load_api_keys())
    dm = DemoMistral()
    multiturn = dm.run_multi_pipeline()
    logger.debug(multiturn)


if __name__ == "__main__":
    main()
