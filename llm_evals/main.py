from os import getenv
import asyncio
from loguru import logger
from dotenv import load_dotenv

from utils import load_api_keys
from eval.eval import DemoMistral
from eval import claude_langsmith


async def main():
    logger.debug("Hello from llm_evals!")
    logger.debug(load_api_keys())
    dm = DemoMistral()
    multiturn = dm.run_multi_pipeline()
    logger.debug(multiturn)
    await claude_langsmith.main()


if __name__ == "__main__":
    asyncio.run(main())
