import time
import asyncio
import backoff
from httpx import HTTPStatusError
from loguru import logger
from tqdm import tqdm
from utils import load_api_keys
from langchain_anthropic import ChatAnthropic
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

MISTRAL_MODELS = [
    "mistral-large-latest",
    "labs-leanstral-2603",
]

ANTHROPIC_MODELS = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]

OPENAI_MODELS = ["gpt-5-nano"]

NVIDIA_MODELS = ["nvidia/nvidia-nemotron-nano-9b-v2"]

TEST_CONTEXT = """
You are a helpful question answering assistant.  Keep your answers as concise as possible. Be sure to support your answer if the query could be considered opaque.
"""

base_message_llm_under_test = [("system", TEST_CONTEXT)]


JUDGE_CONTEXTS = """
                  You are the judge of an assistant. 
                  To judge, you will be provided the same query the assistant recieved, and the answer the assistant gave.
                  You should judge the assistant based on accuracy, precision, and conciseness.
                  Your judgement should be a number from 1-10 where 10 is perfect and 1 is 
                  Do not return anything but a single number.
                  """

base_message_judge = [("system", JUDGE_CONTEXTS)]


def get_test_prompts() -> list[str]:
    return [
        "Explain the difference between a list and a tuple in Python.",
        "What are three common causes of inflation in an economy?",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Write a haiku about autumn leaves falling.",
        "What is the time complexity of quicksort in the average case?",
        "Translate 'Where is the nearest train station?' into French.",
        "What are the main differences between supervised and unsupervised learning?",
        "Give me a recipe for a simple tomato pasta sauce.",
        "Explain what a black hole is to a ten-year-old.",
        "What are three strategies for improving sleep quality?",
    ]


def get_llm_under_test(model_id: str = None, context: str = TEST_CONTEXT):
    """Given a model id, set it up, and return it"""
    return ChatAnthropic(model=ANTHROPIC_MODELS[0])


def get_llm_judges(model_ids: list[str] = []):
    """
    Given a list of model ids, set them up, and return them.
    Given them each different personas via context.
    """
    judges = [
        ChatMistralAI(model=MISTRAL_MODELS[0]),
        ChatNVIDIA(model=NVIDIA_MODELS[0]),
        # ChatOpenAI(model=OPENAI_MODELS[0]),  # I have 0 quota here...
        ChatAnthropic(model=ANTHROPIC_MODELS[1]),
    ]
    return judges


@backoff.on_exception(backoff.expo, HTTPStatusError, base=5, max_value=90)
async def test_prompt(llm_under_test=None, judges=None, prompt: str = None):
    ai_message = await llm_under_test.ainvoke(
        base_message_llm_under_test + [("human", prompt)]
    )
    judge_message = base_message_judge + [
        (
            "human",
            f"Here is the prompt given to the assistant '{prompt}',and the answer from the assistant we are testing '{ai_message.content}'",
        )
    ]
    return [(await judge.ainvoke(judge_message)).content for judge in judges]


async def run_eval_asynch(llm_under_test=None, judges=None):
    """
    use the list of test prompts returned by get_test_prompts()
    as inputs to an LLM, called llm_under_test.  Use every answer to the prompt
    provided by llm_under_test, and the prompt, as input to a panel of judges.
    The judges return a score, low is bad high is good.

    Do this asynchronously.  Account for rate limiting using the backoff library.
    """
    if judges == None or llm_under_test == None:
        logger.error("You must provide pre-initialized judges and llm_under_test")
        exit(1)

    print(f"[async] started at {time.strftime('%X')}")
    async with asyncio.TaskGroup() as tg:
        for prompt in tqdm(get_test_prompts()):
            tg.create_task(
                test_prompt(llm_under_test=llm_under_test, judges=judges, prompt=prompt)
            )
    print(f"[async] finished at {time.strftime('%X')}")


@backoff.on_exception(backoff.expo, HTTPStatusError, max_time=90)
async def run_eval_synchro(llm_under_test=None, judges=None):
    """
    use the list of test prompts returned by get_test_prompts()
    as inputs to an LLM, called llm_under_test.  Use every answer to the prompt
    provided by llm_under_test, and the prompt, as input to a panel of judges.
    The judges return a score, low is bad high is good.
    """
    if judges == None or llm_under_test == None:
        logger.error("You must provide pre-initialized judges and llm_under_test")
        exit(1)
    scores = []
    print(f"[synchro] started at {time.strftime('%X')}")
    for prompt in tqdm(get_test_prompts()):
        # get the answer from the llm under test
        ai_message = llm_under_test.invoke(
            base_message_llm_under_test + [("human", prompt)]
        )
        # logger.debug(f"{prompt}\n\n{ai_message.content}\n")
        judge_messages = base_message_judge + [
            (
                "human",
                f"Here is the prompt given to the assistant '{prompt}',and the answer from the assistant we are testing '{ai_message.content}'",
            )
        ]
        scores.append([judge.invoke(judge_messages).content for judge in judges])
    print(f"[synchro] finished at {time.strftime('%X')}")


async def main():
    load_api_keys()
    llm_under_test = get_llm_under_test()
    judges = get_llm_judges()

    await run_eval_asynch(llm_under_test=llm_under_test, judges=judges)

    await run_eval_synchro(llm_under_test=llm_under_test, judges=judges)


if __name__ == "__main__":
    asyncio.run(main())
