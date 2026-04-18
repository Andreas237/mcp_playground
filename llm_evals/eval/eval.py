from langsmith import traceable

from loguru import logger
from langchain.chat_models import init_chat_model
from langchain_mistralai.chat_models import ChatMistralAI


class DemoMistral:
    # https://docs.langchain.com/oss/python/integrations/chat/mistralai
    def __init__(self):
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_retries=2,
        )

    @traceable
    def format_prompt(self, subject):
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"What's a good name for a store that sells {subject}?",
            },
        ]

    @traceable(run_type="llm")
    def invoke_llm(self, messages):
        return self.llm.invoke(messages)

    @traceable
    def parse_output(self, response):
        return response.content

    @traceable
    def run_pipeline(
        self,
    ):
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        response = self.invoke_llm(messages)
        return self.parse_output(response)

    @traceable
    def run_multi_pipeline(
        self,
    ):
        messages = []
        quit = False
        query = input("Enter a query and hit [Enter]: ")
        while not quit:
            messages.append(("human", query))
            response = self.invoke_llm(messages)
            ai_message = self.parse_output(response)
            messages.append(("ai", ai_message))
            print(f"LLM says: {ai_message}")
            query = input('Enter a query and hit [Enter], or "quit" to exit: ')
            if query.lower() == "quit":
                quit = True
        logger.info("Exiting")
        return messages


if __name__ == "__main__":
    dm = DemoMistral()
    response_content = dm.run_pipeline()
    logger.debug(response_content)

    multiturn = dm.run_multi_pipeline()
    logger.debug(multiturn)
