
import time

from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage)
from custom_types import Wisdom
from services.llm_stream_handler import LlmStreamHandler

from config import settings


class LlmProvider:
    def _wisdom_to_model_name(self, wisdom: Wisdom) -> str:
        if wisdom == wisdom.MEDIUM:
            return "gpt-3.5-turbo"
        elif wisdom == wisdom.HIGH:
            return "gpt-3.5-turbo-16k"
        elif wisdom == wisdom.VERY_HIGH:
            return "gpt-4"
        else:
            raise ValueError(f"Unknown wisdom level: {wisdom}")

    def request_answer(self, prompt: str, reference: str = '', wisdom_level: Wisdom = Wisdom.MEDIUM) -> str:
        start_time = time.time()

        handler = LlmStreamHandler(reference)
        model = self._wisdom_to_model_name(wisdom_level)
        chat = ChatOpenAI(streaming=True, callbacks=[handler], temperature=0.2, model_name=model, client=None, openai_api_key=settings.open_ai_secrets.api_key)
        response = chat([SystemMessage(content=prompt)])

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Function 'prompt' took {elapsed_time:.2f} seconds")

        return response.content

    def get_search_query(self, prompt: str) -> str:
        start_time = time.time()

        chat = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo", client=None, openai_api_key=settings.open_ai_secrets.api_key)
        response = chat([SystemMessage(content=prompt)])

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Function 'prompt' took {elapsed_time:.2f} seconds")

        return response.content

