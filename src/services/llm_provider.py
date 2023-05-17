
from langchain import OpenAI
from config import settings
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage)
from services.llm_stream_handler import LlmStreamHandler


class LlmProvider:
    def prompt(self, text: str, reference: str = '') -> str:
        handler = LlmStreamHandler(reference)
        chat = ChatOpenAI(streaming=True, callbacks=[handler], temperature=0.1, model_name="gpt-3.5-turbo", client=None, openai_api_key=settings.open_ai.api_key)

        response = chat([SystemMessage(content=text)])

        return response.content
