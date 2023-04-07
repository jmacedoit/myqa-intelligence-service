
from langchain import OpenAI
from config import settings


class LlmProvider:
    def prompt(self, text: str):
        llm = OpenAI(temperature=0.5, model_name="gpt-4",
                     client=None, openai_api_key=settings.open_ai.api_key)

        response = llm.generate([text])

        return response 
