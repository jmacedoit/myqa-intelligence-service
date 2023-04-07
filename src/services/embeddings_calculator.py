
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from config import settings

class EmbeddingsCalculator:
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings_maker = OpenAIEmbeddings(
            document_model_name="text-embedding-ada-002", openai_api_key=settings.open_ai.api_key, client=None) 

        embeddings_result = embeddings_maker.embed_documents(documents)

        return embeddings_result
