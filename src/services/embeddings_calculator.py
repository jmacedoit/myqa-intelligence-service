
import time
from sentence_transformers import SentenceTransformer

import numpy as np

from logger import logger

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').eval()

class EmbeddingsCalculator:
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        embeddings_result: list[list[float]] = []

        logger.debug(f"Calculating embeddings for {len(documents)} content segments")
        start_time = time.time()

        embeddings_result = self.calculate_embeddings((documents, 0, len(documents)))

        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.debug(f"Embedding operation took {elapsed_time:.2f} seconds")

        return embeddings_result

    def calculate_embeddings(self, batch: tuple[list[str], int, int]) -> list[list[float]]:
        logger.debug(f"Calculating embeddings for content segments {batch[1]} to {batch[2]}")

        batch_embeddings = model.encode(batch[0], show_progress_bar=False)
        normalized_batch_embeddings = [embedding / np.linalg.norm(embedding) for embedding in batch_embeddings]

        return [embedding.tolist() for embedding in normalized_batch_embeddings]