
import json
import os
import tempfile
import time

from flask import request
from flask import Blueprint
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from services.embeddings_calculator import EmbeddingsCalculator
from services.embeddings_store import ResourceChunkInfo, CollectionEmbeddingsStore
from logger import logger

knowledge_bases_blueprint = Blueprint('knowledge_base', __name__)

@knowledge_bases_blueprint.route('/knowledge-base/<knowledge_base_id>/resource/<resource_id>', methods=['POST'])
def assimilate_resource(knowledge_base_id: str, resource_id: str):
    if 'file' not in request.files:
        return 'Missing file', 400

    file = request.files['file']
    resource_name = file.filename

    # Start timing
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, resource_name)  # type: ignore
        file.save(file_path)

        if file.mimetype == 'application/pdf':
            loader = PyPDFLoader(file_path)

            logger.debug(f"Used pdf file loader for {file_path}")
        else:
            loader = UnstructuredFileLoader(
                file_path,
                strategy='fast'
            )

            logger.debug(f"Used unstructured file loader for {file_path}")

        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        length_function=len,
    )

    texts = text_splitter.split_documents(docs)

    if len(texts) > settings.limits.max_total_chunks:
        return 'Too many chunks', 400
    
    # End timing
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Log the elapsed time
    logger.debug(f"Loading file took {elapsed_time:.2f} seconds")

    embeddings_calculator = EmbeddingsCalculator();
    embeddings_result = embeddings_calculator.embed_documents([text.page_content for text in texts])

    cumulative_character_count = [0]
    for text in texts:
        cumulative_character_count.append(cumulative_character_count[-1] + len(text.page_content))

    if cumulative_character_count[-1] > settings.limits.max_total_characters:
        return 'Too many characters', 400

    rows = [
        ResourceChunkInfo(
            id=None,
            resource_name=str(resource_name),
            data=texts[i].page_content,
            embeddings=embeddings_result[i],
            resource_id=resource_id,
            payload=json.dumps({
                'total_chunks': len(texts),
                'percentage_in': cumulative_character_count[i] / cumulative_character_count[-1],
                'chunk_number': i,
                'resource_mimetype': file.mimetype,
                'page_index': texts[i].metadata.get('page', None) if file.mimetype == 'application/pdf' else None
            })
        ) for i in range(len(texts))
    ]

    start_time = time.time()

    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)
    embeddings_store.setup(True)
    embeddings_store.delete_resource_chunks(resource_id)
    embeddings_store.insert_resource_chunks(rows)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Inserting into database took {elapsed_time:.2f} seconds")

    logger.info(f"Assimilated {resource_name} into knowledge base {knowledge_base_id}")

    processed_data_stats = {
        'total_chunks': len(texts),
        'total_characters': cumulative_character_count[-1]
    }

    return json.dumps(processed_data_stats), 200


@knowledge_bases_blueprint.route('/knowledge-base/<knowledge_base_id>/resource/<resource_id>', methods=['DELETE'])
def remove_resource(knowledge_base_id: str, resource_id: str):
    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id,
                                                 host=settings.milvus.host, port=settings.milvus.host)
    
    embeddings_store.setup()
    embeddings_store.delete_resource_chunks(resource_id)

    logger.info(f"Removed resource {resource_id} from knowledge base {knowledge_base_id}")

    return 'OK', 200


@knowledge_bases_blueprint.route('/knowledge-base/<knowledge_base_id>', methods=['DELETE'])
def remove_knowledge_base(knowledge_base_id: str):
    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id,
                                                 host=settings.milvus.host, port=settings.milvus.host)

    embeddings_store.drop_collection()

    logger.info(f"Removed knowledge base {knowledge_base_id}")

    return 'OK', 200
