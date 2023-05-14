
import json
import os
import tempfile

from flask import request
from flask import Blueprint
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import settings

from services.embeddings_calculator import EmbeddingsCalculator
from services.embeddings_store import ResourceChunkInfo, CollectionEmbeddingsStore

knowledge_bases_blueprint = Blueprint('knowledge_base', __name__)

@knowledge_bases_blueprint.route('/knowledge-base/<knowledge_base_id>/resource/<resource_id>', methods=['POST'])
def assimilate_resource(knowledge_base_id: str, resource_id: str):
    if 'file' not in request.files:
        return 'Missing file', 400

    file = request.files['file']
    resource_name = file.filename

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, resource_name) # type: ignore
        file.save(file_path)
        loader = UnstructuredFileLoader(
            file_path, strategy='fast')
        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        length_function=len,
    )

    texts = text_splitter.create_documents([docs[0].page_content])

    embeddings_calculator = EmbeddingsCalculator();

    embeddings_result = embeddings_calculator.embed_documents(
        [text.page_content for text in texts])
 
    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)

    rows = [
        ResourceChunkInfo(
            resource_name=str(resource_name),
            data=texts[i].page_content,
            embeddings=embeddings_result[i],
            resource_id=resource_id,
            payload=json.dumps({
                'chunk_number': i,
                'resource_mimetype': file.mimetype
            })
        ) for i in range(len(texts))
    ]

    embeddings_store.setup(True) 
    embeddings_store.delete_resource_chunks(resource_id)
    embeddings_store.insert_resource_chunks(rows)

    print(f"Assimilated {resource_name} to knowledge base {knowledge_base_id}")

    return 'OK', 200


@knowledge_bases_blueprint.route('/knowledge-base/<knowledge_base_id>/resource/<resource_id>', methods=['DELETE'])
def remove_resource(knowledge_base_id: str, resource_id: str):
    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id,
                                                 host=settings.milvus.host, port=settings.milvus.host)
    
    embeddings_store.setup()
    embeddings_store.delete_resource_chunks(resource_id)

    return 'OK', 200


@knowledge_bases_blueprint.route('/knowledge-base/<knowledge_base_id>', methods=['DELETE'])
def remove_knowledge_base(knowledge_base_id: str):
    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id,
                                                 host=settings.milvus.host, port=settings.milvus.host)

    embeddings_store.drop_collection()

    return 'OK', 200
