
import json

from flask import request
from flask import Blueprint
from .utils.chunks import group_chunks_by_resource_id, order_and_sew_info_chunks

from services.embeddings_store import CollectionEmbeddingsStore, ResourceChunkInfo

chunks_blueprint = Blueprint('chunks', __name__)

@chunks_blueprint.route('/chunks-retrieval', methods=['POST'])
def retrieve_chunks():
    # Get chunk ids from request
    request_body = request.get_json()

    if 'knowledge_base_id' not in request_body:
        return "Missing knowledge base id", 400

    if 'chunk_ids' not in request_body:
        return "Missing chunk ids", 400

    chunk_ids = request_body['chunk_ids']
    knowledge_base_id = request_body['knowledge_base_id']

    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)
    embeddings_store.setup(True)
    chunks_data = embeddings_store.get_chunks_data(chunk_ids)

    grouped_chunks = group_chunks_by_resource_id(chunks_data)

    for resource_id, chunks in grouped_chunks.items():
        grouped_chunks[resource_id] = order_and_sew_info_chunks(chunks)

    # get list of all chunks from grouped chunks
    all_chunks: list[ResourceChunkInfo] = []
    for resource_id, chunks in grouped_chunks.items():
        all_chunks += chunks

    return {
        'chunks_data': [{
            **chunk_data,
            'id': str(chunk_data['id']),
            'payload': json.loads(chunk_data['payload'])
        } for chunk_data in all_chunks]
    }, 200
