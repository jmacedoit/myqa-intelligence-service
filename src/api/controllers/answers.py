
import json
import threading
from typing import Any, List
from flask import jsonify, request
from flask import Blueprint

from services.embeddings_calculator import EmbeddingsCalculator
from services.embeddings_store import CollectionEmbeddingsStore, ResourceChunkInfo

from services.llm_provider import LlmProvider

answers_blueprint = Blueprint('answers', __name__)

@answers_blueprint.route('/answer-request', methods=['POST'])
def add_answer_request():
    request_data = request.get_json()
    knowledge_base_id = request_data['knowledgeBaseId']
    question = request_data['question']
    reference: str = request_data['reference']

    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)
    embeddings_store.setup() 

    embeddings_calculator = EmbeddingsCalculator()

    question_embeddings = embeddings_calculator.embed_documents([question])

    similar_chunks_with_similarity = embeddings_store.search_similar_chunks(
        question_embeddings[0])
    
    prompt = build_qa_llm_prompt(question, list(map(lambda x: x[0], similar_chunks_with_similarity)))

    llm = LlmProvider();
    response = llm.prompt(prompt, reference)

    return jsonify({
        'answer': response
    }), 200

def order_and_sew_info_chunks(info_chunks: List[ResourceChunkInfo]) -> List[ResourceChunkInfo]:
    # First we need to sort the chunks based on their chunk_number
    sorted_chunks = sorted(
        info_chunks, 
        key=lambda chunk: (chunk['resource_id'], json.loads(chunk['payload'])['chunk_number'])
    )

    sewed_chunks = []
    previous_chunk_number = -1
    previous_resource_id = None

    for i in range(len(sorted_chunks)):
        current_chunk_number = json.loads(sorted_chunks[i]['payload'])['chunk_number']
        current_resource_id = sorted_chunks[i]['resource_id']

        # If it's the first chunk or the chunk belongs to a different resource or 
        # the chunk_number is not one greater than the previous chunk_number, just append it to the list
        if i == 0 or previous_resource_id != current_resource_id or previous_chunk_number + 1 != current_chunk_number:
            sewed_chunks.append(sorted_chunks[i])
        else:
            # If not the first chunk of the resource and the chunk_number is one greater than 
            # the previous chunk_number, we need to find the overlap and merge the non-overlapping part
            previous_chunk_data = sorted_chunks[i - 1]['data']
            current_chunk_data = sorted_chunks[i]['data']
            overlap = find_overlap(previous_chunk_data, current_chunk_data)
            
            # Merge the non-overlapping part of the current chunk with the last chunk in the list
            sewed_chunks[-1]['data'] += current_chunk_data[len(overlap):]

        previous_chunk_number = current_chunk_number
        previous_resource_id = current_resource_id

    return sewed_chunks

def find_overlap(str1: str, str2: str) -> str:
    end_offset = min(len(str1), len(str2))
    for i in range(end_offset, 0, -1):
        if str1.endswith(str2[:i]):
            return str2[:i]
    return ""

def build_qa_llm_prompt(question: str, relevant_info_chunks: List[ResourceChunkInfo]) -> str:
    rearranged_info_chunks = order_and_sew_info_chunks(relevant_info_chunks)

    context = " ".join([f"<<INFO_{i}>>. {chunk['data']} <</INFO_{i}>>\n" for i, chunk in enumerate(rearranged_info_chunks)])

    return f"<<CONTEXT>> {context} <</CONTEXT>> <<QUESTION>> {question} <</QUESTION>> Instruction: First, detect the language of the text inside <<QUESTION>> tags. Then, answer the question using only information from the context and nothing else. The answer must be in the same language as the question (no need to mention the language in the answer). If the answer can't be determined from the context or you are not sure explain you don't know."
