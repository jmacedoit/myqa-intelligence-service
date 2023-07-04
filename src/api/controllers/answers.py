
import json
from collections import defaultdict
from typing import Literal, TypedDict, Union, cast

from flask import jsonify, request
from flask import Blueprint

from services.embeddings_calculator import EmbeddingsCalculator
from services.embeddings_store import CollectionEmbeddingsStore, ResourceChunkInfo
from services.llm_provider import LlmProvider

from logger import logger


answers_blueprint = Blueprint('answers', __name__)

class ConversationEntry(TypedDict):
    sender: Literal['USER, AI_ENGINE']
    content: str

@answers_blueprint.route('/answer-request', methods=['POST'])
async def add_answer_request():
    request_data = request.get_json()
    past_conversation = cast(list[ConversationEntry], request_data['conversation']) if 'conversation' in request_data else None
    knowledge_base_id = request_data['knowledge_base_id']
    question = request_data['question']
    reference: str = request_data['reference']

    if past_conversation is not None:
        search_query = await get_search_query_from_conversation(question, past_conversation)
    else:
        search_query = question

    logger.info(f"Search query: {search_query}")

    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)
    embeddings_store.setup() 

    embeddings_calculator = EmbeddingsCalculator()

    search_query_embeddings = embeddings_calculator.embed_documents([search_query])

    similar_chunks_with_similarity = embeddings_store.search_similar_chunks(search_query_embeddings[0])
    
    prompt = build_qa_llm_prompt(question, list(map(lambda x: x[0], similar_chunks_with_similarity)), past_conversation)

    logger.info(f"Prompt: {prompt}")

    llm = LlmProvider();
    response = llm.prompt(prompt, reference)

    return jsonify({
        'answer': response
    }), 200



def build_search_query_prompt(question: str, past_conversation: list[ConversationEntry]) -> str:
    prompt = ""
    prompt += "<<PAST_CONVERSATION>>\n"

    for entry in past_conversation:
        prompt += f"{entry['sender']}: {entry['content']}\n"

    prompt += "<</PAST_CONVERSATION>>\n\n"
    prompt += f"<<QUESTION>>\n{question}\n<</QUESTION>>\n\nInstruction: Given the question provided in <<QUESTION>> and the past conversation provided in <<PAST_CONVERSATION>>, what would be the best search query to find the answer to the question? Just return a JSON object obeying this format: {{ \"search_query\": <search_query> }}. If you are not sure just return {{ \"search_query\": null }}"

    return prompt

async def get_search_query_from_conversation(question: str, past_conversation: list[ConversationEntry]) -> str:
    prompt = build_search_query_prompt(question, past_conversation)

    llm = LlmProvider()
    response = await llm.async_prompt([prompt])

    return response[0]

def order_and_sew_info_chunks(info_chunks: list[ResourceChunkInfo]) -> list[ResourceChunkInfo]:
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

def build_qa_llm_prompt(question: str, relevant_info_chunks: list[ResourceChunkInfo],  past_conversation: Union[list[ConversationEntry], None]) -> str:
    grouped_chunks = group_chunks_by_resource_id(relevant_info_chunks)
    context = ""

    for resource_id, chunks in grouped_chunks.items():
        resource_name = chunks[0]['resource_name']

        rearranged_info_chunks = order_and_sew_info_chunks(chunks)
        resource_info = "".join([
            f"{segment['data']}\n[...]\n"
            for i, segment in enumerate(rearranged_info_chunks)
        ])
        
        context += f"<<SOURCE {resource_name}>>\n{resource_info}\n<</SOURCE {resource_name}>>\n"


        

    previous_conversation = ""
    if past_conversation is not None:
        for entry in past_conversation:
            previous_conversation += f"{entry['sender']}: {entry['content']}\n"


    return f"<<PREVIOUS_CONVERSATION>>\n{previous_conversation}<</PREVIOUS_CONVERSATION>>\n\n<<SOURCES>>\n{context}<</SOURCES>>\n\n<<QUESTION>>\n{question}\n<</QUESTION>>\n\nInstruction: First, detect the language of the text inside <<QUESTION>> tags. Then, answer the question using only information from the sources and nothing else. The answer must be in the same language as the question (no need to mention the language in the answer). If the answer can't be determined from the sources or you are not sure it can, explain you don't know."


def group_chunks_by_resource_id(chunks: list[ResourceChunkInfo]) -> dict[str, list[ResourceChunkInfo]]:
    grouped_chunks = defaultdict(list)
    for chunk in chunks:
        resource_id = chunk['resource_id']
        grouped_chunks[resource_id].append(chunk)

    return grouped_chunks