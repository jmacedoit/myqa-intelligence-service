
import json
from typing import Literal, TypedDict, Union, cast

from flask import jsonify, request
from flask import Blueprint
from api.controllers.utils.language import get_language_name

from custom_types import Wisdom
from .utils.chunks import group_chunks_by_resource_id, order_and_sew_info_chunks

from config import settings
from services.embeddings_calculator import EmbeddingsCalculator
from services.embeddings_store import CollectionEmbeddingsStore, ResourceChunkInfo
from services.llm_provider import LlmProvider

from logger import logger

answers_blueprint = Blueprint('answers', __name__)

class ConversationEntry(TypedDict):
    sender: Literal['USER, AI_ENGINE']
    content: str

@answers_blueprint.route('/answer-request', methods=['POST'])
def add_answer_request():
    request_data = request.get_json()
    past_conversation = cast(list[ConversationEntry], request_data['conversation']) if 'conversation' in request_data else None
    knowledge_base_id = request_data['knowledge_base_id']
    question = request_data['question']
    reference: str = request_data['reference']

    if past_conversation is not None:
        search_query = get_search_query_from_conversation(question, past_conversation)
    else:
        search_query = question

    language = request_data['language'] if 'language' in request_data else None
    wisdom_level: Wisdom = Wisdom[request_data['wisdom_level']] if 'wisdom_level' in request_data else Wisdom.MEDIUM

    logger.info(f"Search query: {search_query}")

    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)
    embeddings_store.setup() 

    embeddings_calculator = EmbeddingsCalculator()
    search_query_embeddings = embeddings_calculator.embed_documents([search_query])

    def wisdom_to_n_similar_chunks(wisdom: Wisdom) -> int:
        if wisdom == Wisdom.MEDIUM:
            return 7
        elif wisdom == Wisdom.HIGH:
            return 12
        elif wisdom == Wisdom.VERY_HIGH:
            return 12
        else:
            raise ValueError(f"Unknown wisdom level: {wisdom}")

    similar_chunks_with_similarity = embeddings_store.search_similar_chunks(
        search_query_embeddings[0],
        limit=wisdom_to_n_similar_chunks(wisdom_level)
    )
    similar_chunks_with_similarity: list[tuple[ResourceChunkInfo, float]] = list(filter(lambda x: x[1] > settings.answers.minimum_trustable_similarity, similar_chunks_with_similarity))
    similar_chunks = list(map(lambda x: x[0], similar_chunks_with_similarity))

    prompt = build_qa_llm_prompt(question, similar_chunks, past_conversation, language)

    logger.debug(f"Prompt: {prompt}")

    llm = LlmProvider();
    response = llm.request_answer(prompt, reference, wisdom_level)

    sources = []
    for chunk in similar_chunks:
        payload = json.loads(chunk['payload'])
        sources.append({
            'chunk_id': str(chunk['id']),
            'file_name': chunk['resource_name'],
            'resource_name': chunk['resource_name'],
            'resource_id': chunk['resource_id'],
            'chunk_number': payload['chunk_number'],
            'percentage_in': payload['percentage_in'],
            'resource_mimetype': payload['resource_mimetype'],
            'page_index': payload['page_index']
        });

    return jsonify({
        'answer': response,
        'sources': sources
    }), 200



def build_search_query_prompt(question: str, past_conversation: list[ConversationEntry]) -> str:
    prompt = ""
    prompt += "<<PAST_CONVERSATION>>\n"

    for entry in past_conversation[:5]:
        prompt += f"{entry['sender']}: {entry['content']}\n"

    prompt += "<</PAST_CONVERSATION>>\n\n"
    prompt += f"<<QUESTION>>\n{question}\n<</QUESTION>>\n\nInstruction: Given the question provided in <<QUESTION>> and the past conversation provided in <<PAST_CONVERSATION>>, what would be the best search query to find the answer to the question? Just return a JSON object obeying this format: {{ \"search_query\": <search_query> }}. If you are not sure just return {{ \"search_query\": null }}"

    return prompt

def get_search_query_from_conversation(question: str, past_conversation: list[ConversationEntry]) -> str:
    prompt = build_search_query_prompt(question, past_conversation)

    llm = LlmProvider()
    response = llm.get_search_query(prompt)

    return response


def build_qa_llm_prompt(
    question: str,
    relevant_info_chunks: list[ResourceChunkInfo],
    past_conversation: Union[list[ConversationEntry], None],
    language: Union[str, None] = None
) -> str:
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
        for entry in past_conversation[-5:]:
            previous_conversation += f"{entry['sender']}: {entry['content']}\n"

    previous_conversation_part = f"<<PREVIOUS_CONVERSATION>>\n{previous_conversation}<</PREVIOUS_CONVERSATION>>"
    language_part = "The answer must be in the same language as the question (no need to mention the language in the answer)."

    if language is not None:
        language_part = f"The answer must be in {get_language_name(language)} (no need to mention the language in the answer)."

    return f"<<PREVIOUS_CONVERSATION>>\n{previous_conversation}<</PREVIOUS_CONVERSATION>>\n\n<<SOURCES>>\n{context}<</SOURCES>>\n\n{previous_conversation_part}<<QUESTION>>\n{question}\n<</QUESTION>>\n\nInstruction: First, detect the language of the text inside <<QUESTION>> tags. Then, thoroughly answer the question having into account the past conversation using only information from the sources and nothing else. {language_part} If the answer can't be determined from the sources or you are not sure it can, explain you don't know. Use of markdown to format the answer is encouraged, titles, lits, tables, bolds, italics, code blocks etc are allowed."