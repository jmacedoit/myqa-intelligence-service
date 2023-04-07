
from typing import List
from flask import jsonify, request
from flask import Blueprint

from services.embeddings_calculator import EmbeddingsCalculator
from services.embeddings_store import CollectionEmbeddingsStore, ResourceChunkInfo

from langchain.llms import OpenAI

from services.llm_provider import LlmProvider

answers_blueprint = Blueprint('answers', __name__)

@answers_blueprint.route('/answer-request', methods=['POST'])
def add_answer_request():
    request_data = request.get_json()
    knowledge_base_id = request_data['knowledgeBaseId']
    question = request_data['question']

    embeddings_store = CollectionEmbeddingsStore(collection_name=knowledge_base_id)
    embeddings_store.setup() 

    embeddings_calculator = EmbeddingsCalculator()

    question_embeddings = embeddings_calculator.embed_documents([question])

    similar_chunks_with_similarity = embeddings_store.search_similar_chunks(
        question_embeddings[0])
    
    prompt = build_qa_llm_prompt(question, list(
        map(lambda x: x[0], similar_chunks_with_similarity)))

    llm = LlmProvider();
    response = llm.prompt(prompt)

    print(len(prompt))

    return jsonify({
        'answer': response.generations[0][0].text
    }), 200

def build_qa_llm_prompt(question: str, relevant_info_chunks: List[ResourceChunkInfo]) -> str:
    context = " ".join([f"<<INFO {i}>>. {chunk['data']} <<END_INFO {i}>>\n" for i, chunk in enumerate(relevant_info_chunks)])

    return f"Given the following gathered information bits as base knowledge: {context}\n Answer the following question in the same language it is being asked. Don't let the user know you were given an explicit context: {question}"
