import openai
import os
import json
import config

import numpy as np
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load document embeddings and paths
document_embeddings = np.load('document_embeddings.npy', allow_pickle=True)
document_paths = np.load('document_paths.npy', allow_pickle=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup from config
api_base = config.api_base
api_key = config.api_key
deployment_id = config.deployment_id
search_endpoint = config.search_endpoint
search_key = config.search_key
search_index_name = config.search_index_name

# Initialize Azure OpenAI client
client = openai.AzureOpenAI(
    base_url=f"{api_base}/openai/deployments/{deployment_id}/extensions",
    api_key=api_key,
    api_version="2023-08-01-preview",
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def get_query_embedding(query):
    embedding = np.array(model.encode(query), dtype=float)
    return embedding

def get_relevant_document_paths(query_embedding, document_embeddings, threshold=0.55):
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    relevant_doc_indices = np.where(similarities >= threshold)[0]
    relevant_doc_paths = [document_paths[idx] for idx in relevant_doc_indices]
    return relevant_doc_paths

def get_chat_response(question: str):
    completion = client.chat.completions.create(
        model=deployment_id,
        messages=[
            {
                "role": "user",
                "content": question,
            },
        ],
        extra_body={
            "dataSources": [
                {
                    "type": "AzureCognitiveSearch",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "key": search_key,
                        "indexName": search_index_name,
                        "roleInformation": """As a sales engineer at Dutco Tennant, I'm responsible for introducing our product lineup to potential clients through our chatbot. These products, offered in the Abu Dhabi area, are designed to meet local specifications. Dutco Tennant is a leading provider of quality engineering and industrial solutions in the Middle East and Gulf region, known for its wide range of products catering to sectors like construction, infrastructure development, water and wastewater management, and many others. Your responses must be factual, clearly understandable, and helpful. 

                        If a product mentioned does not available, respond with: `Currently, this product is not available. For further details on our comprehensive range of products and services, please visit our website at https://www.dutcotennant.com/.` This approach ensures customers are always provided with a direct line to additional resources and the vast array of solutions Dutco Tennant offers.

                        And at the end of the answer please suggest some questions in python set format like example: {'question1', 'question2', 'question3'} and the questions are related to your knowledge about the product and also which are related to the current answers and questions."""
                    }
                }
            ]
        },
        stream=True
    )

    for chunk in completion:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            if hasattr(choice.delta, 'content') and choice.delta.content:
                yield choice.delta.content

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    
    print(f"question----{question}")

    response_chunks = get_chat_response(question)
    print(f"response_chunks---{response_chunks}")
    full_response = ""
    for chunk in response_chunks:
        full_response += chunk
        print(f"chunk-----{chunk}")
        socketio.emit('response', {'data': chunk})

    query_embedding = get_query_embedding(question)
    most_relevant_document_path = get_relevant_document_paths(query_embedding, document_embeddings)

    if most_relevant_document_path:
        full_response += f"\n\nFor more detailed information, please refer to: {most_relevant_document_path}"
        socketio.emit('response', {'data': f"\n\nFor more detailed information, please refer to: {most_relevant_document_path}"})

    return jsonify({"response": full_response})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
