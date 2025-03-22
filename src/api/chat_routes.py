from flask import Blueprint, request, jsonify
from services.embedding_service import EmbeddingService
from openai import OpenAI
from utils.config import Config

chat_bp = Blueprint('chat', __name__)
client = OpenAI(base_url=Config.OLLAMA_BASE_URL, api_key=Config.OLLAMA_MODEL)
embedding_service = EmbeddingService()

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Get relevant context
        relevant_context = embedding_service.get_relevant_context(user_input)
        
        # Format the message with context
        messages = [
            {"role": "system", "content": "You are a mechanical engineering expert."},
            {"role": "user", "content": f"Context: {relevant_context}\nQuestion: {user_input}"}
        ]

        # Get response from LLM
        response = client.chat.completions.create(
            model=Config.OLLAMA_MODEL,
            messages=messages
        )

        return jsonify({
            'response': response.choices[0].message.content,
            'context': relevant_context
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500 