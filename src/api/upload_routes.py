from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from utils.config import Config
from services.embedding_service import EmbeddingService

upload_bp = Blueprint('upload', __name__)
embedding_service = EmbeddingService()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Reinitialize embeddings after new file upload
            embedding_service.initialize_embeddings()
            
            return jsonify({'message': 'File successfully uploaded'}), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500 