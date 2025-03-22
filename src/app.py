from flask import Flask
from utils.config import Config
from api.chat_routes import chat_bp
from api.cad_routes import cad_bp
from api.upload_routes import upload_bp
from services.embedding_service import EmbeddingService

def create_app():
    app = Flask(__name__, template_folder='../templates')
    
    # Initialize configurations
    Config.initialize_directories()
    
    # Initialize services
    embedding_service = EmbeddingService()
    embedding_service.initialize_embeddings()
    
    # Register blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(cad_bp)
    app.register_blueprint(upload_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
