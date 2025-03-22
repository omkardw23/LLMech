import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    UPLOAD_FOLDER = 'uploads'
    CADOUTPUTS_DIR = 'cadoutputs'
    ALLOWED_EXTENSIONS = {'pdf'}
    OLLAMA_MODEL = "llama3.2:3b"
    OLLAMA_BASE_URL = 'http://localhost:11434/v1'

    @staticmethod
    def initialize_directories():
        for directory in [Config.UPLOAD_FOLDER, Config.CADOUTPUTS_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory) 