import requests
import torch
import ollama
import os
from openai import OpenAI
import argparse
import json
from flask import Flask, request, jsonify, render_template
import logging
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import re
from kittycad.client import Client
import dotenv
import pyvista as pv
import numpy as np
import trimesh
import base64
import io
import tempfile
from datetime import datetime
import shutil
import time
from kittycad.api.ml import create_text_to_cad, get_text_to_cad_model_for_user
from kittycad.client import ClientFromEnv
from kittycad.models.file_export_format import FileExportFormat
from kittycad.models.text_to_cad_create_body import TextToCadCreateBody
from kittycad.models.api_call_status import ApiCallStatus
from kittycad.models.error import Error
from kittycad.models.text_to_cad import TextToCad
from pygltflib import GLTF2 

dotenv.load_dotenv()

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    input_tensor = torch.tensor(input_embedding)
    
    # Ensure dimensions match
    if input_tensor.shape[0] != vault_embeddings.shape[1]:
        # Pad or truncate input_tensor to match vault_embeddings dimension
        if input_tensor.shape[0] < vault_embeddings.shape[1]:
            padding = torch.zeros(vault_embeddings.shape[1] - input_tensor.shape[0])
            input_tensor = torch.cat([input_tensor, padding])
        else:
            input_tensor = input_tensor[:vault_embeddings.shape[1]]
    
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(input_tensor.unsqueeze(0), vault_embeddings)
    
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input_json, conversation_history, ollama_model):
    user_input = json.loads(user_input_json)["Query"]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    rewritten_query = response.choices[0].message.content.strip()
    return json.dumps({"Rewritten Query": rewritten_query})
   
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
        rewritten_query_data = json.loads(rewritten_query_json)
        rewritten_query = rewritten_query_data["Rewritten Query"]
        print(PINK + "Original Query: " + user_input + RESET_COLOR)
        print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3.2:3b", help="Ollama model to use (default: llama3.2:3b)")
args = parser.parse_args()

# Update the Flask app initialization to look for templates in the parent directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Initialize global variables
client = None
vault_embeddings_tensor = None
vault_content = []
conversation_history = []
system_message = """You are a mechanical engineering expert specializing in mechanical design 
and assembly. When given design requirements, provide a concise response in two short paragraphs:
First paragraph should describe all components with their essential dimensions and specifications 
in a flowing, natural way. Second paragraph should describe the assembly process as a clear sequence 
of steps in a natural, flowing way. Keep responses brief but include all critical information. 
Use plain language and avoid bullet points or numbered lists. Focus only on essential specifications 
and clear assembly instructions. Also no explanations allowed even if you don't know something. just answer what you know."""


# Add KittyCAD API configuration
KITTYCAD_API_KEY = os.getenv("ZOO_API_TOKEN")
kittycad_client = Client(token=KITTYCAD_API_KEY)

# Add these configurations after creating the Flask app
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_cad_model(llm_response):
    try:
        logging.info("Starting CAD generation with KittyCAD...")
        
        # Create the initial request
        response = create_text_to_cad.sync(
            client=kittycad_client,
            output_format=FileExportFormat.STEP,
            body=TextToCadCreateBody(
                prompt=llm_response,
            ),
        )

        if isinstance(response, Error) or response is None:
            logging.error(f"Initial KittyCAD error: {response}")
            return None

        result: TextToCad = response

        # Polling to check if the task is complete
        while result.completed_at is None:
            logging.info("Waiting for CAD generation to complete...")
            time.sleep(5)

            response = get_text_to_cad_model_for_user.sync(
                client=kittycad_client,
                id=result.id,
            )

            if isinstance(response, Error) or response is None:
                logging.error(f"Polling error: {response}")
                return None

            result = response

        if result.status == ApiCallStatus.FAILED:
            logging.error(f"Text-to-CAD failed: {result.error}")
            return None

        elif result.status == ApiCallStatus.COMPLETED:
            if result.outputs is None:
                logging.error("Text-to-CAD completed but returned no files.")
                return None

            # Save the files
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_path = os.path.join(CADOUTPUTS_DIR, f"model_{timestamp}")
            
            # Save GLTF file
            gltf_data = result.outputs["source.gltf"].decode("utf-8")
            gltf_path = f"{base_path}.gltf"
            with open(gltf_path, "w", encoding="utf-8") as f:
                f.write(gltf_data)

            logging.info(f"Saved CAD file to: {gltf_path}")
            return gltf_path

    except Exception as e:
        logging.error(f"Error in generate_cad_model: {str(e)}")
        return None

# Add this function for PyVista visualization
def create_pyvista_visualization(step_file_url):
    try:
        # Download the STEP file
        import requests
        response = requests.get(step_file_url)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.step') as temp_step:
            temp_step.write(response.content)
            temp_step.flush()
            
            # Convert STEP to STL using KittyCAD API
            stl_response = kittycad_client.file.convert_file(
                source_format="step",
                target_format="stl",
                input_file=temp_step.name
            )
            
            # Create temporary STL file
            with tempfile.NamedTemporaryFile(suffix='.stl') as temp_stl:
                temp_stl.write(stl_response.content)
                temp_stl.flush()
                
                # Read STL with trimesh then convert to PyVista
                mesh = trimesh.load(temp_stl.name)
                vertices = mesh.vertices
                faces = mesh.faces
                
                # Convert to PyVista mesh
                pv_mesh = pv.PolyData(vertices, faces)
                
                # Create a plotter
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(pv_mesh, color='lightgray')
                plotter.enable_eye_dome_lighting()
                plotter.camera_position = 'iso'
                
                # Export as glTF
                with tempfile.NamedTemporaryFile(suffix='.gltf') as temp_gltf:
                    plotter.export_gltf(temp_gltf.name)
                    with open(temp_gltf.name, 'rb') as f:
                        gltf_data = f.read()
                        
                return base64.b64encode(gltf_data).decode('utf-8')
                
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400

        # Get detailed response for user
        user_response = ollama_chat(
            user_input, 
            system_message, 
            vault_embeddings_tensor, 
            vault_content, 
            "llama3.2:3b",
            conversation_history
        )
        
        # Get CAD-focused response with stricter prompt
        cad_system_message = """You are a CAD design expert. Provide a direct, concise description of the mechanical design in two short sentences:
        
1. First sentence should describe all components with their exact measurements.
2. Second sentence should describe the assembly sequence.

Example response:
The design consists of a main gear (50mm diameter, 20mm height, 24 teeth) and a steel shaft (200mm length, 15mm diameter). To assemble, mount the gear onto the shaft at 50mm from the end and secure it with a set screw.
Also no explanations allowed even if you don't know something. just answer what you know.
Keep it brief and direct. Do not refer to any previous text or add any explanations. Focus only on physical specifications and assembly steps."""
        
        cad_response = ollama_chat(
            user_input,
            cad_system_message,
            vault_embeddings_tensor,
            vault_content,
            "llama3.2:3b",
            []  # Empty conversation history for clean response
        )
        
        return jsonify({
            'response': cad_response,
            'cad_response': user_response,
            'conversation_history': conversation_history
        })
            
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process PDF and update vault
            with open(filepath, 'rb') as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                text = ''
                for page in pdf_reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + " "
                
                # Normalize whitespace and clean up text
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Split text into chunks
                sentences = re.split(r'(?<=[.!?]) +', text)
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 < 1000:
                        current_chunk += (sentence + " ").strip()
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Append to vault.txt
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    for chunk in chunks:
                        vault_file.write(chunk.strip() + "\n")
            
            # Reinitialize the embeddings
            initialize_app()
            
            return jsonify({'message': 'File successfully uploaded and processed'}), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_app():
    global client, vault_embeddings_tensor, vault_content
    
    # Initialize Ollama client with correct model name
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='llama3.2:3b'  # Updated to correct model name
    )
    
    # Load vault content
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()
    
    # Generate embeddings
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])
    
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    return True

# Create cadoutputs directory if it doesn't exist
CADOUTPUTS_DIR = 'cadoutputs'
if not os.path.exists(CADOUTPUTS_DIR):
    os.makedirs(CADOUTPUTS_DIR)

def save_cad_model(url, design_json):
    try:
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Download the STEP file
        response = requests.get(url)
        
        # Create filename based on timestamp and first component name
        component_name = design_json.get('components', [{}])[0].get('name', 'unnamed')
        safe_name = "".join(x for x in component_name if x.isalnum() or x in (' ', '-', '_')).strip()
        filename = f"{safe_name}_{timestamp}"
        
        # Save STEP file
        step_path = os.path.join(CADOUTPUTS_DIR, f"{filename}.step")
        with open(step_path, 'wb') as f:
            f.write(response.content)
        
        # Save design JSON
        json_path = os.path.join(CADOUTPUTS_DIR, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(design_json, f, indent=2)
            
        logging.info(f"Saved CAD model to {step_path}")
        return step_path
        
    except Exception as e:
        logging.error(f"Error saving CAD model: {str(e)}")
        return None

@app.route('/generate-cad', methods=['POST'])
def generate_cad():
    try:
        cad_response = request.json.get('cad_response')
        if not cad_response:
            return jsonify({'error': 'No CAD response provided'}), 400
            
        logging.info("Generating CAD from design response...")
        
        # Generate CAD model using KittyCAD
        gltf_path = generate_cad_model(cad_response)
        
        if gltf_path:
            # Simple two-line visualization
            mesh = pv.read(gltf_path)
            mesh.plot()
            
            return jsonify({
                'message': 'CAD model generated and displayed successfully'
            })
        else:
            return jsonify({'error': 'Failed to generate CAD model'}), 500
            
    except Exception as e:
        logging.error(f"Error generating CAD: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_3d_model(prompt):
    """
    Generate a 3D model using KittyCAD's text-to-CAD API
    Returns the path to the generated GLTF file
    """
    try:
        # Create KittyCAD client
        client = ClientFromEnv()
        
        # Call the text-to-CAD API
        response = create_text_to_cad.sync(
            client=client,
            output_format=FileExportFormat.GLTF,
            body=TextToCadCreateBody(prompt=prompt),
        )

        if isinstance(response, Error) or response is None:
            print(f"Error generating 3D model: {response}")
            return None

        result: TextToCad = response

        # Poll until the task is complete
        while result.completed_at is None:
            time.sleep(5)
            response = get_text_to_cad_model_for_user.sync(
                client=client,
                id=result.id,
            )
            if isinstance(response, Error) or response is None:
                print(f"Error checking model status: {response}")
                return None
            result = response

        if result.status == ApiCallStatus.FAILED:
            print(f"Text-to-CAD failed: {result.error}")
            return None

        if result.status == ApiCallStatus.COMPLETED and result.outputs:
            # Save the GLTF file
            output_path = "generated_model.gltf"
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(result.outputs["source.gltf"].decode("utf-8"))
            return output_path

        return None

    except Exception as e:
        print(f"Error in generate_3d_model: {str(e)}")
        return None

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, port=5000)