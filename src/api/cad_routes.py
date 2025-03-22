from flask import Blueprint, request, jsonify
from services.cad_service import CADService
from utils.logger import logger

cad_bp = Blueprint('cad', __name__)
cad_service = CADService()

@cad_bp.route('/generate-cad', methods=['POST'])
def generate_cad():
    try:
        design_text = request.json.get('design_text')
        if not design_text:
            return jsonify({'error': 'No design text provided'}), 400

        model_path = cad_service.generate_3d_model(design_text)
        
        if model_path:
            return jsonify({
                'success': True,
                'model_path': model_path
            })
        else:
            return jsonify({'error': 'Failed to generate CAD model'}), 500

    except Exception as e:
        logger.error(f"Error in generate_cad: {str(e)}")
        return jsonify({'error': str(e)}), 500 