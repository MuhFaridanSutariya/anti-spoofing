from flask import Blueprint, request, jsonify
import os
from src.services.prediction_service import PredictionService

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    image_path = os.path.join(tmp_dir, image.filename)
    image.save(image_path)

    model_checkpoint = os.path.abspath("checkpoint/seresnext50-v2.ckpt")
    input_shape = (3, 224, 224)
    num_classes = 2

    service = PredictionService(model_checkpoint, input_shape, num_classes)
    result = service.predict(image_path)

    return jsonify(result)
