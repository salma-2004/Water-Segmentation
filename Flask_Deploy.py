from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Load model 
model = torch.load('Water_Segmentation_Model.pth', map_location=torch.device('cpu'))
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    _, buffer = cv2.imencode('.png', mask)
    mask_bytes = buffer.tobytes()

    return mask_bytes, 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
