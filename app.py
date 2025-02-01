from flask import Flask, render_template, request, jsonify
import torch
import os
from PIL import Image
from model import FruitDetectionModel, get_transform
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('best_model.pth', map_location=device)
    model = FruitDetectionModel(num_classes=len(checkpoint['classes'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['classes'], device

# Initialize model
model, classes, device = load_model()
transform = get_transform(is_training=False)

@app.route('/')
def index():
    # Get training statistics
    stats = {
        'accuracy': None,
        'confusion_matrix': None,
        'training_loss': None,
        'validation_accuracy': None
    }
    
    if os.path.exists('static/confusion_matrix.png'):
        stats['confusion_matrix'] = 'confusion_matrix.png'
    if os.path.exists('static/training_loss.png'):
        stats['training_loss'] = 'training_loss.png'
    if os.path.exists('static/validation_accuracy.png'):
        stats['validation_accuracy'] = 'validation_accuracy.png'
    
    return render_template('index.html', stats=stats, classes=classes)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        try:
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_p, top_class = torch.topk(probabilities, 3)
                
                results = []
                for i in range(3):
                    results.append({
                        'class': classes[top_class[0][i]],
                        'probability': f"{top_p[0][i].item()*100:.2f}%"
                    })
                
            return jsonify({
                'success': True,
                'results': results,
                'image_path': f"uploads/{filename}"
            })
            
        except Exception as e:
            return jsonify({'error': f"Error processing image: {str(e)}"})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
