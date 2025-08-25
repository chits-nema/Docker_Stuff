import cv2
import supervision as sv
from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import base64
import json
import time
import logging
from threading import Lock
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Thread safety for model inference
model_lock = Lock()

# Global model instance
model = None

def initialize_model():
    """Initialize YOLO model"""
    global model
    try:
        model = YOLO("yolo11n.pt")
        logger.info("YOLO model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def decode_image(image_data):
    """Decode base64 image data to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        return None

def encode_image(frame, quality=85):
    """Encode OpenCV image to base64"""
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        logger.error(f"Image encode error: {e}")
        return None

@app.route('/detect', methods=['POST'])
def detect():
    start_time = time.time()
    
    try:
        # Get request data
        if request.is_json:
            data = request.json
        else:
            return jsonify({'success': False, 'error': 'Content-Type must be application/json'})
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        # Decode image
        frame = decode_image(data['image'])
        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode image'})
        
        # Thread-safe model inference
        with model_lock:
            if model is None:
                return jsonify({'success': False, 'error': 'Model not initialized'})
            
            # Run detection
            results = model(frame, verbose=False)[0]
        
        # Process detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Create annotations
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
        
        # Create labels with confidence scores
        labels = []
        if len(detections) > 0:
            for class_name, confidence in zip(detections['class_name'], detections.confidence):
                labels.append(f"{class_name} {confidence:.2f}")
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        if labels:
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
        
        # Get detected classes
        detected_classes = []
        class_counts = {}
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                if box.cls is not None:
                    for cls_id in box.cls:
                        class_name = model.names[int(cls_id)]
                        detected_classes.append(class_name)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Encode result image
        encoded_image = encode_image(annotated_frame, quality=80)
        if encoded_image is None:
            return jsonify({'success': False, 'error': 'Failed to encode result image'})
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'annotated_image': encoded_image,
            'detected_classes': detected_classes,
            'class_counts': class_counts,
            'detection_count': len(detected_classes),
            'processing_time': round(processing_time, 3)
        })
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': time.time()
    })

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'service': 'YOLO Detection Server',
        'model': 'yolo11n.pt',
        'version': '1.0.0',
        'endpoints': ['/detect', '/health', '/info']
    })

if __name__ == '__main__':
    # Initialize model on startup
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting.")
        exit(1)
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    