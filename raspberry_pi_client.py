# raspberry_pi_client.py
import cv2
import requests
import base64
import json
import time
import numpy as np
from io import BytesIO

class FrameCaptureClient:
    def __init__(self, server_url, camera_id=0, quality=80, resize_width=640):
        self.server_url = server_url
        self.camera_id = camera_id
        self.quality = quality
        self.resize_width = resize_width
        self.cap = cv2.VideoCapture(camera_id)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    def encode_frame(self, frame):
        """Encode frame to base64 JPEG with compression"""
        # Resize frame to reduce bandwidth
        height, width = frame.shape[:2]
        new_height = int(height * (self.resize_width / width))
        frame = cv2.resize(frame, (self.resize_width, new_height))
        
        # Encode as JPEG with quality setting
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64, frame.shape
    
    def send_frame(self, frame_data, frame_shape):
        """Send frame to processing server"""
        try:
            payload = {
                'frame': frame_data,
                'shape': frame_shape,
                'timestamp': time.time()
            }
            
            response = requests.post(
                f"{self.server_url}/process_frame",
                json=payload,
                timeout=5,
                verify=False  # For self-signed certificates
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Server error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None
    
    def run(self):
        """Main capture and send loop"""
        print("Starting frame capture client...")
        print(f"Server: {self.server_url}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    continue
                
                # Encode frame
                frame_data, frame_shape = self.encode_frame(frame)
                
                # Send to server
                result = self.send_frame(frame_data, frame_shape)
                
                if result:
                    # Display results if available
                    if 'detections' in result:
                        print(f"Detections: {result['detections']}")
                    
                    # Calculate and display FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"Average FPS: {fps:.2f}")
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.033)  # ~30 FPS max
                
        except KeyboardInterrupt:
            print("\nShutting down client...")
        finally:
            self.cap.release()

if __name__ == "__main__":
    # Configuration
    SERVER_URL = "https://YOUR_LAPTOP_IP:8443"  # Replace with your laptop's IP
    CAMERA_ID = 0
    QUALITY = 80  # JPEG quality (1-100)
    RESIZE_WIDTH = 640  # Resize width for bandwidth optimization
    
    client = FrameCaptureClient(SERVER_URL, CAMERA_ID, QUALITY, RESIZE_WIDTH)
    client.run()
