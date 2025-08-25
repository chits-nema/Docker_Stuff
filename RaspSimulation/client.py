import cv2
import requests
import base64
import json
import time
import numpy as np
import threading
from queue import Queue
import argparse

class DetectionClient:
    def __init__(self, server_host, server_port=5000, timeout=10):
        self.server_url = f"http://{server_host}:{server_port}"
        self.detect_url = f"{self.server_url}/detect"
        self.health_url = f"{self.server_url}/health"
        self.session = requests.Session()
        self.timeout = timeout
        
        # Performance tracking
        self.stats = {
            'frames_sent': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'avg_processing_time': 0,
            'total_processing_time': 0
        }
        
    def check_server_health(self):
        """Check if detection server is available"""
        try:
            response = self.session.get(self.health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def encode_image(self, frame, quality=70):
        """Encode OpenCV image to base64"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def decode_image(self, image_data):
        """Decode base64 image data to OpenCV format"""
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    def send_frame_for_detection(self, frame):
        """Send frame to server for detection"""
        try:
            encoded_image = self.encode_image(frame)
            payload = {'image': encoded_image}
            
            response = self.session.post(
                self.detect_url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    self.stats['successful_detections'] += 1
                    self.stats['total_processing_time'] += result.get('processing_time', 0)
                    self.stats['avg_processing_time'] = (
                        self.stats['total_processing_time'] / self.stats['successful_detections']
                    )
                else:
                    self.stats['failed_detections'] += 1
                return result
            else:
                self.stats['failed_detections'] += 1
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except requests.exceptions.Timeout:
            self.stats['failed_detections'] += 1
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            self.stats['failed_detections'] += 1
            return {'success': False, 'error': str(e)}
    
    def print_stats(self):
        """Print performance statistics"""
        total = self.stats['successful_detections'] + self.stats['failed_detections']
        if total > 0:
            success_rate = (self.stats['successful_detections'] / total) * 100
            print(f"\n--- Performance Stats ---")
            print(f"Frames sent: {self.stats['frames_sent']}")
            print(f"Successful detections: {self.stats['successful_detections']}")
            print(f"Failed detections: {self.stats['failed_detections']}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Avg processing time: {self.stats['avg_processing_time']:.3f}s")

def main():
    parser = argparse.ArgumentParser(description='YOLO Detection Client')
    parser.add_argument('--server', required=True, help='Detection server IP address')
    parser.add_argument('--port', default=5000, type=int, help='Server port (default: 5000)')
    parser.add_argument('--camera', default=0, type=int, help='Camera index (default: 0)')
    parser.add_argument('--skip-frames', default=3, type=int, help='Process every N frames (default: 3)')
    parser.add_argument('--width', default=640, type=int, help='Camera width (default: 640)')
    parser.add_argument('--height', default=480, type=int, help='Camera height (default: 480)')
    
    args = parser.parse_args()
    
    print(f"Starting detection client...")
    print(f"Server: {args.server}:{args.port}")
    print(f"Camera: {args.camera} ({args.width}x{args.height})")
    print(f"Processing every {args.skip_frames} frames")
    
    client = DetectionClient(args.server, args.port)
    
    # Check server availability
    print("Checking server health...")
    if not client.check_server_health():
        print(f"Error: Cannot connect to detection server at {args.server}:{args.port}")
        print("Make sure the detection server is running and accessible.")
        return
    
    print("✓ Server is healthy!")
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        return
    
    print("Camera initialized. Press 'q' to quit, 's' to show stats.")
    
    frame_count = 0
    last_detection_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Display original frame
            cv2.imshow("Camera Feed", frame)
            
            # Send frame for detection every N frames
            if frame_count % args.skip_frames == 0:
                client.stats['frames_sent'] += 1
                result = client.send_frame_for_detection(frame)
                
                if result['success']:
                    # Decode and display annotated image
                    try:
                        annotated_frame = client.decode_image(result['annotated_image'])
                        cv2.imshow("Detection Results", annotated_frame)
                        
                        # Print detected objects
                        if result['detected_classes']:
                            classes_str = ', '.join(result['detected_classes'])
                            processing_time = result.get('processing_time', 0)
                            print(f"Detected: {classes_str} ({processing_time:.3f}s)")
                        
                        last_detection_time = time.time()
                        
                    except Exception as e:
                        print(f"Error displaying results: {e}")
                else:
                    print(f"Detection failed: {result['error']}")
            
            frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                client.print_stats()
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.print_stats()

if __name__ == "__main__":
    main()
    