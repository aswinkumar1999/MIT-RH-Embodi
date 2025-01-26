import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
import requests
import time
from collections import Counter 
import threading
import queue

# Replace with your ESP32's IP address
ESP32_IP = "http://10.29.149.177"

# Thread-safe queue for emotion tracking
emotion_queue = queue.Queue()

def post_request_worker():
    """Background worker to send POST requests asynchronously"""
    while True:
        try:
            # Block and wait for emotions to send
            message = emotion_queue.get()
            
            url = "http://localhost:4000"
            data = str(message)

            try:
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    print(f"POST sent: {message}")
                else:
                    print(f"POST Request failed: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
            
            # Mark task as done
            emotion_queue.task_done()
        
        except Exception as e:
            print(f"Unexpected error in post request worker: {e}")

def set_servo_angle(angle):
    if not (0 <= angle <= 180):
        print("Angle must be between 0 and 180")
        return
    
    url = f"{ESP32_IP}/servo?angle={angle}"
    response = requests.get(url)
    
    if response.status_code == 200:
        print(f"Servo moved to {angle} degrees")
    else:
        print(f"Failed to move servo: {response.status_code}")

class EmotionClassifier:
    def __init__(self, model_path='best_model.pth', num_classes=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(num_classes)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.class_names = ['Angry', 'Happy', 'Relaxed', 'Sad']  # Update with actual class names

    def _load_model(self, num_classes):
        model = models.resnet50(pretrained=False)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
        model.eval()
        return model.to(self.device)

    def classify_emotion(self, dog_patch):
        # Preprocess image patch
        input_tensor = self.transform(dog_patch).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1)
        
        return self.class_names[pred.item()], probs[0][pred.item()].item()

class EmotionTracker:
    def __init__(self, sample_window=100, reset_time=10):
        self.emotion_samples = []
        self.emotion_timestamps = []
        self.sample_window = sample_window
        self.reset_time = reset_time
        
        # Custom emotion thresholds
        self.emotion_stability_thresholds = {
            'Angry': 0.3,    # Lower threshold for negative emotions
            'Sad': 0.3,      # Lower threshold for negative emotions
            'Happy': 0.6,    # Higher threshold for positive emotions
            'Relaxed': 0.6   # Higher threshold for positive emotions
        }

    def add_emotion(self, emotion):
        current_time = time.time()
        self.emotion_samples.append(emotion)
        self.emotion_timestamps.append(current_time)
        
        # Remove old samples
        while self.emotion_timestamps and current_time - self.emotion_timestamps[0] > self.reset_time:
            self.emotion_samples.pop(0)
            self.emotion_timestamps.pop(0)
        
        # Keep only the last sample_window entries
        if len(self.emotion_samples) > self.sample_window:
            self.emotion_samples.pop(0)
            self.emotion_timestamps.pop(0)

    def get_stable_emotion(self):
        if not self.emotion_samples:
            return None
        
        # Use Counter to find the most common emotion
        emotion_counts = Counter(self.emotion_samples)
        stable_emotion, count = emotion_counts.most_common(1)[0]
        
        # Calculate the percentage of this emotion
        total_samples = len(self.emotion_samples)
        emotion_percentage = count / total_samples
        
        # Check if the emotion meets its specific stability threshold
        threshold = self.emotion_stability_thresholds.get(stable_emotion, 0.5)
        
        # Return emotion if it meets the threshold, otherwise return None
        return stable_emotion if emotion_percentage >= threshold else None

def process_video_stream():
    # Load YOLOv5 model
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    
    # Initialize emotion classifier and tracker
    emotion_classifier = EmotionClassifier()
    emotion_tracker = EmotionTracker(sample_window=100)
    
    # Open video capture
    cap = cv2.VideoCapture('/dev/video4')
    
    # Set frame rate
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Start with servo at center
    current_servo_angle = 90
    set_servo_angle(current_servo_angle)
    
    frame_count = 0
    last_emotion = None
    last_emotion_confidence = 0
    
    while True:
        # Introduce a small delay to control frame rate
        time.sleep(0.0212)  # approximately 30 FPS
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect dogs with YOLOv5
        results = yolo_model(frame)
        
        # Filter for dogs and sort by confidence
        dogs = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'dog']
        dogs = dogs.sort_values('confidence', ascending=False)
        
        # Take only the most confident dog
        if not dogs.empty:
            dog = dogs.iloc[0]
            
            # Extract bounding box with safety checks
            x1, y1, x2, y2 = map(int, [dog['xmin'], dog['ymin'], dog['xmax'], dog['ymax']])
            
            # Ensure bounding box is within frame
            h, w = frame.shape[:2]
            
            # Add 40 pixels padding with boundary checks
            x1_padded = max(0, x1 - 40)
            y1_padded = max(0, y1 - 40)
            x2_padded = min(w, x2 + 40)
            y2_padded = min(h, y2 + 40)
            
            # Check if padded bounding box is valid
            if x2_padded > x1_padded and y2_padded > y1_padded:
                dog_patch = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                # Additional safety check for dog patch
                if dog_patch.size > 0:
                    # Only run emotion inference every 15 frames
                    if frame_count % 15 == 0:
                        emotion, confidence = emotion_classifier.classify_emotion(dog_patch)
                        
                        # Update emotion only if new confidence is higher
                        if confidence > last_emotion_confidence:
                            last_emotion = emotion
                            last_emotion_confidence = confidence
                    
                    # Calculate servo angle with improved tracking
                    frame_width = frame.shape[1]
                    dog_center_x = (x1 + x2) / 2
                    
                    # Calculate desired servo angle
                    desired_servo_angle = int(90 + 60 * ((dog_center_x / frame_width) - 0.5))
                    desired_servo_angle = max(0, min(180, desired_servo_angle))
                    
                    # Implement smooth servo movement
                    # Add a dampening factor to prevent rapid oscillations
                    servo_movement_speed = 5  # Adjust this value to control responsiveness
                    if abs(desired_servo_angle - current_servo_angle) > servo_movement_speed:
                        if desired_servo_angle > current_servo_angle:
                            current_servo_angle += servo_movement_speed
                        else:
                            current_servo_angle -= servo_movement_speed
                    else:
                        current_servo_angle = desired_servo_angle
                    
                    # Adjust servo to track the most confident dog
                    set_servo_angle(180 - current_servo_angle)
                    
                    # Draw bounding box and emotion
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1_padded, y1_padded), (x2_padded, y2_padded), (255, 0, 0), 1)
                    
                    # Use the most stable emotion found
                    if last_emotion:
                        label = f"{last_emotion} ({last_emotion_confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Display dog patch in separate window
                    cv2.imshow('Dog Patch', dog_patch)
        
        # Increment frame count
        frame_count += 1
        
        # Add last emotion to queue every 5 seconds
        current_time = time.time()
        if last_emotion:
            try:
                # Try to add to queue without blocking
                emotion_queue.put_nowait(last_emotion)
            except queue.Full:
                # If queue is full, skip this iteration
                pass
        
        # Display final frame
        frame_resized = cv2.resize(frame, (1920, 1080))
        cv2.imshow('Dog Emotion Detection', frame_resized)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the POST request worker thread
    post_thread = threading.Thread(target=post_request_worker, daemon=True)
    post_thread.start()
    
    # Run the video processing
    process_video_stream()