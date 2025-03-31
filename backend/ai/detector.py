from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict

class BaseballDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using nano model for speed, can be upgraded to larger models
        
    def detect_objects(self, frame: np.ndarray) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Detect baseball, pitcher, and batter in a frame
        Returns: Dictionary of detected objects with their bounding boxes
        """
        results = self.model(frame)
        detections = {
            'ball': [],
            'person': []  # Will include both pitcher and batter
        }
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # YOLO class 32 is sports ball, 0 is person
                if cls == 32 and conf > 0.5:  # Baseball
                    detections['ball'].append((x1, y1, x2, y2))
                elif cls == 0 and conf > 0.5:  # Person
                    detections['person'].append((x1, y1, x2, y2))
        
        return detections
    
    def track_ball(self, frames: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Track ball movement across multiple frames
        Returns: List of ball positions (x, y)
        """
        ball_positions = []
        for frame in frames:
            detections = self.detect_objects(frame)
            if detections['ball']:
                # Use center of bounding box as ball position
                x1, y1, x2, y2 = detections['ball'][0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                ball_positions.append((center_x, center_y))
            else:
                ball_positions.append(None)
        return ball_positions
    
    def estimate_ball_speed(self, ball_positions: List[Tuple[float, float]], fps: float) -> float:
        """
        Estimate ball speed based on position changes
        Returns: Estimated speed in mph
        """
        if len(ball_positions) < 2:
            return 0.0
            
        # Calculate average distance between consecutive positions
        total_distance = 0
        valid_pairs = 0
        
        for i in range(len(ball_positions) - 1):
            if ball_positions[i] and ball_positions[i + 1]:
                x1, y1 = ball_positions[i]
                x2, y2 = ball_positions[i + 1]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance += distance
                valid_pairs += 1
        
        if valid_pairs == 0:
            return 0.0
            
        avg_distance = total_distance / valid_pairs
        # Convert to mph (assuming 60ft distance from pitcher to plate)
        # This is a rough estimation and would need calibration
        speed_mph = (avg_distance * fps * 60) / 88  # 88 ft/s = 60 mph
        return speed_mph 