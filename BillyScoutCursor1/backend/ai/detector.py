from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

class BaseballDetector:
    def __init__(self):
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # Using nano model for speed, can be upgraded to larger models
        
        # Define class names we're interested in
        self.class_names = ['person', 'sports ball']
        
    def detect(self, frame: np.ndarray) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Detect objects in a frame.
        Returns a dictionary mapping object types to lists of bounding boxes.
        Each bounding box is a tuple of (x1, y1, x2, y2) coordinates.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply binary thresholding with a higher threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        objects = {'ball': []}
        
        # Filter contours
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio and area
            aspect_ratio = float(w)/h
            area = cv2.contourArea(contour)
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter > 0 else 0
            
            print(f"Contour {i}: w={w}, h={h}, ratio={aspect_ratio}, area={area}, circularity={circularity}")
            
            # Filter based on size, aspect ratio, and circularity
            if (15 <= w <= 40 and 15 <= h <= 40 and  # Size constraints
                0.8 <= aspect_ratio <= 1.2 and        # Nearly square
                area >= 150 and                       # Minimum area
                circularity >= 0.7):                  # Must be fairly circular
                
                # Calculate center point
                center_x = x + w/2
                center_y = y + h/2
                
                # Store the actual center point and size
                objects['ball'].append((
                    x, y, x + w, y + h
                ))
        
        return objects
    
    def track_ball(self, frames: List[np.ndarray]) -> List[Tuple[float, float]]:
        """
        Track ball movement across multiple frames
        Returns: List of ball positions (x, y)
        """
        ball_positions = []
        for frame in frames:
            detections = self.detect(frame)
            if 'ball' in detections:
                # Use center of bounding box as ball position
                for box in detections['ball']:
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    ball_positions.append((center_x, center_y))
            else:
                ball_positions.append(None)
        return ball_positions
    
    def estimate_velocity(self, ball_positions: List[Tuple[float, float, float, float]], fps: float = 30.0) -> Optional[float]:
        """
        Estimate the velocity of a pitch based on ball positions
        Args:
            ball_positions: List of ball bounding boxes (x1, y1, x2, y2)
            fps: Frames per second of the video
        Returns:
            Estimated velocity in mph, or None if velocity cannot be calculated
        """
        try:
            if len(ball_positions) < 2:
                return None
            
            if fps <= 0:
                return None
            
            # Get center points of ball positions
            pos1 = ball_positions[0]
            pos2 = ball_positions[1]
            
            # Validate position tuples
            if len(pos1) != 4 or len(pos2) != 4:
                return None
            
            center1 = ((pos1[0] + pos1[2]) / 2, (pos1[1] + pos1[3]) / 2)
            center2 = ((pos2[0] + pos2[2]) / 2, (pos2[1] + pos2[3]) / 2)
            
            # Calculate distance in pixels
            dx = center2[0] - center1[0]
            dy = center2[1] - center1[1]
            distance = np.sqrt(dx * dx + dy * dy)
            
            # If distance is too large, it's likely a jump between pitches
            if distance > 100:  # pixels
                return None
            
            # Convert to real-world units (rough estimation)
            # Assuming the video is 1920x1080 and shows about 60 feet (distance from pitcher to home plate)
            pixels_per_foot = 1920 / 60
            distance_feet = distance / pixels_per_foot
            
            # Calculate velocity (distance/time)
            time = 1.0 / fps  # Time between frames
            velocity_fps = distance_feet / time
            velocity_mph = velocity_fps * (3600 / 5280)  # Convert ft/s to mph
            
            # Validate velocity is in reasonable range (30-120 mph)
            if velocity_mph < 30 or velocity_mph > 120:
                return None
            
            return float(velocity_mph)
        except Exception as e:
            print(f"Error calculating velocity: {str(e)}")
            return None 