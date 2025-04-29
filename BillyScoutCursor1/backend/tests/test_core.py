import os
import sys
import cv2
import numpy as np

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.detector import BaseballDetector
from ai.classifier import PitchClassifier

def test_detector():
    """Test the BaseballDetector class with a single frame"""
    print("\nTesting BaseballDetector...")
    print("=" * 50)
    
    detector = BaseballDetector()
    
    # Create a test image (black background with a white circle for ball)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(test_image, (320, 240), 20, (255, 255, 255), -1)
    
    # Test detection
    objects = detector.detect(test_image)
    print(f"Detected objects: {objects}")
    
    # Test velocity estimation
    if 'ball' in objects:
        velocity = detector.estimate_velocity(objects['ball'])
        print(f"Estimated velocity: {velocity} mph")

def test_classifier():
    """Test the PitchClassifier class with simple test data"""
    print("\nTesting PitchClassifier...")
    print("=" * 50)
    
    classifier = PitchClassifier()
    
    # Create test objects dictionary
    test_objects = {
        'ball': [(100, 100, 120, 120)]  # Simple bounding box
    }
    
    # Test classification
    pitch_type = classifier.classify(test_objects)
    print(f"Classified pitch type: {pitch_type}")
    
    # Test velocity estimation
    velocity = classifier.estimate_velocity(test_objects)
    print(f"Estimated velocity: {velocity} mph")

if __name__ == '__main__':
    print("Starting core component tests...")
    test_detector()
    test_classifier() 