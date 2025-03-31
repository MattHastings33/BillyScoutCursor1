import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple
import cv2

class SwingAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def analyze_swing(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze batter's swing across multiple frames
        Returns: Dictionary containing swing metrics
        """
        swing_metrics = {
            'swing_speed': 0.0,
            'swing_angle': 0.0,
            'aggression': 0.0,
            'keypoints': []
        }
        
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract key points for swing analysis
                landmarks = results.pose_landmarks.landmark
                keypoints = self._extract_keypoints(landmarks)
                swing_metrics['keypoints'].append(keypoints)
        
        if len(swing_metrics['keypoints']) > 1:
            swing_metrics.update(self._calculate_metrics(swing_metrics['keypoints']))
        
        return swing_metrics
    
    def _extract_keypoints(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Extract relevant keypoints for swing analysis"""
        return {
            'right_shoulder': (landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y),
            'right_elbow': (landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y),
            'right_wrist': (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
            'left_hip': (landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y),
            'right_hip': (landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y)
        }
    
    def _calculate_metrics(self, keypoints_list: List[Dict]) -> Dict:
        """Calculate swing metrics from keypoints"""
        metrics = {
            'swing_speed': 0.0,
            'swing_angle': 0.0,
            'aggression': 0.0
        }
        
        if len(keypoints_list) < 2:
            return metrics
            
        # Calculate swing speed (based on wrist movement)
        wrist_positions = [kp['right_wrist'] for kp in keypoints_list]
        total_distance = 0
        for i in range(len(wrist_positions) - 1):
            x1, y1 = wrist_positions[i]
            x2, y2 = wrist_positions[i + 1]
            total_distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        metrics['swing_speed'] = total_distance / (len(wrist_positions) - 1)
        
        # Calculate swing angle (angle between shoulder and wrist)
        first_frame = keypoints_list[0]
        last_frame = keypoints_list[-1]
        
        shoulder = first_frame['right_shoulder']
        wrist_start = first_frame['right_wrist']
        wrist_end = last_frame['right_wrist']
        
        # Calculate vectors
        v1 = np.array([wrist_start[0] - shoulder[0], wrist_start[1] - shoulder[1]])
        v2 = np.array([wrist_end[0] - shoulder[0], wrist_end[1] - shoulder[1]])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        metrics['swing_angle'] = np.degrees(angle)
        
        # Calculate aggression (combination of speed and angle)
        metrics['aggression'] = (metrics['swing_speed'] * metrics['swing_angle']) / 100
        
        return metrics 