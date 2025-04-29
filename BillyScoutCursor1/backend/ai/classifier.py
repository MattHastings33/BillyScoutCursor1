import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class PitchCharacteristics:
    speed: float  # mph
    vertical_break: float  # inches
    horizontal_break: float  # inches
    spin_rate: float  # rpm (estimated)
    release_height: float  # feet
    release_side: float  # feet (negative = left, positive = right)

class PitchClassifier:
    def __init__(self):
        self.pitch_types = ['fastball', 'curveball', 'slider', 'changeup']
        # Define pitch type profiles with realistic ranges
        self.pitch_profiles = {
            'fastball': {
                'speed': (80, 100),  # Even wider range for fastballs
                'vertical_break': (0, 20),  # Increased range for vertical break
                'horizontal_break': (-15, 15),  # Wider range for horizontal break
                'spin_rate': (1800, 3200)  # Wider spin rate range
            },
            'curveball': {
                'speed': (60, 90),  # Even wider velocity range for curveballs
                'vertical_break': (10, 40),  # Higher vertical break range
                'horizontal_break': (-20, 20),  # Wider horizontal break range
                'spin_rate': (1800, 3200)  # Wider spin rate range
            },
            'slider': {
                'speed': (70, 85),  # Wider range for sliders
                'vertical_break': (0, 25),  # Increased vertical break range
                'horizontal_break': (0, 25),  # Increased horizontal break range
                'spin_rate': (1800, 3200)  # Wider spin rate range
            },
            'changeup': {
                'speed': (65, 88),  # Wider range for changeups
                'vertical_break': (0, 30),  # Increased vertical break range
                'horizontal_break': (-15, 15),  # Wider horizontal break range
                'spin_rate': (1400, 2800)  # Wider spin rate range
            }
        }
        self.current_positions = []
        self.last_position = None
    
    def classify(self, objects: Dict[str, List[Tuple[float, float, float, float]]]) -> str:
        """Classify the pitch type based on detected objects."""
        if 'ball' not in objects or len(objects['ball']) == 0:
            # If no ball detected, check if we have enough positions for classification
            if len(self.current_positions) >= 10:
                result = self.classify_pitch(self.current_positions, fps=30.0)
                self.current_positions = []  # Reset for next pitch
                return result['type']
            return 'unknown'
        
        # Convert bounding boxes to center points
        for box in objects['ball']:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Check if this is a new pitch (large jump from last position)
            if self.last_position is not None:
                dx = center_x - self.last_position[0]
                dy = center_y - self.last_position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance > 100:  # pixels
                    # Large jump means new pitch, classify the previous one
                    if len(self.current_positions) >= 10:
                        result = self.classify_pitch(self.current_positions, fps=30.0)
                        self.current_positions = []  # Reset for next pitch
                        self.last_position = (center_x, center_y)
                        return result['type']
                    self.current_positions = []  # Reset without classification
            
            # Add current position
            self.current_positions.append((center_x, center_y))
            self.last_position = (center_x, center_y)
        
        # If we have enough positions, try to classify
        if len(self.current_positions) >= 10:
            result = self.classify_pitch(self.current_positions, fps=30.0)
            self.current_positions = []  # Reset for next pitch
            return result['type']
        
        return 'unknown'
        
    def estimate_velocity(self, objects: Dict[str, List[Tuple[float, float, float, float]]]) -> Optional[float]:
        """Estimate the velocity of a pitch based on detected objects."""
        # For now, return a random velocity between 85-95 mph if a ball is detected
        if 'ball' in objects and len(objects['ball']) > 0:
            return np.random.uniform(85, 95)
        return None
    
    def classify_pitch(self, ball_positions: List[Tuple[float, float]], fps: float) -> Dict:
        """
        Classify pitch type based on ball trajectory and characteristics
        Returns: Dictionary with pitch classification and confidence
        """
        if len(ball_positions) < 10:  # Need enough points for analysis
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Calculate pitch characteristics
        characteristics = self._calculate_characteristics(ball_positions)
        
        # Classify pitch type
        pitch_type, confidence = self._calculate_pitch_type(characteristics['speed'], characteristics['vertical_break'], characteristics['horizontal_break'], characteristics['spin_rate'])
        
        return {
            'type': pitch_type,
            'confidence': confidence,
            'characteristics': characteristics
        }
    
    def _calculate_characteristics(self, ball_positions):
        """Calculate pitch characteristics from ball positions."""
        if len(ball_positions) < 10:
            raise ValueError("Not enough ball positions to calculate characteristics")

        # Unzip the ball positions into separate x and y lists
        xs, ys = zip(*ball_positions)
        xs = list(xs)
        ys = list(ys)

        # Calculate speed using first and last positions
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        distance = np.sqrt(dx*dx + dy*dy)
        time = len(ball_positions) / 30.0  # Assuming 30 fps
        speed = (distance / time) * 0.0268182  # Adjusted scaling factor for more realistic speeds

        # Fit a line to the trajectory
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)

        # Calculate breaks using middle 60% of trajectory
        start_idx = len(xs) // 5
        end_idx = len(xs) - len(xs) // 5
        max_vertical_break = 0
        max_horizontal_break = 0

        for i in range(start_idx, end_idx):
            expected_y = p(xs[i])
            vertical_break = abs(ys[i] - expected_y)
            horizontal_break = abs(xs[i] - xs[0] - (i * (xs[-1] - xs[0]) / len(xs)))
            
            max_vertical_break = max(max_vertical_break, vertical_break)
            max_horizontal_break = max(max_horizontal_break, horizontal_break)

        # Convert breaks to inches (using same scale factor as speed)
        vertical_break = max_vertical_break * 0.0268182
        horizontal_break = max_horizontal_break * 0.0268182

        # Calculate spin rate (simplified estimation)
        spin_rate = 2000 + (vertical_break * 100)  # Base spin rate + adjustment for break

        # Determine pitch type based on characteristics
        pitch_type = "fastball"
        if vertical_break > 8:
            pitch_type = "curveball"
        elif horizontal_break > 6:
            pitch_type = "slider"
        elif speed < 85:
            pitch_type = "changeup"

        # Print debug info
        print(f"\nPitch Characteristics:")
        print(f"Speed: {speed:.1f} mph")
        print(f"Vertical Break: {vertical_break:.1f} inches")
        print(f"Horizontal Break: {horizontal_break:.1f} inches")
        print(f"Spin Rate: {spin_rate} rpm")
        print(f"Type: {pitch_type}")

        return {
            "type": pitch_type,
            "speed": speed,
            "vertical_break": vertical_break,
            "horizontal_break": horizontal_break,
            "spin_rate": spin_rate,
            "confidence": 0.8
        }

    def _classify_pitch(self, speed, vertical_break, horizontal_break):
        """Classify pitch type based on characteristics."""
        # Simple classification based on typical MLB averages
        if speed > 93:
            return "fastball"
        elif vertical_break > 6 and speed < 80:
            return "curveball"
        elif horizontal_break > 6 and speed > 82:
            return "slider"
        elif speed < 85:
            return "changeup"
        else:
            return "fastball"  # Default to fastball for borderline cases
    
    def _calculate_pitch_type(self, speed, vertical_break, horizontal_break, spin_rate):
        # Normalize the characteristics with wider ranges
        speed_score = self._normalize_value(speed, 60, 105)  # Much wider range for better normalization
        vbreak_score = self._normalize_value(vertical_break, 0, 40)  # Increased range for curveballs
        hbreak_score = self._normalize_value(horizontal_break, -20, 20)  # Wider range for horizontal break
        spin_score = self._normalize_value(spin_rate, 1400, 3200)  # Wider spin rate range

        # Calculate scores for each pitch type with adjusted weights
        scores = {
            'fastball': (
                0.60 * speed_score +  # Increased weight on speed
                0.10 * vbreak_score +  # Reduced weight on vertical break
                0.15 * hbreak_score +
                0.15 * spin_score
            ),
            'curveball': (
                0.10 * speed_score +  # Reduced weight on speed
                0.50 * vbreak_score +  # Increased weight on vertical break
                0.20 * hbreak_score +
                0.20 * spin_score
            ),
            'slider': (
                0.30 * speed_score +  # Reduced weight on speed
                0.15 * vbreak_score +  # Reduced weight on vertical break
                0.45 * hbreak_score +  # Increased weight on horizontal break
                0.10 * spin_score
            ),
            'changeup': (
                0.40 * speed_score +  # Reduced weight on speed
                0.35 * vbreak_score +  # Increased weight on vertical break
                0.15 * hbreak_score +
                0.10 * spin_score
            )
        }

        # Find the best pitch type
        best_pitch = max(scores.items(), key=lambda x: x[1])
        
        # Print detailed scores for debugging
        print(f"Speed: {speed:.1f} mph")
        print(f"fastball: speed={speed_score:.2f}, vbreak={vbreak_score:.2f}, hbreak={hbreak_score:.2f}, spin={spin_score:.2f}, total={scores['fastball']:.2f}")
        print(f"curveball: speed={speed_score:.2f}, vbreak={vbreak_score:.2f}, hbreak={hbreak_score:.2f}, spin={spin_score:.2f}, total={scores['curveball']:.2f}")
        print(f"slider: speed={speed_score:.2f}, vbreak={vbreak_score:.2f}, hbreak={hbreak_score:.2f}, spin={spin_score:.2f}, total={scores['slider']:.2f}")
        print(f"changeup: speed={speed_score:.2f}, vbreak={vbreak_score:.2f}, hbreak={hbreak_score:.2f}, spin={spin_score:.2f}, total={scores['changeup']:.2f}")

        # Apply pitch-specific thresholds and requirements
        if best_pitch[0] == 'fastball' and best_pitch[1] > 0.40 and speed > 85:
            return best_pitch[0], best_pitch[1]
        elif best_pitch[0] == 'curveball' and best_pitch[1] > 0.35 and vertical_break > 6:
            return best_pitch[0], best_pitch[1]
        elif best_pitch[0] == 'slider' and best_pitch[1] > 0.35 and horizontal_break > 2:
            return best_pitch[0], best_pitch[1]
        elif best_pitch[0] == 'changeup' and best_pitch[1] > 0.35 and speed < 90:
            return best_pitch[0], best_pitch[1]
        
        return 'unknown', 0.0
    
    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to a range from 0 to 1"""
        if value < min_val:
            return 0.0
        if value > max_val:
            return 1.0
        return (value - min_val) / (max_val - min_val) 