import numpy as np
from typing import List, Tuple, Dict
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
        # Define pitch type characteristics (can be refined with more data)
        self.pitch_profiles = {
            'fastball': {
                'speed_range': (85, 100),
                'vertical_break_range': (-5, 5),
                'horizontal_break_range': (-5, 5),
                'spin_rate_range': (2000, 2500)
            },
            'curveball': {
                'speed_range': (70, 85),
                'vertical_break_range': (5, 15),
                'horizontal_break_range': (-5, 5),
                'spin_rate_range': (2500, 3000)
            },
            'slider': {
                'speed_range': (80, 90),
                'vertical_break_range': (-5, 5),
                'horizontal_break_range': (5, 15),
                'spin_rate_range': (2500, 3000)
            },
            'changeup': {
                'speed_range': (75, 85),
                'vertical_break_range': (-5, 5),
                'horizontal_break_range': (-5, 5),
                'spin_rate_range': (1500, 2000)
            }
        }
    
    def classify_pitch(self, ball_positions: List[Tuple[float, float]], fps: float) -> Dict:
        """
        Classify pitch type based on ball trajectory and characteristics
        Returns: Dictionary with pitch classification and confidence
        """
        if len(ball_positions) < 10:  # Need enough points for analysis
            return {'type': 'unknown', 'confidence': 0.0}
        
        # Calculate pitch characteristics
        characteristics = self._calculate_characteristics(ball_positions, fps)
        
        # Classify pitch type
        pitch_type, confidence = self._determine_pitch_type(characteristics)
        
        return {
            'type': pitch_type,
            'confidence': confidence,
            'characteristics': characteristics.__dict__
        }
    
    def _calculate_characteristics(self, positions: List[Tuple[float, float]], fps: float) -> PitchCharacteristics:
        """Calculate pitch characteristics from ball positions"""
        # Convert positions to numpy array for easier calculations
        pos_array = np.array(positions)
        
        # Calculate speed (assuming 60ft distance)
        dx = np.diff(pos_array[:, 0])
        dy = np.diff(pos_array[:, 1])
        distances = np.sqrt(dx**2 + dy**2)
        avg_speed = np.mean(distances) * fps * 60 / 88  # Convert to mph
        
        # Calculate vertical and horizontal break
        # This is a simplified calculation - would need calibration
        vertical_break = (pos_array[-1, 1] - pos_array[0, 1]) * 12  # Convert to inches
        horizontal_break = (pos_array[-1, 0] - pos_array[0, 0]) * 12  # Convert to inches
        
        # Estimate spin rate (very rough estimation)
        # In reality, this would need more sophisticated calculation
        spin_rate = 2000 + (avg_speed * 50)  # Rough estimation
        
        # Estimate release point (simplified)
        release_height = 6.0  # feet
        release_side = 0.0  # feet (assuming right-handed pitcher)
        
        return PitchCharacteristics(
            speed=avg_speed,
            vertical_break=vertical_break,
            horizontal_break=horizontal_break,
            spin_rate=spin_rate,
            release_height=release_height,
            release_side=release_side
        )
    
    def _determine_pitch_type(self, characteristics: PitchCharacteristics) -> Tuple[str, float]:
        """Determine pitch type and confidence based on characteristics"""
        best_match = ('unknown', 0.0)
        max_confidence = 0.0
        
        for pitch_type, profile in self.pitch_profiles.items():
            confidence = self._calculate_confidence(characteristics, profile)
            if confidence > max_confidence:
                max_confidence = confidence
                best_match = (pitch_type, confidence)
        
        return best_match
    
    def _calculate_confidence(self, characteristics: PitchCharacteristics, profile: Dict) -> float:
        """Calculate confidence score for a pitch type match"""
        confidences = []
        
        # Speed confidence
        speed_range = profile['speed_range']
        speed_conf = self._normalize_to_range(characteristics.speed, speed_range)
        confidences.append(speed_conf)
        
        # Vertical break confidence
        vb_range = profile['vertical_break_range']
        vb_conf = self._normalize_to_range(characteristics.vertical_break, vb_range)
        confidences.append(vb_conf)
        
        # Horizontal break confidence
        hb_range = profile['horizontal_break_range']
        hb_conf = self._normalize_to_range(characteristics.horizontal_break, hb_range)
        confidences.append(hb_conf)
        
        # Spin rate confidence
        sr_range = profile['spin_rate_range']
        sr_conf = self._normalize_to_range(characteristics.spin_rate, sr_range)
        confidences.append(sr_conf)
        
        # Return average confidence
        return np.mean(confidences)
    
    def _normalize_to_range(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Normalize a value to a confidence score within a range"""
        min_val, max_val = range_tuple
        if value < min_val or value > max_val:
            return 0.0
        return 1.0 - (abs(value - (min_val + max_val) / 2) / ((max_val - min_val) / 2)) 