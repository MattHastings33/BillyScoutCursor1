import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2

class HeatmapGenerator:
    def __init__(self):
        self.points = []
        
    def generate_point(self, frame: np.ndarray, objects: Dict, poses: List[Dict]) -> Optional[Tuple[float, float]]:
        """
        Generate a heatmap point from ball position
        Returns: (x, y) coordinates or None
        """
        # For now, just use the first ball's center position
        if objects.get('ball'):
            x1, y1, x2, y2 = objects['ball'][0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            self.points.append((center_x, center_y))
            return (center_x, center_y)
        return None
        
    def generate_heatmap(self, width: int, height: int) -> np.ndarray:
        """
        Generate a heatmap from collected points
        Returns: Heatmap image
        """
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for x, y in self.points:
            x = int(x)
            y = int(y)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(heatmap, (x, y), 10, (1.0), -1)
        
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        
        return heatmap

class StrikeZoneHeatmap(HeatmapGenerator):
    def __init__(self):
        super().__init__()
        self.strike_zone = {
            'top': 0.5,    # Top of strike zone (normalized)
            'bottom': 0.2, # Bottom of strike zone (normalized)
            'left': 0.3,   # Left edge of strike zone (normalized)
            'right': 0.7   # Right edge of strike zone (normalized)
        }
    
    def add_pitch(self, x: float, y: float, value: float = 1.0, pitch_type: str = None):
        """
        Add a pitch to the strike zone heatmap
        x, y should be normalized coordinates (0-1)
        """
        # Adjust value based on pitch type
        if pitch_type:
            value = self._adjust_value_by_pitch_type(value, pitch_type)
        
        super().generate_point(None, {'ball': [(x, y, x, y)]}, [])
    
    def _adjust_value_by_pitch_type(self, value: float, pitch_type: str) -> float:
        """Adjust heatmap value based on pitch type"""
        # Different pitch types might have different visual weights
        weights = {
            'fastball': 1.0,
            'curveball': 0.8,
            'slider': 0.9,
            'changeup': 0.7
        }
        return value * weights.get(pitch_type, 1.0)
    
    def get_strike_zone_overlay(self) -> Dict:
        """Get strike zone boundaries for overlay"""
        return {
            'top': self.strike_zone['top'],
            'bottom': self.strike_zone['bottom'],
            'left': self.strike_zone['left'],
            'right': self.strike_zone['right']
        }
    
    def get_heatmap_data(self) -> Dict:
        """Get heatmap data with strike zone information"""
        data = {
            'points': self.points,
            'width': 100,
            'height': 100,
            'strike_zone': self.get_strike_zone_overlay()
        }
        return data
    
    def get_image(self) -> np.ndarray:
        """Get heatmap as an image (for debugging or direct display)"""
        heatmap = self.generate_heatmap(100, 100)
        
        # Convert to 8-bit image
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
    
    def clear(self):
        """Clear the heatmap"""
        self.points = []

    def normalize(self):
        """Normalize the heatmap values to 0-1 range"""
        if self.generate_heatmap(100, 100).max() > 0:
            self.generate_heatmap(100, 100)
    
    def get_heatmap_data(self) -> Dict:
        """Get heatmap data in format suitable for frontend visualization"""
        self.normalize()
        
        # Convert to list of points with values
        points = []
        for y in range(100):
            for x in range(100):
                if self.generate_heatmap(100, 100)[y, x] > 0.01:  # Only include significant points
                    points.append({
                        'x': x / (100 - 1),  # Normalize back to 0-1
                        'y': y / (100 - 1),
                        'value': float(self.generate_heatmap(100, 100)[y, x])
                    })
        
        return {
            'points': points,
            'width': 100,
            'height': 100
        } 