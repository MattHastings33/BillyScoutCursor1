import numpy as np
from typing import List, Tuple, Dict
import cv2

class HeatmapGenerator:
    def __init__(self, width: int = 100, height: int = 100):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        
    def add_pitch(self, x: float, y: float, value: float = 1.0):
        """
        Add a pitch to the heatmap
        x, y should be normalized coordinates (0-1)
        """
        grid_x = int(x * (self.width - 1))
        grid_y = int(y * (self.height - 1))
        
        # Ensure coordinates are within bounds
        grid_x = max(0, min(grid_x, self.width - 1))
        grid_y = max(0, min(grid_y, self.height - 1))
        
        # Add value to grid with gaussian distribution
        self._add_gaussian(grid_x, grid_y, value)
    
    def _add_gaussian(self, center_x: int, center_y: int, value: float):
        """Add a gaussian distribution to the grid"""
        sigma = 5  # Standard deviation of the gaussian
        
        # Create meshgrid for gaussian calculation
        x, y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        
        # Calculate gaussian
        gaussian = value * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Add to grid
        self.grid += gaussian
    
    def normalize(self):
        """Normalize the heatmap values to 0-1 range"""
        if self.grid.max() > 0:
            self.grid = self.grid / self.grid.max()
    
    def get_heatmap_data(self) -> Dict:
        """Get heatmap data in format suitable for frontend visualization"""
        self.normalize()
        
        # Convert to list of points with values
        points = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] > 0.01:  # Only include significant points
                    points.append({
                        'x': x / (self.width - 1),  # Normalize back to 0-1
                        'y': y / (self.height - 1),
                        'value': float(self.grid[y, x])
                    })
        
        return {
            'points': points,
            'width': self.width,
            'height': self.height
        }
    
    def get_image(self) -> np.ndarray:
        """Get heatmap as an image (for debugging or direct display)"""
        self.normalize()
        
        # Convert to 8-bit image
        heatmap = (self.grid * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        return heatmap
    
    def clear(self):
        """Clear the heatmap"""
        self.grid = np.zeros((self.height, self.width))

class StrikeZoneHeatmap(HeatmapGenerator):
    def __init__(self):
        super().__init__(width=100, height=100)
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
        
        super().add_pitch(x, y, value)
    
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
        data = super().get_heatmap_data()
        data['strike_zone'] = self.get_strike_zone_overlay()
        return data 