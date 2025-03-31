from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class PitcherChange:
    timestamp: float
    pitcher_name: str
    team: str
    jersey_number: str
    inning: int
    score: str

@dataclass
class PitchData:
    type: str
    speed: float
    confidence: float
    timestamp: float
    result: str  # "hit", "out", "walk", etc.
    count: str  # e.g., "0-0", "0-1", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"
    pitcher_name: str
    velocity: float  # Pitch velocity in mph
    max_velocity: float  # Maximum velocity in the game
    velocity_trend: List[float]  # List of velocities for trend analysis

@dataclass
class SwingData:
    speed: float
    angle: float
    aggression: float
    timing: float  # relative to pitch arrival

class ScoutingReport:
    def __init__(self):
        self.pitch_data: List[PitchData] = []
        self.swing_data: List[SwingData] = []
        self.pitcher_changes: List[PitcherChange] = []
        self.current_pitcher: Optional[PitcherChange] = None
        
    def add_pitcher_change(self, change: PitcherChange):
        """Add a pitcher change event"""
        self.pitcher_changes.append(change)
        self.current_pitcher = change
        
    def add_pitch(self, pitch: PitchData):
        """Add pitch data with pitcher information"""
        if self.current_pitcher:
            pitch.pitcher_name = self.current_pitcher.pitcher_name
        self.pitch_data.append(pitch)
        
    def add_swing(self, swing: SwingData):
        self.swing_data.append(swing)
        
    def generate_report(self) -> Dict:
        """Generate a comprehensive scouting report"""
        return {
            'pitcher_report': self._generate_pitcher_report(),
            'batter_report': self._generate_batter_report(),
            'pitch_sequence': self._generate_pitch_sequence(),
            'heatmap_data': self._generate_heatmap_data()
        }
    
    def _generate_pitcher_report(self) -> Dict:
        """Generate pitcher-specific insights"""
        if not self.pitch_data:
            return {}
            
        # Group pitches by pitcher
        pitcher_data = {}
        for pitch in self.pitch_data:
            if pitch.pitcher_name not in pitcher_data:
                pitcher_data[pitch.pitcher_name] = {
                    'pitch_data': [],
                    'count_based_selection': {},
                    'pitch_types': {},
                    'velocity_stats': {
                        'max': 0.0,
                        'avg': 0.0,
                        'min': float('inf'),
                        'by_pitch_type': {},
                        'trend': []
                    }
                }
            pitcher_data[pitch.pitcher_name]['pitch_data'].append(pitch)
            
            # Update velocity statistics
            stats = pitcher_data[pitch.pitcher_name]['velocity_stats']
            stats['max'] = max(stats['max'], pitch.velocity)
            stats['min'] = min(stats['min'], pitch.velocity)
            stats['trend'].append(pitch.velocity)
            
            if pitch.type not in stats['by_pitch_type']:
                stats['by_pitch_type'][pitch.type] = {
                    'max': 0.0,
                    'avg': 0.0,
                    'min': float('inf'),
                    'count': 0
                }
            
            pitch_type_stats = stats['by_pitch_type'][pitch.type]
            pitch_type_stats['max'] = max(pitch_type_stats['max'], pitch.velocity)
            pitch_type_stats['min'] = min(pitch_type_stats['min'], pitch.velocity)
            pitch_type_stats['count'] += 1
        
        # Calculate averages
        for pitcher_name, data in pitcher_data.items():
            stats = data['velocity_stats']
            stats['avg'] = sum(stats['trend']) / len(stats['trend'])
            
            for pitch_type, pitch_stats in stats['by_pitch_type'].items():
                pitch_stats['avg'] = sum(pitch_stats['trend']) / pitch_stats['count']
        
        # Generate report for each pitcher
        pitcher_reports = {}
        for pitcher_name, data in pitcher_data.items():
            pitch_data = data['pitch_data']
            total_pitches = len(pitch_data)
            velocity_stats = data['velocity_stats']
            
            # Calculate pitch type distribution
            pitch_types = {}
            count_based_selection = {}
            
            for pitch in pitch_data:
                # Track pitch types
                if pitch.type not in pitch_types:
                    pitch_types[pitch.type] = {
                        'count': 0,
                        'avg_speed': 0.0,
                        'success_rate': 0.0
                    }
                pitch_types[pitch.type]['count'] += 1
                pitch_types[pitch.type]['avg_speed'] += pitch.speed
                
                # Track count-based pitch selection
                if pitch.count not in count_based_selection:
                    count_based_selection[pitch.count] = {
                        'total': 0,
                        'types': {}
                    }
                count_based_selection[pitch.count]['total'] += 1
                if pitch.type not in count_based_selection[pitch.count]['types']:
                    count_based_selection[pitch.count]['types'][pitch.type] = 0
                count_based_selection[pitch.count]['types'][pitch.type] += 1
            
            # Calculate averages and success rates
            for pitch_type in pitch_types:
                data = pitch_types[pitch_type]
                data['percentage'] = (data['count'] / total_pitches) * 100
                data['avg_speed'] /= data['count']
                
                successful_pitches = sum(1 for p in pitch_data 
                                       if p.type == pitch_type and p.result in ['out', 'strikeout'])
                data['success_rate'] = (successful_pitches / data['count']) * 100
            
            # Calculate count-based percentages
            for count in count_based_selection:
                total = count_based_selection[count]['total']
                for pitch_type in count_based_selection[count]['types']:
                    count_based_selection[count]['types'][pitch_type] = (
                        count_based_selection[count]['types'][pitch_type] / total * 100
                    )
            
            pitcher_reports[pitcher_name] = {
                'pitch_types': pitch_types,
                'total_pitches': total_pitches,
                'count_based_selection': count_based_selection,
                'velocity_stats': velocity_stats,
                'summary': self._generate_pitcher_summary(pitch_types, count_based_selection, velocity_stats)
            }
        
        return {
            'pitcher_reports': pitcher_reports,
            'pitcher_changes': [
                {
                    'timestamp': change.timestamp,
                    'pitcher_name': change.pitcher_name,
                    'team': change.team,
                    'jersey_number': change.jersey_number,
                    'inning': change.inning,
                    'score': change.score
                }
                for change in self.pitcher_changes
            ]
        }
    
    def _generate_batter_report(self) -> Dict:
        """Generate batter-specific insights"""
        if not self.swing_data:
            return {}
            
        # Calculate swing metrics
        avg_speed = sum(s.speed for s in self.swing_data) / len(self.swing_data)
        avg_angle = sum(s.angle for s in self.swing_data) / len(self.swing_data)
        avg_aggression = sum(s.aggression for s in self.swing_data) / len(self.swing_data)
        
        # Analyze timing
        early_swings = sum(1 for s in self.swing_data if s.timing < -0.1)
        late_swings = sum(1 for s in self.swing_data if s.timing > 0.1)
        on_time = len(self.swing_data) - early_swings - late_swings
        
        return {
            'swing_metrics': {
                'average_speed': avg_speed,
                'average_angle': avg_angle,
                'average_aggression': avg_aggression
            },
            'timing_analysis': {
                'early_swings': early_swings,
                'late_swings': late_swings,
                'on_time': on_time,
                'timing_percentage': (on_time / len(self.swing_data)) * 100
            },
            'summary': self._generate_batter_summary(avg_speed, avg_angle, avg_aggression, 
                                                   early_swings, late_swings, on_time)
        }
    
    def _generate_pitch_sequence(self) -> List[Dict]:
        """Generate chronological sequence of pitches"""
        return [
            {
                'timestamp': pitch.timestamp,
                'type': pitch.type,
                'speed': pitch.speed,
                'result': pitch.result
            }
            for pitch in sorted(self.pitch_data, key=lambda x: x.timestamp)
        ]
    
    def _generate_heatmap_data(self) -> Dict:
        """Generate data for pitch location heatmap"""
        # This would normally use actual pitch location data
        # For now, return a simplified version
        return {
            'zones': [
                {'x': 0.2, 'y': 0.2, 'value': 5},
                {'x': 0.5, 'y': 0.5, 'value': 8},
                {'x': 0.8, 'y': 0.8, 'value': 3}
            ]
        }
    
    def _generate_pitcher_summary(self, pitch_types: Dict, count_based_selection: Dict, velocity_stats: Dict) -> str:
        """Generate natural language summary for pitcher"""
        summary = []
        
        # Add velocity summary
        summary.append("Velocity Analysis:")
        summary.append(f"- Max Velocity: {velocity_stats['max']:.1f} mph")
        summary.append(f"- Average Velocity: {velocity_stats['avg']:.1f} mph")
        summary.append(f"- Min Velocity: {velocity_stats['min']:.1f} mph")
        
        summary.append("\nVelocity by Pitch Type:")
        for pitch_type, stats in velocity_stats['by_pitch_type'].items():
            summary.append(f"- {pitch_type.capitalize()}:")
            summary.append(f"  Max: {stats['max']:.1f} mph")
            summary.append(f"  Avg: {stats['avg']:.1f} mph")
            summary.append(f"  Min: {stats['min']:.1f} mph")
        
        # Add pitch type distribution
        summary.append("\nPitch Type Distribution:")
        for pitch_type, data in pitch_types.items():
            summary.append(f"- {pitch_type.capitalize()}: {data['percentage']:.1f}% "
                         f"(avg speed: {data['avg_speed']:.1f} mph, "
                         f"success rate: {data['success_rate']:.1f}%)")
        
        # Add count-based pitch selection analysis
        summary.append("\nCount-Based Pitch Selection:")
        for count, data in count_based_selection.items():
            if data['total'] >= 5:  # Only show counts with enough data
                summary.append(f"\n{count} Count:")
                for pitch_type, percentage in data['types'].items():
                    summary.append(f"- {pitch_type.capitalize()}: {percentage:.1f}%")
        
        # Add overall assessment
        total_pitches = sum(data['count'] for data in pitch_types.values())
        summary.append(f"\nTotal Pitches: {total_pitches}")
        
        return "\n".join(summary)
    
    def _generate_batter_summary(self, avg_speed: float, avg_angle: float, 
                               avg_aggression: float, early_swings: int, 
                               late_swings: int, on_time: int) -> str:
        """Generate natural language summary for batter"""
        summary = []
        
        # Add swing characteristics
        summary.append("Swing Characteristics:")
        summary.append(f"- Average Speed: {avg_speed:.1f}")
        summary.append(f"- Average Angle: {avg_angle:.1f}°")
        summary.append(f"- Average Aggression: {avg_aggression:.1f}")
        
        # Add timing analysis
        total_swings = early_swings + late_swings + on_time
        timing_percentage = (on_time / total_swings) * 100 if total_swings > 0 else 0
        
        summary.append("\nTiming Analysis:")
        summary.append(f"- Early Swings: {early_swings}")
        summary.append(f"- Late Swings: {late_swings}")
        summary.append(f"- On Time: {on_time}")
        summary.append(f"- Timing Percentage: {timing_percentage:.1f}%")
        
        return "\n".join(summary) 