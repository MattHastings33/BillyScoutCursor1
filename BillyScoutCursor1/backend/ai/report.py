from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import cv2
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
from .detector import BaseballDetector
from .pose import PoseEstimator
from .classifier import PitchClassifier
from .clip_tagger import ClipTagger
from .heatmap import HeatmapGenerator
import os

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
    vertical_break: float
    horizontal_break: float
    spin_rate: float
    release_pos: List[float]
    heatmap_points: List[List[float]]

@dataclass
class SwingData:
    speed: float
    angle: float
    aggression: float
    timing: float  # relative to pitch arrival

class ScoutingReport:
    def __init__(
        self,
        video_path: str,
        detector: BaseballDetector,
        pose_estimator: PoseEstimator,
        pitch_classifier: PitchClassifier,
        clip_tagger: ClipTagger,
        heatmap_generator: HeatmapGenerator,
        box_score_data: Optional[Dict] = None
    ):
        self.video_path = video_path
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.pitch_classifier = pitch_classifier
        self.clip_tagger = clip_tagger
        self.heatmap_generator = heatmap_generator
        self.box_score_data = box_score_data
        self.pitch_data: List[PitchData] = []
        self.swing_data: List[SwingData] = []
        self.pitcher_changes: List[PitcherChange] = []
        self.current_pitcher: Optional[PitcherChange] = None
        self.chunk_size = 300  # Process 10 minutes at 30fps
        self.frame_skip = 2  # Process every other frame for efficiency
        self.executor = ThreadPoolExecutor(max_workers=4)  # For parallel processing
        
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
        
    async def process_chunk(self, start_frame: int, end_frame: int) -> Dict:
        """Process a chunk of frames"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        chunk_data = {
            'pitches': [],
            'clips': [],
            'heatmap_points': []
        }
        
        frame_count = start_frame
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames for efficiency
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            # Process frame
            objects = self.detector.detect(frame)
            poses = self.pose_estimator.estimate(frame, objects)
            
            # Classify pitch type
            pitch_type = self.pitch_classifier.classify(objects)
            if pitch_type:
                chunk_data['pitches'].append({
                    'frame': frame_count,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                    'type': pitch_type,
                    'velocity': self.pitch_classifier.estimate_velocity(objects)
                })
            
            # Tag important clips
            clip_tags = self.clip_tagger.tag(frame, objects, poses)
            if clip_tags:
                chunk_data['clips'].append({
                    'frame': frame_count,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                    'tags': clip_tags
                })
            
            # Generate heatmap points
            heatmap_point = self.heatmap_generator.generate_point(frame, objects, poses)
            if heatmap_point:
                chunk_data['heatmap_points'].append(heatmap_point)
            
            frame_count += 1
        
        cap.release()
        return chunk_data

    async def generate(self) -> Dict:
        """Generate a complete scouting report with chunked processing"""
        # Get video information
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()

        # Calculate chunks
        chunks = []
        for start_frame in range(0, total_frames, self.chunk_size):
            end_frame = min(start_frame + self.chunk_size, total_frames)
            chunks.append((start_frame, end_frame))

        # Process chunks in parallel
        chunk_results = await asyncio.gather(
            *[self.process_chunk(start, end) for start, end in chunks]
        )

        # Combine results
        pitches = []
        clips = []
        heatmap_points = []
        
        for chunk in chunk_results:
            pitches.extend(chunk['pitches'])
            clips.extend(chunk['clips'])
            heatmap_points.extend(chunk['heatmap_points'])

        # Generate the report
        report = {
            'video_info': {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': len(pitches) * self.frame_skip
            },
            'pitches': pitches,
            'clips': clips,
            'heatmap_points': heatmap_points
        }

        # Integrate box score data if available
        if self.box_score_data:
            # Match pitches with pitchers based on timestamps
            innings_duration = duration / self.box_score_data['total_innings']
            
            for pitch in pitches:
                inning_number = int(pitch['timestamp'] / innings_duration) + 1
                
                # Find the pitcher for this inning
                for pitcher in self.box_score_data['pitchers']:
                    if pitcher['start_inning'] <= inning_number <= pitcher['end_inning']:
                        pitch['pitcher'] = pitcher['name']
                        break

        # Clean up resources
        gc.collect()
        return report

    def __del__(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        gc.collect()
    
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
            if hasattr(pitch, 'velocity_trend') and pitch.velocity_trend:
                stats['trend'].extend(pitch.velocity_trend)
            else:
                stats['trend'].append(pitch.velocity)
            
            if pitch.type not in stats['by_pitch_type']:
                stats['by_pitch_type'][pitch.type] = {
                    'max': 0.0,
                    'avg': 0.0,
                    'min': float('inf'),
                    'count': 0,
                    'trend': []
                }
            
            pitch_type_stats = stats['by_pitch_type'][pitch.type]
            pitch_type_stats['max'] = max(pitch_type_stats['max'], pitch.velocity)
            pitch_type_stats['min'] = min(pitch_type_stats['min'], pitch.velocity)
            pitch_type_stats['count'] += 1
            if hasattr(pitch, 'velocity_trend') and pitch.velocity_trend:
                pitch_type_stats['trend'].extend(pitch.velocity_trend)
            else:
                pitch_type_stats['trend'].append(pitch.velocity)
        
        # Calculate averages
        for pitcher_name, data in pitcher_data.items():
            stats = data['velocity_stats']
            if stats['trend']:
                stats['avg'] = sum(stats['trend']) / len(stats['trend'])
            
            for pitch_type, pitch_stats in stats['by_pitch_type'].items():
                if pitch_stats['trend']:
                    pitch_stats['avg'] = sum(pitch_stats['trend']) / len(pitch_stats['trend'])
        
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

    def save(self, report_path: str):
        """Save the report to a JSON file."""
        try:
            # Generate the report data
            report_data = {
                'pitcher_report': self._generate_pitcher_report(),
                'batter_report': self._generate_batter_report(),
                'pitch_sequence': self._generate_pitch_sequence(),
                'heatmap_data': self._generate_heatmap_data()
            }
            
            # Save to file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            print(f"\nDetailed report saved to {os.path.basename(report_path)}")
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False 