import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class Clip:
    start_frame: int
    end_frame: int
    timestamp: float
    confidence: float

class ClipTagger:
    def __init__(self):
        self.motion_threshold = 30  # Threshold for motion detection
        self.min_clip_length = 30   # Minimum frames for a pitch clip
        self.max_clip_length = 150  # Maximum frames for a pitch clip
        self.tags = ['pitch', 'swing', 'hit', 'catch', 'throw']
        
    def detect_clips(self, video_path: str) -> List[Clip]:
        """
        Detect pitch clips in a video file
        Returns: List of detected clips with start/end frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize variables
        clips = []
        motion_scores = []
        frame_count = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
                
                # Keep only recent motion scores
                if len(motion_scores) > self.max_clip_length:
                    motion_scores.pop(0)
            
            prev_frame = gray
            frame_count += 1
            
            # Process motion scores to detect clips
            if len(motion_scores) >= self.min_clip_length:
                self._process_motion_scores(motion_scores, clips, frame_count, fps)
        
        cap.release()
        return self._merge_overlapping_clips(clips)
    
    def _process_motion_scores(self, motion_scores: List[float], 
                             clips: List[Clip], frame_count: int, fps: float):
        """Process motion scores to detect potential pitch clips"""
        # Calculate average motion in recent frames
        recent_motion = np.mean(motion_scores[-self.min_clip_length:])
        
        # If motion is above threshold, potential pitch detected
        if recent_motion > self.motion_threshold:
            # Look for start of motion
            start_frame = frame_count - self.min_clip_length
            for i in range(len(motion_scores) - 1, -1, -1):
                if motion_scores[i] < self.motion_threshold:
                    start_frame = frame_count - i - 1
                    break
            
            # Look for end of motion
            end_frame = frame_count
            for i in range(len(motion_scores)):
                if motion_scores[i] < self.motion_threshold:
                    end_frame = frame_count - len(motion_scores) + i
                    break
            
            # Create clip if valid length
            clip_length = end_frame - start_frame
            if self.min_clip_length <= clip_length <= self.max_clip_length:
                clips.append(Clip(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    timestamp=start_frame / fps,
                    confidence=recent_motion / self.motion_threshold
                ))
    
    def _merge_overlapping_clips(self, clips: List[Clip]) -> List[Clip]:
        """Merge overlapping clips to avoid duplicates"""
        if not clips:
            return []
            
        # Sort clips by start frame
        clips.sort(key=lambda x: x.start_frame)
        
        merged = []
        current = clips[0]
        
        for next_clip in clips[1:]:
            # If clips overlap, merge them
            if next_clip.start_frame <= current.end_frame:
                current.end_frame = max(current.end_frame, next_clip.end_frame)
                current.confidence = max(current.confidence, next_clip.confidence)
            else:
                merged.append(current)
                current = next_clip
        
        merged.append(current)
        return merged
    
    def extract_clip(self, video_path: str, clip: Clip) -> np.ndarray:
        """
        Extract a specific clip from the video
        Returns: Array of frames for the clip
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Set frame position to start of clip
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame)
        
        frames = []
        for _ in range(clip.end_frame - clip.start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def preview_clip(self, video_path: str, clip: Clip, output_path: str = None):
        """
        Preview a detected clip by saving it as a video file
        """
        frames = self.extract_clip(video_path, clip)
        if not frames.size:
            return
            
        # Get video properties from original video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create video writer
        if output_path is None:
            output_path = f"clip_{clip.start_frame}_{clip.end_frame}.mp4"
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()

    def tag(self, frame: np.ndarray, objects: Dict, poses: List[Dict]) -> List[str]:
        """
        Tag important events in a frame
        Returns: List of tags for the frame
        """
        tags = []
        
        # For now, just tag frames with balls and people
        if objects.get('ball'):
            tags.append('pitch')
        if objects.get('person'):
            tags.append('player')
            
        return tags 