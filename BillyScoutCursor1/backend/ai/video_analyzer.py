import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import yt_dlp
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .detector import BaseballDetector
from .report import ScoutingReport, PitchData, SwingData
from .pose import PoseEstimator
from .classifier import PitchClassifier
from .clip_tagger import ClipTagger
from .heatmap import HeatmapGenerator
import json
import re
import math
import gc
import datetime
import psutil
import time
import platform

class KalmanFilter:
    """Kalman filter for baseball trajectory tracking."""
    def __init__(self, dt=1.0, process_noise=1e-5, measurement_noise=1e-1):
        # State transition matrix (x, y, vx, vy)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (only x, y are observed)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State estimate
        self.x = np.zeros((4, 1))
        
        # Error covariance
        self.P = np.eye(4)
        
    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()
    
    def update(self, measurement):
        """Update state with measurement."""
        y = measurement.reshape(2, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[:2].flatten()

class VideoAnalyzer:
    def __init__(self, debug=False):
        """Initialize the video analyzer."""
        self.debug = debug
        self.detector = BaseballDetector()
        self.pose_estimator = PoseEstimator()
        self.pitch_classifier = PitchClassifier()
        self.clip_tagger = ClipTagger()
        self.heatmap_generator = HeatmapGenerator()
        
        # Performance settings
        self.frame_buffer_size = 10
        self.max_parallel_frames = 4
        self.memory_threshold = 0.8
        self.disk_threshold = 0.9
        
        # Trajectory settings
        self.kalman_filter = KalmanFilter()
        self.trajectory_history = []
        self.max_trajectory_length = 30
        self.min_confidence = 0.7
        self.max_prediction_frames = 3
        
        # Initialize monitoring
        self.start_time = None
        self.frame_times = []
        self.memory_usage = []
        
        self.report = ScoutingReport(
            video_path="",
            detector=self.detector,
            pose_estimator=self.pose_estimator,
            pitch_classifier=self.pitch_classifier,
            clip_tagger=self.clip_tagger,
            heatmap_generator=self.heatmap_generator
        )

    def check_resources(self):
        """Check system resources and adjust processing parameters if needed."""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                self.frame_buffer_size = max(5, self.frame_buffer_size // 2)
                if self.debug:
                    print(f"High memory usage ({memory.percent}%), reducing buffer size to {self.frame_buffer_size}")

            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > self.disk_threshold * 100:
                raise RuntimeError(f"Insufficient disk space: {disk.percent}% used")

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.max_parallel_frames = max(2, self.max_parallel_frames - 1)
                if self.debug:
                    print(f"High CPU usage ({cpu_percent}%), reducing parallel processing to {self.max_parallel_frames}")

        except Exception as e:
            if self.debug:
                print(f"Resource check error: {str(e)}")
            raise

    def enhance_frame(self, frame):
        """Enhance frame quality for better detection."""
        try:
            # Convert to LAB color space for better contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply adaptive thresholding
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            if self.debug:
                print(f"Frame enhancement error: {str(e)}")
            return frame

    def process_frame_batch(self, frames):
        """Process a batch of frames in parallel."""
        try:
            results = []
            for frame in frames:
                # Enhance frame
                enhanced = self.enhance_frame(frame)
                
                # Detect objects
                objects = self.detector.detect(enhanced)
                
                # Track ball position
                if 'ball' in objects and len(objects['ball']) > 0:
                    x, y, w, h = objects['ball'][0]
                    center_x = x + w/2
                    center_y = y + h/2
                    results.append((center_x, center_y))
                else:
                    results.append(None)
                    
            return results
            
        except Exception as e:
            if self.debug:
                print(f"Batch processing error: {str(e)}")
            return [None] * len(frames)

    def get_video_url(self, url: str) -> Optional[str]:
        """Extract the video URL from a page URL."""
        try:
            # For YouTube videos, return the URL as is
            if 'youtube.com' in url or 'youtu.be' in url:
                return url
            
            # For other pages, try to find video URL in HTML
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for video elements
                video_elements = soup.find_all('video')
                for video in video_elements:
                    if video.get('src'):
                        return video['src']
                
                # Look for iframes
                iframes = soup.find_all('iframe')
                for iframe in iframes:
                    if iframe.get('src'):
                        return iframe['src']
            
            print("No video URL found in page")
            return None
            
        except Exception as e:
            print(f"Error extracting video URL: {str(e)}")
            return None
    
    def download_video(self, url: str) -> Optional[str]:
        """Download a video from a URL."""
        try:
            # For YouTube videos, download directly
            if 'youtube.com' in url or 'youtu.be' in url:
                print(f"Downloading YouTube video: {url}")
                ydl_opts = {
                    'format': 'best[height<=720]',  # Limit to 720p for faster processing
                    'outtmpl': 'temp_video.%(ext)s',
                    'quiet': True,
                    'no_warnings': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return f"temp_video.{info['ext']}"
            
            # For other URLs, try to get the actual video URL first
            video_url = self.get_video_url(url)
            if not video_url:
                print("Could not find video URL")
                return None
            
            print(f"Downloading video from: {video_url}")
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': 'temp_video.%(ext)s',
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                return f"temp_video.{info['ext']}"
                
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None
    
    def preprocess_video(self, video_path):
        """Preprocess video for optimal analysis."""
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Get video properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get original video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            duration = total_frames / fps if fps > 0 else 0

            print(f"\nOriginal video properties:")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps:.1f}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Total frames: {total_frames}")
            print(f"Codec: {codec}")

            # Create temporary directory for processed video
            temp_dir = os.path.join(os.path.dirname(video_path), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            processed_path = os.path.join(temp_dir, "processed_" + os.path.basename(video_path))

            # Calculate optimal resolution
            # Target resolution for optimal performance and quality
            max_width = 1280
            max_height = 720
            
            # Define common aspect ratios and their tolerances
            common_ratios = {
                '16:9': (16/9, 0.05),    # Standard widescreen
                '4:3': (4/3, 0.05),      # Standard fullscreen
                '1:1': (1, 0.01),        # Square
                '9:16': (9/16, 0.05),    # Vertical video
                '3:4': (3/4, 0.05),      # Vertical video
                '21:9': (21/9, 0.05),    # Ultra widescreen
                '2.35:1': (2.35, 0.05),  # Cinemascope
                '2.39:1': (2.39, 0.05)   # Anamorphic
            }
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Find closest common aspect ratio
            closest_match = None
            min_diff = float('inf')
            for name, (ratio, tolerance) in common_ratios.items():
                diff = abs(ratio - aspect_ratio)
                if diff < min_diff and diff <= tolerance:
                    min_diff = diff
                    closest_match = (name, ratio, diff)
            
            # Determine format classification
            is_standard = closest_match is not None
            standard_name = closest_match[0] if is_standard else "Non-standard"
            standard_ratio = closest_match[1] if is_standard else aspect_ratio
            ratio_diff = closest_match[2] if is_standard else 0
            
            # Determine orientation
            is_landscape = aspect_ratio > 1
            is_portrait = aspect_ratio < 1
            is_square = abs(aspect_ratio - 1) < 0.01
            
            print(f"\nVideo format analysis:")
            print(f"Original aspect ratio: {aspect_ratio:.3f}")
            if is_standard:
                print(f"Standard format: {standard_name} ({standard_ratio:.3f})")
                print(f"Difference from standard: {ratio_diff:.3f}")
            else:
                print(f"Non-standard format")
                print(f"Custom aspect ratio: {aspect_ratio:.3f}")
            print(f"Orientation: {'Landscape' if is_landscape else 'Portrait' if is_portrait else 'Square'}")
            
            # Determine target dimensions
            if width > max_width or height > max_height:
                if is_standard:
                    # Use standard ratio for scaling
                    if is_landscape:
                        target_width = max_width
                        target_height = int(target_width / standard_ratio)
                        if target_height > max_height:
                            target_height = max_height
                            target_width = int(target_height * standard_ratio)
                    elif is_portrait:
                        target_height = max_height
                        target_width = int(target_height * standard_ratio)
                        if target_width > max_width:
                            target_width = max_width
                            target_height = int(target_width / standard_ratio)
                    else:
                        target_width = min(max_width, max_height)
                        target_height = target_width
                else:
                    # Maintain original aspect ratio
                    if is_landscape:
                        target_width = max_width
                        target_height = int(target_width / aspect_ratio)
                        if target_height > max_height:
                            target_height = max_height
                            target_width = int(target_height * aspect_ratio)
                    elif is_portrait:
                        target_height = max_height
                        target_width = int(target_height * aspect_ratio)
                        if target_width > max_width:
                            target_width = max_width
                            target_height = int(target_width / aspect_ratio)
                    else:
                        target_width = min(max_width, max_height)
                        target_height = target_width
            else:
                # Keep original dimensions if within limits
                target_width = width
                target_height = height

            # Ensure dimensions are even numbers
            target_width = target_width - (target_width % 2)
            target_height = target_height - (target_height % 2)

            # Verify aspect ratio preservation
            final_aspect_ratio = target_width / target_height
            aspect_ratio_diff = abs(final_aspect_ratio - aspect_ratio)
            ratio_preserved = aspect_ratio_diff < 0.01

            if not ratio_preserved:
                print(f"Warning: Aspect ratio changed from {aspect_ratio:.3f} to {final_aspect_ratio:.3f}")
                print(f"Difference: {aspect_ratio_diff:.3f}")

            # Set target properties
            target_fps = 30.0
            target_codec = cv2.VideoWriter_fourcc(*'mp4v')

            print(f"\nProcessing video to:")
            print(f"Resolution: {target_width}x{target_height}")
            print(f"FPS: {target_fps}")
            print(f"Codec: MP4V")
            print(f"Original aspect ratio: {aspect_ratio:.3f}")
            print(f"Final aspect ratio: {final_aspect_ratio:.3f}")
            print(f"Scale factor: {target_width/width:.2f}x")
            print(f"Aspect ratio preserved: {'Yes' if ratio_preserved else 'No'}")

            # Initialize video writer
            out = cv2.VideoWriter(
                processed_path,
                target_codec,
                target_fps,
                (target_width, target_height)
            )

            # Process frames
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame with high-quality interpolation
                if width != target_width or height != target_height:
                    frame = cv2.resize(frame, (target_width, target_height), 
                                     interpolation=cv2.INTER_LANCZOS4)

                # Enhance frame
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

                # Write processed frame
                out.write(frame)
                frame_count += 1

                # Update progress every 10%
                progress = int((frame_count / total_frames) * 100)
                if progress % 10 == 0:
                    print(f"Progress: {progress}%")

            # Release resources
            cap.release()
            out.release()

            # Verify processed video
            if not os.path.exists(processed_path):
                raise ValueError("Failed to create processed video")

            processed_size = os.path.getsize(processed_path)
            original_size = os.path.getsize(video_path)
            compression_ratio = original_size / processed_size if processed_size > 0 else 1

            print(f"\nProcessing complete!")
            print(f"Original size: {original_size / 1024 / 1024:.1f} MB")
            print(f"Processed size: {processed_size / 1024 / 1024:.1f} MB")
            print(f"Compression ratio: {compression_ratio:.1f}x")

            return processed_path

        except Exception as e:
            print(f"Error preprocessing video: {str(e)}")
            raise

    def analyze_broadcast(self, video_path):
        """Analyze a baseball broadcast video with improved performance."""
        try:
            self.start_time = time.time()
            self.frame_times = []
            self.memory_usage = []
            
            # Check resources before starting
            self.check_resources()
            
            # Preprocess video
            processed_path = self.preprocess_video(video_path)
            
            if not os.path.exists(processed_path):
                raise FileNotFoundError(f"Processed video file not found: {processed_path}")

            cap = cv2.VideoCapture(processed_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open processed video: {processed_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Initialize tracking variables
            processed_frames = 0
            ball_positions = []
            frame_buffer = []
            last_progress = 0

            print(f"\nAnalyzing video with optimized settings:")
            print(f"Buffer size: {self.frame_buffer_size}")
            print(f"Parallel processing: {self.max_parallel_frames} frames")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps:.1f}")

            while cap.isOpened():
                # Check resources periodically
                if processed_frames % 100 == 0:
                    self.check_resources()
                
                ret, frame = cap.read()
                if not ret:
                    break

                # Add frame to buffer
                frame_buffer.append(frame)
                
                # Process when buffer is full
                if len(frame_buffer) >= self.frame_buffer_size:
                    # Process frames in parallel
                    batch_results = self.process_frame_batch(frame_buffer)
                    
                    # Update ball positions
                    for result in batch_results:
                        if result is not None:
                            ball_positions.append(result)
                    
                    # Clear buffer and force garbage collection
                    frame_buffer.clear()
                    gc.collect()
                    
                    # Update progress
                    processed_frames += len(batch_results)
                    progress = int((processed_frames / total_frames) * 100)
                    if progress > last_progress and progress % 5 == 0:
                        print(f"Progress: {progress}% - Detected {len(ball_positions)} ball positions")
                        last_progress = progress
                        
                        # Log performance metrics
                        self.frame_times.append(time.time() - self.start_time)
                        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)

            cap.release()

            # Calculate performance metrics
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            max_memory = max(self.memory_usage) if self.memory_usage else 0

            # Smooth trajectory
            smoothed_positions = []
            if len(ball_positions) > 2:
                # Apply moving average smoothing
                window_size = 3
                for i in range(len(ball_positions)):
                    start = max(0, i - window_size // 2)
                    end = min(len(ball_positions), i + window_size // 2 + 1)
                    window = ball_positions[start:end]
                    avg_x = sum(x for x, y in window) / len(window)
                    avg_y = sum(y for x, y in window) / len(window)
                    smoothed_positions.append((avg_x, avg_y))
            else:
                smoothed_positions = ball_positions

            # Calculate pitch characteristics
            characteristics = self.pitch_classifier._calculate_characteristics(smoothed_positions)
            
            # Generate heatmap
            heatmap = self.heatmap_generator.generate_heatmap(smoothed_positions)
            heatmap_path = os.path.join(os.path.dirname(video_path), "pitch_heatmap.png")
            cv2.imwrite(heatmap_path, heatmap)
            
            # Create report
            report = {
                "metadata": {
                    "analysis_info": {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "version": "1.0",
                        "analysis_type": "baseball_pitch",
                        "analysis_id": f"ANL-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "status": "completed",
                        "parameters": {
                            "frame_buffer_size": self.frame_buffer_size,
                            "max_parallel_frames": self.max_parallel_frames,
                            "memory_threshold": self.memory_threshold,
                            "disk_threshold": self.disk_threshold
                        },
                        "performance": {
                            "start_time": self.start_time,
                            "processing_duration": f"{duration:.1f} seconds",
                            "frames_per_second": f"{processed_frames/duration:.1f}",
                            "memory_usage": f"{avg_memory:.1f} MB"
                        },
                        "performance_metrics": {
                            "average_frame_time": f"{avg_frame_time:.3f} seconds",
                            "average_memory_usage": f"{avg_memory:.1f} MB",
                            "peak_memory_usage": f"{max_memory:.1f} MB",
                            "total_processing_time": f"{time.time() - self.start_time:.1f} seconds",
                            "frames_per_second": f"{processed_frames / (time.time() - self.start_time):.1f}"
                        }
                    },
                    "video_info": {
                        "source": video_path,
                        "format": video_format,
                        "resolution": f"{frame_width}x{frame_height}",
                        "frame_rate": fps,
                        "duration": f"{duration:.2f} seconds",
                        "total_frames": total_frames,
                        "processed_frames": processed_frames,
                        "frame_skip": frame_skip,
                        "processing_rate": f"{(processed_frames / total_frames * 100):.1f}%",
                        "file_size": f"{os.path.getsize(video_path) / (1024 * 1024):.1f} MB"
                    },
                    "detection_stats": {
                        "ball_detections": len(ball_positions),
                        "detection_rate": f"{(len(ball_positions) / processed_frames * 100):.1f}%",
                        "average_confidence": f"{sum(d['confidence'] for d in ball_positions) / len(ball_positions):.2f}",
                        "min_confidence": f"{min(d['confidence'] for d in ball_positions):.2f}",
                        "max_confidence": f"{max(d['confidence'] for d in ball_positions):.2f}",
                        "average_processing_time": f"{(duration / processed_frames * 1000):.1f} ms/frame",
                        "total_processing_time": f"{duration:.1f} seconds",
                        "frames_per_second": f"{processed_frames/duration:.1f}",
                        "memory_usage": f"{avg_memory:.1f} MB"
                    }
                },
                "video_info": {
                    "duration": duration,
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "ball_positions_detected": len(ball_positions),
                    "smoothed_positions": len(smoothed_positions)
                },
                "pitch": {
                    "type": characteristics["type"],
                    "velocity": characteristics["speed"],
                    "vertical_break": characteristics["vertical_break"],
                    "horizontal_break": characteristics["horizontal_break"],
                    "spin_rate": characteristics["spin_rate"],
                    "confidence": characteristics["confidence"]
                },
                "heatmap_points": [{"x": x, "y": y} for x, y in smoothed_positions],
                "pitch_analysis": {
                    "total_pitches": len(ball_positions),
                    "average_velocity": f"{sum(p['velocity'] for p in ball_positions) / len(ball_positions):.1f} mph",
                    "min_velocity": f"{min(p['velocity'] for p in ball_positions):.1f} mph",
                    "max_velocity": f"{max(p['velocity'] for p in ball_positions):.1f} mph",
                    "average_vertical_break": f"{sum(p['vertical_break'] for p in ball_positions) / len(ball_positions):.1f} inches",
                    "average_horizontal_break": f"{sum(p['horizontal_break'] for p in ball_positions) / len(ball_positions):.1f} inches",
                    "pitch_types": {
                        "fastball": sum(1 for p in ball_positions if p['type'] == 'fastball'),
                        "curveball": sum(1 for p in ball_positions if p['type'] == 'curveball'),
                        "slider": sum(1 for p in ball_positions if p['type'] == 'slider'),
                        "changeup": sum(1 for p in ball_positions if p['type'] == 'changeup')
                    }
                },
                "heatmap": {
                    "path": heatmap_path,
                    "resolution": f"{heatmap.shape[1]}x{heatmap.shape[0]}",
                    "points": len(smoothed_positions)
                },
                "processing_stats": {
                    "total_frames": total_frames,
                    "processed_frames": processed_frames,
                    "processing_time": time.time() - self.start_time,
                    "frames_per_second": processed_frames / (time.time() - self.start_time) if (time.time() - self.start_time) > 0 else 0,
                    "memory_usage": f"{avg_memory:.1f} MB"
                },
                "pitches": ball_positions
            }
            
            print("\nAnalysis complete!")
            print(f"Detected {len(ball_positions)} ball positions")
            print(f"Smoothed to {len(smoothed_positions)} positions")
            print(f"Pitch type: {characteristics['type']}")
            print(f"Velocity: {characteristics['speed']:.1f} mph")
            print(f"Vertical break: {characteristics['vertical_break']:.1f} inches")
            print(f"Horizontal break: {characteristics['horizontal_break']:.1f} inches")
            
            # Clean up temporary files
            try:
                os.remove(processed_path)
                os.rmdir(os.path.dirname(processed_path))
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {str(e)}")
            
            return report

        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            raise 

    def predict_trajectory(self, current_position: Tuple[float, float], 
                         velocity: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Predict future trajectory points using Kalman filter."""
        try:
            # Initialize Kalman filter with current state
            self.kalman_filter.x = np.array([
                [current_position[0]],
                [current_position[1]],
                [velocity[0]],
                [velocity[1]]
            ])
            
            predicted_points = []
            for _ in range(self.max_prediction_frames):
                predicted = self.kalman_filter.predict()
                predicted_points.append((predicted[0], predicted[1]))
            
            return predicted_points
            
        except Exception as e:
            if self.debug:
                print(f"Trajectory prediction error: {str(e)}")
            return []

    def interpolate_missing_frames(self, positions: List[Tuple[float, float]], 
                                 frame_numbers: List[int]) -> List[Tuple[float, float]]:
        """Interpolate missing positions between detected frames."""
        try:
            if len(positions) < 2:
                return positions
                
            interpolated = []
            for i in range(len(positions) - 1):
                # Add current position
                interpolated.append(positions[i])
                
                # Calculate gap between frames
                gap = frame_numbers[i + 1] - frame_numbers[i]
                if gap > 1:
                    # Calculate step sizes
                    dx = (positions[i + 1][0] - positions[i][0]) / gap
                    dy = (positions[i + 1][1] - positions[i][1]) / gap
                    
                    # Interpolate intermediate positions
                    for j in range(1, gap):
                        x = positions[i][0] + dx * j
                        y = positions[i][1] + dy * j
                        interpolated.append((x, y))
            
            # Add last position
            interpolated.append(positions[-1])
            
            return interpolated
            
        except Exception as e:
            if self.debug:
                print(f"Frame interpolation error: {str(e)}")
            return positions

    def calculate_velocity(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                         time_interval: float) -> Tuple[float, float]:
        """Calculate velocity between two positions."""
        try:
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            vx = dx / time_interval
            vy = dy / time_interval
            return (vx, vy)
        except Exception as e:
            if self.debug:
                print(f"Velocity calculation error: {str(e)}")
            return (0, 0)

    def analyze_trajectory(self, positions: List[Tuple[float, float]], 
                         frame_numbers: List[int], fps: float) -> Dict:
        """Analyze complete trajectory with prediction and interpolation."""
        try:
            if len(positions) < 2:
                return {"error": "Insufficient positions for trajectory analysis"}
            
            # Interpolate missing frames
            interpolated_positions = self.interpolate_missing_frames(positions, frame_numbers)
            
            # Calculate velocities
            velocities = []
            for i in range(len(interpolated_positions) - 1):
                v = self.calculate_velocity(
                    interpolated_positions[i],
                    interpolated_positions[i + 1],
                    1.0 / fps
                )
                velocities.append(v)
            
            # Predict future trajectory
            last_position = interpolated_positions[-1]
            last_velocity = velocities[-1] if velocities else (0, 0)
            predicted_points = self.predict_trajectory(last_position, last_velocity)
            
            # Calculate trajectory metrics
            total_distance = 0
            max_velocity = 0
            avg_velocity = 0
            
            for i in range(len(interpolated_positions) - 1):
                dx = interpolated_positions[i + 1][0] - interpolated_positions[i][0]
                dy = interpolated_positions[i + 1][1] - interpolated_positions[i][1]
                distance = math.sqrt(dx*dx + dy*dy)
                total_distance += distance
                
                if velocities:
                    speed = math.sqrt(velocities[i][0]**2 + velocities[i][1]**2)
                    max_velocity = max(max_velocity, speed)
                    avg_velocity += speed
            
            avg_velocity /= len(velocities) if velocities else 1
            
            return {
                "interpolated_positions": interpolated_positions,
                "predicted_points": predicted_points,
                "velocities": velocities,
                "metrics": {
                    "total_distance": total_distance,
                    "max_velocity": max_velocity,
                    "average_velocity": avg_velocity,
                    "total_frames": len(interpolated_positions),
                    "interpolated_frames": len(interpolated_positions) - len(positions),
                    "predicted_frames": len(predicted_points)
                }
            }
            
        except Exception as e:
            if self.debug:
                print(f"Trajectory analysis error: {str(e)}")
            return {"error": str(e)} 