import os
import sys
import cv2
import numpy as np
import asyncio
from pathlib import Path
import shutil
import yt_dlp

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.video_analyzer import VideoAnalyzer

def create_test_video(output_path: str):
    """Create a test video with different pitch types"""
    # Video parameters
    width = 1920
    height = 1080
    fps = 30
    duration = 12  # seconds
    frames = duration * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Parameters for different pitch types
    pitch_types = {
        'fastball': {
            'vertical_break': -2,  # Slight downward break
            'horizontal_break': 0,  # Minimal horizontal movement
            'speed_factor': 1.0,
            'frames_per_pitch': 30  # 1 second per pitch at 30fps
        },
        'curveball': {
            'vertical_break': 15,  # Strong downward break
            'horizontal_break': -5,  # Slight break inside to RHH
            'speed_factor': 0.8,
            'frames_per_pitch': 40  # Slower pitch = more frames
        },
        'slider': {
            'vertical_break': 5,  # Moderate downward break
            'horizontal_break': 10,  # Strong horizontal break
            'speed_factor': 0.9,
            'frames_per_pitch': 35
        },
        'changeup': {
            'vertical_break': 8,  # Moderate downward break
            'horizontal_break': -3,  # Slight fade
            'speed_factor': 0.85,
            'frames_per_pitch': 38
        }
    }

    ball_size = 10
    start_x = width // 6  # Start from pitcher's mound
    start_y = height // 2 - 100  # Start slightly above center for gravity effect

    for pitch_type, params in pitch_types.items():
        # Create multiple pitches of each type
        for pitch_num in range(3):  # 3 pitches of each type
            # Reset ball position for each pitch
            current_x = float(start_x)
            current_y = float(start_y)
            
            # Calculate trajectory points
            for frame in range(params['frames_per_pitch']):
                frame_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Calculate progress through pitch (0 to 1)
                progress = float(frame) / float(params['frames_per_pitch'])
                
                # Basic forward motion
                x_distance = float(width * 2/3 - start_x)  # Distance to travel
                current_x = float(start_x + (x_distance * progress))
                
                # Add vertical break
                # Use a quadratic function for vertical movement
                # Start with slight upward movement, then break downward
                vertical_offset = float(params['vertical_break']) * float(progress * progress - progress)
                
                # Add horizontal break
                # Use a cubic function for horizontal movement to create a late break
                horizontal_offset = float(params['horizontal_break']) * float(progress * progress * progress)
                
                # Calculate final position
                x = int(current_x + horizontal_offset)
                y = int(current_y + (height/3 * progress) + vertical_offset)
                
                # Keep ball within frame bounds
                x = max(ball_size, min(width - ball_size, x))
                y = max(ball_size, min(height - ball_size, y))
                
                # Draw white baseball
                cv2.circle(frame_img, (x, y), ball_size, (255, 255, 255), -1)
                
                # Add frame to video
                out.write(frame_img)
            
            # Add some blank frames between pitches
            for _ in range(10):  # 10 frames of blank space between pitches
                blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(blank_frame)

    out.release()

def test_pitch_classification():
    """Test pitch classification with different pitch types"""
    # Create test directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test video
    video_path = test_dir / "test_pitches.mp4"
    create_test_video(str(video_path))
    
    # Initialize video analyzer
    analyzer = VideoAnalyzer()
    
    # Analyze video
    report = analyzer.analyze_broadcast(str(video_path))
    results = asyncio.run(report.generate())
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Total frames processed: {results['video_info']['processed_frames']}")
    print(f"Total pitches detected: {len(results['pitches'])}")
    
    # Calculate pitch type distribution
    pitch_types = {}
    velocities = {}
    
    for pitch in results['pitches']:
        pitch_type = pitch['type']
        velocity = pitch['velocity']
        
        if pitch_type not in pitch_types:
            pitch_types[pitch_type] = 0
            velocities[pitch_type] = []
        
        pitch_types[pitch_type] += 1
        if velocity is not None:
            velocities[pitch_type].append(velocity)
    
    print("\nPitch Type Distribution:")
    for pitch_type, count in pitch_types.items():
        vel_list = velocities[pitch_type]
        if vel_list:
            avg_vel = sum(vel_list) / len(vel_list)
            max_vel = max(vel_list)
            min_vel = min(vel_list)
            print(f"{pitch_type}: {count} pitches")
            print(f"  Average velocity: {avg_vel:.1f} mph")
            print(f"  Max velocity: {max_vel:.1f} mph")
            print(f"  Min velocity: {min_vel:.1f} mph")
        else:
            print(f"{pitch_type}: {count} pitches (no velocity data)")
    
    # Clean up
    shutil.rmtree(test_dir)

def test_youtube_video(url: str):
    """Test pitch classification with a YouTube video"""
    # Create test directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Download YouTube video
    video_path = test_dir / "youtube_pitches.mp4"
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': str(video_path),
    }
    
    print(f"Downloading video from {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Initialize video analyzer
    analyzer = VideoAnalyzer()
    
    # Analyze video
    report = analyzer.analyze_broadcast(str(video_path))
    results = asyncio.run(report.generate())
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Total frames processed: {results['video_info']['processed_frames']}")
    print(f"Total pitches detected: {len(results['pitches'])}")
    
    # Calculate pitch type distribution
    pitch_types = {}
    velocities = {}
    
    for pitch in results['pitches']:
        pitch_type = pitch['type']
        velocity = pitch['velocity']
        
        if pitch_type not in pitch_types:
            pitch_types[pitch_type] = 0
            velocities[pitch_type] = []
        
        pitch_types[pitch_type] += 1
        if velocity is not None:
            velocities[pitch_type].append(velocity)
    
    print("\nPitch Type Distribution:")
    for pitch_type, count in pitch_types.items():
        vel_list = velocities[pitch_type]
        if vel_list:
            avg_vel = sum(vel_list) / len(vel_list)
            max_vel = max(vel_list)
            min_vel = min(vel_list)
            print(f"{pitch_type}: {count} pitches")
            print(f"  Average velocity: {avg_vel:.1f} mph")
            print(f"  Max velocity: {max_vel:.1f} mph")
            print(f"  Min velocity: {min_vel:.1f} mph")
        else:
            print(f"{pitch_type}: {count} pitches (no velocity data)")
    
    # Clean up
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If URL provided, test YouTube video
        test_youtube_video(sys.argv[1])
    else:
        # Otherwise run standard test
        test_pitch_classification() 