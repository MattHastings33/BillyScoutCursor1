import os
import sys
import json
import asyncio
import numpy as np
import cv2

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.video_analyzer import VideoAnalyzer

def create_test_video(video_path):
    """Create a test video with a moving baseball."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Initialize video writer with lower resolution and quality
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480), True)

        # Create frames for a moving baseball
        for i in range(60):  # 2 seconds at 30 fps
            # Create a dark gray background
            frame = np.full((480, 640, 3), 50, dtype=np.uint8)
            
            # Calculate ball position
            x = 50 + i * 10  # Move right
            y = 240 + int(50 * np.sin(i * 0.2))  # Oscillate vertically
            
            # Draw baseball
            cv2.circle(frame, (x, y), 10, (255, 255, 255), -1)  # White circle
            cv2.circle(frame, (x, y), 10, (0, 0, 0), 1)  # Black outline
            
            # Draw red stitching
            cv2.line(frame, (x-5, y), (x+5, y), (0, 0, 255), 1)
            cv2.line(frame, (x, y-5), (x, y+5), (0, 0, 255), 1)
            
            out.write(frame)

        out.release()
        print(f"Created test video: {video_path}")
        
    except Exception as e:
        print(f"Error creating test video: {str(e)}")
        raise

async def analyze_kenyon_pitchers():
    """Test function to analyze baseball pitches."""
    try:
        # Create test data directory if it doesn't exist
        test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Create test video if it doesn't exist
        video_path = os.path.join(test_data_dir, "test_pitch.mp4")
        if not os.path.exists(video_path):
            print("Creating test video...")
            create_test_video(video_path)
        
        print("\nAnalyzing baseball video...")
        print("=" * 50)
        print(f"Video path: {video_path}")
        
        # Initialize video analyzer
        analyzer = VideoAnalyzer()
        
        # Analyze video
        report = analyzer.analyze_broadcast(video_path)
        
        if report:
            # Save report to file
            report_path = os.path.join(test_data_dir, "baseball_analysis_report.json")
            report.save(report_path)
            print(f"\nDetailed report saved to {os.path.basename(report_path)}")
        else:
            print("Failed to analyze video")
            
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")

def test_analyze_kenyon_pitchers():
    """Test the video analyzer with a synthetic baseball video."""
    try:
        # Create test video
        video_path = os.path.join(os.path.dirname(__file__), "test_data", "test_pitch.mp4")
        create_test_video(video_path)

        # Initialize video analyzer
        detector = BaseballDetector()
        pose_estimator = PoseEstimator()
        pitch_classifier = PitchClassifier()
        clip_tagger = ClipTagger()
        heatmap_generator = HeatmapGenerator()

        analyzer = VideoAnalyzer(
            detector=detector,
            pose_estimator=pose_estimator,
            pitch_classifier=pitch_classifier,
            clip_tagger=clip_tagger,
            heatmap_generator=heatmap_generator
        )

        print("\nAnalyzing baseball video...")
        print("=" * 50)
        print(f"Video path: {video_path}")

        # Analyze video
        report = analyzer.analyze_broadcast(video_path)
        
        # Save report to JSON file
        report_path = os.path.join(os.path.dirname(__file__), "test_data", "baseball_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        print("\nAnalysis complete!")
        print(f"Report saved to: {report_path}")
        print("\nReport contents:")
        print(json.dumps(report, indent=4))
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        raise
    finally:
        # Cleanup
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Cleaned up test video: {video_path}")
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")

if __name__ == "__main__":
    test_analyze_kenyon_pitchers() 