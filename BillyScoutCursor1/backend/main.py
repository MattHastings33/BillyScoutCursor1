from fastapi import FastAPI, UploadFile, File, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os
from typing import List, Optional, Dict
import json
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError
import uuid
import asyncio
import yt_dlp
from ai.report import ScoutingReport
from ai.detector import ObjectDetector
from ai.pose import PoseEstimator
from ai.classifier import PitchClassifier
from ai.clip_tagger import ClipTagger
from ai.heatmap import HeatmapGenerator
from models.player_model import PlayerModel
from scrapers.box_score import BoxScoreScraper

app = FastAPI(title="BillyScout API", description="AI-powered baseball scouting platform")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store analysis progress
analysis_progress: Dict[str, Dict] = {}

@app.get("/")
async def root():
    return {"message": "Welcome to BillyScout API"}

@app.post("/upload")
async def upload_video(file: Optional[UploadFile] = File(None), video_url: Optional[str] = Body(None)):
    if not file and not video_url:
        raise HTTPException(status_code=400, detail="Either a file or video URL must be provided")
    
    try:
        if file:
            if not file.filename.endswith('.mp4'):
                raise HTTPException(status_code=400, detail="Only MP4 files are allowed")
            
            file_path = UPLOAD_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            return {"message": "File uploaded successfully", "filename": file.filename}
        
        elif video_url:
            # Generate a unique filename
            unique_id = str(uuid.uuid4())
            output_path = UPLOAD_DIR / f"{unique_id}.mp4"
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
            }
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Get box score data if it's a supported broadcast platform
            box_score_data = {}
            scraper = BoxScoreScraper()
            if any(platform in video_url for platform in scraper.supported_platforms.keys()):
                box_score_data = scraper.get_pitcher_info(video_url)
            
            return {
                "message": "Video URL processed successfully",
                "filename": output_path.name,
                "report_id": unique_id,
                "box_score_data": box_score_data,
                "platform": box_score_data.get('platform', 'unknown')
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_video_background(filename: str, report_id: str):
    """Background task to analyze video"""
    try:
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            analysis_progress[report_id]['status'] = 'error'
            analysis_progress[report_id]['error'] = "Video file not found"
            return

        # Initialize AI components
        detector = ObjectDetector()
        pose_estimator = PoseEstimator()
        pitch_classifier = PitchClassifier()
        clip_tagger = ClipTagger()
        heatmap_generator = HeatmapGenerator()
        
        # Process the video
        report = ScoutingReport(
            video_path=str(file_path),
            detector=detector,
            pose_estimator=pose_estimator,
            pitch_classifier=pitch_classifier,
            clip_tagger=clip_tagger,
            heatmap_generator=heatmap_generator
        )
        
        # Generate report
        report_data = await report.generate()
        
        # If we have box score data, integrate it with the report
        if hasattr(report, 'box_score_data') and report.box_score_data:
            report_data['pitcher_info'] = report.box_score_data
        
        # Store the final report
        analysis_progress[report_id]['status'] = 'completed'
        analysis_progress[report_id]['report'] = report_data
        
    except Exception as e:
        analysis_progress[report_id]['status'] = 'error'
        analysis_progress[report_id]['error'] = str(e)

@app.post("/analyze/{filename}")
async def start_analysis(filename: str, background_tasks: BackgroundTasks):
    """Start video analysis in the background"""
    report_id = str(uuid.uuid4())
    
    # Initialize progress tracking
    analysis_progress[report_id] = {
        'status': 'processing',
        'progress': 0,
        'filename': filename
    }
    
    # Start background task
    background_tasks.add_task(analyze_video_background, filename, report_id)
    
    return {
        "message": "Analysis started",
        "report_id": report_id
    }

@app.get("/analysis/{report_id}")
async def get_analysis_status(report_id: str):
    """Get the status of a video analysis"""
    if report_id not in analysis_progress:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_progress[report_id]

@app.get("/reports/{player_id}")
async def get_player_reports(player_id: str):
    # TODO: Implement player report retrieval
    return {"message": f"Reports for player {player_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 