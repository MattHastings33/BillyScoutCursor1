# BillyScout

BillyScout is an AI-powered baseball scouting platform that analyzes game videos to generate automated scouting reports. The platform provides detailed insights about pitchers, batters, and game strategies.

## Features

- **Video Analysis**: Upload game videos or provide broadcast URLs for analysis
- **Pitcher Tracking**: 
  - Identify and track multiple pitchers in a game
  - Analyze pitch types, velocities, and tendencies
  - Count-based pitch selection analysis
- **Batter Analysis**:
  - Swing mechanics analysis
  - Contact point tracking
  - Performance metrics
- **Heatmap Visualization**:
  - Pitch location heatmaps
  - Contact point visualization
- **Comprehensive Reports**:
  - Individual pitcher reports
  - Batter performance analysis
  - Game strategy insights

## Tech Stack

- **Backend**:
  - FastAPI
  - OpenCV
  - YOLOv8
  - MediaPipe
  - PostgreSQL
  - AWS S3

- **Frontend**:
  - React.js
  - Tailwind CSS
  - Chart.js
  - React Player

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- PostgreSQL
- AWS Account (for S3)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/billyscout.git
cd billyscout
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend/billyscout-ui
npm install
```

4. Set up environment variables:
```bash
# Backend (.env)
DATABASE_URL=postgresql://user:password@localhost:5432/billyscout
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_BUCKET_NAME=your_bucket_name

# Frontend (.env)
REACT_APP_API_URL=http://localhost:8000
```

5. Start the development servers:
```bash
# Backend
cd backend
uvicorn main:app --reload

# Frontend
cd frontend/billyscout-ui
npm start
```

## Usage

1. Navigate to `http://localhost:3000`
2. Upload a game video or provide a broadcast URL
3. Wait for the analysis to complete
4. View the generated scouting report

## Supported Video Sources

- Local video files (MP4)
- Broadcast URLs from:
  - North Coast Network
  - Hudl
  - YouTube
  - Vimeo

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 for object detection
- MediaPipe for pose estimation
- Chart.js for data visualization