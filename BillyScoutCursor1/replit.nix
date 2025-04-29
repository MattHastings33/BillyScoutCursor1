{ pkgs }: {
    deps = [
        pkgs.python39
        pkgs.python39Packages.pip
        pkgs.python39Packages.uvicorn
        pkgs.python39Packages.fastapi
        pkgs.python39Packages.opencv
        pkgs.python39Packages.numpy
        pkgs.python39Packages.pandas
        pkgs.python39Packages.psycopg2
        pkgs.python39Packages.boto3
        pkgs.python39Packages.ultralytics
        pkgs.python39Packages.mediapipe
        pkgs.python39Packages.python-jose
        pkgs.python39Packages.passlib
        pkgs.python39Packages.python-dotenv
        pkgs.python39Packages.yt-dlp
        pkgs.python39Packages.beautifulsoup4
        pkgs.python39Packages.requests
    ];
} 