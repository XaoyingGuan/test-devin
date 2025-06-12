# Multilingual Audio/Video Transcriber

A web application that transcribes audio and video files into subtitle files (.srt) with speaker diarization support. Built with FastAPI backend and React frontend.

## Features

- **Multilingual Support**: 13+ languages with auto-detection
- **File Format Support**: MP3, WAV, AAC, MP4, MOV, AVI, M4A, FLAC
- **Speaker Diarization**: Basic alternating speaker identification
- **Subtitle Generation**: Creates .srt files with speaker tags and timestamps
- **No API Keys Required**: Uses open-source Whisper model locally

## Architecture

- **Backend**: FastAPI with Hugging Face Transformers (Whisper tiny model)
- **Frontend**: React with TypeScript and Tailwind CSS
- **Audio Processing**: ffmpeg for video-to-audio extraction
- **Model**: OpenAI Whisper tiny model for speech recognition

## Prerequisites

- Python 3.12+
- Node.js 18+
- Poetry (for Python dependency management)
- npm/pnpm (for frontend dependencies)
- ffmpeg (for audio/video processing)

## Installation & Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd transcriber-backend
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Start the FastAPI development server:
```bash
poetry run fastapi dev app/main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd transcriber-frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

1. **Start both servers** (backend on port 8000, frontend on port 5173)
2. **Open your browser** to `http://localhost:5173`
3. **Upload an audio/video file** using drag-and-drop or file browser
4. **Select language** (optional - auto-detect works well)
5. **Click "Upload File"** then **"Start Transcription"**
6. **Download the generated .srt file** with speaker tags

## Supported Languages

- Auto-detect
- English, Spanish, French, German, Italian, Portuguese
- Russian, Chinese, Japanese, Korean, Arabic, Hindi, Dutch

## Technical Details

### Backend Endpoints

- `GET /healthz` - Health check
- `POST /upload` - Upload audio/video file
- `POST /transcribe/{file_id}` - Start transcription
- `GET /download/{filename}` - Download subtitle file

### Model Information

- **Model**: OpenAI Whisper tiny (~39MB)
- **Device**: CPU-only (CUDA if available)
- **Memory Usage**: ~200MB during transcription
- **Processing**: Lazy loading - model loads on first transcription request

### Speaker Diarization

The application implements basic speaker diarization by:
1. Splitting transcription into sentences
2. Alternating speaker assignments (Speaker 1, Speaker 2)
3. Distributing timestamps evenly across segments

## File Structure

```
multilingual-transcriber/
├── transcriber-backend/          # FastAPI backend
│   ├── app/
│   │   └── main.py              # Main application file
│   ├── pyproject.toml           # Python dependencies
│   └── uploads/                 # Uploaded files (created at runtime)
├── transcriber-frontend/         # React frontend
│   ├── src/
│   │   ├── App.tsx              # Main React component
│   │   └── components/ui/       # UI components
│   ├── package.json             # Node.js dependencies
│   └── .env                     # Environment variables
└── README.md                    # This file
```

## Development Notes

- The Whisper model downloads automatically on first use (~39MB)
- Uploaded files are stored temporarily in `uploads/` directory
- Generated subtitle files are saved in `outputs/` directory
- The application uses CPU-only PyTorch for broader compatibility

## Troubleshooting

### Backend Issues
- Ensure Poetry is installed: `pip install poetry`
- Check Python version: `python --version` (should be 3.12+)
- Verify ffmpeg installation: `ffmpeg -version`

### Frontend Issues
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 18+)
- Ensure backend is running on port 8000

### Model Loading Issues
- First transcription may take longer as the model downloads
- Ensure stable internet connection for initial model download
- Check available disk space (model requires ~100MB)

## Performance

- **Startup Time**: ~2-3 seconds (without model loading)
- **First Transcription**: ~30-60 seconds (includes model download)
- **Subsequent Transcriptions**: ~10-30 seconds depending on file length
- **Memory Usage**: ~200MB during active transcription

## License

This project uses open-source components:
- FastAPI (MIT License)
- React (MIT License)
- Hugging Face Transformers (Apache 2.0)
- OpenAI Whisper (MIT License)
