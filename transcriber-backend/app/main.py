from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import aiofiles
import ffmpeg
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import tempfile
import uuid
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multilingual Transcriber API")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

processor = None
model = None
device = None

def load_whisper_model():
    """Load Whisper model lazily when needed"""
    global processor, model, device
    if processor is not None and model is not None:
        return
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model on device: {device}")
        
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.to(device)
        
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

def extract_audio_from_video(input_path: str, output_path: str):
    """Extract audio from video file using ffmpeg"""
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        raise HTTPException(status_code=500, detail="Failed to extract audio from video")

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> list:
    """Transcribe audio file using Whisper"""
    try:
        load_whisper_model()
        
        import torchaudio
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        audio = waveform.squeeze().numpy()
        
        if processor is None or model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        
        if language and language != "auto":
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        else:
            predicted_ids = model.generate(input_features)
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        segments = []
        if transcription and transcription[0]:
            text = transcription[0]
            sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            duration = len(audio) / 16000  # 16kHz sample rate
            
            if sentences:
                segment_duration = duration / len(sentences)
                
                for i, sentence in enumerate(sentences):
                    if sentence:
                        start_time = i * segment_duration
                        end_time = min((i + 1) * segment_duration, duration)
                        
                        speaker_id = (i % 2) + 1
                        
                        segments.append({
                            "start": start_time,
                            "end": end_time,
                            "text": sentence + '.',
                            "speaker": f"Speaker {speaker_id}"
                        })
            else:
                segments.append({
                    "start": 0.0,
                    "end": duration,
                    "text": text,
                    "speaker": "Speaker 1"
                })
        
        return segments
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

def generate_srt(segments: list) -> str:
    """Generate SRT subtitle format"""
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        speaker = segment["speaker"]
        text = segment["text"]
        
        srt_content += f"{i}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"[{speaker}]: {text}\n\n"
    
    return srt_content

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Initialize app without loading model"""
    logger.info("FastAPI app started - model will be loaded on first transcription request")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model_loaded": model is not None, "ready": True}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload audio or video file"""
    try:
        allowed_extensions = {'.mp3', '.wav', '.aac', '.mp4', '.mov', '.avi', '.m4a', '.flac'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        file_id = str(uuid.uuid4())
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join("uploads", filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "message": "File uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/transcribe/{file_id}")
async def transcribe_file(file_id: str, language: Optional[str] = Form(None)):
    """Transcribe uploaded file and generate subtitles"""
    try:
        upload_files = [f for f in os.listdir("uploads") if f.startswith(file_id)]
        if not upload_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        input_path = os.path.join("uploads", upload_files[0])
        file_extension = os.path.splitext(upload_files[0])[1].lower()
        
        audio_path = input_path
        if file_extension in {'.mp4', '.mov', '.avi'}:
            audio_path = os.path.join("uploads", f"{file_id}_audio.wav")
            extract_audio_from_video(input_path, audio_path)
        
        segments = transcribe_audio(audio_path, language)
        
        srt_content = generate_srt(segments)
        srt_filename = f"{file_id}.srt"
        srt_path = os.path.join("outputs", srt_filename)
        
        async with aiofiles.open(srt_path, 'w', encoding='utf-8') as f:
            await f.write(srt_content)
        
        if audio_path != input_path and os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            "file_id": file_id,
            "subtitle_file": srt_filename,
            "segments_count": len(segments),
            "message": "Transcription completed successfully"
        }
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe file: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated subtitle file"""
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/")
async def root():
    return {"message": "Multilingual Transcriber API", "status": "running"}
