from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Header, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
import os
import uvicorn
import uuid
import tempfile
from datetime import datetime


def verify_internal_key(request: Request, x_internal_key: str = Header(None)):
    if request.url.path in ("/", "/health"):
        return
    expected = os.getenv("INTERNAL_API_KEY", "dev-internal-key")
    if not x_internal_key or x_internal_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing internal key")


# Initialize FastAPI
app = FastAPI(
    title="LexiAssist Audio Service",
    description="Speech-to-Text using SpeechRecognition + pydub (supports ALL formats)",
    version="2.1.0",
    dependencies=[Depends(verify_internal_key)],
)

# Create recognizer
recognizer = sr.Recognizer()

# Create temp directory
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".webm", ".flac", ".aac"}

# Pydantic models
class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0

class SpeechToTextResponse(BaseModel):
    text: str
    confidence: float
    language: str
    original_format: str

class TextToSpeechResponse(BaseModel):
    audio_file_url: str
    message: str

# Health check
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "audio",
        "port": 5004,
        "version": "2.1.0",
        "engine": "SpeechRecognition + pydub",
        "supported_formats": sorted(list(SUPPORTED_FORMATS)),
        "max_file_size_mb": MAX_FILE_SIZE_MB
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "SpeechRecognition + pydub + gTTS",
        "features": {
            "speech_to_text": "available (all formats)",
            "text_to_speech": "available (gTTS)"
        }
    }

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """
    Convert ANY audio format to WAV using pydub.
    Requires ffmpeg to be installed for non-WAV formats.
    """
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False

def cleanup_files(*paths):
    """Safely remove temporary files."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

@app.post("/api/v1/ai/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    audio: UploadFile = File(..., description="Audio file (MP3, WAV, M4A, OGG, etc.)"),
    language: str = Form("en-US", description="Language code (en-US, es-ES, fr-FR, etc.)")
):
    """
    Convert uploaded audio file to text.
    Supports: MP3, WAV, M4A, OGG, MP4, WEBM, FLAC, AAC
    Max file size: 50MB
    """
    input_path = None
    wav_path = None

    try:
        # Validate file extension
        file_ext = os.path.splitext(audio.filename or "")[1].lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{file_ext}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        # Read and validate file size
        content = await audio.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(content) / 1024 / 1024:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB"
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

        # Save uploaded file using full UUID to prevent collision
        temp_id = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"input_{temp_id}{file_ext}")
        wav_path = os.path.join(TEMP_DIR, f"converted_{temp_id}.wav")

        with open(input_path, "wb") as f:
            f.write(content)

        print(f"\n🎤 Processing audio: {audio.filename}")
        print(f"   Format: {file_ext}")
        print(f"   Size: {len(content)} bytes")
        print(f"   Language: {language}")

        # Convert to WAV if not already WAV
        if file_ext == '.wav':
            wav_path = input_path
            print("   Already WAV format")
        else:
            print(f"   Converting {file_ext} to WAV...")
            success = convert_to_wav(input_path, wav_path)
            if not success:
                raise HTTPException(status_code=400, detail=f"Could not convert {file_ext} to WAV. Is ffmpeg installed?")
            print("   ✅ Conversion successful")

        # Process with speech_recognition
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        # Use Google Speech Recognition (free tier)
        text = recognizer.recognize_google(audio_data, language=language)

        print(f"   ✅ Transcription: {text[:100]}...")

        return SpeechToTextResponse(
            text=text,
            # Note: Google free STT API does not expose confidence scores.
            # This is a fixed estimate. Use Google Cloud Speech-to-Text for real confidence.
            confidence=0.85,
            language=language,
            original_format=file_ext
        )

    except HTTPException:
        raise
    except sr.UnknownValueError:
        raise HTTPException(
            status_code=400,
            detail="Could not understand audio. Try speaking more clearly or check audio quality."
        )
    except sr.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Google Speech API unavailable: {str(e)}. Check your internet connection."
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    finally:
        # Always clean up temp files
        if wav_path != input_path:
            cleanup_files(input_path, wav_path)
        else:
            cleanup_files(input_path)

class TextToSpeechJSONRequest(BaseModel):
    text: str
    language: str = "en"
    slow: bool = False

@app.post("/api/v1/ai/text-to-speech")
async def text_to_speech(request: TextToSpeechJSONRequest):
    """
    Convert text to speech using Google Text-to-Speech (gTTS).
    Returns MP3 audio file.
    """
    text = request.text
    language = request.language
    slow = request.slow
    output_path = None
    
    try:
        # Validate text length
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long. Maximum 5000 characters.")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
        # Generate unique filename
        temp_id = str(uuid.uuid4())[:8]
        output_path = os.path.join(TEMP_DIR, f"tts_{temp_id}.mp3")
        
        print(f"\n🔊 Generating TTS:")
        print(f"   Text length: {len(text)} chars")
        print(f"   Language: {language}")
        print(f"   Slow: {slow}")
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=language, slow=slow)
        tts.save(output_path)
        
        print(f"   ✅ TTS generated: {output_path}")
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename=f"speech_{temp_id}.mp3",
            background=None
        )
        
    except Exception as e:
        # Cleanup on error
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        print(f"❌ TTS Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# Keep old endpoint for backward compatibility
@app.post("/text-to-speech")
async def text_to_speech_legacy(request: TextToSpeechRequest):
    """
    Legacy endpoint - redirects to new implementation.
    """
    return await text_to_speech(
        TextToSpeechJSONRequest(
            text=request.text,
            language=request.voice_id[:2] if request.voice_id else "en",
            slow=request.speed < 1.0
        )
    )

@app.get("/api/v1/ai/languages")
async def list_languages():
    """
    List supported languages for speech recognition.
    """
    languages = {
        "en-US": "English (US)",
        "en-GB": "English (UK)",
        "es-ES": "Spanish",
        "fr-FR": "French",
        "de-DE": "German",
        "it-IT": "Italian",
        "pt-BR": "Portuguese (Brazil)",
        "ja-JP": "Japanese",
        "zh-CN": "Chinese (Simplified)",
        "ko-KR": "Korean",
        "ar-SA": "Arabic",
        "hi-IN": "Hindi",
        "ru-RU": "Russian",
        "auto": "Auto-detect"
    }
    return {"supported_languages": languages}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5004, reload=True)
