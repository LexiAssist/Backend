import io
import mimetypes
import os
import struct
from typing import Literal, Optional, Dict, Any
from gtts import gTTS
from pathlib import Path


class TTSGenerator:
    """Handles text-to-speech generation for the reading assistant using gTTS."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the TTS generator."""
        # gTTS does not require an API key for basic usage
        self.available_voices = {
            "Zephyr": "Warm, friendly, and approachable",
            "Puck": "Energetic, playful, slightly whimsical",
            "Athena": "Clear, articulate, academic tone",
            "Aria": "Soft, melodic, soothing",
            "Nova": "Confident, authoritative, professional"
        }
    
    def generate_audio(
        self,
        text:str,
        voice: str = "Zephyr",
        speaker_label: str = "Reader",
        output_file: Optional[str] = None,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate audio from text using gTTS (in-memory, no disk I/O).
        """
        
        # Validate voice selection (kept for API compatibility)
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not found. Available: {list(self.available_voices.keys())}")
        
        # gTTS generates MP3 audio directly into an in-memory buffer
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_data = buf.read()
        
        return {
            "audio_path": "memory://generated.mp3",
            "audio_data": audio_data,
            "mime_type": "audio/mpeg",
            "voice": voice,
            "speaker": speaker_label,
            "text_length": len(text)
        }
    
    def generate_multi_speaker_audio(
        self,
        text_segments: list,
        voice_assignments: list,
        output_file: Optional[str] = None,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate multi-speaker audio using gTTS (in-memory, no disk I/O).
        gTTS does not support per-speaker voices, so we flatten to a single narration.
        """
        
        # Flatten segments into one text block
        full_text = "\n".join(
            f"{segment['speaker']}: {segment['text']}" for segment in text_segments
        )
        
        tts = gTTS(text=full_text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_data = buf.read()
        
        return {
            "audio_path": "memory://generated.mp3",
            "audio_data": audio_data,
            "mime_type": "audio/mpeg",
            "voice_assignments": voice_assignments,
            "text_segments": text_segments
        }
    
    def _save_audio_file(self, audio_data: bytes, mime_type: str, output_path: str) -> str:
        """Save audio data to file, converting to WAV if needed."""
        
        file_extension = mimetypes.guess_extension(mime_type)
        
        if file_extension is None:
            # Convert to WAV if format not recognized
            file_extension = ".wav"
            audio_data = self._convert_to_wav(audio_data, mime_type)
        
        # Ensure correct extension
        if not output_path.endswith(file_extension):
            output_path = output_path.replace(Path(output_path).suffix, file_extension)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        # Save file
        with open(output_path, "wb") as f:
            f.write(audio_data)
        
        print(f"✅ Audio saved to: {output_path}")
        return output_path
    
    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Convert raw audio to WAV format."""
        
        parameters = self._parse_audio_mime_type(mime_type)
        bits_per_sample = parameters["bits_per_sample"]
        sample_rate = parameters["rate"]
        num_channels = 1
        data_size = len(audio_data)
        bytes_per_sample = bits_per_sample // 8
        block_align = num_channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        chunk_size = 36 + data_size
        
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",          # ChunkID
            chunk_size,       # ChunkSize
            b"WAVE",          # Format
            b"fmt ",          # Subchunk1ID
            16,               # Subchunk1Size
            1,                # AudioFormat (PCM)
            num_channels,     # NumChannels
            sample_rate,      # SampleRate
            byte_rate,        # ByteRate
            block_align,      # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",          # Subchunk2ID
            data_size         # Subchunk2Size
        )
        return header + audio_data
    
    def _parse_audio_mime_type(self, mime_type: str) -> dict:
        """Parse audio MIME type for parameters."""
        bits_per_sample = 16
        rate = 24000
        
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass
        
        return {"bits_per_sample": bits_per_sample, "rate": rate}
    
    def list_voices(self) -> Dict[str, str]:
        """Return available voices with descriptions."""
        return self.available_voices