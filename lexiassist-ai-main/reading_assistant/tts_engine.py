import mimetypes
import os
import struct
import time
from typing import Literal, Optional, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
import tempfile
from pathlib import Path


class TTSGenerator:
    """Handles text-to-speech generation for the reading assistant."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the TTS generator with Gemini API."""
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY")
        )
        self.model = "gemini-2.5-flash-preview-tts"
        
        # Available voices for customization
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
        Generate audio from text using Gemini TTS.
        
        Args:
            text: The text to convert to speech
            voice: Voice name (Zephyr, Puck, Athena, Aria, Nova)
            speaker_label: Label for the speaker (e.g., "Narrator", "Assistant")
            output_file: Path to save audio file (if None, uses temp file)
            temperature: Controls randomness (0.0-1.0)
        
        Returns:
            Dictionary with audio data and metadata
        """
        
        # Validate voice selection
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not found. Available: {list(self.available_voices.keys())}")
        
        # Prepare content for TTS
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=text),
                ],
            ),
        ]
        
        # Configure speech generation
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice
                                )
                            ),
                        ),
            )
        
        # Generate audio stream with retry logic
        audio_data = None
        audio_mime_type = None
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if chunk.parts is None:
                        continue
                        
                    if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
                        audio_data = chunk.parts[0].inline_data.data
                        audio_mime_type = chunk.parts[0].inline_data.mime_type
                        break  # Only process first audio chunk
                    else:
                        # Handle any text responses (if any)
                        print(chunk.text)
                
                if audio_data is not None:
                    break  # Success, exit retry loop
                    
            except ServerError as e:
                last_error = e
                print(f"⚠️ Gemini API server error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            except ClientError as e:
                last_error = e
                print(f"⚠️ Gemini API client error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
        
        if audio_data is None:
            if last_error:
                raise RuntimeError(f"No audio data generated after {max_retries} attempts. Last error: {last_error}")
            raise RuntimeError("No audio data generated")
        
        # Determine output file path
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(
                suffix=".wav", 
                delete=False
            ).name
        
        # Convert and save audio
        output_path = self._save_audio_file(audio_data, audio_mime_type, output_file)
        
        return {
            "audio_path": output_path,
            "audio_data": audio_data,
            "mime_type": audio_mime_type,
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
        Generate multi-speaker audio with different voices for different speakers.
        
        Args:
            text_segments: List of text segments with speaker labels
            voice_assignments: List of speaker voice configurations
            output_file: Path to save audio file
            temperature: Controls randomness
        
        Returns:
            Dictionary with audio data and metadata
        """
        
        # Build multi-speaker content
        content_parts = []
        for segment in text_segments:
            content_parts.append(
                types.Part.from_text(
                    text=f"{segment['speaker']}: {segment['text']}"
                )
            )
        
        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]
        
        # Configure multi-speaker voices
        speaker_configs = []
        for assignment in voice_assignments:
            speaker_configs.append(
                types.SpeakerVoiceConfig(
                    speaker=assignment["speaker"],
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=assignment["voice"]
                        )
                    ),
                )
            )
        
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_configs
                ),
            ),
        )
        
        # Generate audio
        audio_data = None
        audio_mime_type = None
        
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.parts is None:
                continue
                
            if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
                audio_data = chunk.parts[0].inline_data.data
                audio_mime_type = chunk.parts[0].inline_data.mime_type
                break
        
        if audio_data is None:
            raise RuntimeError("No audio data generated")
        
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        
        output_path = self._save_audio_file(audio_data, audio_mime_type, output_file)
        
        return {
            "audio_path": output_path,
            "audio_data": audio_data,
            "mime_type": audio_mime_type,
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