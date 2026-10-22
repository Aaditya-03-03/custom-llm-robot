"""
Voice STT & Audio Processing API Router for IOFT Humanoid Robot.
Provides dedicated, robust server-side speech recognition for Bluetooth headsets and local microphones.
"""

import io
import logging
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
from pydantic import BaseModel
import speech_recognition as sr

logger = logging.getLogger("custom_llm_robot.api.voice")
router = APIRouter()


class STTResponse(BaseModel):
    success: bool
    transcript: str
    language: str


@router.post(
    "/voice/stt",
    response_model=STTResponse,
    summary="Transcribe uploaded audio via dedicated Python STT engine",
    description="Transcribes 16-bit PCM WAV audio from client microphone using SpeechRecognition library.",
)
async def voice_stt_endpoint(
    file: UploadFile = File(...),
    language: str = Query("en-IN", description="Language code, e.g. en-IN, en-US, en-GB"),
):
    try:
        content = await file.read()
        if not content or len(content) < 44:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty or invalid audio payload.",
            )

        audio_buf = io.BytesIO(content)
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_buf) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            audio = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio, language=language)
            logger.info(f"Server-side STT successful: '{transcript}' [lang={language}]")
            return STTResponse(
                success=True,
                transcript=transcript.strip(),
                language=language,
            )
        except sr.UnknownValueError:
            logger.info("Server-side STT: Audio heard but no speech recognized (silence or unintelligible).")
            return STTResponse(
                success=False,
                transcript="",
                language=language,
            )
        except sr.RequestError as req_err:
            logger.error(f"STT Service request failed: {req_err}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Speech Recognition service error: {req_err}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected STT error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio decoding or STT failed: {e}",
        )
