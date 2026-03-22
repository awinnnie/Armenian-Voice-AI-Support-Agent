import os
import tempfile
import numpy as np
from livekit.agents import stt, utils
from livekit.agents.types import APIConnectOptions
from livekit.agents.stt import SpeechEvent, SpeechEventType, SpeechData
from livekit.rtc import AudioFrame

# NeMo model loads once at import time
_asr_model = None

def _get_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr
        _asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="nvidia/stt_hy_fastconformer_hybrid_large_pc"
        )
        _asr_model.eval()
    return _asr_model


def _audio_buffer_to_wav_bytes(buffer) -> bytes:
    """Convert LiveKit AudioBuffer to WAV bytes."""
    if isinstance(buffer, AudioFrame):
        return buffer.to_wav_bytes()
    elif isinstance(buffer, list):
        # Merge multiple frames
        if not buffer:
            return b""
        all_data = bytearray()
        sr = buffer[0].sample_rate
        nc = buffer[0].num_channels
        total_samples = 0
        for frame in buffer:
            all_data.extend(frame.data)
            total_samples += frame.samples_per_channel
        merged = AudioFrame(
            data=bytes(all_data),
            sample_rate=sr,
            num_channels=nc,
            samples_per_channel=total_samples,
        )
        return merged.to_wav_bytes()
    return b""


class NeMoArmenianSTT(stt.STT):
    """Custom STT using NVIDIA NeMo FastConformer for Armenian."""

    def __init__(self):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))

    async def _recognize_impl(
        self,
        buffer,
        *,
        language="hy",
        conn_options=APIConnectOptions(),
    ) -> SpeechEvent:
        import asyncio

        wav_bytes = _audio_buffer_to_wav_bytes(buffer)
        if not wav_bytes:
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(language="hy", text="", start_time=0, end_time=0, confidence=0.0)],
            )

        # Write WAV to temp file for NeMo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            # Run NeMo inference in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._transcribe, tmp_path)
        finally:
            os.unlink(tmp_path)

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language="hy",
                    text=text,
                    start_time=0.0,
                    end_time=0.0,
                    confidence=0.9,
                )
            ],
        )

    def _transcribe(self, wav_path: str) -> str:
        model = _get_model()
        result = model.transcribe([wav_path], batch_size=1)
        # NeMo returns list of strings or list of Hypothesis objects
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], str):
                return result[0]
            elif hasattr(result[0], "text"):
                return result[0].text
            else:
                return str(result[0])
        return ""