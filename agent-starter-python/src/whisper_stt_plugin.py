from __future__ import annotations
import os
os.environ["TORCH_USE_NNPACK"] = "0"
import asyncio
import logging
import weakref
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

os.environ["PYTORCH_ENABLE_NNPACK!"] = "0"

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

logger = logging.getLogger(__name__)


@dataclass
class STTOptions:
    sample_rate: int
    buffer_size_seconds: float
    model_name: str
    device: str
    compute_type: str
    beam_size: int
    language: str | None
    vad_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        model_name: Literal["tiny", "base", "small", "medium", "large-v2", "large-v3"] = "base",
        device: Literal["cpu", "cuda"] = "cpu",
        compute_type: str = "int8",
        beam_size: int = 3,
        language: str | None = None,
        sample_rate: int = 16000,
        buffer_size_seconds: float = 0.05,
        vad_threshold: float = 0.40,
        min_speech_duration_ms: int = 100,
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 30,
    ) -> None:

        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=False),
        )

        if sample_rate != 16000:
            raise ValueError("Sample rate must be 16000 Hz for Silero VAD")

        logger.info(f"Loading Whisper model ({model_name}) on {device}...")

        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

        # Warm up
        logger.info("Warming up Whisper model...")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            list(self._model.transcribe(dummy_audio, beam_size=1, language=language))
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

        logger.info("Whisper model loaded and warmed up")

        # Load Silero VAD
        logger.info("Loading Silero VAD...")
        try:
            self._vad_model, vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._get_speech_timestamps = vad_utils[0]
            logger.info("Silero VAD loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            logger.info("Falling back to energy-based VAD")
            self._vad_model = None
            self._get_speech_timestamps = None

        self._opts = STTOptions(
            sample_rate=sample_rate,
            buffer_size_seconds=buffer_size_seconds,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            language=language,
            vad_threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        self._streams = weakref.WeakSet()  #keeps track of streams for later updates. WeakSet means streams can be garbage collected.

    @property
    def model(self) -> str:
        return f"faster-whisper-{self._opts.model_name}"

    @property
    def provider(self) -> str:
        return "Faster-Whisper"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:

        lang = language if is_given(language) else self._opts.language
        #converts bytes into numpy
        audio_int16 = np.frombuffer(buffer.data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        if audio_float32.size < self._opts.sample_rate:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=lang or "en", text="")],
            )

        def transcribe():
            segments, info = self._model.transcribe(
                audio=audio_float32,
                beam_size=self._opts.beam_size,
                language=lang,
                temperature=0.0,
                no_speech_threshold=0.7,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_threshold=0.2,
                suppress_blank=True,
                suppress_non_speech_tokens=True,
            )
            return segments, info

        #This runs CPU-bound WHISPER for transcription on a threadpool, preventing blocking of the asyncio loop.
        segments, info = await asyncio.get_event_loop().run_in_executor(None, transcribe)

        text = " ".join(seg.text.strip() for seg in segments if seg.text)
        detected_lang = getattr(info, "language", lang or "en")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=detected_lang, text=text)],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        lang = language if is_given(language) else self._opts.language

        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=self._opts,
            model=self._model,
            vad_model=self._vad_model,
            get_speech_timestamps=self._get_speech_timestamps,
            language=lang,
        )

        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        beam_size: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:

        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(beam_size):
            self._opts.beam_size = beam_size
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms

        for stream in self._streams:
            stream.update_options(
                buffer_size_seconds=buffer_size_seconds,
                beam_size=beam_size,
                vad_threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
            )


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        model,
        vad_model,
        get_speech_timestamps,
        language: str | None,
    ) -> None:

        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        #Key attributes for speech stream 
        self._opts = opts
        self._model = model
        self._vad_model = vad_model
        self._get_speech_timestamps = get_speech_timestamps

        self._language = language
        self._speech_duration = 0.0

        self._reconnect_event = asyncio.Event()

        self._audio_buffer = []
        self._is_recording = False
        self._silence_samples = 0

        self._min_silence_samples = int(
            opts.sample_rate * opts.min_silence_duration_ms / 1000
        )

        self._vad_chunk_size = 512
        self._vad_buffer = bytearray()

    def update_options(
        self,
        *,
        buffer_size_seconds: NotGivenOr[float] = NOT_GIVEN,
        beam_size: NotGivenOr[int] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:

        if is_given(buffer_size_seconds):
            self._opts.buffer_size_seconds = buffer_size_seconds
        if is_given(beam_size):
            self._opts.beam_size = beam_size
        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold
        if is_given(min_speech_duration_ms):
            self._opts.min_speech_duration_ms = min_speech_duration_ms
        if is_given(min_silence_duration_ms):
            self._opts.min_silence_duration_ms = min_silence_duration_ms
            self._min_silence_samples = int(
                self._opts.sample_rate * min_silence_duration_ms / 1000
            )

        self._reconnect_event.set()

    async def _run(self) -> None:
        #samples_per_buffer calculation
        samples_per_buffer = (
            self._opts.sample_rate // round(1 / self._opts.buffer_size_seconds)
        )

        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=samples_per_buffer,
        )

        try:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    if self._audio_buffer:
                        await self._process_accumulated_audio()
                        self._audio_buffer.clear()
                        self._is_recording = False
                        self._silence_samples = 0
                        self._vad_buffer = bytearray()
                    continue

                frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    self._speech_duration += frame.duration

                    audio_int16 = np.frombuffer(frame.data.tobytes(), dtype=np.int16)
                    self._vad_buffer.extend(audio_int16.tobytes())

                    # Process VAD in chunks of 512 samples
                    while len(self._vad_buffer) >= self._vad_chunk_size * 2:
                        chunk = bytes(self._vad_buffer[: self._vad_chunk_size * 2])
                        self._vad_buffer = self._vad_buffer[self._vad_chunk_size * 2 :]

                        samples = np.frombuffer(chunk, dtype=np.int16)
                        audio_f32 = samples.astype(np.float32) / 32768.0

                        speech_prob = self._detect_speech(audio_f32)

                        if speech_prob > self._opts.vad_threshold:
                            self._is_recording = True
                            self._silence_samples = 0
                            self._audio_buffer.append(samples)
                            self._limit_buffer()
                        else:
                            if self._is_recording:
                                self._silence_samples += len(samples)

                                if self._silence_samples <= self._opts.sample_rate * 0.1:
                                    self._audio_buffer.append(samples)
                                    self._limit_buffer()

                                if self._silence_samples >= self._min_silence_samples:
                                    await self._process_accumulated_audio()
                                    self._audio_buffer = []
                                    self._is_recording = False
                                    self._silence_samples = 0

        except Exception as e:
            logger.exception(f"SpeechStream._run error: {e}")

        finally:
            if self._audio_buffer:
                await self._process_accumulated_audio()


    def _limit_buffer(self):
        max_len = self._opts.sample_rate * 10   # 10 seconds
        if sum(len(a) for a in self._audio_buffer) > max_len:
            cat = np.concatenate(self._audio_buffer)
            cat = cat[-max_len:]  # keep last 10 sec
            self._audio_buffer = [cat]


    def _detect_speech(self, audio_chunk: np.ndarray) -> float:
        try:
            if self._vad_model is None:
                energy = np.sqrt(np.mean(audio_chunk ** 2))
                return min(1.0, energy * 10)

            if len(audio_chunk) != 512:
                logger.warning(f"Invalid VAD chunk size: {len(audio_chunk)}")
                return 0.0

            audio_tensor = torch.from_numpy(audio_chunk)

            with torch.no_grad():
                prob = self._vad_model(audio_tensor, self._opts.sample_rate).item()

            return prob

        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return 0.0

    #prepare audio and call transcription
    async def _process_accumulated_audio(self):
        if not self._audio_buffer:
            return

        audio_int16 = np.concatenate(self._audio_buffer)
        self._vad_buffer = bytearray()

        max_len = 16000 * 10  #keep last 10 seconds if audio longer than 10s.
        if len(audio_int16) > max_len:
            audio_int16 = audio_int16[-max_len:]

        min_samples = int(self._opts.sample_rate * 0.25)
        if audio_int16.size < min_samples:
            logger.debug(f"Skipping too short audio: {audio_int16.size} samples")
            return

        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        await self._transcribe_and_send(audio_f32)

    async def _transcribe_and_send(self, audio_float32: np.ndarray):
        try:
            def transcribe():
                segments, info = self._model.transcribe(
                    audio=audio_float32,
                    beam_size=self._opts.beam_size,
                    language=self._language,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                    vad_filter=True,
                )
                return segments, info

            segments, info = await asyncio.get_event_loop().run_in_executor(
                None, transcribe
            )

            text = " ".join(seg.text.strip() for seg in segments if seg.text)
            detected_lang = getattr(info, "language", self._language or "en")

            if text:
                logger.debug(f"Transcription: {text}")

                final_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=detected_lang, text=text)],
                )
                self._event_ch.send_nowait(final_event)

                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )

                if self._speech_duration > 0.0:
                    usage_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        alternatives=[],
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=self._speech_duration
                        ),
                    )
                    self._event_ch.send_nowait(usage_event)
                    self._speech_duration = 0.0

            else:
                logger.debug("Empty transcription result")

        except Exception as e:
            logger.exception(f"Transcription error: {e}")
