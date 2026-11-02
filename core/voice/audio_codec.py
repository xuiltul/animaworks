from __future__ import annotations

# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0

"""NumPy-only PCM and G.711 μ-law conversion helpers for phone audio."""

import io
import struct
import wave

import numpy as np

TWILIO_SAMPLE_RATE = 8_000
VOICE_SAMPLE_RATE = 16_000
TWILIO_FRAME_MS = 20
MULAW_SAMPLES_PER_FRAME = TWILIO_SAMPLE_RATE * TWILIO_FRAME_MS // 1000


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 μ-law bytes to little-endian mono PCM16 samples."""
    encoded = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    if encoded.size == 0:
        return b""

    value = np.bitwise_xor(encoded, 0xFF)
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    magnitude = ((mantissa << 3) + 0x84) << exponent
    magnitude -= 0x84
    samples = np.where(sign != 0, -magnitude, magnitude)
    return samples.astype("<i2").tobytes()


def pcm16_to_mulaw(data: bytes) -> bytes:
    """Encode little-endian mono PCM16 samples as G.711 μ-law bytes."""
    if len(data) % 2:
        raise ValueError("PCM16 data must contain complete samples")
    samples = np.frombuffer(data, dtype="<i2").astype(np.int32)
    if samples.size == 0:
        return b""

    sign = np.where(samples < 0, 0x80, 0).astype(np.int32)
    magnitude = np.minimum(np.abs(samples), 32_635) + 0x84
    thresholds = np.array([0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF], dtype=np.int32)
    exponent = np.searchsorted(thresholds, magnitude, side="left").astype(np.int32)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = np.bitwise_not(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8).tobytes()


def resample_pcm16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample little-endian mono PCM16 while preserving duration."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if len(data) % 2:
        raise ValueError("PCM16 data must contain complete samples")
    samples = np.frombuffer(data, dtype="<i2")
    if samples.size == 0 or source_rate == target_rate:
        return bytes(data)

    target_count = int(round(samples.size * target_rate / source_rate))
    if target_count <= 0:
        return b""

    source_samples = samples.astype(np.float64)
    if source_rate > target_rate:
        # A short windowed-sinc low-pass filter prevents high-frequency speech
        # components from folding into the narrow telephone band on decimation.
        half_width = 31
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
        cutoff = target_rate / (2 * source_rate)
        kernel = 2 * cutoff * np.sinc(2 * cutoff * offsets) * np.hamming(offsets.size)
        kernel /= kernel.sum()
        padded = np.pad(source_samples, (half_width, half_width), mode="edge")
        source_samples = np.convolve(padded, kernel, mode="valid")

    positions = np.arange(target_count, dtype=np.float64) * (source_rate / target_rate)
    positions = np.minimum(positions, source_samples.size - 1)
    resampled = np.interp(positions, np.arange(source_samples.size, dtype=np.float64), source_samples)
    return np.rint(resampled).astype("<i2").tobytes()


def upsample_8k_to_16k(data: bytes) -> bytes:
    """Upsample Twilio's 8 kHz PCM16 mono audio for the voice pipeline."""
    return resample_pcm16(data, TWILIO_SAMPLE_RATE, VOICE_SAMPLE_RATE)


def downsample_16k_to_8k(data: bytes) -> bytes:
    """Downsample mono PCM16 audio from the voice rate to Twilio's 8 kHz rate."""
    return resample_pcm16(data, VOICE_SAMPLE_RATE, TWILIO_SAMPLE_RATE)


def _wav_pcm16(data: bytes) -> tuple[bytes, int]:
    """Read uncompressed PCM WAV data and normalize it to mono PCM16."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            raw = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error) as exc:
        raise ValueError("Twilio Media Streams requires PCM WAV audio") from exc

    if channels < 1 or sample_rate < 1:
        raise ValueError("WAV audio has invalid channel or sample-rate metadata")
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) << 8
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if octets.size % 3:
            raise ValueError("WAV audio contains an incomplete 24-bit sample")
        triples = octets.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        samples = ((values ^ 0x800000) - 0x800000) >> 8
    elif sample_width == 4:
        samples = (np.frombuffer(raw, dtype="<i4").astype(np.int64) >> 16).astype(np.int32)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if samples.size % channels:
        raise ValueError("WAV audio contains an incomplete multi-channel frame")
    if channels > 1:
        samples = np.rint(samples.reshape(-1, channels).mean(axis=1)).astype(np.int32)
    return np.clip(samples, -32_768, 32_767).astype("<i2").tobytes(), sample_rate


def split_mulaw_frames(data: bytes, *, frame_ms: int = TWILIO_FRAME_MS, pad_final: bool = True) -> list[bytes]:
    """Split 8 kHz μ-law audio into frames, padding the final frame by default."""
    if frame_ms <= 0 or TWILIO_SAMPLE_RATE * frame_ms % 1000:
        raise ValueError("frame_ms must produce a whole number of 8 kHz samples")
    frame_bytes = TWILIO_SAMPLE_RATE * frame_ms // 1000
    frames = [data[offset : offset + frame_bytes] for offset in range(0, len(data), frame_bytes)]
    if pad_final and frames and len(frames[-1]) < frame_bytes:
        frames[-1] += b"\xff" * (frame_bytes - len(frames[-1]))
    return frames


def wav_to_mulaw_frames(data: bytes, *, frame_ms: int = TWILIO_FRAME_MS) -> list[bytes]:
    """Convert a PCM WAV segment to 8 kHz μ-law and split it into Twilio frames."""
    pcm16, sample_rate = _wav_pcm16(data)
    pcm8 = resample_pcm16(pcm16, sample_rate, TWILIO_SAMPLE_RATE)
    return split_mulaw_frames(pcm16_to_mulaw(pcm8), frame_ms=frame_ms)


def wav_to_mulaw_wav(data: bytes) -> bytes:
    """Convert a PCM WAV to an 8 kHz mono μ-law WAV, which is all a phone line carries."""
    pcm16, sample_rate = _wav_pcm16(data)
    encoded = pcm16_to_mulaw(resample_pcm16(pcm16, sample_rate, TWILIO_SAMPLE_RATE))
    body = b"".join(
        (
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHHH", 18, 7, 1, TWILIO_SAMPLE_RATE, TWILIO_SAMPLE_RATE, 1, 8, 0),
            b"fact",
            struct.pack("<II", 4, len(encoded)),
            b"data",
            struct.pack("<I", len(encoded)),
            encoded,
            b"\x00" * (len(encoded) % 2),
        )
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


__all__ = [
    "MULAW_SAMPLES_PER_FRAME",
    "TWILIO_FRAME_MS",
    "TWILIO_SAMPLE_RATE",
    "VOICE_SAMPLE_RATE",
    "downsample_16k_to_8k",
    "mulaw_to_pcm16",
    "pcm16_to_mulaw",
    "resample_pcm16",
    "split_mulaw_frames",
    "upsample_8k_to_16k",
    "wav_to_mulaw_frames",
    "wav_to_mulaw_wav",
]
