from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from core.voice.audio_codec import (
    downsample_16k_to_8k,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    resample_pcm16,
    split_mulaw_frames,
    upsample_8k_to_16k,
    wav_to_mulaw_frames,
    wav_to_mulaw_wav,
)


def _wav(pcm: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def test_mulaw_zero_and_empty_roundtrip() -> None:
    silence = np.zeros(160, dtype="<i2").tobytes()

    encoded = pcm16_to_mulaw(silence)

    assert encoded == b"\xff" * 160
    assert mulaw_to_pcm16(encoded) == silence
    assert pcm16_to_mulaw(b"") == b""
    assert mulaw_to_pcm16(b"") == b""


def test_mulaw_roundtrip_error_is_within_companding_tolerance() -> None:
    source = np.linspace(-32_000, 32_000, 4_001, dtype=np.int16).astype("<i2")

    decoded = np.frombuffer(mulaw_to_pcm16(pcm16_to_mulaw(source.tobytes())), dtype="<i2").astype(np.int32)
    error = np.abs(decoded - source.astype(np.int32))

    assert float(error.mean()) < 180
    assert int(error.max()) < 1_100


def test_pcm16_resampler_preserves_exact_8k_to_16k_to_8k_length() -> None:
    source = np.arange(800, dtype="<i2").tobytes()

    upsampled = upsample_8k_to_16k(source)
    restored = downsample_16k_to_8k(upsampled)

    assert len(upsampled) == 1_600 * 2
    assert len(restored) == len(source)


def test_resample_rejects_bad_rates_and_partial_samples() -> None:
    with pytest.raises(ValueError, match="sample rates"):
        resample_pcm16(b"\x00\x00", 0, 16_000)
    with pytest.raises(ValueError, match="complete samples"):
        resample_pcm16(b"\x00", 8_000, 16_000)


def test_downsampling_suppresses_frequencies_above_telephone_nyquist() -> None:
    sample_rate = 24_000
    times = np.arange(sample_rate, dtype=np.float64) / sample_rate
    high_tone = (np.sin(2 * np.pi * 7_000 * times) * 16_000).astype("<i2")

    downsampled = np.frombuffer(resample_pcm16(high_tone.tobytes(), sample_rate, 8_000), dtype="<i2")

    assert np.sqrt(np.mean(downsampled.astype(np.float64) ** 2)) < 500


def test_split_mulaw_frames_pads_final_twenty_millisecond_frame() -> None:
    frames = split_mulaw_frames(bytes(range(200)))

    assert [len(frame) for frame in frames] == [160, 160]
    assert frames[0] == bytes(range(160))
    assert frames[1][:40] == bytes(range(160, 200))
    assert frames[1][40:] == b"\xff" * 120


def test_split_mulaw_frames_can_leave_final_frame_unpadded() -> None:
    assert split_mulaw_frames(b"x" * 165, pad_final=False) == [b"x" * 160, b"x" * 5]


def test_split_mulaw_frames_rejects_invalid_duration() -> None:
    with pytest.raises(ValueError, match="whole number"):
        split_mulaw_frames(b"audio", frame_ms=0)


def test_wav_conversion_resamples_24khz_pcm_and_splits_into_twenty_ms_frames() -> None:
    source = np.full(480, 2_000, dtype="<i2").tobytes()  # 20 ms at 24 kHz

    frames = wav_to_mulaw_frames(_wav(source, sample_rate=24_000))

    assert len(frames) == 1
    assert len(frames[0]) == 160
    assert np.all(np.frombuffer(mulaw_to_pcm16(frames[0]), dtype="<i2") > 0)


def test_wav_conversion_downmixes_stereo_to_mono() -> None:
    stereo = np.array([[1_000, -1_000]] * 480, dtype="<i2").tobytes()

    frames = wav_to_mulaw_frames(_wav(stereo, sample_rate=24_000, channels=2))

    assert len(frames) == 1
    decoded = np.frombuffer(mulaw_to_pcm16(frames[0]), dtype="<i2")
    assert np.max(np.abs(decoded)) < 10


def test_wav_to_mulaw_wav_shrinks_audio_into_a_telephone_wav() -> None:
    pcm = (np.sin(np.linspace(0, 200, 24_000)) * 8_000).astype("<i2").tobytes()
    source = _wav(pcm, sample_rate=24_000)

    converted = wav_to_mulaw_wav(source)

    assert converted[:4] == b"RIFF" and converted[8:12] == b"WAVE"
    assert struct.unpack("<I", converted[4:8])[0] == len(converted) - 8
    assert struct.unpack("<HHIIHH", converted[20:36]) == (7, 1, 8_000, 8_000, 1, 8)
    data_at = converted.index(b"data")
    assert struct.unpack("<I", converted[data_at + 4 : data_at + 8])[0] == 8_000
    assert converted[data_at + 8 :] == b"".join(wav_to_mulaw_frames(source))
    assert len(converted) < len(source) / 5


def test_wav_to_mulaw_wav_pads_odd_length_audio_to_a_chunk_boundary() -> None:
    converted = wav_to_mulaw_wav(_wav(b"\x00\x00" * 3, sample_rate=8_000))

    assert len(converted) % 2 == 0
    assert struct.unpack("<I", converted[4:8])[0] == len(converted) - 8


def test_wav_to_mulaw_wav_rejects_non_wav_audio() -> None:
    with pytest.raises(ValueError):
        wav_to_mulaw_wav(b"ID3-not-a-wav")


def test_wav_conversion_rejects_non_wav_audio() -> None:
    with pytest.raises(ValueError, match="PCM WAV"):
        wav_to_mulaw_frames(b"not a wav")
