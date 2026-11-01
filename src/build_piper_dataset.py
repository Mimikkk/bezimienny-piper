"""Pair Bezimienny's recordings with the Polish lines and write an LJSpeech set.

Piper reads metadata.csv plus wav/ and later exports one onnx, same kind of
file as pl_PL-gosia-medium.
"""
import struct
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

DATA = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Gothic II\Data")
OU = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Gothic II"
    r"\_work\Data\Scripts\Content\Cutscene\Ou.csl"
)
VDFS = ["Speech1.vdf", "Speech2.vdf", "Speech_Addon.vdf"]
OUT = Path(__file__).resolve().parents[1] / "datasets" / "piper" / "pl" / "bezimienny"
TARGET_RATE = 22050
MIN_SECONDS = 0.6
MAX_SECONDS = 12.0

STEP = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
    253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
    1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
    3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487,
    12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
]
INDEX = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8]


def hero_wav(name: str) -> bool:
    upper = name.upper()
    if not upper.endswith(".WAV") or "_15_" not in upper:
        return False
    tail = upper.rsplit("_15_", 1)[-1][:-4]
    return tail.isdigit()


def load_lines(path: Path) -> dict:
    text = path.read_text(encoding="cp1250", errors="replace")
    lines = {}
    pending = None
    for raw in text.splitlines():
        if raw.startswith("\t\t\t\ttext=string:"):
            pending = " ".join(raw.split("text=string:", 1)[1].split())
        elif raw.startswith("\t\t\t\tname=string:") and pending:
            wav = raw.split("name=string:", 1)[1].strip()
            if hero_wav(wav) and any(ch.isalpha() for ch in pending) and len(pending) >= 3:
                lines.setdefault(wav.upper(), pending)
            pending = None
    return lines


def vdf_entries(path: Path):
    with path.open("rb") as handle:
        handle.seek(272)
        entry_count, _, _, _, table_off, _ = struct.unpack("<6I", handle.read(24))
        handle.seek(table_off)
        raw = handle.read(entry_count * 80)
    entries = []
    for i in range(entry_count):
        chunk = raw[i * 80 : (i + 1) * 80]
        name = chunk[:64].split(b"\x00", 1)[0].decode("latin1").strip()
        offset, size, typ, _attr = struct.unpack_from("<4I", chunk, 64)
        if typ & 0x80000000:
            continue
        if hero_wav(name):
            entries.append((name, offset, size))
    return entries


def decode_ima_block(block: bytes, samples_per_block: int):
    if len(block) < 4:
        return []
    predictor = struct.unpack_from("<h", block, 0)[0]
    index = block[2]
    if index > 88:
        index = 88
    out = [predictor]
    nibble_i = 0
    data = block[4:]
    limit = len(data) * 2
    while len(out) < samples_per_block and nibble_i < limit:
        byte = data[nibble_i // 2]
        nibble = (byte & 0x0F) if (nibble_i % 2 == 0) else (byte >> 4)
        nibble_i += 1
        step = STEP[index]
        diff = step >> 3
        if nibble & 1:
            diff += step >> 2
        if nibble & 2:
            diff += step >> 1
        if nibble & 4:
            diff += step
        predictor += -diff if (nibble & 8) else diff
        if predictor > 32767:
            predictor = 32767
        elif predictor < -32768:
            predictor = -32768
        index += INDEX[nibble & 7]
        if index < 0:
            index = 0
        elif index > 88:
            index = 88
        out.append(predictor)
    return out


def decode_wav(blob: bytes):
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return None
    i = 12
    fmt = None
    data = None
    while i + 8 <= len(blob):
        cid = blob[i : i + 4]
        size = int.from_bytes(blob[i + 4 : i + 8], "little")
        body = blob[i + 8 : i + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
            break
        i += 8 + size + (size & 1)
    if not fmt or data is None or len(fmt) < 16:
        return None
    tag, channels, rate, _br, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if channels != 1:
        return None
    if tag == 1 and bits == 16:
        count = len(data) // 2
        return rate, np.frombuffer(data[: count * 2], dtype="<i2").astype(np.float32)
    if tag != 17 or align <= 4:
        return None
    samples_per_block = struct.unpack_from("<H", fmt, 18)[0] if len(fmt) >= 20 else (align - 4) * 2 + 1
    samples = []
    for off in range(0, len(data) - align + 1, align):
        samples.extend(decode_ima_block(data[off : off + align], samples_per_block))
    if not samples:
        return None
    return rate, np.asarray(samples, dtype=np.float32)


def trim(samples: np.ndarray, rate: int) -> np.ndarray:
    if samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples)))
    if peak < 400:
        return samples[:0]
    thresh = max(400.0, peak * 0.02)
    idx = np.flatnonzero(np.abs(samples) >= thresh)
    if idx.size == 0:
        return samples[:0]
    pad = int(rate * 0.04)
    start = max(0, int(idx[0]) - pad)
    end = min(samples.size, int(idx[-1]) + 1 + pad)
    return samples[start:end]


def main():
    lines = load_lines(OU)
    print(f"polish hero lines in dialogue file: {len(lines)}", flush=True)
    wav_dir = OUT / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    missing_text = 0
    skipped = 0
    total = 0.0
    for vdf_name in VDFS:
        path = DATA / vdf_name
        entries = vdf_entries(path)
        print(f"{vdf_name}: {len(entries)} hero recordings", flush=True)
        with path.open("rb") as handle:
            for name, offset, size in entries:
                key = name.upper()
                if key in seen:
                    continue
                text = lines.get(key)
                if not text:
                    missing_text += 1
                    continue
                handle.seek(offset)
                decoded = decode_wav(handle.read(size))
                if decoded is None:
                    skipped += 1
                    continue
                rate, samples = decoded
                samples = trim(samples, rate)
                dur = samples.size / rate if rate else 0
                if dur < MIN_SECONDS or dur > MAX_SECONDS:
                    skipped += 1
                    continue
                if rate != TARGET_RATE:
                    samples = resample_poly(samples, TARGET_RATE, rate)
                peak = float(np.max(np.abs(samples))) if samples.size else 0
                if peak < 400:
                    skipped += 1
                    continue
                pcm = np.clip(samples, -32768, 32767).astype(np.int16)
                stem = Path(name).stem
                out = wav_dir / f"{stem}.wav"
                with wave.open(str(out), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(TARGET_RATE)
                    wav_file.writeframes(pcm.tobytes())
                rows.append((stem, text))
                seen.add(key)
                total += pcm.size / TARGET_RATE
                if len(rows) % 200 == 0:
                    print(f"kept {len(rows)}  {total / 60:.1f} min", flush=True)
    meta = OUT / "metadata.csv"
    with meta.open("w", encoding="utf-8", newline="\n") as handle:
        for stem, text in rows:
            handle.write(f"{stem}|{text}\n")
    print(
        f"DONE lines={len(rows)} minutes={total / 60:.1f} "
        f"missing_text={missing_text} skipped={skipped}",
        flush=True,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for stem, text in rows[:8]:
        print(f"  {stem} | {text}", flush=True)


if __name__ == "__main__":
    main()
