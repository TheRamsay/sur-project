"""
Generate synthetic m431 audio via ElevenLabs + spectral matching.

Steps:
  1. Re-clone voice using all 20 training WAVs
  2. Generate 10 English utterances
  3. Spectral-match each clip to the average VUTBR training spectrum

Usage: uv run python generate_synthetic_m431.py
"""
import os, wave
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from elevenlabs.client import ElevenLabs

# ── Load .env ─────────────────────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ.get("ELEVEN_API_KEY")
if not API_KEY:
    raise SystemExit("Set ELEVEN_API_KEY in .env or environment.")

client = ElevenLabs(api_key=API_KEY)

# ── All 20 training WAVs as reference ────────────────────────────────────────
TRAIN_DIR = Path("data/target_train")
REFS = sorted(TRAIN_DIR.glob("m431_*.wav"))
print(f"Using {len(REFS)} reference clips for voice clone")

# ── English prompts — match language of m431 recordings ──────────────────────
PROMPTS = [
    "The weather today is quite pleasant and sunny.",
    "I enjoy reading books about science and technology.",
    "Please repeat the sentence slowly and clearly.",
    "The quick brown fox jumps over the lazy dog.",
    "Speech recognition systems have improved significantly in recent years.",
    "I am a student at the technical university.",
    "Tomorrow morning I will attend an important meeting.",
    "The microphone captures the sound of my voice accurately.",
    "Natural language processing is a fascinating field of research.",
    "I had soup and a main course for lunch today.",
]

OUT_RAW = Path("eval_synthetic/audio_raw")
OUT     = Path("eval_synthetic/audio")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

N_FFT      = 2048
HOP_LENGTH = 512
SR         = 16000

# ── Spectral matching helpers ─────────────────────────────────────────────────
def mean_spectrum(wav_paths):
    """Average magnitude spectrum across a list of WAV files."""
    spectra = []
    for p in wav_paths:
        y, _ = librosa.load(str(p), sr=SR, mono=True)
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        spectra.append(S.mean(axis=1))
    return np.mean(spectra, axis=0)

def spectral_match(src_path, target_spectrum, dst_path):
    """Equalize src_path to match target_spectrum, write to dst_path."""
    y, _ = librosa.load(str(src_path), sr=SR, mono=True)
    D     = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S     = np.abs(D)
    phase = np.angle(D)

    src_spectrum = S.mean(axis=1)
    eq = target_spectrum / (src_spectrum + 1e-8)
    eq = gaussian_filter1d(eq, sigma=8)          # smooth to avoid ringing
    eq = np.clip(eq, 0.1, 10.0)                  # limit boost/cut range

    S_eq = S * eq[:, np.newaxis]
    D_eq = S_eq * np.exp(1j * phase)
    y_eq = librosa.istft(D_eq, hop_length=HOP_LENGTH)

    # Normalise to -3 dBFS peak
    peak = np.max(np.abs(y_eq)) + 1e-8
    y_eq = y_eq / peak * 0.708

    sf.write(str(dst_path), y_eq, SR)

def pcm_to_wav(pcm_bytes, path, sample_rate=16000, channels=1, sampwidth=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)

# ── Step 1: compute target spectrum from training data ────────────────────────
print("Computing target spectrum from training data...")
target_spectrum = mean_spectrum(REFS)
print(f"  Target spectrum shape: {target_spectrum.shape}")

# ── Step 2: clone voice with all 20 clips ─────────────────────────────────────
print("Cloning m431 voice from all training clips...")
file_handles = [open(p, "rb") for p in REFS]
try:
    voice = client.voices.ivc.create(
        name="m431_clone_v2",
        files=file_handles,
        description="m431 full-data voice clone",
    )
finally:
    for f in file_handles:
        f.close()
print(f"Voice created: {voice.voice_id}")

# ── Step 3: generate utterances ───────────────────────────────────────────────
print("\nGenerating utterances...")
for i, text in enumerate(PROMPTS):
    raw  = OUT_RAW / f"m431_synth_{i:03d}.wav"
    final = OUT    / f"m431_synth_{i:03d}.wav"
    print(f"[{i+1}/{len(PROMPTS)}] {text[:55]}...")

    audio_bytes = b"".join(
        client.text_to_speech.convert(
            voice_id=voice.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_16000",
        )
    )
    pcm_to_wav(audio_bytes, raw)

    # Step 4: spectral match
    spectral_match(raw, target_spectrum, final)
    print(f"  raw → {raw.name}  |  matched → {final.name}")

print(f"\nDone. {len(PROMPTS)} clips in {OUT}/")
print(f"Voice ID for reuse: {voice.voice_id}")
print("\nValidate:")
print("  uv run python predict_audio.py --eval-dir eval_synthetic/audio/")
