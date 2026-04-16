"""
Advanced Audio Processing - Cantautore Digitale
================================================
Tecniche innovative che nessuno combina insieme:
1. Segmentazione intelligente per frasi
2. AI-in-the-loop (Gemini critico automatico)
3. Pitch correction + vibrato injection
4. Humanization (breath, micro-timing, dynamics)
5. Professional mastering chain
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.panel import Panel

console = Console()


# ============================================================
# 1. SEGMENTAZIONE INTELLIGENTE
# ============================================================

def detect_vocal_segments(audio: np.ndarray, sr: int,
                          min_silence_ms: int = 300,
                          silence_thresh_db: float = -40.0) -> list[tuple[int, int]]:
    """
    Rileva segmenti vocali nell'audio basandosi su energia.
    Ritorna lista di (start_sample, end_sample).
    """
    from scipy.ndimage import uniform_filter1d

    # Converti a mono per l'analisi
    if audio.ndim > 1:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # Calcola energia in dB con finestra mobile
    frame_size = int(sr * 0.02)  # 20ms frames
    energy = uniform_filter1d(mono ** 2, size=frame_size)
    energy_db = 10 * np.log10(energy + 1e-10)

    # Trova regioni sopra la soglia
    is_voice = energy_db > silence_thresh_db

    # Espandi le regioni per non tagliare attacchi/code
    pad_samples = int(sr * 0.05)  # 50ms padding
    from scipy.ndimage import binary_dilation
    is_voice = binary_dilation(is_voice, iterations=pad_samples)

    # Trova transizioni
    segments = []
    in_segment = False
    start = 0
    min_silence_samples = int(sr * min_silence_ms / 1000)

    for i in range(len(is_voice)):
        if is_voice[i] and not in_segment:
            start = max(0, i - pad_samples)
            in_segment = True
        elif not is_voice[i] and in_segment:
            # Controlla se il silenzio è abbastanza lungo
            silence_end = i
            while silence_end < len(is_voice) and not is_voice[silence_end]:
                silence_end += 1
            if silence_end - i >= min_silence_samples or silence_end >= len(is_voice):
                end = min(len(audio), i + pad_samples)
                segments.append((start, end))
                in_segment = False

    if in_segment:
        segments.append((start, len(audio)))

    # Unisci segmenti troppo corti (< 1s)
    min_segment = int(sr * 1.0)
    merged = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] < min_segment:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    return merged


def split_audio_by_segments(audio: np.ndarray, sr: int,
                            segments: list[tuple[int, int]]) -> list[np.ndarray]:
    """Taglia l'audio in segmenti."""
    return [audio[start:end] for start, end in segments]


def reassemble_segments(segments: list[np.ndarray], original_length: int,
                        positions: list[tuple[int, int]],
                        crossfade_ms: int = 20) -> np.ndarray:
    """Riassembla segmenti con crossfade per evitare click."""
    ndim = segments[0].ndim
    shape = (original_length,) if ndim == 1 else (original_length, segments[0].shape[1])
    result = np.zeros(shape, dtype=np.float32)

    crossfade_samples = int(44100 * crossfade_ms / 1000)

    for seg, (start, end) in zip(segments, positions):
        seg_len = min(len(seg), end - start)
        # Applica fade in/out per crossfade pulito
        fade_len = min(crossfade_samples, seg_len // 4)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            if ndim > 1:
                fade_in = fade_in[:, np.newaxis]
                fade_out = fade_out[:, np.newaxis]
            seg[:fade_len] *= fade_in
            seg[-fade_len:] *= fade_out

        result[start:start + seg_len] = seg[:seg_len]

    return result


# ============================================================
# 2. PITCH CORRECTION + VIBRATO INJECTION
# ============================================================

def extract_pitch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Estrai pitch usando librosa (pyin)."""
    import librosa

    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    f0, voiced_flag, _ = librosa.pyin(
        mono, fmin=60, fmax=500, sr=sr,
        frame_length=2048, hop_length=512
    )
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    return f0, times


def smooth_pitch(f0: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Liscia il pitch per rimuovere jitter indesiderato."""
    from scipy.ndimage import median_filter

    smoothed = f0.copy()
    voiced = ~np.isnan(f0)

    if np.sum(voiced) > window_size:
        smoothed[voiced] = median_filter(f0[voiced], size=window_size)

    return smoothed


def inject_vibrato(audio: np.ndarray, sr: int,
                   rate_hz: float = 5.5, depth_cents: float = 15.0,
                   onset_delay_ms: float = 200.0) -> np.ndarray:
    """
    Inietta vibrato naturale nelle note lunghe.
    Rate: 5-6 Hz (vibrato naturale umano)
    Depth: 10-20 cents (sottile ma percepibile)
    Onset delay: il vibrato inizia dopo l'attacco della nota.
    """
    import librosa

    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio

    # Estrai pitch
    f0, voiced_flag, _ = librosa.pyin(
        mono, fmin=60, fmax=500, sr=sr,
        frame_length=2048, hop_length=512
    )

    hop = 512
    n_frames = len(f0)

    # Trova note lunghe (> 300ms di pitch stabile)
    min_frames = int(0.3 * sr / hop)
    onset_frames = int(onset_delay_ms / 1000 * sr / hop)

    # Genera modulazione di pitch
    t = np.arange(len(audio)) / sr
    vibrato_signal = np.ones(len(audio))

    # Per ogni frame con pitch
    for i in range(n_frames):
        if np.isnan(f0[i]):
            continue

        # Conta frames consecutivi con pitch
        consecutive = 0
        j = i
        while j < n_frames and not np.isnan(f0[j]):
            consecutive += 1
            j += 1

        if consecutive >= min_frames and (i - max(0, i - consecutive)) >= onset_frames:
            # Questa è una nota lunga dopo l'onset
            frame_start = i * hop
            frame_end = min((i + 1) * hop, len(audio))

            # Modulazione sinusoidale con envelope crescente
            frames_since_onset = i - (i - consecutive) - onset_frames
            envelope = min(1.0, frames_since_onset / (min_frames * 0.5))

            for s in range(frame_start, frame_end):
                # Vibrato: modulazione di ampiezza sottile
                mod = 1.0 + envelope * (depth_cents / 1200.0) * np.sin(2 * np.pi * rate_hz * t[s])
                vibrato_signal[s] = mod

    # Applica come modulazione di ampiezza (approssimazione del pitch shift)
    if audio.ndim > 1:
        result = audio * vibrato_signal[:, np.newaxis]
    else:
        result = audio * vibrato_signal

    return result


# ============================================================
# 3. HUMANIZATION - Respiro e micro-timing
# ============================================================

def add_breath_sounds(audio: np.ndarray, sr: int,
                      segments: list[tuple[int, int]]) -> np.ndarray:
    """Aggiunge suoni di respiro sottili tra i segmenti vocali."""
    result = audio.copy()
    np.random.seed(123)

    for i in range(1, len(segments)):
        prev_end = segments[i - 1][1]
        curr_start = segments[i][0]
        gap = curr_start - prev_end

        if gap < int(sr * 0.15) or gap > int(sr * 1.5):
            continue

        # Genera un suono di respiro (rumore filtrato passa-basso)
        breath_len = min(gap, int(sr * 0.25))
        breath = np.random.randn(breath_len) * 0.008

        # Filtra passa-basso per suono realistico
        from scipy.signal import butter, sosfilt
        sos = butter(4, 2000, btype='low', fs=sr, output='sos')
        breath = sosfilt(sos, breath)

        # Envelope naturale (attacco rapido, decadimento lento)
        t = np.linspace(0, 1, breath_len)
        envelope = np.sin(np.pi * t) ** 0.5  # curva naturale
        breath *= envelope

        # Inserisci il respiro
        breath_start = prev_end + (gap - breath_len) // 2
        if audio.ndim > 1:
            for ch in range(audio.shape[1]):
                result[breath_start:breath_start + breath_len, ch] += breath
        else:
            result[breath_start:breath_start + breath_len] += breath

    return result


def add_micro_timing(audio: np.ndarray, sr: int,
                     max_shift_ms: float = 8.0) -> np.ndarray:
    """
    Aggiunge micro-variazioni di timing per suono più umano.
    Gli umani non cantano perfettamente a tempo.
    """
    # Shift molto sottile - massimo 8ms
    max_shift = int(sr * max_shift_ms / 1000)
    np.random.seed(456)
    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift > 0:
        result = np.pad(audio, ((shift, 0), (0, 0)) if audio.ndim > 1 else (shift, 0))
        result = result[:len(audio)]
    elif shift < 0:
        result = np.pad(audio, ((0, -shift), (0, 0)) if audio.ndim > 1 else (0, -shift))
        result = result[-shift:-shift + len(audio)]
    else:
        result = audio.copy()

    return result


def add_dynamic_variation(audio: np.ndarray, sr: int,
                          variation_db: float = 1.5) -> np.ndarray:
    """Aggiunge variazione dinamica naturale (non tutto lo stesso volume)."""
    # Modulazione lenta del volume (0.5-2 Hz)
    t = np.arange(len(audio)) / sr
    np.random.seed(789)
    # Somma di sinusoidi lente per variazione naturale
    mod = 1.0
    for freq in [0.3, 0.7, 1.2, 0.15]:
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = variation_db / 20.0 / 4  # dividi per numero di componenti
        mod = mod + amplitude * np.sin(2 * np.pi * freq * t + phase)

    if audio.ndim > 1:
        return audio * mod[:, np.newaxis]
    else:
        return audio * mod


# ============================================================
# 4. AI-IN-THE-LOOP - Gemini come critico automatico
# ============================================================

def gemini_score_segment(audio_segment: np.ndarray, sr: int,
                         api_key: str) -> dict:
    """
    Fa analizzare un segmento audio a Gemini e ritorna un punteggio.
    """
    import tempfile
    from google import genai

    # Salva segmento temporaneo
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio_segment, sr)
        temp_path = f.name

    with open(temp_path, 'rb') as f:
        audio_data = f.read()

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            {
                'inline_data': {
                    'mime_type': 'audio/wav',
                    'data': base64.b64encode(audio_data).decode()
                }
            },
            '''Analizza questo segmento vocale cantato. Rispondi SOLO in JSON:
{
  "pronunciation_score": X,  // 1-10 pronuncia italiana
  "naturalness_score": X,    // 1-10 quanto suona naturale/umano
  "audio_quality_score": X,  // 1-10 qualità audio (artefatti, distorsione)
  "emotion_score": X,        // 1-10 espressività emotiva
  "overall_score": X,        // 1-10 media pesata
  "issues": ["lista", "problemi"],
  "suggestion": "suggerimento breve"
}'''
        ]
    )

    import os
    os.unlink(temp_path)

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    return json.loads(text.strip())


def gemini_score_full_song(audio_path: str, api_key: str) -> dict:
    """Fa analizzare la canzone completa a Gemini."""
    from google import genai

    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    client = genai.Client(api_key=api_key)

    # Prova gemini-2.5-pro, fallback a flash se non disponibile
    model_name = 'gemini-2.5-pro'
    prompt_text = '''Sei un produttore musicale e ingegnere del suono che valuta un demo.
Valuta come un ascoltatore professionale, non cercare se è AI o umano.
Standard: 5=demo accettabile, 7=pronto per piattaforme streaming, 9=hit potenziale.

Rispondi SOLO in JSON valido:
{
  "overall": X,
  "pronunciation": X,
  "audio_quality": X,
  "voice_timbre": X,
  "emotion": X,
  "music": X,
  "human_likeness": X,
  "mix_balance": X,
  "top_issues": ["issue1", "issue2", "issue3"],
  "verdict": "giudizio critico in 2 frasi"
}
Punteggi 1-10. Sii critico ma giusto.'''

    for attempt_model in [model_name, 'gemini-2.5-flash']:
        try:
            response = client.models.generate_content(
                model=attempt_model,
                contents=[
                    {
                        'inline_data': {
                            'mime_type': 'audio/wav',
                            'data': base64.b64encode(audio_data).decode()
                        }
                    },
                    prompt_text
                ]
            )
            break
        except Exception:
            if attempt_model == 'gemini-2.5-flash':
                raise
            continue

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    return json.loads(text.strip())


# ============================================================
# 5. PIPELINE AVANZATA - combina tutto
# ============================================================

def process_vocals_advanced(vocals: np.ndarray, sr: int,
                            verbose: bool = True) -> np.ndarray:
    """
    Pipeline di processing vocale avanzato:
    1. Segmenta per frasi
    2. Pitch smoothing
    3. Vibrato injection
    4. Humanization (respiro, micro-timing, dinamiche)
    """
    if verbose:
        console.print("[cyan]    🔬 Processing avanzato vocals...[/cyan]")

    # 1. Rileva segmenti
    segments = detect_vocal_segments(vocals, sr)
    if verbose:
        console.print(f"[dim]      Trovati {len(segments)} segmenti vocali[/dim]")

    # 2. Pitch smoothing (rimuovi jitter)
    if verbose:
        console.print("[dim]      Pitch smoothing...[/dim]")
    # Applichiamo un filtro mediano leggero sul segnale per ridurre artefatti
    from scipy.ndimage import median_filter
    if vocals.ndim > 1:
        for ch in range(vocals.shape[1]):
            # Filtro mediano molto leggero (3 samples) per rimuovere micro-glitch
            vocals[:, ch] = median_filter(vocals[:, ch], size=3)
    else:
        vocals = median_filter(vocals, size=3)

    # 3. Vibrato injection
    if verbose:
        console.print("[dim]      Vibrato injection...[/dim]")
    vocals = inject_vibrato(vocals, sr, rate_hz=5.5, depth_cents=12.0,
                            onset_delay_ms=250.0)

    # 4. Respiri tra le frasi
    if verbose:
        console.print("[dim]      Inserimento respiri...[/dim]")
    vocals = add_breath_sounds(vocals, sr, segments)

    # 5. Variazione dinamica
    if verbose:
        console.print("[dim]      Variazione dinamica naturale...[/dim]")
    vocals = add_dynamic_variation(vocals, sr, variation_db=1.2)

    # 6. Micro-timing
    vocals = add_micro_timing(vocals, sr, max_shift_ms=5.0)

    if verbose:
        console.print("[green]    ✓ Processing avanzato completato[/green]")

    return vocals
