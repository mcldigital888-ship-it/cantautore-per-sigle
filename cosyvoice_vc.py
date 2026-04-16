"""
CosyVoice Voice Conversion - processa audio lunghi a segmenti.
Gestisce il limite di 30s di CosyVoice dividendo in chunk con overlap.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from rich.console import Console

# Patch torchaudio prima di tutto
sys.path.insert(0, str(Path(__file__).parent))
import patch_torchaudio  # noqa: F401

console = Console()

COSYVOICE_MODEL = None  # singleton


def _get_model():
    """Carica CosyVoice una sola volta (lazy loading)."""
    global COSYVOICE_MODEL
    if COSYVOICE_MODEL is None:
        project_dir = Path(__file__).parent
        sys.path.insert(0, str(project_dir / "models" / "CosyVoice"))
        sys.path.insert(0, str(project_dir / "models" / "CosyVoice" / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import CosyVoice2
        model_path = str(project_dir / "models" / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
        COSYVOICE_MODEL = CosyVoice2(model_path, load_jit=False, load_trt=False)
    return COSYVOICE_MODEL


def convert_voice_cosyvoice(source_path: str, reference_path: str,
                            output_path: str, segment_sec: float = 25.0,
                            crossfade_sec: float = 0.5) -> str:
    """
    Converti la voce dell'intero file audio usando CosyVoice VC.
    Processa a segmenti di segment_sec con crossfade.

    Args:
        source_path: path audio sorgente (vocals separate)
        reference_path: path reference voice (max 30s)
        output_path: path output
        segment_sec: durata segmento in secondi (max 25)
        crossfade_sec: crossfade tra segmenti
    """
    model = _get_model()

    # Assicura reference < 30s
    ref_data, ref_sr = sf.read(reference_path)
    max_ref_samples = int(ref_sr * 28)  # 28s per sicurezza
    if len(ref_data) > max_ref_samples:
        ref_data = ref_data[:max_ref_samples]
        ref_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(ref_temp.name, ref_data, ref_sr)
        reference_path = ref_temp.name

    # Carica source
    source_data, source_sr = sf.read(source_path)
    total_samples = len(source_data)
    total_duration = total_samples / source_sr

    console.print(f"[dim]      Audio totale: {total_duration:.1f}s, procedo a segmenti di {segment_sec}s[/dim]")

    # Calcola segmenti con overlap
    segment_samples = int(source_sr * segment_sec)
    crossfade_samples = int(source_sr * crossfade_sec)
    step_samples = segment_samples - crossfade_samples

    segments_starts = list(range(0, total_samples, step_samples))
    n_segments = len(segments_starts)

    console.print(f"[dim]      {n_segments} segmenti da processare[/dim]")

    # Output sample rate di CosyVoice
    output_sr = 22050

    # Processa ogni segmento
    all_converted = []

    for i, start in enumerate(segments_starts):
        end = min(start + segment_samples, total_samples)
        segment = source_data[start:end]

        console.print(f"[dim]      Segmento {i+1}/{n_segments} ({start/source_sr:.1f}s-{end/source_sr:.1f}s)...[/dim]")

        # Salva segmento temporaneo
        seg_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(seg_temp.name, segment, source_sr)

        # Converti con CosyVoice
        chunks = []
        for chunk in model.inference_vc(
            source_wav=seg_temp.name,
            prompt_wav=reference_path,
            stream=False
        ):
            chunks.append(chunk['tts_speech'])

        converted = torch.cat(chunks, dim=-1).cpu().numpy().squeeze()

        import os
        os.unlink(seg_temp.name)

        all_converted.append(converted)

    # Riassembla con crossfade
    console.print("[dim]      Riassemblaggio con crossfade...[/dim]")
    result = _crossfade_segments(all_converted, output_sr, crossfade_sec)

    # Converti a stereo
    if result.ndim == 1:
        result = np.stack([result, result], axis=-1)

    sf.write(output_path, result, output_sr)
    info = sf.info(output_path)
    console.print(f"[green]    ✓ CosyVoice VC completato: {info.duration:.1f}s, {info.samplerate}Hz[/green]")

    return output_path


def _crossfade_segments(segments: list[np.ndarray], sr: int,
                        crossfade_sec: float) -> np.ndarray:
    """Unisci segmenti con crossfade lineare."""
    if len(segments) == 1:
        return segments[0]

    crossfade_samples = int(sr * crossfade_sec)

    # Calcola lunghezza totale
    total_len = len(segments[0])
    for seg in segments[1:]:
        total_len += len(seg) - crossfade_samples

    result = np.zeros(total_len, dtype=np.float32)
    pos = 0

    for i, seg in enumerate(segments):
        if i == 0:
            result[:len(seg)] = seg
            pos = len(seg) - crossfade_samples
        else:
            # Crossfade region
            if crossfade_samples > 0 and crossfade_samples <= len(seg):
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)

                result[pos:pos + crossfade_samples] *= fade_out
                result[pos:pos + crossfade_samples] += seg[:crossfade_samples] * fade_in

                # Resto del segmento
                remaining = seg[crossfade_samples:]
                result[pos + crossfade_samples:pos + crossfade_samples + len(remaining)] = remaining
            else:
                result[pos:pos + len(seg)] = seg

            pos += len(seg) - crossfade_samples

    return result[:pos + crossfade_samples] if pos + crossfade_samples <= total_len else result
