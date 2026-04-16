"""
Export Kit - Cantautore Digitale
================================
Esporta tutti gli stems separati di una canzone per uso professionale.
Ogni canzone viene esportata come kit completo importabile in qualsiasi DAW
(Logic Pro, Ableton, FL Studio, Pro Tools, GarageBand, Reaper, etc).

Uso:
  # Esporta stems da un file audio
  python export_kit.py separa canzone.wav

  # Esporta stems da un file audio in una cartella specifica
  python export_kit.py separa canzone.wav --output cartella_kit

  # Esporta stems con modello Demucs 6-stems (più separazione)
  python export_kit.py separa canzone.wav --model htdemucs_6s

  # Esporta solo vocals e instrumental (2 stems)
  python export_kit.py separa canzone.wav --two-stems

  # Info su un file audio
  python export_kit.py info canzone.wav

  # Converti formato (wav→mp3, cambio sample rate, etc)
  python export_kit.py converti canzone.wav --formato mp3 --sr 44100
"""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROJECT_DIR = Path(__file__).parent
TEMP_DIR = PROJECT_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def get_audio_info(audio_path: Path) -> dict:
    """Ritorna info dettagliate su un file audio."""
    info = sf.info(str(audio_path))
    data, sr = sf.read(str(audio_path))
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))
    peak_db = 20 * np.log10(peak + 1e-10)
    rms_db = 20 * np.log10(rms + 1e-10)

    return {
        "file": str(audio_path),
        "duration_sec": info.duration,
        "duration_min": f"{int(info.duration // 60)}:{int(info.duration % 60):02d}",
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "frames": info.frames,
        "peak_db": peak_db,
        "rms_db": rms_db,
        "bit_depth": info.subtype,
        "file_size_mb": audio_path.stat().st_size / (1024 * 1024),
    }


def separa_stems(audio_path: Path, output_dir: Path,
                 model: str = "htdemucs", two_stems: bool = False,
                 format_wav: bool = True, sample_rate: int | None = None) -> dict:
    """
    Separa un file audio in stems usando Demucs.

    Modelli disponibili:
      - htdemucs: 4 stems (drums, bass, other, vocals) — default, migliore
      - htdemucs_6s: 6 stems (drums, bass, other, vocals, guitar, piano)
      - htdemucs_ft: 4 stems fine-tuned (leggermente migliore, più lento)

    Ritorna dict con i path degli stems generati.
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"File non trovato: {audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Separazione Stems[/bold]\n"
        f"File: {audio_path.name}\n"
        f"Modello: {model}\n"
        f"Output: {output_dir}",
        style="cyan"
    ))

    # Carica modello
    console.print(f"[cyan]Caricamento modello {model}...[/cyan]")
    demucs_model = get_model(model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    demucs_model.to(device)

    # Carica audio
    console.print("[cyan]Caricamento audio...[/cyan]")
    audio_data, sr = sf.read(str(audio_path))

    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=-1)

    # Demucs vuole (batch, channels, samples)
    wav = torch.tensor(audio_data.T, dtype=torch.float32).unsqueeze(0).to(device)

    # Resample se necessario (Demucs vuole 44100)
    if sr != demucs_model.samplerate:
        import torchaudio
        wav = torchaudio.functional.resample(
            wav.squeeze(0), sr, demucs_model.samplerate
        ).unsqueeze(0)
        sr = demucs_model.samplerate

    # Separa
    console.print("[cyan]Separazione in corso (può richiedere qualche minuto)...[/cyan]")
    sources = apply_model(demucs_model, wav, device=device)
    source_names = demucs_model.sources

    stems = {}
    info = sf.info(str(audio_path))

    # Target sample rate
    out_sr = sample_rate or sr

    # Salva ogni stem
    for i, name in enumerate(source_names):
        stem_data = sources[0, i].cpu().numpy().T  # (samples, channels)

        # Resample se richiesto
        if out_sr != sr:
            import librosa
            stem_data_l = librosa.resample(stem_data[:, 0], orig_sr=sr, target_sr=out_sr)
            stem_data_r = librosa.resample(stem_data[:, 1], orig_sr=sr, target_sr=out_sr)
            stem_data = np.stack([stem_data_l, stem_data_r], axis=-1)

        stem_path = output_dir / f"{name}.wav"
        sf.write(str(stem_path), stem_data, out_sr, subtype='PCM_24')
        stems[name] = stem_path
        console.print(f"  [green]✓ {name}.wav[/green]")

    # Crea anche mix strumentale (tutto tranne vocals)
    vocals_idx = source_names.index("vocals")
    instrumental = sum(
        sources[0, i] for i in range(len(source_names)) if i != vocals_idx
    ).cpu().numpy().T

    if out_sr != sr:
        import librosa
        inst_l = librosa.resample(instrumental[:, 0], orig_sr=sr, target_sr=out_sr)
        inst_r = librosa.resample(instrumental[:, 1], orig_sr=sr, target_sr=out_sr)
        instrumental = np.stack([inst_l, inst_r], axis=-1)

    inst_path = output_dir / "instrumental.wav"
    sf.write(str(inst_path), instrumental, out_sr, subtype='PCM_24')
    stems["instrumental"] = inst_path
    console.print(f"  [green]✓ instrumental.wav[/green]")

    # Crea anche vocals isolati puliti (mono centrato per DAW)
    vocals_stereo = sources[0, vocals_idx].cpu().numpy().T
    vocals_mono = np.mean(vocals_stereo, axis=1)
    vocals_mono_stereo = np.stack([vocals_mono, vocals_mono], axis=-1)

    if out_sr != sr:
        import librosa
        vm = librosa.resample(vocals_mono, orig_sr=sr, target_sr=out_sr)
        vocals_mono_stereo = np.stack([vm, vm], axis=-1)

    vocals_center_path = output_dir / "vocals_center.wav"
    sf.write(str(vocals_center_path), vocals_mono_stereo, out_sr, subtype='PCM_24')
    stems["vocals_center"] = vocals_center_path
    console.print(f"  [green]✓ vocals_center.wav (mono centrato)[/green]")

    # Salva info del kit
    kit_info = {
        "source_file": str(audio_path),
        "model": model,
        "sample_rate": out_sr,
        "bit_depth": "24-bit",
        "duration": info.duration,
        "stems": list(stems.keys()),
        "created": datetime.now().isoformat(),
        "note": "Importa tutti i .wav nella tua DAW sullo stesso punto di inizio (beat 1). Sono già allineati.",
    }
    kit_info_path = output_dir / "kit_info.json"
    with open(kit_info_path, "w", encoding="utf-8") as f:
        json.dump(kit_info, f, ensure_ascii=False, indent=2)

    # Riepilogo
    table = Table(title=f"Kit Stems - {audio_path.stem}")
    table.add_column("Stem", style="cyan")
    table.add_column("File", style="green")
    table.add_column("Uso in DAW")

    stem_descriptions = {
        "drums": "Batteria e percussioni",
        "bass": "Basso",
        "other": "Synth, archi, pad, effetti",
        "vocals": "Voce principale + cori (stereo)",
        "vocals_center": "Voce principale (mono centrato)",
        "instrumental": "Tutto tranne la voce",
        "guitar": "Chitarre",
        "piano": "Pianoforte e tastiere",
    }

    for name, path in stems.items():
        table.add_row(
            name,
            path.name,
            stem_descriptions.get(name, ""),
        )

    console.print(f"\n")
    console.print(table)
    console.print(f"\n[green]✓ Kit esportato in: {output_dir}[/green]")
    console.print(f"[dim]  Importa tutti i .wav nella DAW sullo stesso punto (sono allineati)[/dim]")

    return stems


def converti_audio(input_path: Path, output_path: Path | None = None,
                   formato: str = "wav", sample_rate: int | None = None,
                   bit_depth: str = "PCM_24", mono: bool = False):
    """Converti formato audio, sample rate, bit depth, stereo/mono."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File non trovato: {input_path}")

    data, sr = sf.read(str(input_path))

    # Resample
    if sample_rate and sample_rate != sr:
        import librosa
        if data.ndim > 1:
            channels = []
            for ch in range(data.shape[1]):
                channels.append(librosa.resample(data[:, ch], orig_sr=sr, target_sr=sample_rate))
            data = np.stack(channels, axis=-1)
        else:
            data = librosa.resample(data, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    # Mono
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # Output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_converted.{formato}"
    else:
        output_path = Path(output_path)

    # Salva
    if formato == "wav":
        sf.write(str(output_path), data, sr, subtype=bit_depth)
    elif formato == "flac":
        sf.write(str(output_path), data, sr, format="FLAC")
    elif formato == "mp3":
        # Salva come wav temporaneo poi converti con ffmpeg
        temp_wav = TEMP_DIR / "temp_convert.wav"
        sf.write(str(temp_wav), data, sr)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(temp_wav), "-b:a", "320k", str(output_path)],
            capture_output=True, text=True
        )
        temp_wav.unlink(missing_ok=True)
        if result.returncode != 0:
            console.print(f"[red]Errore ffmpeg: {result.stderr[-200:]}[/red]")
            console.print("[yellow]Installalo con: apt install ffmpeg[/yellow]")
            return
    else:
        sf.write(str(output_path), data, sr)

    info = sf.info(str(output_path)) if output_path.exists() else None
    console.print(f"[green]✓ Convertito: {output_path}[/green]")
    if info:
        console.print(f"  {info.samplerate}Hz, {info.channels}ch, {info.duration:.1f}s")


def cmd_info(audio_path: str):
    """Mostra info dettagliate su un file audio."""
    path = Path(audio_path)
    if not path.exists():
        console.print(f"[red]File non trovato: {audio_path}[/red]")
        sys.exit(1)

    info = get_audio_info(path)

    console.print(Panel(
        f"[bold]Info Audio: {path.name}[/bold]\n\n"
        f"Durata: {info['duration_min']} ({info['duration_sec']:.1f}s)\n"
        f"Sample rate: {info['sample_rate']}Hz\n"
        f"Canali: {info['channels']} ({'stereo' if info['channels'] == 2 else 'mono'})\n"
        f"Formato: {info['format']} ({info['bit_depth']})\n"
        f"Peak: {info['peak_db']:.1f} dB\n"
        f"RMS: {info['rms_db']:.1f} dB\n"
        f"Dimensione: {info['file_size_mb']:.1f} MB",
        style="cyan"
    ))


def cmd_separa(audio_path: str, output: str | None = None,
               model: str = "htdemucs", two_stems: bool = False,
               sample_rate: int | None = None):
    """Separa un file audio in stems."""
    path = Path(audio_path)
    if not path.exists():
        console.print(f"[red]File non trovato: {audio_path}[/red]")
        sys.exit(1)

    if output:
        output_dir = Path(output)
    else:
        output_dir = path.parent / f"{path.stem}_kit"

    separa_stems(path, output_dir, model=model, two_stems=two_stems,
                 sample_rate=sample_rate)


def cmd_converti(audio_path: str, output: str | None = None,
                 formato: str = "wav", sr: int | None = None,
                 mono: bool = False):
    """Converti un file audio."""
    path = Path(audio_path)
    if not path.exists():
        console.print(f"[red]File non trovato: {audio_path}[/red]")
        sys.exit(1)

    out = Path(output) if output else None
    converti_audio(path, out, formato=formato, sample_rate=sr, mono=mono)


def main():
    parser = argparse.ArgumentParser(
        description="Export Kit - Separa, converti e gestisci stems audio"
    )
    subparsers = parser.add_subparsers(dest="command")

    # separa
    p_sep = subparsers.add_parser("separa", help="Separa audio in stems (drums, bass, vocals, etc)")
    p_sep.add_argument("audio", help="File audio da separare (.wav)")
    p_sep.add_argument("--output", "-o", help="Cartella output per il kit")
    p_sep.add_argument("--model", "-m", default="htdemucs",
                       choices=["htdemucs", "htdemucs_6s", "htdemucs_ft"],
                       help="Modello Demucs (default: htdemucs)")
    p_sep.add_argument("--two-stems", action="store_true",
                       help="Solo 2 stems: vocals + instrumental")
    p_sep.add_argument("--sr", type=int, help="Sample rate output (default: originale)")

    # info
    p_info = subparsers.add_parser("info", help="Info su un file audio")
    p_info.add_argument("audio", help="File audio")

    # converti
    p_conv = subparsers.add_parser("converti", help="Converti formato/sample rate")
    p_conv.add_argument("audio", help="File audio da convertire")
    p_conv.add_argument("--output", "-o", help="File output")
    p_conv.add_argument("--formato", "-f", default="wav",
                        choices=["wav", "mp3", "flac"],
                        help="Formato output (default: wav)")
    p_conv.add_argument("--sr", type=int, help="Sample rate output")
    p_conv.add_argument("--mono", action="store_true", help="Converti a mono")

    args = parser.parse_args()

    if args.command == "separa":
        cmd_separa(args.audio, args.output, args.model, args.two_stems, args.sr)
    elif args.command == "info":
        cmd_info(args.audio)
    elif args.command == "converti":
        cmd_converti(args.audio, args.output, args.formato, args.sr, args.mono)
    else:
        parser.print_help()
        console.print(
            "\n[yellow]Esempi:[/yellow]\n"
            "  python export_kit.py separa canzone.wav                    # separa in stems\n"
            "  python export_kit.py separa canzone.wav --model htdemucs_6s  # 6 stems (+ guitar, piano)\n"
            "  python export_kit.py separa canzone.wav -o mio_kit         # output custom\n"
            "  python export_kit.py info canzone.wav                       # info file\n"
            "  python export_kit.py converti canzone.wav -f mp3            # converti a mp3\n"
            "  python export_kit.py converti canzone.wav --sr 44100        # cambio sample rate\n"
        )


if __name__ == "__main__":
    main()
