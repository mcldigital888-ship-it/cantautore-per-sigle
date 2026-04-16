"""
Gestione Voci - Cantautore Digitale
====================================
Aggiungi, rimuovi, testa e seleziona profili voce.

Uso:
  python manage_voices.py list                              # lista voci disponibili
  python manage_voices.py add mia_voce registrazione.wav    # aggiungi la tua voce
  python manage_voices.py add voce_soul altra_voce.wav      # aggiungi altra voce
  python manage_voices.py test mia_voce                     # testa la voce con Gemini
  python manage_voices.py set-default mia_voce              # imposta come default
  python manage_voices.py info mia_voce                     # info su una voce
  python manage_voices.py remove voce_vecchia               # rimuovi un profilo
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROJECT_DIR = Path(__file__).parent
VOICES_DIR = PROJECT_DIR / "voices"
VOICES_DIR.mkdir(exist_ok=True)

# Vecchia cartella voice per compatibilità
LEGACY_VOICE_DIR = PROJECT_DIR / "voice"

CONFIG_FILE = VOICES_DIR / "voices_config.json"


def load_config() -> dict:
    """Carica la configurazione delle voci."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"default_voice": None, "profiles": {}}


def save_config(config: dict):
    """Salva la configurazione delle voci."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_voice_path(voice_name: str) -> Path | None:
    """Ritorna il path del file reference per una voce."""
    voice_dir = VOICES_DIR / voice_name
    ref = voice_dir / "reference.wav"
    if ref.exists():
        return ref
    # Fallback: cerca qualsiasi .wav nella cartella
    wavs = list(voice_dir.glob("*.wav"))
    if wavs:
        return wavs[0]
    # Legacy: vecchia cartella voice/
    legacy = LEGACY_VOICE_DIR / "artist_voice.wav"
    if voice_name == "default" and legacy.exists():
        return legacy
    return None


def validate_audio(wav_path: Path) -> dict:
    """Valida un file audio e ritorna info."""
    try:
        info = sf.info(str(wav_path))
        data, sr = sf.read(str(wav_path))

        # Calcola RMS per verificare che non sia silenzio
        rms = np.sqrt(np.mean(data ** 2))

        result = {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "rms": float(rms),
            "is_silent": rms < 0.001,
            "is_too_short": info.duration < 5,
            "is_too_long": info.duration > 120,
            "is_good_quality": info.samplerate >= 16000,
        }

        # Warnings
        warnings = []
        if result["is_silent"]:
            warnings.append("Il file sembra silenzioso — verifica la registrazione")
        if result["is_too_short"]:
            warnings.append(f"Troppo corto ({info.duration:.1f}s) — minimo 10 secondi consigliati")
        if result["is_too_long"]:
            warnings.append(f"Molto lungo ({info.duration:.1f}s) — verrà trimmato a 60s per Seed-VC")
        if not result["is_good_quality"]:
            warnings.append(f"Sample rate basso ({info.samplerate}Hz) — consigliato almeno 16kHz")
        if info.duration >= 10 and info.duration <= 60 and rms > 0.01 and info.samplerate >= 16000:
            warnings.append("✓ Qualità ottima per voice conversion")

        result["warnings"] = warnings
        return result

    except Exception as e:
        return {"error": str(e)}


def cmd_add(name: str, wav_path: str):
    """Aggiungi un profilo voce."""
    source = Path(wav_path)
    if not source.exists():
        console.print(f"[red]File non trovato: {wav_path}[/red]")
        sys.exit(1)

    # Valida
    console.print(f"[cyan]Analisi audio: {source.name}...[/cyan]")
    info = validate_audio(source)

    if "error" in info:
        console.print(f"[red]Errore: {info['error']}[/red]")
        sys.exit(1)

    console.print(f"  Durata: {info['duration']:.1f}s")
    console.print(f"  Sample rate: {info['sample_rate']}Hz")
    console.print(f"  Canali: {info['channels']}")
    for w in info.get("warnings", []):
        color = "green" if "✓" in w else "yellow"
        console.print(f"  [{color}]{w}[/{color}]")

    # Crea cartella profilo
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Copia file
    dest = voice_dir / "reference.wav"
    shutil.copy2(str(source), str(dest))

    # Se è troppo lungo, crea anche una versione trimmata
    if info["duration"] > 60:
        data, sr = sf.read(str(source))
        trimmed = data[:int(60 * sr)]
        sf.write(str(voice_dir / "reference_trimmed.wav"), trimmed, sr)
        console.print(f"  [dim]Creata versione trimmata (60s) per Seed-VC[/dim]")

    # Aggiorna config
    config = load_config()
    config["profiles"][name] = {
        "added": datetime.now().isoformat(),
        "duration": info["duration"],
        "sample_rate": info["sample_rate"],
        "source_file": source.name,
    }

    # Se è la prima voce, impostala come default
    if config["default_voice"] is None:
        config["default_voice"] = name
        console.print(f"  [green]Impostata come voce default[/green]")

    save_config(config)

    # Copia anche nella vecchia cartella per compatibilità
    LEGACY_VOICE_DIR.mkdir(exist_ok=True)
    if config["default_voice"] == name:
        shutil.copy2(str(dest), str(LEGACY_VOICE_DIR / "artist_voice.wav"))

    console.print(f"\n[green]✓ Voce '{name}' aggiunta![/green]")
    console.print(f"  Usa con: [cyan]python cantautore.py --tema '...' --voce {name}[/cyan]")


def cmd_list():
    """Lista tutti i profili voce."""
    config = load_config()
    default = config.get("default_voice")

    table = Table(title="Profili Voce Disponibili")
    table.add_column("Nome", style="cyan")
    table.add_column("Default", justify="center")
    table.add_column("Durata", justify="right")
    table.add_column("Qualità", justify="center")
    table.add_column("Aggiunta")

    # Cerca cartelle in voices/
    found = False
    for voice_dir in sorted(VOICES_DIR.iterdir()):
        if not voice_dir.is_dir():
            continue

        name = voice_dir.name
        ref = get_voice_path(name)

        if ref and ref.exists():
            found = True
            info = validate_audio(ref)
            duration = f"{info.get('duration', 0):.1f}s" if "duration" in info else "?"
            quality = "✓" if info.get("is_good_quality") and not info.get("is_silent") else "⚠"
            is_default = "★" if name == default else ""

            profile = config.get("profiles", {}).get(name, {})
            added = profile.get("added", "?")[:10]

            table.add_row(name, is_default, duration, quality, added)

    # Controlla anche la legacy voice
    legacy = LEGACY_VOICE_DIR / "artist_voice.wav"
    if legacy.exists() and not found:
        info = validate_audio(legacy)
        duration = f"{info.get('duration', 0):.1f}s"
        table.add_row("(legacy)", "★", duration, "✓", "originale")
        found = True

    if found:
        console.print(table)
    else:
        console.print("[yellow]Nessuna voce trovata.[/yellow]")
        console.print("Aggiungi la tua voce con:")
        console.print("  [cyan]python manage_voices.py add mia_voce registrazione.wav[/cyan]")


def cmd_set_default(name: str):
    """Imposta una voce come default."""
    ref = get_voice_path(name)
    if not ref:
        console.print(f"[red]Voce '{name}' non trovata[/red]")
        sys.exit(1)

    config = load_config()
    config["default_voice"] = name
    save_config(config)

    # Copia nella legacy dir per compatibilità
    LEGACY_VOICE_DIR.mkdir(exist_ok=True)
    shutil.copy2(str(ref), str(LEGACY_VOICE_DIR / "artist_voice.wav"))

    console.print(f"[green]✓ Voce default impostata: {name}[/green]")


def cmd_info(name: str):
    """Mostra info dettagliate su una voce."""
    ref = get_voice_path(name)
    if not ref:
        console.print(f"[red]Voce '{name}' non trovata[/red]")
        sys.exit(1)

    info = validate_audio(ref)
    config = load_config()
    profile = config.get("profiles", {}).get(name, {})

    console.print(Panel(
        f"[bold]Profilo Voce: {name}[/bold]\n\n"
        f"File: {ref}\n"
        f"Durata: {info.get('duration', '?'):.1f}s\n"
        f"Sample rate: {info.get('sample_rate', '?')}Hz\n"
        f"Canali: {info.get('channels', '?')}\n"
        f"RMS: {info.get('rms', 0):.4f}\n"
        f"Default: {'sì' if config.get('default_voice') == name else 'no'}\n"
        f"Aggiunta: {profile.get('added', '?')[:19]}",
        style="cyan"
    ))

    for w in info.get("warnings", []):
        color = "green" if "✓" in w else "yellow"
        console.print(f"  [{color}]{w}[/{color}]")


def cmd_test(name: str):
    """Testa la qualità della voce con Gemini."""
    ref = get_voice_path(name)
    if not ref:
        console.print(f"[red]Voce '{name}' non trovata[/red]")
        sys.exit(1)

    console.print(f"[cyan]Test voce '{name}' con Gemini...[/cyan]")

    try:
        import base64
        from google import genai
        from config import GEMINI_API_KEY

        client = genai.Client(api_key=GEMINI_API_KEY)

        with open(ref, 'rb') as f:
            audio_data = f.read()

        prompt = '''Analizza questa voce. Rispondi in JSON:
{
  "language": "lingua rilevata",
  "gender": "M/F",
  "estimated_age": "range età",
  "voice_quality": X,
  "clarity": X,
  "background_noise": X,
  "suitability_for_singing": X,
  "notes": "commento breve"
}
Punteggi 1-10. 10 = ottimo.'''

        for model_name in ['gemini-2.5-flash', 'gemini-2.5-pro']:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {'inline_data': {'mime_type': 'audio/wav',
                                         'data': base64.b64encode(audio_data).decode()}},
                        prompt
                    ]
                )
                break
            except Exception:
                continue

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        scores = json.loads(text.strip())
        console.print(Panel(
            f"[bold]Analisi Voce: {name}[/bold]\n\n"
            f"Lingua: {scores.get('language', '?')}\n"
            f"Genere: {scores.get('gender', '?')}\n"
            f"Età stimata: {scores.get('estimated_age', '?')}\n"
            f"Qualità voce: {scores.get('voice_quality', '?')}/10\n"
            f"Chiarezza: {scores.get('clarity', '?')}/10\n"
            f"Rumore fondo: {scores.get('background_noise', '?')}/10\n"
            f"Adatta al canto: {scores.get('suitability_for_singing', '?')}/10\n"
            f"Note: {scores.get('notes', '?')}",
            style="green"
        ))

    except Exception as e:
        console.print(f"[red]Errore nel test: {e}[/red]")
        console.print("[yellow]Assicurati di avere GEMINI_API_KEY nel file .env[/yellow]")


def cmd_remove(name: str):
    """Rimuovi un profilo voce."""
    voice_dir = VOICES_DIR / name
    if not voice_dir.exists():
        console.print(f"[red]Voce '{name}' non trovata[/red]")
        sys.exit(1)

    shutil.rmtree(voice_dir)

    config = load_config()
    config["profiles"].pop(name, None)
    if config.get("default_voice") == name:
        config["default_voice"] = None
        # Imposta la prossima voce disponibile come default
        for d in VOICES_DIR.iterdir():
            if d.is_dir() and (d / "reference.wav").exists():
                config["default_voice"] = d.name
                break
    save_config(config)

    console.print(f"[green]✓ Voce '{name}' rimossa[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Gestione profili voce - Cantautore Digitale"
    )
    subparsers = parser.add_subparsers(dest="command")

    # list
    subparsers.add_parser("list", help="Lista voci disponibili")

    # add
    p_add = subparsers.add_parser("add", help="Aggiungi profilo voce")
    p_add.add_argument("name", help="Nome del profilo (es: mia_voce)")
    p_add.add_argument("wav_file", help="Path al file .wav della registrazione")

    # set-default
    p_def = subparsers.add_parser("set-default", help="Imposta voce default")
    p_def.add_argument("name", help="Nome del profilo")

    # info
    p_info = subparsers.add_parser("info", help="Info su un profilo voce")
    p_info.add_argument("name", help="Nome del profilo")

    # test
    p_test = subparsers.add_parser("test", help="Testa voce con Gemini AI")
    p_test.add_argument("name", help="Nome del profilo")

    # remove
    p_rm = subparsers.add_parser("remove", help="Rimuovi profilo voce")
    p_rm.add_argument("name", help="Nome del profilo")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "add":
        cmd_add(args.name, args.wav_file)
    elif args.command == "set-default":
        cmd_set_default(args.name)
    elif args.command == "info":
        cmd_info(args.name)
    elif args.command == "test":
        cmd_test(args.name)
    elif args.command == "remove":
        cmd_remove(args.name)
    else:
        parser.print_help()
        console.print(
            "\n[yellow]Esempi:[/yellow]\n"
            "  python manage_voices.py list\n"
            "  python manage_voices.py add mia_voce registrazione.wav\n"
            "  python manage_voices.py test mia_voce\n"
            "  python manage_voices.py set-default mia_voce\n"
        )


if __name__ == "__main__":
    main()
