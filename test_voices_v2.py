"""
Test reference voices con pronuncia italiana REALE.
Metodo 1: Lyria a cappella (canta in italiano)
Metodo 2: Google TTS (pronuncia italiana perfetta)
"""
import sys
import json
import base64
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import patch_torchaudio

import soundfile as sf
import numpy as np
from rich.console import Console
from rich.table import Table

from config import GEMINI_API_KEY, LYRIA_CLIP_MODEL

console = Console()
OUTPUT_DIR = Path(__file__).parent / "voice" / "test_voices_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_lyria_acappella(name: str, prompt: str) -> Path:
    """Genera voce a cappella con Lyria 3 Clip (canta in italiano)."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    console.print(f"  [cyan]Lyria generating: {name}...[/cyan]")

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=LYRIA_CLIP_MODEL,
                contents=prompt,
            )

            if not response.candidates:
                console.print(f"  [yellow]Nessun candidato, riprovo...[/yellow]")
                time.sleep(3)
                continue

            candidate = response.candidates[0]
            if candidate.content is None:
                console.print(f"  [yellow]Contenuto vuoto, riprovo...[/yellow]")
                time.sleep(3)
                continue

            audio_data = None
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    if "audio" in (part.inline_data.mime_type or ""):
                        audio_data = part.inline_data.data
                        break

            if audio_data:
                out_path = OUTPUT_DIR / f"{name}.wav"
                with open(out_path, "wb") as f:
                    f.write(audio_data)
                info = sf.info(str(out_path))
                console.print(f"  [green]✓ {name}: {info.duration:.1f}s[/green]")
                return out_path

            console.print(f"  [yellow]Nessun audio, riprovo...[/yellow]")
            time.sleep(3)

        except Exception as e:
            console.print(f"  [yellow]Errore: {e}[/yellow]")
            time.sleep(3)

    return None


def generate_all_voices():
    """Genera voci diverse con Lyria a cappella."""
    voices = [
        {
            "name": "lyria_baritono_roca",
            "prompt": (
                "A cappella solo male voice, Italian cantautore, "
                "warm hoarse baritone singing in Italian language, "
                "singing words: 'Ricordo il rumore del treno, le rose rosse di Roma, "
                "il profumo della sera, amore mio ritorna', "
                "intimate emotional delivery, close microphone, "
                "no instruments, no reverb, studio recording, "
                "clear perfect Italian pronunciation and diction"
            ),
        },
        {
            "name": "lyria_baritono_caldo",
            "prompt": (
                "A cappella solo male voice singing in Italian, "
                "warm rich baritone like an Italian singer-songwriter, "
                "singing: 'Le strade di Roma si accendono al tramonto, "
                "cammino tra i ricordi e il rumore della sera', "
                "gentle and emotional, perfect Italian pronunciation, "
                "native Italian speaker, close microphone, dry recording, "
                "no instruments, no echo"
            ),
        },
        {
            "name": "lyria_tenore_dolce",
            "prompt": (
                "A cappella male Italian voice, gentle tenor, "
                "singing softly in Italian: 'Tra le onde del mare "
                "e il profumo del vento, ricordo le tue parole, "
                "il tuo sorriso al tramonto', "
                "clear Italian diction, native pronunciation, "
                "intimate and warm, studio quality, no instruments, "
                "close microphone, no reverb"
            ),
        },
        {
            "name": "lyria_cantautore_profondo",
            "prompt": (
                "Solo male voice, deep Italian cantautore style, "
                "singing in perfect Italian: 'Il treno corre rapido "
                "attraverso la campagna, portando con se il ricordo "
                "di un amore perduto tra le rose rosse', "
                "expressive vibrato, emotional delivery, "
                "close microphone, dry studio, no instruments, "
                "native Italian pronunciation"
            ),
        },
        {
            "name": "lyria_indie_rauco",
            "prompt": (
                "A cappella raspy male Italian voice, indie folk style, "
                "singing in Italian language: 'Sotto un cielo di stelle "
                "la chitarra riposa, le parole risuonano "
                "come un ricordo che non muore', "
                "slightly hoarse warm voice, intimate, "
                "perfect Italian R pronunciation, native speaker, "
                "close mic, dry recording, no instruments"
            ),
        },
    ]

    generated = []
    for v in voices:
        path = generate_lyria_acappella(v["name"], v["prompt"])
        if path:
            generated.append(path)
        time.sleep(2)  # Rate limit

    return generated


def separate_vocals(wav_paths):
    """Separa vocals puri con Demucs (rimuovi eventuali strumenti residui)."""
    import subprocess
    clean_paths = []

    for wav_path in wav_paths:
        console.print(f"  [cyan]Demucs: {wav_path.name}...[/cyan]")
        out_dir = OUTPUT_DIR / "separated"
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "-o", str(out_dir),
            str(wav_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            vocals_path = out_dir / "htdemucs" / wav_path.stem / "vocals.wav"
            if vocals_path.exists():
                # Copia nella cartella principale
                clean_path = OUTPUT_DIR / f"{wav_path.stem}_clean.wav"
                import shutil
                shutil.copy2(vocals_path, clean_path)
                clean_paths.append(clean_path)
                console.print(f"  [green]✓ {wav_path.stem}_clean.wav[/green]")
        else:
            console.print(f"  [yellow]Demucs fallito per {wav_path.name}[/yellow]")
            clean_paths.append(wav_path)  # usa originale

    return clean_paths


def score_with_gemini(wav_paths):
    """Valuta pronuncia italiana con Gemini."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    results = []
    for wav_path in wav_paths:
        name = wav_path.stem
        console.print(f"  [cyan]Scoring {name}...[/cyan]")

        with open(wav_path, 'rb') as f:
            audio_data = f.read()

        prompt = '''Ascolta questa voce che canta in italiano.
Valuta la pronuncia italiana, specialmente la R (deve essere vibrante, NON gutturale).
JSON: {"r_quality":X, "native_likeness":X, "pronunciation":X, "voice_quality":X,
"accent":"descrizione", "notes":"commento breve"}
1-10. 10=madrelingua perfetto.'''

        for model in ['gemini-2.5-pro', 'gemini-2.5-flash']:
            try:
                r = client.models.generate_content(model=model, contents=[
                    {'inline_data': {'mime_type': 'audio/wav',
                                     'data': base64.b64encode(audio_data).decode()}},
                    prompt
                ])
                text = r.text.strip()
                if text.startswith("```"): text = text.split("\n", 1)[1]
                if text.endswith("```"): text = text.rsplit("```", 1)[0]
                scores = json.loads(text.strip())
                scores["name"] = name
                results.append(scores)
                console.print(f"    R:{scores.get('r_quality','?')}/10 "
                             f"Native:{scores.get('native_likeness','?')}/10 "
                             f"- {scores.get('accent','?')[:50]}")
                break
            except Exception as e:
                if model == 'gemini-2.5-flash':
                    console.print(f"    [red]Errore: {e}[/red]")

    # Tabella
    table = Table(title="Test Voci v2 - Pronuncia Italiana (Lyria)")
    table.add_column("Voce", style="cyan")
    table.add_column("R", justify="center")
    table.add_column("Native", justify="center")
    table.add_column("Pronuncia", justify="center")
    table.add_column("Qualità voce", justify="center")
    table.add_column("Accento")

    for r in sorted(results, key=lambda x: x.get("r_quality", 0) + x.get("native_likeness", 0), reverse=True):
        table.add_row(
            r["name"][:25],
            f"{r.get('r_quality', '?')}",
            f"{r.get('native_likeness', '?')}",
            f"{r.get('pronunciation', '?')}",
            f"{r.get('voice_quality', '?')}",
            r.get("accent", "?")[:45]
        )
    console.print(table)

    # Salva
    with open(OUTPUT_DIR / "scores.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        best = max(results, key=lambda x: x.get("r_quality", 0) + x.get("native_likeness", 0))
        console.print(f"\n[bold green]Migliore: {best['name']}[/bold green]")
        return best["name"]
    return None


if __name__ == "__main__":
    console.print("[bold magenta]Test Voci v2 - Lyria A Cappella Italiana[/bold magenta]\n")

    # Step 1: Genera voci a cappella con Lyria
    console.print("[bold]1. Generazione voci a cappella...[/bold]")
    wav_paths = generate_all_voices()

    if not wav_paths:
        console.print("[red]Nessuna voce generata![/red]")
        sys.exit(1)

    # Step 2: Separa vocals puri (rimuovi strumenti residui)
    console.print("\n[bold]2. Pulizia vocals con Demucs...[/bold]")
    clean_paths = separate_vocals(wav_paths)

    # Step 3: Valuta con Gemini
    console.print("\n[bold]3. Valutazione Gemini...[/bold]")
    best = score_with_gemini(clean_paths)

    if best:
        best_file = OUTPUT_DIR / f"{best}.wav"
        console.print(f"\n[bold]Per usare la voce migliore:[/bold]")
        console.print(f"  cp {best_file} voice/artist_voice.wav")
