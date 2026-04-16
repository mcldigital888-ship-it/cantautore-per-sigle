"""
Import Base - Cantautore Digitale
==================================
Importa basi/beat esterni e ci canta sopra con la tua voce.
Puoi portare una base da YouTube, un beat comprato, una strumentale tua,
e il sistema genera testi + ci canta sopra con la tua voce.

Uso:
  # Canta sulla tua base con testo generato
  python import_base.py --base beat.wav --tema "amore perduto a Roma"

  # Canta con la tua voce
  python import_base.py --base beat.wav --tema "notte in città" --voce mia_voce

  # Usa testo personalizzato (scritto da te)
  python import_base.py --base beat.wav --testo mio_testo.txt --voce mia_voce

  # Genera solo il testo (senza musica) per una base
  python import_base.py --base beat.wav --tema "nostalgia" --solo-testo

  # Analizza una base e suggerisci tema/mood
  python import_base.py --analizza beat.wav
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_DIR = Path(__file__).parent
TEMP_DIR = PROJECT_DIR / "temp"
OUTPUT_DIR = PROJECT_DIR / "output"
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def analizza_base(base_path: Path) -> dict:
    """Analizza una base strumentale con Gemini per suggerire tema/mood/BPM."""
    import base64
    from google import genai
    from config import GEMINI_API_KEY

    client = genai.Client(api_key=GEMINI_API_KEY)

    console.print("[cyan]Analisi della base con AI...[/cyan]")

    with open(base_path, 'rb') as f:
        audio_data = f.read()

    prompt = '''Analizza questa base strumentale/beat. Rispondi SOLO in JSON:
{
  "bpm_stimato": X,
  "tonalita": "es: Am, C, Dm...",
  "genere": "genere musicale",
  "mood": "mood in inglese per produzione",
  "energia": "low/medium/high",
  "strumenti_rilevati": ["strumento1", "strumento2"],
  "temi_suggeriti": ["tema1 per il testo", "tema2", "tema3"],
  "stile_vocale_consigliato": "descrizione dello stile vocale adatto",
  "note": "commento breve sulla produzione"
}'''

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

    analysis = json.loads(text.strip())
    return analysis


def genera_testo_per_base(base_path: Path, tema: str,
                          analysis: dict | None = None) -> dict:
    """Genera testo ottimizzato per una base specifica."""
    from google import genai
    from config import GEMINI_API_KEY, ARTIST
    from artist_brain import ArtistBrain

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Analizza la base se non già fatto
    if analysis is None:
        analysis = analizza_base(base_path)

    brain = ArtistBrain("cantautore_digitale", PROJECT_DIR)
    dna_prompt = brain.get_dna_prompt()
    memory_prompt = brain.get_memory_prompt()

    bpm = analysis.get("bpm_stimato", 100)
    mood = analysis.get("mood", "melancholic")
    genere = analysis.get("genere", "unknown")
    tonalita = analysis.get("tonalita", "unknown")
    stile_vocale = analysis.get("stile_vocale_consigliato", "")

    prompt = f"""Sei un cantautore italiano. Devi scrivere un testo per una BASE GIÀ ESISTENTE.
NON stai creando la musica — la musica è già fatta. Tu scrivi SOLO il testo che va cantato sopra.

{dna_prompt}

{memory_prompt}

INFORMAZIONI SULLA BASE:
- BPM: {bpm}
- Tonalità: {tonalita}
- Genere: {genere}
- Mood: {mood}
- Stile vocale consigliato: {stile_vocale}

ARTISTA: {ARTIST['name']}
TEMA RICHIESTO: {tema}

REGOLE CRITICHE PER CANTARE SU UNA BASE ESISTENTE:
1. Il RITMO delle parole deve combaciare col BPM ({bpm}). Conta le sillabe.
2. Le frasi devono avere RESPIRI — non puoi cantare frasi infinite su un beat
3. Lascia SPAZI per la strumentale — non riempire ogni secondo
4. Il mood del testo DEVE combaciare col mood della base ({mood})
5. Struttura classica: [Verse 1] [Chorus] [Verse 2] [Chorus] [Bridge] [Chorus] [Outro]

STRUTTURA (adatta al BPM {bpm}):
[Verse 1] (6-8 righe, frasi che seguono il groove)
[Chorus] (4-5 righe, hook memorabile)
[Verse 2] (6-8 righe, nuova prospettiva)
[Chorus]
[Bridge] (3-4 righe, cambio emotivo)
[Chorus] (variazione)
[Outro] (2-3 righe che sfumano)

Rispondi SOLO in JSON:
{{
    "titolo": "...",
    "testo": "[Verse 1]\\n...\\n\\n[Chorus]\\n...",
    "mood": "{mood}",
    "bpm": {bpm}
}}"""

    console.print("[cyan]Generazione testo per la base...[/cyan]")
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]

    result = json.loads(text.strip())
    console.print(f"  [green]✓ \"{result['titolo']}\" - {result.get('mood', '')}, {result.get('bpm', '')} BPM[/green]")

    # Aggiorna memoria
    brain.digest_song(result, tema)

    return result


def genera_vocals_su_base(base_path: Path, song_data: dict,
                          output_dir: Path) -> Path:
    """Genera vocals cantati usando Lyria e li sovrappone alla base."""
    from google import genai
    from config import GEMINI_API_KEY, LYRIA_MODEL, ARTIST
    from cantautore import pulisci_testo_per_lyria

    client = genai.Client(api_key=GEMINI_API_KEY)

    testo = pulisci_testo_per_lyria(song_data["testo"])
    style = ARTIST["style_prompt"]
    mood = song_data.get("mood", "melancholic")
    bpm = song_data.get("bpm", 100)

    # Genera canzone con Lyria (con testo)
    prompt = f"""{style}.
Mood: {mood}. Tempo: {bpm} BPM.
Create a full song with these lyrics. Focus on the VOCAL PERFORMANCE.
The singer should be expressive, emotional, with clear Italian diction.
Full production with dynamic arrangement.

Lyrics:
{testo}"""

    console.print("[cyan]Generazione vocals con Lyria 3 Pro...[/cyan]")
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=LYRIA_MODEL,
                contents=prompt,
            )

            if not response.candidates:
                console.print("[yellow]  Nessun candidato, riprovo...[/yellow]")
                import time; time.sleep(5)
                continue

            candidate = response.candidates[0]
            if candidate.content is None:
                import time; time.sleep(5)
                continue

            audio_data = None
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    if "audio" in (part.inline_data.mime_type or ""):
                        audio_data = part.inline_data.data
                        break

            if audio_data:
                break

            import time; time.sleep(5)

        except Exception as e:
            console.print(f"[yellow]  Errore: {e}[/yellow]")
            import time; time.sleep(5)
    else:
        raise RuntimeError("Lyria non ha generato audio dopo 3 tentativi")

    # Salva il raw generato
    raw_path = output_dir / "lyria_raw.wav"
    with open(raw_path, "wb") as f:
        f.write(audio_data)

    console.print(f"  [green]✓ Audio generato[/green]")
    return raw_path


def mixa_vocals_su_base(base_path: Path, vocals_path: Path,
                        output_path: Path,
                        vocal_gain_db: float = -1.0,
                        base_gain_db: float = 0.0) -> Path:
    """Mixa vocals estratti sulla base originale importata."""
    from scipy import signal as sig
    from scipy.ndimage import uniform_filter1d

    console.print("[cyan]Mix vocals sulla base...[/cyan]")

    base, sr_base = sf.read(str(base_path))
    vocals, sr_voc = sf.read(str(vocals_path))

    # Resample se necessario
    if sr_voc != sr_base:
        import librosa
        if vocals.ndim > 1:
            vocals = np.stack([
                librosa.resample(vocals[:, ch], orig_sr=sr_voc, target_sr=sr_base)
                for ch in range(vocals.shape[1])
            ], axis=-1)
        else:
            vocals = librosa.resample(vocals, orig_sr=sr_voc, target_sr=sr_base)

    # Stereo
    if base.ndim == 1:
        base = np.stack([base, base], axis=-1)
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=-1)

    # Allinea lunghezze (usa la base come riferimento)
    if len(vocals) > len(base):
        vocals = vocals[:len(base)]
    elif len(vocals) < len(base):
        pad = np.zeros((len(base) - len(vocals), vocals.shape[1]))
        vocals = np.concatenate([vocals, pad])

    # Vocal processing: high-pass 80Hz
    sos_hp = sig.butter(4, 80, btype='high', fs=sr_base, output='sos')
    vocals = sig.sosfilt(sos_hp, vocals, axis=0)

    # Presenza vocale: boost 2-5kHz
    sos_pres = sig.butter(2, [2000, 5000], btype='band', fs=sr_base, output='sos')
    presence = sig.sosfilt(sos_pres, vocals, axis=0)
    vocals = vocals + 0.2 * presence

    # Sidechain ducking sulla base
    vocal_mono = np.mean(np.abs(vocals), axis=1)
    envelope = uniform_filter1d(vocal_mono, size=int(sr_base * 0.05))
    duck_amount = np.clip(envelope / (np.max(envelope) + 1e-10), 0, 1)
    duck_gain = 1.0 - 0.35 * duck_amount
    base = base * duck_gain[:, np.newaxis]

    # Gain
    base = base * (10 ** (base_gain_db / 20))
    vocals = vocals * (10 ** (vocal_gain_db / 20))

    # Mix
    mixed = base + vocals

    # Mastering leggero
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95
    mixed = np.tanh(mixed * 1.05) / np.tanh(1.05)
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed * (0.89 / peak)

    sf.write(str(output_path), mixed, sr_base, subtype='PCM_24')

    info = sf.info(str(output_path))
    console.print(f"  [green]✓ Mix finale: {info.duration:.1f}s[/green]")
    return output_path


def pipeline_importa_base(base_path: str, tema: str | None = None,
                          testo_file: str | None = None,
                          voice_name: str | None = None,
                          solo_testo: bool = False):
    """Pipeline completa: importa base → genera testo → canta → mixa."""
    from config import get_voice_reference

    base = Path(base_path)
    if not base.exists():
        console.print(f"[red]File base non trovato: {base_path}[/red]")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = TEMP_DIR / f"import_{timestamp}"
    work_dir.mkdir(exist_ok=True)

    console.print(Panel(
        f"[bold]Import Base[/bold]\n"
        f"Base: {base.name}\n"
        f"Tema: {tema or '(da analisi)'}\n"
        f"Voce: {voice_name or 'default'}",
        style="magenta"
    ))

    # Step 1: Analizza la base
    console.print("\n[bold]Step 1: Analisi base[/bold]")
    analysis = analizza_base(base)

    console.print(f"  BPM: {analysis.get('bpm_stimato', '?')}")
    console.print(f"  Tonalità: {analysis.get('tonalita', '?')}")
    console.print(f"  Genere: {analysis.get('genere', '?')}")
    console.print(f"  Mood: {analysis.get('mood', '?')}")
    console.print(f"  Strumenti: {', '.join(analysis.get('strumenti_rilevati', []))}")

    if tema is None:
        temi = analysis.get("temi_suggeriti", ["la vita notturna"])
        tema = temi[0]
        console.print(f"  [cyan]Tema auto-suggerito: {tema}[/cyan]")

    # Step 2: Genera o carica testo
    console.print("\n[bold]Step 2: Testo[/bold]")
    if testo_file:
        # Carica testo personalizzato
        testo_path = Path(testo_file)
        if not testo_path.exists():
            console.print(f"[red]File testo non trovato: {testo_file}[/red]")
            sys.exit(1)
        testo_raw = testo_path.read_text(encoding="utf-8")
        song_data = {
            "titolo": testo_path.stem.replace("_", " ").title(),
            "testo": testo_raw,
            "mood": analysis.get("mood", "melancholic"),
            "bpm": analysis.get("bpm_stimato", 100),
        }
        console.print(f"  [green]✓ Testo caricato da {testo_path.name}[/green]")
    else:
        song_data = genera_testo_per_base(base, tema, analysis)

    # Salva testo
    testo_json = work_dir / "testo.json"
    with open(testo_json, "w", encoding="utf-8") as f:
        json.dump(song_data, f, ensure_ascii=False, indent=2)

    if solo_testo:
        # Mostra il testo e esci
        console.print(Panel(
            f"[bold]{song_data['titolo']}[/bold]\n\n{song_data['testo']}",
            style="green"
        ))
        testo_output = OUTPUT_DIR / f"{song_data['titolo'].replace(' ', '_')}_{timestamp}_testo.json"
        shutil.copy2(testo_json, testo_output)
        console.print(f"[green]✓ Testo salvato: {testo_output}[/green]")
        return

    # Step 3: Genera vocals con Lyria
    console.print("\n[bold]Step 3: Generazione vocals[/bold]")
    raw_path = genera_vocals_su_base(base, song_data, work_dir)

    # Step 4: Separa vocals dal generato di Lyria
    console.print("\n[bold]Step 4: Estrazione vocals[/bold]")
    from cantautore import separa_vocals
    separated_dir = work_dir / "separated"
    vocals_path, _ = separa_vocals(raw_path, separated_dir)

    # Step 5: Voice Conversion (se voce disponibile)
    try:
        voice_ref = get_voice_reference(voice_name)
        console.print(f"\n[bold]Step 5: Voice Conversion[/bold] ({voice_ref.parent.name})")
        from cantautore import converti_voce
        converted_path = work_dir / "vocals_converted.wav"
        converti_voce(vocals_path, converted_path, voice_name)
        vocals_path = converted_path
    except FileNotFoundError:
        console.print("\n[yellow]  Voice conversion saltata (nessuna voce di riferimento)[/yellow]")

    # Step 6: Mixa vocals sulla BASE ORIGINALE importata
    console.print("\n[bold]Step 6: Mix sulla tua base[/bold]")
    final_path = work_dir / "final.wav"
    mixa_vocals_su_base(base, vocals_path, final_path)

    # Step 7: Esporta kit completo
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in song_data["titolo"])
    kit_dir = OUTPUT_DIR / f"{safe_title}_{timestamp}_kit"
    kit_dir.mkdir(parents=True, exist_ok=True)

    # Copia tutti i file nel kit
    shutil.copy2(str(final_path), str(kit_dir / "mix_finale.wav"))
    shutil.copy2(str(base), str(kit_dir / "base_originale.wav"))
    shutil.copy2(str(vocals_path), str(kit_dir / "vocals.wav"))
    shutil.copy2(str(testo_json), str(kit_dir / "testo.json"))

    # Salva anche analisi
    with open(kit_dir / "analisi_base.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    console.print(Panel(
        f"[bold green]Canzone completata![/bold green]\n\n"
        f"Titolo: {song_data['titolo']}\n"
        f"Kit: {kit_dir}\n\n"
        f"File nel kit:\n"
        f"  mix_finale.wav      — canzone finita\n"
        f"  base_originale.wav  — la tua base\n"
        f"  vocals.wav          — solo voce (per DAW)\n"
        f"  testo.json          — testo della canzone\n"
        f"  analisi_base.json   — analisi AI della base",
        style="green"
    ))

    return kit_dir


def main():
    parser = argparse.ArgumentParser(
        description="Import Base - Canta sulla tua base con AI"
    )
    parser.add_argument("--base", type=str, help="File audio della base/beat (.wav)")
    parser.add_argument("--tema", type=str, help="Tema per il testo (o auto-suggerito)")
    parser.add_argument("--testo", type=str, help="File .txt con testo personalizzato")
    parser.add_argument("--voce", type=str, help="Nome profilo voce (es: mia_voce)")
    parser.add_argument("--solo-testo", action="store_true",
                        help="Genera solo il testo, senza musica")
    parser.add_argument("--analizza", type=str,
                        help="Analizza una base e mostra suggerimenti")

    args = parser.parse_args()

    if args.analizza:
        path = Path(args.analizza)
        if not path.exists():
            console.print(f"[red]File non trovato: {args.analizza}[/red]")
            sys.exit(1)
        analysis = analizza_base(path)
        console.print(Panel(
            f"[bold]Analisi Base: {path.name}[/bold]\n\n"
            f"BPM: {analysis.get('bpm_stimato', '?')}\n"
            f"Tonalità: {analysis.get('tonalita', '?')}\n"
            f"Genere: {analysis.get('genere', '?')}\n"
            f"Mood: {analysis.get('mood', '?')}\n"
            f"Energia: {analysis.get('energia', '?')}\n"
            f"Strumenti: {', '.join(analysis.get('strumenti_rilevati', []))}\n"
            f"\nTemi suggeriti:\n" +
            "\n".join(f"  - {t}" for t in analysis.get("temi_suggeriti", [])) +
            f"\n\nStile vocale: {analysis.get('stile_vocale_consigliato', '?')}\n"
            f"Note: {analysis.get('note', '?')}",
            style="cyan"
        ))
    elif args.base:
        pipeline_importa_base(
            args.base,
            tema=args.tema,
            testo_file=args.testo,
            voice_name=args.voce,
            solo_testo=args.solo_testo,
        )
    else:
        parser.print_help()
        console.print(
            "\n[yellow]Esempi:[/yellow]\n"
            "  python import_base.py --analizza beat.wav\n"
            "  python import_base.py --base beat.wav --tema 'amore perduto'\n"
            "  python import_base.py --base beat.wav --tema 'notte' --voce mia_voce\n"
            "  python import_base.py --base beat.wav --testo mio_testo.txt\n"
            "  python import_base.py --base beat.wav --tema 'Roma' --solo-testo\n"
        )


if __name__ == "__main__":
    main()
