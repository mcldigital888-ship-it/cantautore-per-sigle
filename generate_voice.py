"""
Genera la voce unica dell'artista digitale usando CosyVoice.
Esegui una volta sola: python generate_voice.py

Genera un file audio di riferimento (artist_voice.wav) che verrà usato
come reference per Seed-VC in tutte le canzoni future.
"""
import sys
import os
import shutil
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from config import ARTIST, VOICE_DIR, VOICE_REFERENCE_FILE, MODELS_DIR, DEVICE

console = Console()


def add_cosyvoice_to_path():
    """Aggiunge CosyVoice al Python path."""
    cosyvoice_dir = MODELS_DIR / "CosyVoice"
    if not cosyvoice_dir.exists():
        console.print("[red]CosyVoice non trovato! Esegui prima: python setup_models.py[/red]")
        sys.exit(1)

    # Aggiungi al path
    sys.path.insert(0, str(cosyvoice_dir))
    # CosyVoice ha bisogno di third_party nel path
    third_party = cosyvoice_dir / "third_party" / "Matcha-TTS"
    if third_party.exists():
        sys.path.insert(0, str(third_party))


def generate_voice_cosyvoice():
    """Genera la voce dell'artista con CosyVoice instruct mode."""
    add_cosyvoice_to_path()

    from cosyvoice.cli.cosyvoice import CosyVoice2

    console.print(Panel(
        f"[bold]Generazione voce per: {ARTIST['name']}[/bold]\n\n"
        f"Descrizione: {ARTIST['voice_description']}",
        style="cyan"
    ))

    model_dir = str(MODELS_DIR / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
    console.print("[yellow]Caricamento modello CosyVoice2...[/yellow]")
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False)

    # Testi italiani che l'artista "direbbe" - per catturare il timbro
    sample_texts = [
        "Le strade di Roma si accendono al tramonto, "
        "e ogni angolo racconta una storia che nessuno ha mai scritto. "
        "Cammino tra i vicoli stretti, cercando quella melodia "
        "che mi porto dentro da sempre.",

        "Il treno parte e il paesaggio scorre come un sogno, "
        "lascio dietro di me le luci della citta. "
        "Il vento porta con se il profumo del mare, "
        "e io chiudo gli occhi, lasciandomi cullare.",

        "Sotto un cielo di stelle, la chitarra suona piano, "
        "le parole escono come un fiume in piena. "
        "Ogni nota e un ricordo, ogni accordo un abbraccio, "
        "la musica e il mio modo di dirti che ci sono.",
    ]

    all_audio = []
    for i, text in enumerate(sample_texts):
        console.print(f"[cyan]Generazione campione {i+1}/{len(sample_texts)}...[/cyan]")

        # Instruct mode: genera voce basata sulla descrizione
        instruct_text = ARTIST["voice_description"]

        for chunk in model.inference_instruct2(
            text=text,
            instruct_text=instruct_text,
            stream=False,
        ):
            all_audio.append(chunk["tts_speech"])

    # Concatena tutti i campioni
    full_audio = torch.cat(all_audio, dim=-1)

    # Assicura formato corretto e salva
    if full_audio.dim() == 1:
        full_audio = full_audio.unsqueeze(0)

    sample_rate = 22050  # CosyVoice default
    torchaudio.save(str(VOICE_REFERENCE_FILE), full_audio, sample_rate)

    duration = full_audio.shape[-1] / sample_rate
    console.print(f"\n[green]✓ Voce generata: {VOICE_REFERENCE_FILE}[/green]")
    console.print(f"[green]  Durata: {duration:.1f} secondi[/green]")
    console.print(f"[green]  Sample rate: {sample_rate} Hz[/green]")

    return VOICE_REFERENCE_FILE


def generate_voice_fallback():
    """
    Fallback: genera la voce usando Lyria 3 stesso.
    Genera un breve brano a cappella e lo usa come reference.
    """
    from google import genai

    from config import GEMINI_API_KEY, LYRIA_CLIP_MODEL

    console.print(Panel(
        "[bold]Fallback: generazione voce con Lyria 3 Clip[/bold]\n"
        "Genero un breve brano a cappella per estrarre la voce",
        style="yellow"
    ))

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = (
        f"A cappella solo male voice, Italian singer-songwriter, "
        f"warm baritone, intimate and emotional, no instruments, "
        f"just voice singing a gentle Italian melody, 'la la la' humming, "
        f"close microphone, studio recording quality"
    )

    console.print(f"[cyan]Generazione con Lyria 3 Clip...[/cyan]")
    response = client.models.generate_content(
        model=LYRIA_CLIP_MODEL,
        contents=prompt,
    )

    # Estrai audio dalla risposta
    audio_data = None
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            mime = getattr(part.inline_data, "mime_type", "") or ""
            if "audio" in mime:
                audio_data = part.inline_data.data
                break

    if audio_data is None:
        raise RuntimeError(
            f"Lyria non ha generato audio. Risposta: "
            f"{[p.text if hasattr(p, 'text') and p.text else str(type(p)) for p in response.candidates[0].content.parts]}"
        )

    raw_file = VOICE_DIR / "lyria_acappella.wav"

    with open(raw_file, "wb") as f:
        f.write(audio_data)

    # Il prompt genera una voce a cappella (solo voce, no strumentali)
    # Quindi il file è già un reference vocale puro
    shutil.copy2(raw_file, VOICE_REFERENCE_FILE)

    info = sf.info(str(VOICE_REFERENCE_FILE))
    console.print(f"[green]✓ Voce reference generata: {VOICE_REFERENCE_FILE}[/green]")
    console.print(f"[green]  Durata: {info.duration:.1f}s, {info.samplerate}Hz[/green]")

    return VOICE_REFERENCE_FILE


def main():
    console.print(Panel(
        f"[bold magenta]Generazione Voce - {ARTIST['name']}[/bold magenta]\n\n"
        f"Genere: {ARTIST['genre']}\n"
        f"Voce: {ARTIST['voice_description'][:80]}...",
        style="magenta"
    ))

    if VOICE_REFERENCE_FILE.exists():
        if not Confirm.ask(
            f"[yellow]La voce esiste già ({VOICE_REFERENCE_FILE}). Rigenerare?[/yellow]"
        ):
            console.print("[green]Voce esistente mantenuta.[/green]")
            return

    # Prova prima con CosyVoice (miglior qualità)
    cosyvoice_dir = MODELS_DIR / "CosyVoice"
    if cosyvoice_dir.exists():
        try:
            generate_voice_cosyvoice()
            return
        except Exception as e:
            console.print(f"[yellow]CosyVoice fallito: {e}[/yellow]")
            console.print("[yellow]Tento con metodo fallback (Lyria 3)...[/yellow]")

    # Fallback: usa Lyria 3 Clip per generare una voce
    try:
        generate_voice_fallback()
    except Exception as e:
        console.print(f"[red]Errore: {e}[/red]")
        console.print(
            "\n[yellow]Puoi anche fornire manualmente un file .wav di 15-30 secondi\n"
            f"e copiarlo in: {VOICE_REFERENCE_FILE}[/yellow]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
