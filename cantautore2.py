"""
Cantautore Digitale 2 - Stile Soul/Jazz Atmosferico (alla Mario Biondi)
========================================================================
Artista con voce profondissima, soul/jazz, atmosfere notturne.
Canta in italiano (a differenza di Biondi che canta in inglese).

Test: python cantautore2.py --tema "una notte d'estate al porto"
"""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import patch_torchaudio  # noqa: F401

import soundfile as sf
import numpy as np
from rich.console import Console
from rich.panel import Panel

from config import (
    GEMINI_API_KEY, LYRIA_MODEL,
    DEMUCS_MODEL, DEVICE, TEMP_DIR, MODELS_DIR,
    get_voice_reference,
)
from artist_brain import ArtistBrain

console = Console()

# === BRAIN ===
BASE_DIR = Path(__file__).parent
brain = ArtistBrain("luca_notturno", BASE_DIR)

# === OUTPUT DIR DEDICATA ===
OUTPUT_DIR_2 = Path(__file__).parent / "output_artist2"
OUTPUT_DIR_2.mkdir(exist_ok=True)

# === PROFILO ARTISTA 2 ===
ARTIST2 = {
    "name": "Luca Notturno",
    "genre": "soul jazz italiano, smooth jazz, R&B atmosferico",
    "style_prompt": (
        "Professional studio recording of an Italian male singer, "
        "extremely deep bass-baritone voice, rich warm velvety tone, "
        "smooth soulful jazz vocal delivery, elegant and expressive, "
        "deep resonant chest voice with incredible low range, "
        "silky smooth phrasing, intimate and warm, "
        "layered production: Fender Rhodes electric piano, walking upright bass, "
        "muted trumpet, warm tenor saxophone, lush string section, "
        "brushed jazz drums with subtle groove, Hammond B3 organ, "
        "brass section (trumpet, trombone, alto sax), "
        "smooth sophisticated soul-jazz production, nightclub atmosphere, "
        "warm analog sound, professional mix, "
        "clear Italian diction, close microphone, dry vocal with subtle warmth"
    ),
}


# ============================================================
# STEP 1: Generazione testi stile soul/jazz italiano
# ============================================================

def genera_testo_soul(tema: str, titolo: str | None = None) -> dict:
    """Genera testo stile soul/jazz atmosferico in italiano."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Carica personalità e memoria
    dna_prompt = brain.get_dna_prompt()
    memory_prompt = brain.get_memory_prompt()
    evolution_prompt = brain.get_evolution_prompt()

    prompt_gen = f"""Sei un cantautore italiano con l'anima soul. La tua voce è la più profonda che esista.
Scrivi canzoni notturne, atmosferiche, eleganti. Il tuo mondo è fatto di luci soffuse, jazz club,
whisky, pioggia sui vetri, sguardi lunghi. Ma non sei mai banale — sotto la superficie c'è
sempre qualcosa di più profondo.

{dna_prompt}

{evolution_prompt}

{memory_prompt}

ARTISTA: {ARTIST2['name']}
GENERE: {ARTIST2['genre']}
LINGUA: italiano elegante ma caldo, mai freddo. Come parlare in italiano con l'anima soul.
Non aulico, non accademico. L'italiano di chi sa vivere bene. Sensuale senza essere volgare.

TEMA: {tema}
{f'TITOLO: {titolo}' if titolo else 'Scegli un titolo evocativo, morbido, che suoni come una carezza.'}

STILE MUSICALE E LIRICO:
1. ATMOSFERA PRIMA DI TUTTO — ogni canzone deve creare un LUOGO. Chi ascolta deve sentire
   il profumo del locale, il calore delle luci, il sapore del drink. Non racconti una storia:
   costruisci un mondo.
2. LA VOCE COME STRUMENTO — scrivi per una voce bassissima. Vocali aperte, frasi lunghe che
   si possono tenere. Parole che vibrano nel petto. Evita parole acute o stridenti.
3. IL GROOVE NELLE PAROLE — le sillabe devono seguire un groove soul/jazz. Alterna frasi
   lunghe e respiri corti. Il ritmo è più lento, più morbido, più sinuoso del pop.
4. ELEGANZA EMOTIVA — l'emozione è sottintesa, mai esplicita. Un gesto, un profumo,
   un dettaglio che dice tutto. Come un film francese: suggerisci, non mostrare.
   ATTENZIONE: evita parole che possano essere bloccate dai filtri (niente 'sensuale', 'erotico', 'labbra', 'corpo' in modo esplicito). Usa metafore.
5. METAFORE LIQUIDE — usa immagini legate all'acqua, alla notte, alla luce, al calore.
   Ma mai ovvie. Non "la luna" ma "l'ombra della luna sul ghiaccio del bicchiere".
6. IL DETTAGLIO CHE UCCIDE — un particolare preciso che rende tutto reale. Il rumore
   del cucchiaino, il colore del rossetto, il numero del taxi.

RIFERIMENTI (livello da raggiungere):
- Mario Biondi (atmosfera, groove, profondità vocale) MA in italiano
- Barry White (sensualità, voce profonda)
- Lucio Dalla (visionarietà poetica italiana)
- Paolo Conte (eleganza, ironia, jazz)
- Al Jarreau (versatilità, scat, gioia)
- Fred Buscaglione (Italia + swing + ironia)

ERRORI DA EVITARE:
- Mai testi da cantautore folk/pop — questo è SOUL, è un altro mondo
- Mai troppo triste o malinconico — il soul è caldo anche quando è triste
- Mai elenchi poetici — ogni verso deve avere un groove interno
- Mai rime forzate — meglio assonanze morbide
- Mai didascalico — mostra, non raccontare

STRUTTURA (IMPORTANTE: la canzone deve essere LUNGA, almeno 40-50 righe totali di testo cantato):
[Intro] (3-4 righe sussurrate lente, vocalizzazioni, crea l'atmosfera con calma)
[Verse 1] (7-8 righe, entra in una scena notturna, un luogo, un momento — prendi tempo)
[Chorus] (4-5 righe, l'hook morbido ma che resta. Cantabile, profondo)
[Instrumental] (indica: "solo di sassofono" o "piano solo" — 8-12 battute)
[Verse 2] (7-8 righe, cambia prospettiva, avvicinati, nuovi dettagli)
[Chorus]
[Verse 3] (6-7 righe, il momento più intimo, il dettaglio che rivela tutto)
[Bridge] (3-4 righe, il cambio emotivo, la rivelazione)
[Chorus] (variazione, aggiungi una riga in più)
[Outro] (4-5 righe di vocalizzazione, scat, ripetizioni che sfumano lentamente)

Rispondi SOLO in JSON:
{{
    "titolo": "...",
    "testo": "[Intro]\\n...\\n\\n[Verse 1]\\n...\\n\\n[Chorus]\\n...",
    "mood": "mood in inglese per produzione musicale soul/jazz (NO parole come sensual/erotic/sexy)",
    "bpm": 72
}}

NOTA SUL BPM: scegli un BPM tra 65 e 80. Il soul/jazz è LENTO. Mai sopra 85."""

    console.print("[cyan]  Generazione testo con Gemini 2.5 Pro (step 1: bozza soul)...[/cyan]")
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt_gen,
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    draft = json.loads(text)

    # === STEP 2: RAFFINAMENTO ===
    console.print("[cyan]  Raffinamento testo (step 2: critico soul)...[/cyan]")

    prompt_refine = f"""Sei un produttore musicale leggendario che ha lavorato con i più grandi del soul e del jazz.
Ti hanno dato questa bozza di canzone soul in italiano da perfezionare.

BOZZA:
Titolo: {draft['titolo']}
{draft['testo']}

COSA DEVI FARE:
1. GROOVE: Le parole scorrono come miele? Ogni sillaba deve cadere nel groove giusto. Leggi a voce alta con un ritmo soul — se inciampi, riscrivi.
2. ATMOSFERA: Chiudi gli occhi. Vedi il luogo? Senti il profumo? Se no, aggiungi il dettaglio che manca.
3. LA VOCE: Questo testo sarà cantato dalla voce più profonda d'Italia. Le vocali devono vibrare, le frasi devono avere respiro. Niente staccato, tutto legato.
4. HOOK: Il ritornello si canticchia sotto la doccia? Se no, semplificalo. Il soul vive di melodia.
5. QUEL MOMENTO: C'è un punto dove il cuore si stringe? Se no, crealo. Un dettaglio, un gesto, una parola.

REGOLE ASSOLUTE:
- Mantieni TUTTE le sezioni: [Intro] [Verse 1] [Chorus] [Instrumental] [Verse 2] [Chorus] [Verse 3] [Bridge] [Chorus] [Outro]
- NON accorciare — se una strofa ha meno di 6 righe, ALLUNGALA
- Il testo TOTALE deve avere almeno 40 righe cantate (escluse indicazioni strumentali)
- L'[Instrumental] è fondamentale: "solo di sassofono" o "piano solo" — 8-12 battute
- L'[Outro] deve essere lungo: ripetizioni, scat, vocalizzazioni che sfumano
- L'italiano deve essere caldo, elegante — non da cantautore indie
- Il risultato deve suonare come un disco soul registrato a mezzanotte

Rispondi SOLO in JSON (stesso formato):
{{
    "titolo": "...",
    "testo": "[Intro]\\n...\\n\\n[Verse 1]\\n...\\n\\n[Chorus]\\n...",
    "mood": "{draft.get('mood', 'smooth soul jazz')}",
    "bpm": {draft.get('bpm', 90)}
}}"""

    response2 = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt_refine,
    )

    text2 = (response2.text or "").strip()
    if text2.startswith("```"):
        text2 = text2.split("\n", 1)[1]
    if text2.endswith("```"):
        text2 = text2.rsplit("```", 1)[0]
    text2 = text2.strip()

    try:
        result = json.loads(text2)
    except (json.JSONDecodeError, ValueError):
        console.print("[yellow]  ⚠ Raffinamento fallito, uso bozza originale[/yellow]")
        result = draft

    if titolo:
        result["titolo"] = titolo

    console.print(f"  [green]✓ \"{result['titolo']}\" - {result['mood']}, {result['bpm']} BPM[/green]")
    return result


# ============================================================
# STEP 2: Generazione musica (riusa genera_musica ma con ARTIST2)
# ============================================================

def genera_musica_soul(song_data: dict, output_path: Path, max_retries: int = 3) -> Path:
    """Genera la canzone soul/jazz con Lyria 3 Pro."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    style = ARTIST2["style_prompt"]
    mood = song_data.get("mood", "smooth soul jazz")
    bpm = min(song_data.get("bpm", 72), 80)  # Cap a 80 BPM per soul/jazz
    from cantautore import pulisci_testo_per_lyria
    testo = pulisci_testo_per_lyria(song_data["testo"])

    prompt = f"""{style}.
Mood: {mood}. Tempo: {bpm} BPM.
Create a full-length soul jazz song of at least 3 minutes and 30 seconds.
Lush warm production with rich instrumentation throughout.
Deep soulful male vocal performance with incredible bass range.
The song must have a clear intro, multiple verses, choruses, a bridge, and a smooth outro.

[0:00 - 0:25] Intro: atmospheric Rhodes piano, walking bass, brushed drums setting the mood
[0:25 - 1:10] Verse 1 with deep vocals - intimate, close, building
[1:10 - 1:45] Chorus - brass section enters, fuller arrangement, soulful hook
[1:45 - 2:30] Verse 2 with vocals - saxophone weaving between phrases
[2:30 - 3:05] Chorus - full band, strings swell, bigger
[3:05 - 3:40] Verse 3 or saxophone solo - emotional build, unexpected
[3:40 - 4:00] Bridge - stripped back, voice and piano, emotional peak
[4:00 - 4:35] Final Chorus - maximum soul, full brass and strings
[4:35 - 4:55] Outro - voice ad-libs and scat over fading groove

Lyrics:
{testo}"""

    for attempt in range(max_retries):
        console.print(f"[cyan]  Generazione musica soul con Lyria 3 Pro (tentativo {attempt+1}/{max_retries})...[/cyan]")
        try:
            response = client.models.generate_content(
                model=LYRIA_MODEL,
                contents=prompt,
            )

            if not response.candidates:
                block_reason = getattr(response, 'prompt_feedback', None)
                console.print(f"[yellow]  ⚠ Nessun candidato. Feedback: {block_reason}[/yellow]")
                time.sleep(5)
                continue

            candidate = response.candidates[0]
            if candidate.content is None:
                finish_reason = getattr(candidate, 'finish_reason', 'unknown')
                console.print(f"[yellow]  ⚠ Contenuto vuoto. Motivo: {finish_reason}[/yellow]")
                time.sleep(5)
                continue

            audio_data = None
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    if "audio" in (part.inline_data.mime_type or ""):
                        audio_data = part.inline_data.data
                        break

            if audio_data is not None:
                break

            console.print("[yellow]  ⚠ Risposta senza audio, riprovo...[/yellow]")
            time.sleep(5)

        except Exception as e:
            console.print(f"[yellow]  ⚠ Errore: {e}[/yellow]")
            time.sleep(5)
    else:
        raise RuntimeError(f"Generazione musica fallita dopo {max_retries} tentativi")

    # Salva come WAV
    import io
    import wave

    audio_bytes = io.BytesIO(audio_data)
    try:
        with wave.open(audio_bytes) as wav_file:
            params = wav_file.getparams()
            audio_np = np.frombuffer(wav_file.readframes(params.nframes), dtype=np.int16)
            if params.nchannels == 2:
                audio_np = audio_np.reshape(-1, 2)
            audio_float = audio_np.astype(np.float32) / 32768.0
            sf.write(str(output_path), audio_float, params.framerate)
    except Exception:
        output_path.write_bytes(audio_data)

    info = sf.info(str(output_path))
    console.print(f"  [green]✓ Canzone generata: {info.duration:.1f}s, {info.samplerate}Hz[/green]")
    return output_path


# ============================================================
# PIPELINE - riusa separa_vocals e mix_finale da cantautore.py
# ============================================================

def genera_canzone_soul(tema: str, titolo: str | None = None, voice_name: str | None = None) -> Path:
    """Pipeline completa per artista 2 (soul/jazz)."""
    from cantautore import separa_vocals, mix_finale, converti_voce

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    song_temp = TEMP_DIR / f"artist2_{timestamp}"
    song_temp.mkdir(exist_ok=True)

    console.print(Panel(
        f"[bold]Nuova canzone soul/jazz[/bold]\n"
        f"Artista: {ARTIST2['name']}\n"
        f"Tema: {tema}",
        style="blue"
    ))

    try:
        # Step 1: Testi
        console.print("\n[bold]Step 1: Testi[/bold]")
        song_data = genera_testo_soul(tema, titolo)
        song_title = song_data["titolo"]
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in song_title)

        testo_file = song_temp / "testo.json"
        with open(testo_file, "w", encoding="utf-8") as f:
            json.dump(song_data, f, ensure_ascii=False, indent=2)

        # Digest: aggiorna memoria dell'artista
        console.print("  [dim]Aggiornamento memoria artista...[/dim]")
        brain.digest_song(song_data, tema)

        # Step 2: Musica
        console.print("\n[bold]Step 2: Musica[/bold]")
        raw_song = song_temp / "raw_song.wav"
        genera_musica_soul(song_data, raw_song)

        # Step 3: Separazione
        console.print("\n[bold]Step 3: Separazione[/bold]")
        separated_dir = song_temp / "separated"
        vocals_path, instrumental_path = separa_vocals(raw_song, separated_dir)

        # Step 3b: Voice Conversion (se voce disponibile)
        try:
            voice_ref = get_voice_reference(voice_name)
            console.print(f"\n[bold]Step 3b: Voice Conversion[/bold] ({voice_ref.parent.name})")
            converted_path = song_temp / "vocals_converted.wav"
            converti_voce(vocals_path, converted_path, voice_name)
            vocals_path = converted_path
        except FileNotFoundError:
            console.print("\n[yellow]  Voice conversion saltata (nessuna voce di riferimento)[/yellow]")

        # Step 4: Mix
        console.print("\n[bold]Step 4: Mix[/bold]")
        ver_final = song_temp / "final.wav"
        mix_finale(instrumental_path, vocals_path, ver_final)

        # Output
        final_path = OUTPUT_DIR_2 / f"{safe_title}_{timestamp}.wav"
        shutil.copy2(str(ver_final), str(final_path))

        testo_output = OUTPUT_DIR_2 / f"{safe_title}_{timestamp}_testo.json"
        shutil.copy2(testo_file, testo_output)

        console.print(Panel(
            f"[bold green]Canzone completata![/bold green]\n\n"
            f"Titolo: {song_title}\n"
            f"File: {final_path}\n"
            f"Testo: {testo_output}",
            style="green"
        ))

        return final_path

    except Exception as e:
        console.print(f"\n[red]Errore nella pipeline: {e}[/red]")
        raise
    finally:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Cantautore Digitale 2 - Soul/Jazz Atmosferico Italiano"
    )
    parser.add_argument("--tema", type=str, help="Tema della canzone")
    parser.add_argument("--titolo", type=str, help="Titolo (opzionale)")
    parser.add_argument("--voce", type=str, help="Nome profilo voce (es: mia_voce)")

    args = parser.parse_args()

    if args.tema:
        genera_canzone_soul(args.tema, args.titolo, args.voce)
    else:
        parser.print_help()
        console.print(
            "\n[yellow]Esempi:[/yellow]\n"
            "  python cantautore2.py --tema 'una notte al porto con il profumo del mare'\n"
            "  python cantautore2.py --tema 'un incontro in un jazz club di Milano a mezzanotte'\n"
        )


if __name__ == "__main__":
    main()
