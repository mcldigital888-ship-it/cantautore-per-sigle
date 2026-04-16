"""
Cantautore Digitale - Pipeline Automatica
==========================================
Genera canzoni complete usando:
  Gemini (testi) → Lyria 3 Pro (musica) → Demucs (separazione) → Mix professionale

Uso:
  python cantautore.py --tema "nostalgia di un viaggio in treno"
  python cantautore.py --tema "amore perduto in una notte d'estate" --titolo "Stelle Cadenti"
  python cantautore.py --genera 5   # genera 5 canzoni con temi casuali
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import patch_torchaudio  # noqa: F401 - patch torchaudio.save per usare soundfile

import soundfile as sf
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import (
    ARTIST, GEMINI_API_KEY, LYRIA_MODEL,
    SEED_VC, DEMUCS_MODEL, DEVICE,
    OUTPUT_DIR, TEMP_DIR, VOICE_REFERENCE_FILE, MODELS_DIR,
    get_voice_reference,
)
from artist_brain import ArtistBrain

console = Console()

# === BRAIN ===
BASE_DIR = Path(__file__).parent
brain = ArtistBrain("cantautore_digitale", BASE_DIR)


def time_stretch(input_path: Path, output_path: Path, min_duration: float = 210.0, max_stretch: float = 1.15):
    """Allunga leggermente una canzone se è sotto la durata minima.
    Usa librosa per pitch-preserving time stretch (max 15%).
    min_duration: durata minima target in secondi (default 3:30)
    max_stretch: fattore massimo di stretching (default 1.15 = +15%)"""
    info = sf.info(str(input_path))
    current_duration = info.duration

    if current_duration >= min_duration:
        # Già abbastanza lunga
        if input_path != output_path:
            shutil.copy2(str(input_path), str(output_path))
        return current_duration

    stretch_factor = min_duration / current_duration
    if stretch_factor > max_stretch:
        stretch_factor = max_stretch  # Non stretchiamo oltre il 15%

    console.print(f"  [cyan]Time-stretch: {current_duration:.0f}s → {current_duration * stretch_factor:.0f}s ({stretch_factor:.0%})[/cyan]")

    import librosa
    audio, sr = librosa.load(str(input_path), sr=None, mono=False)
    # time_stretch con rate < 1 = rallenta
    if audio.ndim == 1:
        stretched = librosa.effects.time_stretch(audio, rate=1.0/stretch_factor)
    else:
        # Stereo: stretcha canale per canale
        stretched = np.array([
            librosa.effects.time_stretch(audio[ch], rate=1.0/stretch_factor)
            for ch in range(audio.shape[0])
        ])
        stretched = stretched.T  # (samples, channels)

    if stretched.ndim == 1:
        sf.write(str(output_path), stretched, sr)
    else:
        sf.write(str(output_path), stretched, sr)

    new_info = sf.info(str(output_path))
    console.print(f"  [green]✓ Durata finale: {new_info.duration:.0f}s[/green]")
    return new_info.duration


def pulisci_testo_per_lyria(testo: str) -> str:
    """Rimuove annotazioni di regia e parole trigger dal testo prima di inviarlo a Lyria.
    Lyria canta TUTTO quello che riceve — le indicazioni tra parentesi
    come (sussurrato) o (la voce si rompe) verrebbero cantate letteralmente.
    Inoltre sostituisce parole che possono triggerare i safety filter."""
    import re
    # Rimuovi tutto tra parentesi tonde (indicazioni di scena)
    testo = re.sub(r'\([^)]*\)', '', testo)
    # Sostituisci parole trigger per safety filter
    trigger_subs = {
        'sigaretta': 'stella',
        'sigarette': 'stelle',
        'fumo': 'nebbia',
        'fumare': 'respirare',
        'whisky': 'caffè',
        'whiskey': 'caffè',
        'droga': 'nebbia',
        'sangue': 'vino',
        'morire': 'partire',
        'morte': 'notte',
        'uccide': 'cancella',
        'pistola': 'chitarra',
    }
    for trigger, replacement in trigger_subs.items():
        testo = re.sub(rf'\b{trigger}\b', replacement, testo, flags=re.IGNORECASE)
    # Rimuovi accenti NON finali — in italiano l'accento si scrive solo
    # sull'ultima lettera (maturità, perché, così, città, più).
    # Accenti nel mezzo della parola (ronzìo) sono aggiunte di Gemini
    # che Lyria pronuncia male. Quelli finali vanno MANTENUTI.
    accent_to_plain = {
        'à': 'a', 'á': 'a', 'è': 'e', 'é': 'e',
        'ì': 'i', 'í': 'i', 'ò': 'o', 'ó': 'o',
        'ù': 'u', 'ú': 'u',
        'À': 'A', 'Á': 'A', 'È': 'E', 'É': 'E',
        'Ì': 'I', 'Í': 'I', 'Ò': 'O', 'Ó': 'O',
        'Ù': 'U', 'Ú': 'U',
    }
    accented_chars = ''.join(accent_to_plain.keys())
    # Sostituisci accenti SOLO se NON sono l'ultimo carattere della parola
    # (guardando se il carattere successivo è ancora una lettera)
    def fix_accent(m):
        char = m.group(1)
        after = m.group(2)
        return accent_to_plain.get(char, char) + after
    testo = re.sub(f'([{re.escape(accented_chars)}])([a-zA-Z\u00c0-\u00ff])', fix_accent, testo)
    # Rimuovi righe vuote multiple
    testo = re.sub(r'\n{3,}', '\n\n', testo)
    # Rimuovi spazi a inizio/fine riga
    testo = '\n'.join(line.strip() for line in testo.split('\n'))
    return testo.strip()


# ============================================================
# STEP 1: Generazione testi con Gemini
# ============================================================

def genera_testo(tema: str, titolo: str | None = None) -> dict:
    """Genera testo e titolo della canzone usando Gemini."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Carica personalità e memoria
    dna_prompt = brain.get_dna_prompt()
    memory_prompt = brain.get_memory_prompt()
    evolution_prompt = brain.get_evolution_prompt()

    # === STEP 1: GENERAZIONE GREZZA ===
    prompt_gen = f"""Sei un cantautore italiano con l'anima di chi ha vissuto troppo e amato male.
Cinico e romantico allo stesso tempo. Elegante senza sforzo. Dici cose devastanti sottovoce.
Le tue canzoni hanno la profondità di Califano, il groove di Sade, l'ironia di Paolo Conte.
Non urli mai. L'emozione è SOTTO la superficie. Lo spazio e il silenzio contano quanto le parole.

{dna_prompt}

{evolution_prompt}

{memory_prompt}

ARTISTA: {ARTIST['name']}
GENERE: {ARTIST['genre']}
LINGUA: italiano vero, parlato, diretto. L'artista ha origini romane — può usare qualche espressione romana qua e là (ma NON in ogni canzone, e mai forzato). Quando esce il romanesco deve essere naturale, musicale, come in Mannarino o Califano. Il dialetto è un colore, non una maschera.

TEMA: {tema}
{f'TITOLO: {titolo}' if titolo else 'Scegli un titolo che suoni come una frase che diresti a qualcuno.'}

COSA RENDE UNA CANZONE UNA HIT (studia bene):
1. L'HOOK RITMICO — il ritornello deve avere un ritmo che si stampa nel cervello. Non solo le parole, ma COME suonano. Sillabe che battono come un tamburo. "Minuetto" (Califano), "La donna cannone" (De Gregori), "Almeno tu nell'universo" (Mia Martini). Il ritmo delle parole è più importante del significato.
2. METAFORE POTENTI E ORIGINALI — non usare mai metafore ovvie. Inventa immagini che nessuno ha mai accostato. De André diceva "dai diamanti non nasce niente, dal letame nascono i fior". Battisti: "e ti vengo a cercare anche solo per vederti o parlare". Crea metafore che richiedono un secondo per essere capite, che lasciano un dubbio, che rivelano un significato nuovo ad ogni ascolto.
3. OSCURITÀ ELEGANTE — i testi migliori NON sono di facile comprensione. Hanno strati. Significati nascosti. Frasi che capisci dopo il quinto ascolto. Come un quadro che vedi qualcosa di nuovo ogni volta. "Bocca di rosa" non parla davvero di una rosa. "Via del Campo" non è una via. Scrivi così: il primo livello è una storia, il secondo è una verità più profonda.
4. IL CONTRASTO VIOLENTO — accosta il sacro e il profano. Il bello e lo schifo. L'amore e il tradimento nella stessa frase. Ma anche: verità scomode dette con il sorriso. Il paradosso è l'arma più potente — una canzone allegra che parla di morte, una melodia da ballare che racconta la miseria. De André cantava dei bordelli con eleganza. Califano raccontava il vuoto ballando. Dalla cantava "Attenti al lupo" ridendo. Le canzoni migliori NON sono tutte d'amore — parlano di soldi, ipocrisia, lavoro, politica, paura, ma sempre con poesia.
5. IL RITMO CHE NON TI MOLLA — ogni verso deve avere una cadenza interna che trascina. Conta le sillabe. Alterna versi lunghi e corti. Usa la ripetizione come un'arma: una parola che torna, un suono che insiste, un pattern che il cervello non riesce a dimenticare.
6. MAI SPIEGARE — non dire mai cosa provi. Mostra una scena e lascia che sia chi ascolta a sentire. Non "ero disperato". Sì "ho contato le crepe sul muro finché non ha fatto mattina". Il non-detto è più potente di qualsiasi parola.

ERRORI DA NON FARE MAI (se li fai, si capisce subito che è AI):
- Mai elenchi di immagini belle una dopo l'altra (sembra un generatore automatico)
- Mai simmetria perfetta tra strofe (gli umani sono disordinati)
- Mai rime troppo pulite ad ogni verso (nella vita non rima niente)
- Mai metafore generiche o da primo livello: "oceano", "stelle", "volare", "luce/ombra", "pioggia=tristezza"
- Mai frasi che "suonano bene" ma non dicono niente di specifico
- Mai terminare le strofe con una morale o una conclusione. Lascia in sospeso
- Mai essere troppo chiari. Se il significato è ovvio al primo ascolto, è banale

TECNICHE DEI MAESTRI (usale):
- PERSONIFICAZIONE: dai vita a oggetti, stanze, città, animali. "Questa stanza non ha più pareti, ma alberi infiniti" (Paoli). La stanza respira, la città ti guarda, il treno ti aspetta. Gli oggetti sentono, i luoghi ricordano. Ma deve avere senso emotivo, non essere decorazione.
- IL RITUALE CONCRETO CHE SI SVUOTA: "La barba fatta con maggiore cura, la macchina a lavare" (Califano) — descrivi i gesti precisi, poi il vuoto dopo. Il dettaglio banale che diventa poesia.
- LA PAROLA-MANTRA: una parola o frase che torna ossessivamente. "Noia" in Califano. "Cielo" in Paoli. Scegli la tua e martellala nel ritornello finché non diventa filosofia.
- DESCRIVERE SENZA NOMINARE: Paoli descrive l'orgasmo senza mai dire "amore" o "piacere". Descrivi lo stato d'animo facendo vivere lo SPAZIO intorno. Se sei triste, non dire "sono triste" — fai tremare i muri.
- LO SBALZO: dolcezza → vuoto. Preparazione → delusione. Califano costruisce l'attesa e poi la smonta con una frase. Il contrasto emotivo è l'arma più potente.
- LA RIPETIZIONE IPNOTICA: "Parole, parole, parole" (Mina). Una parola ripetuta fino a diventare il significato stesso. La ripetizione non è pigrizia, è martello. Usala nel ritornello.
- L'IRONIA CHE TAGLIA: Mina risponde ai violini con "caramelle non ne voglio più". Distruggi la retorica con una frase quotidiana. L'ironia è più potente della poesia.
- IL DUBBIO COME HOOK: "Sarà perché ti amo" — non afferma, non nega. Il dubbio è più forte della certezza. Un ritornello che è una domanda senza punto interrogativo resta in testa per sempre.
- IL CORPO PRIMA DELLA MENTE: "Stringimi forte" (Ricchi e Poveri). "La mia schiena è più dritta" (Mannarino). Non pensieri astratti ma gesti fisici, sensazioni concrete, il corpo che parla prima delle parole.
- LA SINESTESIA: mescola i sensi. "L'odore dell'ombra", "rimbomba l'odore" (Mannarino). L'ombra non ha odore, l'odore non rimbomba. Ma funziona perché è emotivamente vero. Accosta sensi che non c'entrano niente: il risultato è poesia che non si dimentica.
- LA TRATTENUTA DI SADE: "No Ordinary Love" dice cose devastanti con voce piatta. Mai urlare. Mai enfatizzare. Più l'emozione è grande, più la voce è calma. Il testo deve funzionare SUSSURRATO. Se ha bisogno del volume per funzionare, è debole.
- LO SPAZIO: in musica e in poesia, quello che NON dici conta più di quello che dici. Lascia respirare. Non riempire ogni riga. A volte una riga corta dopo una lunga colpisce più forte.

STILE: De André (profondità e metafore), Califano (cinismo e poesia di strada), Sade (eleganza, trattenuta, groove), Paolo Conte (ironia raffinata, jazz), Dalla (visionarietà), Battisti/Mogol (ritmo delle parole perfetto), Paoli (personificazione e surreale), Mannarino (sinestesia, corpo, dialetto come musica). Il livello è QUESTO. Non meno.

STRUTTURA (IMPORTANTE: la canzone deve essere LUNGA, almeno 40-50 righe totali di testo cantato):
[Intro] (2-3 righe sussurrate o parlate, crea l'atmosfera con calma)
[Verse 1] (7-8 righe, entra in una scena precisa, un momento, un posto — prendi tempo)
[Chorus] (4-5 righe, l'hook che resta. Semplice. Ripetibile. Emotivo)
[Verse 2] (7-8 righe, cambia angolazione, sorprendi, ribalta, nuovi dettagli)
[Chorus]
[Instrumental] (indica: "solo di chitarra" o "pausa strumentale" — 8 battute)
[Verse 3] (6-7 righe, approfondisci, vai dove non ti aspetti)
[Bridge] (3-4 righe, il pugno allo stomaco. La frase che spacca)
[Chorus] (variazione, aggiungi una riga in più rispetto al chorus precedente)
[Outro] (3-4 righe, ripeti un frammento del chorus che sfuma, una frase sospesa che si perde)

Rispondi SOLO in JSON:
{{
    "titolo": "...",
    "testo": "[Verse 1]\\n...\\n\\n[Chorus]\\n...",
    "mood": "mood in inglese per produzione musicale",
    "bpm": 100
}}"""

    console.print("[cyan]  Generazione testo con Gemini 2.5 Pro (step 1: bozza)...[/cyan]")
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

    # === STEP 2: RAFFINAMENTO CON CRITICO ===
    console.print("[cyan]  Raffinamento testo (step 2: critico)...[/cyan]")

    prompt_refine = f"""Sei un paroliere italiano leggendario. Ti hanno dato questa bozza di canzone da migliorare.
Il tuo lavoro è trasformare una buona canzone in una canzone INDIMENTICABILE.

BOZZA:
Titolo: {draft['titolo']}
{draft['testo']}

COSA DEVI FARE:
1. HOOK: Il ritornello è abbastanza forte da restare in testa per giorni? Se no, riscrivilo. L'hook deve essere una frase che potresti dire in una conversazione e che tutti ricordano.
2. QUEL VERSO: Cerca il verso più potente del testo. Se non ce n'è uno che ti ferma il respiro, crealo. Un'immagine nuova, mai sentita, che dice una verità universale.
3. NATURALEZZA: Leggi il testo ad alta voce. Suona come parlerebbe una persona vera? Elimina qualsiasi cosa suoni "scritta" o "poetica in modo artificiale".
4. SORPRESA: C'è almeno un momento in cui il testo va dove non te lo aspetti? Un cambio di prospettiva, un dettaglio inaspettato, una parola fuori posto che funziona.
5. RITMO: Le sillabe funzionano per essere cantate? Le vocali aperte cadono sugli accenti giusti?

REGOLE ASSOLUTE:
- Mantieni la struttura [Intro] [Verse 1] [Chorus] [Verse 2] [Chorus] [Verse 3] [Bridge] [Chorus] [Outro]
- NON rimuovere sezioni, la canzone deve essere lunga
- Puoi cambiare tutto o quasi niente — segui il tuo istinto
- Il risultato deve sembrare scritto da un essere umano alle 3 di notte, non da un algoritmo
- Italiano parlato, niente accenti regionali

Rispondi SOLO in JSON (stesso formato):
{{
    "titolo": "...",
    "testo": "[Verse 1]\\n...\\n\\n[Chorus]\\n...",
    "mood": "{draft.get('mood', 'melancholic')}",
    "bpm": {draft.get('bpm', 100)}
}}"""

    response2 = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt_refine,
    )

    text2 = response2.text.strip()
    if text2.startswith("```"):
        text2 = text2.split("\n", 1)[1]
    if text2.endswith("```"):
        text2 = text2.rsplit("```", 1)[0]
    text2 = text2.strip()

    result = json.loads(text2)

    if titolo:
        result["titolo"] = titolo

    console.print(f"  [green]✓ \"{result['titolo']}\" - {result['mood']}, {result['bpm']} BPM[/green]")
    return result


# ============================================================
# STEP 2: Generazione musica con Lyria 3 Pro
# ============================================================

def genera_musica(song_data: dict, output_path: Path, max_retries: int = 3) -> Path:
    """Genera la canzone completa con Lyria 3 Pro."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    style = ARTIST["style_prompt"]
    mood = song_data.get("mood", "melancholic and intimate")
    bpm = song_data.get("bpm", 100)
    testo = pulisci_testo_per_lyria(song_data["testo"])

    prompt = f"""{style}.
Mood: {mood}. Tempo: {bpm} BPM.
Create a full-length song of at least 3 minutes and 30 seconds.
Full professional production with dynamic arrangement - build from quiet intro to emotional chorus.
Rich layered instrumentation throughout. Expressive passionate vocal performance.
The song must have a clear intro, multiple verses, choruses, a bridge, and a fading outro.

[0:00 - 0:20] Intro: atmospheric, build anticipation slowly
[0:20 - 1:05] Verse 1 with vocals - intimate start
[1:05 - 1:40] Chorus - full energy, memorable hook
[1:40 - 2:25] Verse 2 with vocals - new angle, build
[2:25 - 3:00] Chorus - full band, bigger arrangement
[3:00 - 3:35] Verse 3 with vocals - deeper, unexpected
[3:35 - 3:55] Bridge - stripped back, emotional peak
[3:55 - 4:30] Final Chorus - maximum intensity, variation
[4:30 - 4:50] Outro - fade out slowly

Lyrics:
{testo}"""

    for attempt in range(max_retries):
        console.print(f"[cyan]  Generazione musica con Lyria 3 Pro (tentativo {attempt+1}/{max_retries})...[/cyan]")
        try:
            response = client.models.generate_content(
                model=LYRIA_MODEL,
                contents=prompt,
            )

            # Controlla blocchi/errori
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

            # Estrai audio dalla risposta
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
            console.print(f"[yellow]  ⚠ Errore: {e}, riprovo...[/yellow]")
            time.sleep(5)
    else:
        raise RuntimeError(f"Lyria 3 non ha generato audio dopo {max_retries} tentativi.")

    with open(output_path, "wb") as f:
        f.write(audio_data)

    # Verifica durata
    info = sf.info(str(output_path))
    console.print(f"  [green]✓ Canzone generata: {info.duration:.1f}s, {info.samplerate}Hz[/green]")


def estendi_con_strumentale(song_path: Path, instrumental_path: Path,
                             output_path: Path, intro_sec: float = 20.0,
                             outro_sec: float = 25.0, crossfade_sec: float = 3.0):
    """Estende la canzone usando la sua stessa strumentale per intro e outro.
    Prende i primi N secondi della strumentale come intro (con fade-in)
    e gli ultimi N secondi come outro (con fade-out).
    Stesso stile, stessa chiave, stessa produzione — perfettamente coerente."""

    song, sr = sf.read(str(song_path))
    instr, sr_i = sf.read(str(instrumental_path))

    # Assicura stesso sample rate
    if sr_i != sr:
        import librosa
        instr = librosa.resample(instr.T if instr.ndim > 1 else instr, orig_sr=sr_i, target_sr=sr)
        if instr.ndim > 1:
            instr = instr.T

    # Assicura stereo
    if song.ndim == 1:
        song = np.column_stack([song, song])
    if instr.ndim == 1:
        instr = np.column_stack([instr, instr])

    intro_samples = int(intro_sec * sr)
    outro_samples = int(outro_sec * sr)
    crossfade_samples = int(crossfade_sec * sr)

    # === INTRO: primi N secondi della strumentale con fade-in lento ===
    intro = instr[:min(intro_samples, len(instr))].copy()
    # Fade-in graduale (da silenzio)
    fade_in_len = len(intro)
    fade_in = np.linspace(0.0, 1.0, fade_in_len)[:, None]
    intro = intro * fade_in

    # === OUTRO: ultimi N secondi della strumentale con fade-out lento ===
    outro = instr[-min(outro_samples, len(instr)):].copy()
    # Fade-out graduale (al silenzio)
    fade_out_len = len(outro)
    fade_out = np.linspace(1.0, 0.0, fade_out_len)[:, None]
    outro = outro * fade_out

    # === ASSEMBLA: intro → crossfade → canzone → crossfade → outro ===
    cf = min(crossfade_samples, len(intro), len(song))

    # Crossfade intro → song
    xf_out = np.linspace(1.0, 0.0, cf)[:, None]
    xf_in = np.linspace(0.0, 1.0, cf)[:, None]
    merged = np.concatenate([
        intro[:-cf],
        intro[-cf:] * xf_out + song[:cf] * xf_in,
        song[cf:]
    ])

    # Crossfade song → outro
    cf2 = min(crossfade_samples, len(merged), len(outro))
    xf_out2 = np.linspace(1.0, 0.0, cf2)[:, None]
    xf_in2 = np.linspace(0.0, 1.0, cf2)[:, None]
    final = np.concatenate([
        merged[:-cf2],
        merged[-cf2:] * xf_out2 + outro[:cf2] * xf_in2,
        outro[cf2:]
    ])

    # Normalizza
    peak = np.max(np.abs(final))
    if peak > 0:
        final = final * (0.95 / peak)

    sf.write(str(output_path), final, sr)
    info = sf.info(str(output_path))
    console.print(f"  [green]✓ Canzone estesa: {info.duration:.0f}s ({info.duration/60:.1f} min)[/green]")
    return output_path


# ============================================================
# STEP 3: Separazione vocale con Demucs
# ============================================================

def separa_vocals(audio_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Separa vocals e strumentali con Demucs (API Python diretta)."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    console.print("[cyan]  Separazione vocals/strumentali con Demucs...[/cyan]")

    # Carica modello
    model = get_model(DEMUCS_MODEL)
    model.to(DEVICE)

    # Carica audio
    audio_data, sr = sf.read(str(audio_path))
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=-1)
    # Demucs vuole (batch, channels, samples)
    wav = torch.tensor(audio_data.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Resample se necessario (Demucs vuole 44100)
    if sr != model.samplerate:
        import torchaudio
        wav = torchaudio.functional.resample(wav.squeeze(0), sr, model.samplerate).unsqueeze(0)
        sr = model.samplerate

    # Separa
    sources = apply_model(model, wav, device=DEVICE)
    # sources shape: (batch, n_sources, channels, samples)
    # Ordine tipico: drums, bass, other, vocals
    source_names = model.sources
    vocals_idx = source_names.index("vocals")

    vocals = sources[0, vocals_idx].cpu().numpy().T  # (samples, channels)
    # Strumentali = somma di tutto tranne vocals
    instrumental = sum(
        sources[0, i] for i in range(len(source_names)) if i != vocals_idx
    ).cpu().numpy().T

    # Salva con soundfile
    output_dir.mkdir(parents=True, exist_ok=True)
    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "no_vocals.wav"
    sf.write(str(vocals_path), vocals, sr)
    sf.write(str(instrumental_path), instrumental, sr)

    console.print("  [green]✓ Separazione completata[/green]")
    return vocals_path, instrumental_path


# ============================================================
# STEP 3b: De-reverb vocals
# ============================================================

def dereverb_vocals(vocals_path: Path, output_path: Path) -> Path:
    """Rimuovi riverbero dalle vocals usando spectral gating."""
    console.print("[cyan]  De-reverb vocals...[/cyan]")
    from scipy import signal as sig

    data, sr = sf.read(str(vocals_path))
    mono = np.mean(data, axis=1) if data.ndim > 1 else data

    # STFT
    nperseg = 2048
    hop = 512
    f, t_frames, Zxx = sig.stft(mono, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Stima il noise floor (reverb tail) dalle parti più silenziose
    frame_energy = np.sum(magnitude ** 2, axis=0)
    quiet_threshold = np.percentile(frame_energy, 15)
    quiet_frames = frame_energy < quiet_threshold
    if np.sum(quiet_frames) > 0:
        noise_profile = np.mean(magnitude[:, quiet_frames], axis=1, keepdims=True)
    else:
        noise_profile = np.min(magnitude, axis=1, keepdims=True)

    # Spectral subtraction aggressiva per rimuovere riverbero
    alpha = 3.0  # aggressività
    cleaned_magnitude = np.maximum(magnitude - alpha * noise_profile, 0.05 * magnitude)

    # Ricostruisci
    Zxx_clean = cleaned_magnitude * np.exp(1j * phase)
    _, cleaned = sig.istft(Zxx_clean, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)

    # Torna a stereo
    cleaned = cleaned[:len(mono)]
    if data.ndim > 1:
        cleaned = np.stack([cleaned, cleaned], axis=-1)

    sf.write(str(output_path), cleaned, sr)
    console.print("  [green]✓ De-reverb completato[/green]")
    return output_path


# ============================================================
# STEP 4: Voice Conversion con CosyVoice VC (o fallback Seed-VC)
# ============================================================

def converti_voce(vocals_path: Path, output_path: Path, voice_name: str | None = None) -> Path:
    """Converti la voce usando Seed-VC (migliore intonazione per canto) con fallback CosyVoice."""
    voice_ref = get_voice_reference(voice_name)
    if not voice_ref.exists():
        raise FileNotFoundError(
            f"Reference voice non trovata: {voice_ref}\n"
            "Aggiungi la tua voce: python manage_voices.py add mia_voce file.wav\n"
            "Oppure genera voce AI: python generate_voice.py"
        )

    # Seed-VC: migliore per canto (f0_condition preserva melodia)
    console.print("[cyan]  Conversione voce con Seed-VC...[/cyan]")
    seed_vc_dir = MODELS_DIR / "seed-vc"
    inference_script = seed_vc_dir / "inference.py"

    if not inference_script.exists():
        raise FileNotFoundError(
            f"Seed-VC non trovato: {seed_vc_dir}\n"
            "Esegui prima: python setup_models.py"
        )

    wrapper_script = Path(__file__).parent / "run_seed_vc.py"

    cmd = [
        sys.executable, str(wrapper_script),
        "--source", str(vocals_path),
        "--target", str(voice_ref),
        "--output", str(output_path.parent),
        "--diffusion-steps", str(SEED_VC["diffusion_steps"]),
        "--f0-condition", str(SEED_VC["f0_condition"]),
        "--auto-f0-adjust", str(SEED_VC["auto_f0_adjust"]),
        "--semi-tone-shift", str(SEED_VC["semi_tone_shift"]),
        "--inference-cfg-rate", str(SEED_VC["inference_cfg_rate"]),
        "--length-adjust", str(SEED_VC["length_adjust"]),
        "--fp16", str(SEED_VC["fp16"]),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(seed_vc_dir))

    if result.returncode != 0:
        console.print(f"[red]  Errore Seed-VC: {result.stderr[-500:]}[/red]")
        raise RuntimeError("Seed-VC fallito")

    converted_files = list(output_path.parent.glob("*.wav"))
    if not converted_files:
        raise FileNotFoundError("Seed-VC non ha prodotto output")

    latest = max(converted_files, key=lambda f: f.stat().st_mtime)
    if latest != output_path:
        shutil.move(str(latest), str(output_path))

    console.print("  [green]✓ Voce convertita (Seed-VC)[/green]")
    return output_path


# ============================================================
# STEP 5: Mix finale
# ============================================================

def _apply_vocal_processing(vocals: np.ndarray, sr: int) -> np.ndarray:
    """Applica processing professionale alle vocals: pitch smoothing, EQ, compressione, riverbero."""
    from scipy import signal
    from scipy.ndimage import uniform_filter1d, median_filter

    # 0. PITCH SMOOTHING - rimuovi micro-jitter/tremolo da Seed-VC
    # Applichiamo un leggero smoothing temporale per eliminare le oscillazioni rapide
    smooth_window = int(sr * 0.003)  # 3ms window
    if smooth_window > 1:
        for ch in range(vocals.shape[1] if vocals.ndim > 1 else 1):
            channel = vocals[:, ch] if vocals.ndim > 1 else vocals
            smoothed = uniform_filter1d(channel, size=smooth_window)
            # Blend: 70% originale + 30% smoothed (mantieni transienti)
            blended = 0.7 * channel + 0.3 * smoothed
            if vocals.ndim > 1:
                vocals[:, ch] = blended
            else:
                vocals = blended

    # 1. HIGH-PASS FILTER - rimuovi frequenze sotto 80Hz
    sos_hp = signal.butter(4, 80, btype='high', fs=sr, output='sos')
    vocals = signal.sosfilt(sos_hp, vocals, axis=0)

    # 2. DE-ESSER molto leggero - attenua sibilanti (5-8kHz)
    sos_deess = signal.butter(2, [5000, 8000], btype='band', fs=sr, output='sos')
    sibilants = signal.sosfilt(sos_deess, vocals, axis=0)
    vocals = vocals - 0.15 * sibilants

    # 3. COMPRESSIONE leggera (riduce picchi, uniforma volume)
    threshold = 0.25
    ratio = 3.0
    for ch in range(vocals.shape[1] if vocals.ndim > 1 else 1):
        channel = vocals[:, ch] if vocals.ndim > 1 else vocals
        abs_signal = np.abs(channel)
        envelope = uniform_filter1d(abs_signal, size=int(sr * 0.01))
        gain = np.ones_like(envelope)
        mask = envelope > threshold
        gain[mask] = threshold + (envelope[mask] - threshold) / ratio
        gain[mask] = gain[mask] / envelope[mask]
        if vocals.ndim > 1:
            vocals[:, ch] = channel * gain
        else:
            vocals = channel * gain

    # 4. MICRO-REVERB room (ambiente naturale, non cattedrale)
    from scipy.signal import fftconvolve
    reverb_time = 0.08  # 80ms = piccola stanza
    reverb_len = int(sr * reverb_time)
    t = np.arange(reverb_len) / sr
    np.random.seed(42)
    impulse = np.random.randn(reverb_len) * np.exp(-25.0 * t)
    impulse[0] = 1.0
    impulse = impulse / np.max(np.abs(impulse))
    wet_mix = 0.04  # 4% - appena percepibile

    if vocals.ndim > 1:
        reverbed = np.zeros_like(vocals)
        for ch in range(vocals.shape[1]):
            reverbed[:, ch] = fftconvolve(vocals[:, ch], impulse, mode='full')[:len(vocals)]
    else:
        reverbed = fftconvolve(vocals, impulse, mode='full')[:len(vocals)]
    vocals = (1 - wet_mix) * vocals + wet_mix * reverbed

    return vocals


def mix_finale(instrumental_path: Path, vocals_path: Path, output_path: Path,
               vocal_gain_db: float = -2.5,
               instrumental_gain_db: float = 1.0) -> Path:
    """Mixa strumentali + vocals con bilanciamento professionale."""
    console.print("[cyan]  Mix finale...[/cyan]")
    from scipy import signal as sig
    from scipy.ndimage import uniform_filter1d

    # Carica audio
    instrumental, sr_inst = sf.read(str(instrumental_path))
    vocals, sr_voc = sf.read(str(vocals_path))

    # Resample se necessario
    if sr_inst != sr_voc:
        import librosa
        vocals = librosa.resample(
            vocals.T if vocals.ndim > 1 else vocals,
            orig_sr=sr_voc, target_sr=sr_inst
        )
        if vocals.ndim > 1:
            vocals = vocals.T

    # Allinea lunghezze
    min_len = min(len(instrumental), len(vocals))
    instrumental = instrumental[:min_len]
    vocals = vocals[:min_len]

    # Converti a stereo se necessario
    if instrumental.ndim == 1:
        instrumental = np.stack([instrumental, instrumental], axis=-1)
    if vocals.ndim == 1:
        vocals = np.stack([vocals, vocals], axis=-1)

    # === VOCAL PROCESSING ===
    console.print("[cyan]    Vocal processing...[/cyan]")

    # High-pass 80Hz
    sos_hp = sig.butter(4, 80, btype='high', fs=sr_inst, output='sos')
    vocals = sig.sosfilt(sos_hp, vocals, axis=0)

    # Presenza vocale: boost leggero 2-5kHz per chiarezza
    sos_pres = sig.butter(2, [2000, 5000], btype='band', fs=sr_inst, output='sos')
    presence = sig.sosfilt(sos_pres, vocals, axis=0)
    vocals = vocals + 0.2 * presence  # boost 20% nella zona di presenza

    # === INSTRUMENTAL PROCESSING ===
    console.print("[cyan]    Instrumental processing...[/cyan]")

    # Boost strumentali per non farli sembrare "thin"
    inst_gain = 10 ** (instrumental_gain_db / 20)
    instrumental = instrumental * inst_gain

    # Mid-side EQ: scava spazio per la voce nella zona 200-4000Hz degli strumentali
    sos_vocal_range = sig.butter(2, [200, 4000], btype='band', fs=sr_inst, output='sos')
    vocal_freq_band = sig.sosfilt(sos_vocal_range, instrumental, axis=0)
    instrumental = instrumental - 0.15 * vocal_freq_band  # -1.5dB nella zona vocale

    # Sidechain ducking: abbassa strumentali quando c'è voce
    vocal_mono = np.mean(np.abs(vocals), axis=1) if vocals.ndim > 1 else np.abs(vocals)
    envelope = uniform_filter1d(vocal_mono, size=int(sr_inst * 0.05))
    # Quando la voce è forte, riduci strumentali di 4-5dB
    duck_amount = np.clip(envelope / (np.max(envelope) + 1e-10), 0, 1)
    duck_gain = 1.0 - 0.45 * duck_amount  # max -5dB di ducking
    if instrumental.ndim > 1:
        instrumental = instrumental * duck_gain[:, np.newaxis]
    else:
        instrumental = instrumental * duck_gain

    # === VOCAL GAIN ===
    voc_gain = 10 ** (vocal_gain_db / 20)
    vocals = vocals * voc_gain

    # === MIX ===
    console.print("[cyan]    Mixaggio bilanciato...[/cyan]")
    mixed = instrumental + vocals

    # === MASTERING ===
    console.print("[cyan]    Mastering...[/cyan]")

    # Normalizza
    peak = np.max(np.abs(mixed))
    if peak > 0.0:
        mixed = mixed / peak * 0.95

    # Soft clipping dolce
    mixed = np.tanh(mixed * 1.05) / np.tanh(1.05)

    # Normalizza finale a -1dB
    peak = np.max(np.abs(mixed))
    if peak > 0.0:
        mixed = mixed * (0.89 / peak)  # -1dB

    # Salva WAV 24-bit
    sf.write(str(output_path), mixed, sr_inst, subtype='PCM_24')

    info = sf.info(str(output_path))
    console.print(f"  [green]✓ Brano finale: {info.duration:.1f}s, {info.samplerate}Hz[/green]")
    return output_path


# ============================================================
# PIPELINE COMPLETA
# ============================================================

def genera_canzone(tema: str, titolo: str | None = None, voice_name: str | None = None, export_stems: bool = False) -> Path:
    """Pipeline completa: tema → canzone finita con voce dell'artista."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    song_temp = TEMP_DIR / timestamp
    song_temp.mkdir(exist_ok=True)

    console.print(Panel(
        f"[bold]Nuova canzone[/bold]\n"
        f"Artista: {ARTIST['name']}\n"
        f"Tema: {tema}",
        style="magenta"
    ))

    try:
        # Step 1: Genera testi
        console.print("\n[bold]Step 1: Testi[/bold]")
        song_data = genera_testo(tema, titolo)
        song_title = song_data["titolo"]
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in song_title)

        # Salva i testi
        testo_file = song_temp / "testo.json"
        with open(testo_file, "w", encoding="utf-8") as f:
            json.dump(song_data, f, ensure_ascii=False, indent=2)

        # Digest: aggiorna memoria dell'artista
        console.print("  [dim]Aggiornamento memoria artista...[/dim]")
        brain.digest_song(song_data, tema)

        # Step 2: Genera musica
        console.print("\n[bold]Step 2: Musica[/bold]")
        raw_song = song_temp / "raw_song.wav"
        genera_musica(song_data, raw_song)

        # Step 3: Separa vocals
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
            console.print("  [dim]Usa la voce generata da Lyria. Per usare la tua voce:[/dim]")
            console.print("  [dim]  python manage_voices.py add mia_voce registrazione.wav[/dim]")

        # Step 4: Mix
        console.print("\n[bold]Step 4: Mix[/bold]")
        ver_final = song_temp / "final.wav"
        mix_finale(instrumental_path, vocals_path, ver_final)

        # Step 5: Estensione con strumentale propria
        console.print("\n[bold]Step 5: Estensione durata[/bold]")
        ver_extended = song_temp / "extended.wav"
        estendi_con_strumentale(ver_final, instrumental_path, ver_extended,
                                 intro_sec=20.0, outro_sec=25.0, crossfade_sec=3.0)

        # Copia nell'output
        final_path = OUTPUT_DIR / f"{safe_title}_{timestamp}.wav"
        shutil.copy2(str(ver_extended), str(final_path))

        # Copia testo nella cartella output
        testo_output = OUTPUT_DIR / f"{safe_title}_{timestamp}_testo.json"
        shutil.copy2(testo_file, testo_output)

        # Step 6: Export Kit (stems separati per DAW)
        kit_dir = None
        if export_stems:
            console.print("\n[bold]Step 6: Export Kit Stems[/bold]")
            from export_kit import separa_stems
            kit_dir = OUTPUT_DIR / f"{safe_title}_{timestamp}_kit"
            separa_stems(final_path, kit_dir)
            # Copia anche vocals e instrumental già separati
            shutil.copy2(str(vocals_path), str(kit_dir / "vocals_original.wav"))
            shutil.copy2(str(instrumental_path), str(kit_dir / "instrumental_original.wav"))
            shutil.copy2(testo_file, str(kit_dir / "testo.json"))

        console.print(Panel(
            f"[bold green]Canzone completata![/bold green]\n\n"
            f"Titolo: {song_title}\n"
            f"File: {final_path}\n"
            f"Testo: {testo_output}"
            + (f"\nKit stems: {kit_dir}" if kit_dir else ""),
            style="green"
        ))

        return final_path

    except Exception as e:
        console.print(f"\n[red]Errore nella pipeline: {e}[/red]")
        raise
    finally:
        # Pulizia temp (opzionale - commenta per debug)
        # shutil.rmtree(song_temp, ignore_errors=True)
        pass


def genera_album(n: int = 10, nome_album: str = None):
    """Genera un album completo con N canzoni variegate."""
    # Album bilanciato: mix di temi romantici, sociali, allegri, malinconici
    temi_album = [
        # ROMANTICHE/MALINCONICHE
        "una donna che ti ha lasciato e che rivedi dopo 10 anni in un supermercato — non sai se salutarla o scappare",
        "l'ultima sigaretta fumata insieme a qualcuno che non vedrai mai più, sul balcone alle 5 di mattina",
        "un uomo che parla con la segreteria telefonica della ex perché è l'unico modo di sentire la sua voce",
        # VERITÀ SCOMODE (tono allegro/ironico)
        "l'Italia che va a rotoli ma tutti ballano — i politici rubano, la gente si lamenta al bar e poi vota uguale. Mood ALLEGRO, ironico, da ballare",
        "i social media che ci stanno mangiando il cervello ma non riusciamo a smettere — detto ridendo come se fosse una barzelletta. Mood UPBEAT",
        "il lavoro che ti ruba la vita, 40 anni di mutuo per un appartamento che odii, la sveglia alle 6 che suona come una condanna — ma raccontato come fosse una festa. Mood ALLEGRO",
        # ESISTENZIALI
        "un uomo che a 40 anni si accorge che ha passato la vita a inseguire cose sbagliate e adesso è troppo tardi per ricominciare, ma forse no",
        "la paura di morire non è morire, è morire senza aver mai vissuto davvero — realizzarlo una notte in macchina fermo al semaforo",
        # STORIE
        "la storia di un barista che conosce i segreti di tutti nel quartiere ma nessuno conosce i suoi",
        "due sconosciuti sul treno notturno che si raccontano tutto perché tanto non si rivedranno mai — e invece",
        # ROMA
        "Roma di notte, i sampietrini bagnati, il rumore del motorino in lontananza, una città che è troppo bella per chi ci vive davvero",
        "il mercato di Porta Portese la domenica mattina — le facce, le voci, la vita che si compra e si vende",
        # AMICIZIA/VITA
        "un amico che non senti da anni ti chiama alle 3 di notte — capisci tutto dal tono della voce",
        "la differenza tra chi sei alle 8 di mattina in ufficio e chi sei alle 2 di notte quando nessuno ti guarda",
        "invecchiare non è il corpo che cede ma gli amici che diventano estranei, le feste che finiscono prima, il silenzio che pesa di più",
    ]

    random.shuffle(temi_album)
    temi = temi_album[:n]

    if not nome_album:
        nome_album = f"Album_{datetime.now().strftime('%Y%m%d')}"

    album_dir = OUTPUT_DIR / nome_album
    album_dir.mkdir(exist_ok=True)

    console.print(Panel(
        f"[bold]ALBUM: {nome_album}[/bold]\n"
        f"Artista: {ARTIST['name']}\n"
        f"Tracce: {n}\n"
        f"Costo stimato: ~€{n * 0.10:.2f}",
        style="magenta"
    ))

    results = []
    for i, tema in enumerate(temi):
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Traccia {i+1}/{n}[/bold]")
        console.print(f"{'='*60}")
        try:
            path = genera_canzone(tema)
            # Sposta nell'album dir con numero traccia
            if path and path.exists():
                track_name = f"{i+1:02d}_{path.name}"
                track_path = album_dir / track_name
                shutil.copy2(str(path), str(track_path))
                results.append(("✓", tema, str(track_path)))
            else:
                results.append(("✗", tema, "Nessun file generato"))
        except Exception as e:
            results.append(("✗", tema, str(e)))
            console.print(f"[red]Traccia {i+1} fallita, continuo...[/red]")

        # Pausa tra generazioni per rispettare rate limits
        if i < n - 1:
            time.sleep(3)

    # Riepilogo album
    ok = sum(1 for s, _, _ in results if s == "✓")
    console.print(f"\n{'='*60}")
    console.print(Panel(
        f"[bold]ALBUM COMPLETATO: {nome_album}[/bold]\n\n"
        f"Tracce riuscite: {ok}/{n}\n"
        f"Cartella: {album_dir}",
        style="green" if ok == n else "yellow"
    ))
    for i, (status, tema, info) in enumerate(results):
        color = "green" if status == "✓" else "red"
        console.print(f"  [{color}]{status}[/{color}] {i+1:02d}. {tema[:60]}")
        console.print(f"    [dim]{info}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Cantautore Digitale - Genera canzoni con voce AI coerente"
    )
    parser.add_argument("--tema", type=str, help="Tema della canzone da generare")
    parser.add_argument("--titolo", type=str, help="Titolo della canzone (opzionale)")
    parser.add_argument("--voce", type=str, help="Nome profilo voce (es: mia_voce). Vedi: python manage_voices.py list")
    parser.add_argument("--export-stems", action="store_true", dest="export_stems",
                        help="Esporta kit con tutti gli stems separati (drums, bass, vocals, etc)")
    parser.add_argument("--album", type=int, help="Genera un album con N tracce")
    parser.add_argument("--nome-album", type=str, dest="nome_album", help="Nome dell'album")

    args = parser.parse_args()

    if args.album:
        genera_album(args.album, args.nome_album)
    elif args.tema:
        genera_canzone(args.tema, args.titolo, args.voce, args.export_stems)
    else:
        parser.print_help()
        console.print(
            "\n[yellow]Esempi:[/yellow]\n"
            "  python cantautore.py --tema 'nostalgia di un viaggio in treno'\n"
            "  python cantautore.py --tema 'amore perduto' --titolo 'Stelle Cadenti'\n"
            "  python cantautore.py --tema 'Roma di notte' --voce mia_voce\n"
            "  python cantautore.py --album 10 --nome-album 'Impronte Digitali'"
        )


if __name__ == "__main__":
    main()
