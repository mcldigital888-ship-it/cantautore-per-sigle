"""
Configurazione del Cantautore Digitale
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === PERCORSI ===
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"
VOICE_DIR = PROJECT_DIR / "voice"
TEMP_DIR = PROJECT_DIR / "temp"

# Crea le cartelle se non esistono
for d in [MODELS_DIR, OUTPUT_DIR, VOICE_DIR, TEMP_DIR]:
    d.mkdir(exist_ok=True)

# === API KEYS ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY non trovata! Crea un file .env con:\n"
        "GEMINI_API_KEY=la_tua_chiave_qui\n"
        "Oppure esporta la variabile d'ambiente."
    )

# === PROFILO ARTISTA ===
ARTIST = {
    "name": "Cantautore Digitale",
    "genre": "cantautore italiano, soul jazz, sophisticated pop",
    "voice_description": (
        "Un uomo italiano di 32 anni con voce calda e leggermente roca, "
        "tono intimo e trattenuto, dice cose potenti sottovoce. "
        "Voce media-bassa, naturale e espressiva, mai forzata."
    ),
    "themes": [
        "nostalgia", "viaggi", "amore perduto", "vita quotidiana",
        "notti in citta", "ricordi", "treni", "tempo che passa",
        "ipocrisia sociale", "soldi e apparenze",
        "amicizie false", "il lavoro che ti mangia", "la paura di invecchiare",
        "la morte detta ridendo", "il paese che va a rotoli ma noi balliamo",
        "la notte come verita", "dettagli che nessuno nota", "Roma rotta e bella"
    ],
    "language": "italiano",
    "style_prompt": (
        "Professional studio recording of an Italian male singer, "
        "warm low raspy voice, intimate restrained delivery, "
        "never shouts, emotion always underneath the surface, "
        "smooth elegant phrasing, restrained emotional delivery, "
        "between speaking and singing, world-weary romantic, "
        "sophisticated soul-jazz production, smooth and warm: "
        "Fender Rhodes electric piano, clean warm electric guitar, "
        "deep walking upright bass, brushed jazz drums with subtle groove, "
        "warm strings arranged elegantly, occasional muted trumpet or tenor sax, "
        "SPACE in the arrangement - not every moment needs to be filled, "
        "silence is an instrument, let the music breathe, "
        "warm analog sound, intimate nightclub atmosphere, "
        "polished radio-ready mix, clear Italian diction, "
        "close microphone, dry vocal with subtle warmth"
    ),
}

# === REFERENCE VOICE ===
VOICE_REFERENCE_FILE = VOICE_DIR / "artist_voice.wav"

# === PARAMETRI GENERAZIONE ===
LYRIA_MODEL = "lyria-3-pro-preview"
LYRIA_CLIP_MODEL = "lyria-3-clip-preview"

# Seed-VC parametri per singing voice conversion
SEED_VC = {
    "diffusion_steps": 100,      # piu passi = pronuncia e qualita migliore
    "f0_condition": True,        # True per singing voice conversion
    "auto_f0_adjust": False,     # False per singing
    "semi_tone_shift": 0,        # shift di tonalita in semitoni
    "inference_cfg_rate": 0.4,
    "length_adjust": 1.0,
    "fp16": True,
}

# Demucs
DEMUCS_MODEL = "htdemucs"  # miglior modello per separazione vocale

# GPU
DEVICE = "cuda:0"  # usa la prima GPU
