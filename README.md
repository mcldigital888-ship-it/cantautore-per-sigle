# Cantautore Digitale

Pipeline automatica per creare il tuo artista digitale italiano: genera testi, musica, separa le tracce e mixa tutto in una canzone completa.

**Stack**: Gemini 2.5 Pro (testi) → Lyria 3 Pro (musica) → Demucs (separazione vocals) → Mix professionale

## Cosa fa

- Genera testi originali con personalità coerente (sistema "Artist Brain" con DNA, memoria, evoluzione)
- Produce musica completa con Lyria 3 Pro (soul-jazz, pop, cantautore...)
- Separa vocals e strumentali con Demucs
- Mixa con processing professionale (EQ, compressione, sidechain ducking)
- Estende la durata con intro/outro strumentali dalla canzone stessa
- Ogni canzone costa ~$0.08

## Setup

### 1. Clona e installa

```bash
git clone https://github.com/TUO_USERNAME/Cantautore_digitale.git
cd Cantautore_digitale

# Crea virtual environment
python3 -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Configura API Key

```bash
# Copia il template
cp .env.example .env

# Modifica .env con la tua chiave
# Ottienila gratis su: https://aistudio.google.com/apikey
nano .env
```

### 3. Scarica modelli AI

```bash
python setup_models.py
```

### 4. (Opzionale) Genera la voce dell'artista

```bash
python generate_voice.py
```

## Uso

```bash
# Genera una canzone con un tema
python cantautore.py --tema "nostalgia di un viaggio in treno"

# Genera con titolo specifico
python cantautore.py --tema "amore perduto" --titolo "Stelle Cadenti"

# Genera 5 canzoni automaticamente
python cantautore.py --genera 5
```

## Personalizza il tuo artista

### Profilo (`config.py`)
Modifica nome, genere, temi, voce e style_prompt per creare il tuo artista unico.

### DNA (`brain/cantautore_digitale/dna.json`)
La personalità profonda dell'artista: backstory, ossessioni, vocabolario, parole vietate, luoghi, abitudini. Il sistema inietta tutto questo nei prompt per mantenere coerenza.

### Memoria (`brain/*/memory.json`)
Si genera automaticamente: dopo ogni canzone, il sistema analizza il testo e salva temi, metafore, frasi killer. Le canzoni successive "ricordano" quelle precedenti.

## Costo per canzone

| Componente | Costo |
|-----------|-------|
| Testi (Gemini 2.5 Pro) | ~$0.01 |
| Musica (Lyria 3 Pro) | ~$0.08 |
| Separazione (Demucs) | $0 (locale) |
| Mix + Mastering | $0 (locale) |
| **Totale** | **~$0.09** |

## Requisiti

- Python 3.10+
- GPU NVIDIA con CUDA (per Demucs e voice conversion)
- Google Gemini API key (gratis su aistudio.google.com)
- ~10GB spazio disco per modelli

## Google Colab (GPU gratis)

Il modo più semplice per iniziare — **zero costi GPU**:

1. Apri `cantautore_colab.ipynb` su [Google Colab](https://colab.research.google.com)
2. `Runtime → Change runtime type → T4 GPU`
3. Premi `Ctrl+F9` (Run All)
4. Inserisci la tua API Key quando richiesto
5. Genera canzoni!

**Costo:** ~€0.10 per canzone (solo API Google)

## Dashboard Web

Per un'esperienza visuale con player, progress bar e gestione album:

```bash
python web_app.py
# Apri http://localhost:5000
```

## Producer Toolkit

```bash
# Separa qualsiasi canzone in stems (drums, bass, vocals, other)
python export_kit.py separa canzone.wav

# Importa una base esterna e cantaci sopra
python import_base.py --base beat.wav --tema "amore perduto" --voce mia_voce

# Genera con export stems automatico
python cantautore.py --tema "Roma di notte" --voce mia_voce --export-stems

# Gestisci voci
python manage_voices.py add mia_voce registrazione.wav
python manage_voices.py list
```

## Struttura

```
├── config.py             # Configurazione artista e parametri
├── cantautore.py         # Pipeline principale
├── cantautore2.py        # Pipeline soul/jazz
├── artist_brain.py       # Sistema personalità + memoria
├── web_app.py            # Dashboard web
├── templates/index.html  # UI dashboard
├── export_kit.py         # Separazione stems per DAW
├── import_base.py        # Importa basi esterne
├── manage_voices.py      # Gestione profili voce
├── cantautore_colab.ipynb # Notebook Google Colab
├── setup_runpod.sh       # Setup automatico RunPod
├── brain/                # DNA e memoria degli artisti
├── voices/               # Profili voce
├── models/               # Modelli AI (gitignored)
├── output/               # Canzoni generate (gitignored)
└── temp/                 # File temporanei (gitignored)
```

## License

MIT
