# Guida alla Registrazione della Tua Voce

## Cosa serve

La tua voce registrata in un file `.wav` o `.mp3`. Il sistema la usa come riferimento per convertire tutte le canzoni generate nel tuo timbro vocale.

## Requisiti minimi

| Cosa | Minimo | Consigliato |
|------|--------|-------------|
| **Durata** | 15 secondi | 30-60 secondi |
| **Formato** | .wav o .mp3 | .wav (no compressione) |
| **Ambiente** | Silenzioso | Stanza chiusa, niente eco |
| **Microfono** | Telefono | Microfono USB (~30-50 euro) |

## Come registrare

### Opzione 1: Dal telefono (per iniziare)
1. Usa l'app **Memo Vocali** (iPhone) o **Registratore** (Android)
2. Metti il telefono a ~20cm dalla bocca
3. Stanza silenziosa, chiudi finestre
4. **Canta** una melodia in italiano per 30-60 secondi
5. Esporta come `.wav` o `.m4a`

### Opzione 2: Con microfono USB (consigliato)
- **Budget**: Fifine K669 (~25 euro), Maono AU-A04 (~35 euro)
- **Medio**: Rode NT-USB Mini (~90 euro), Blue Yeti (~100 euro)
- Usa Audacity (gratis) o GarageBand per registrare
- Esporta in WAV 44100Hz 16-bit

### Cosa cantare
- **Meglio cantare che parlare** — il sistema converte voci cantate
- Canta una canzone italiana che conosci bene (anche La la la va bene)
- Varietà: parti basse e alte, piano e forte
- Non sussurrare, non urlare — voce naturale
- Se sbagli, non importa — il sistema prende solo il timbro

### Errori da evitare
- **NO** musica di sottofondo
- **NO** TV, ventilatore, traffico
- **NO** troppo vicino al microfono (distorsione)
- **NO** troppo lontano (riverbero ambientale)
- **NO** file troppo corti (< 10 secondi)

## Come caricare la voce

### Su RunPod
1. Nella sidebar di RunPod, usa il **File Manager** per caricare il file
2. Poi nel terminale:
```bash
python manage_voices.py add mia_voce /path/al/file.wav
```

### Gestione voci
```bash
# Lista voci disponibili
python manage_voices.py list

# Aggiungi voce
python manage_voices.py add mia_voce registrazione.wav

# Testa qualità con AI
python manage_voices.py test mia_voce

# Imposta come default
python manage_voices.py set-default mia_voce

# Info su una voce
python manage_voices.py info mia_voce
```

### Usa la tua voce nelle canzoni
```bash
# Con la tua voce
python cantautore.py --tema "Roma di notte" --voce mia_voce

# Con voce default
python cantautore.py --tema "Roma di notte"

# Artista 2 con la tua voce
python cantautore2.py --tema "notte al porto" --voce mia_voce
```

## Come migliorare nel tempo

| Fase | Cosa fare | Risultato |
|------|-----------|-----------|
| **V1** | Registra 30s dal telefono | Funziona, timbro approssimativo |
| **V2** | Registra 1-2 minuti cantando | Molto meglio |
| **V3** | Usa microfono USB, stanza trattata | Salto di qualità |
| **V4** | Registra 3-5 min con stili diversi | Voce quasi perfetta |

Per aggiornare la voce, basta ricaricarla:
```bash
python manage_voices.py add mia_voce nuova_registrazione.wav
```

Il file vecchio viene sovrascritto. Le canzoni future useranno la nuova voce.

## Profili multipli

Puoi avere più voci e scegliere quale usare:
```bash
python manage_voices.py add mia_voce voce_marco.wav
python manage_voices.py add voce_femminile voce_donna.wav
python manage_voices.py add voce_profonda voce_bassa.wav

# Usa quella che vuoi
python cantautore.py --tema "..." --voce mia_voce
python cantautore.py --tema "..." --voce voce_femminile
```
