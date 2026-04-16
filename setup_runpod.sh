#!/bin/bash
# ============================================================
# Setup Automatico RunPod - Cantautore Digitale
# ============================================================
# Esegui questo script UNA VOLTA quando crei un nuovo pod:
#   bash setup_runpod.sh
#
# Dopo il setup, genera canzoni con:
#   python cantautore.py --tema "nostalgia di un viaggio in treno"
#   python cantautore.py --tema "amore perduto" --voce mia_voce
# ============================================================

set -e  # Esci al primo errore

echo "============================================================"
echo "  SETUP CANTAUTORE DIGITALE - RunPod"
echo "============================================================"
echo ""

# === 1. VERIFICA GPU ===
echo "[1/6] Verifica GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  ✓ GPU trovata"
else
    echo "  ✗ ERRORE: Nessuna GPU trovata! Assicurati di aver scelto un pod con GPU."
    exit 1
fi
echo ""

# === 2. INSTALLA DIPENDENZE ===
echo "[2/6] Installazione dipendenze Python..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install google-genai -q
echo "  ✓ Dipendenze installate"
echo ""

# === 3. SETUP MODELLI AI ===
echo "[3/6] Download modelli AI (può richiedere 5-10 minuti)..."
python setup_models.py
echo ""

# === 4. CONFIGURA API KEY ===
echo "[4/6] Configurazione API Key..."
if [ -f .env ]; then
    echo "  ✓ File .env già presente"
else
    echo ""
    echo "  ╔══════════════════════════════════════════════════╗"
    echo "  ║  SERVE LA TUA GEMINI API KEY                    ║"
    echo "  ║                                                  ║"
    echo "  ║  1. Vai su: https://aistudio.google.com/apikey  ║"
    echo "  ║  2. Crea una nuova API key (gratis)             ║"
    echo "  ║  3. Copiala e incollala qui sotto                ║"
    echo "  ╚══════════════════════════════════════════════════╝"
    echo ""
    read -p "  Incolla la tua GEMINI_API_KEY: " api_key
    echo "GEMINI_API_KEY=${api_key}" > .env
    echo "  ✓ API Key salvata in .env"
fi
echo ""

# === 5. CREA CARTELLE ===
echo "[5/6] Creazione cartelle..."
mkdir -p voices/default
mkdir -p voices/mia_voce
mkdir -p output
mkdir -p output_artist2
mkdir -p temp
echo "  ✓ Cartelle create"
echo ""

# === 6. GENERA VOCE DEFAULT ===
echo "[6/6] Generazione voce default..."
if [ -f voice/artist_voice.wav ]; then
    echo "  ✓ Voce default già presente"
else
    echo "  Generazione voce AI default (fallback)..."
    python generate_voice.py || echo "  ⚠ Voce default non generata (puoi caricare la tua)"
fi
echo ""

echo "============================================================"
echo "  ✓ SETUP COMPLETATO!"
echo "============================================================"
echo ""
echo "  PROSSIMI PASSI:"
echo ""
echo "  1. CARICA LA TUA VOCE (consigliato):"
echo "     python manage_voices.py add mia_voce /path/to/tua_registrazione.wav"
echo ""
echo "  2. GENERA UNA CANZONE:"
echo "     python cantautore.py --tema 'nostalgia di un viaggio in treno'"
echo ""
echo "  3. GENERA CON LA TUA VOCE:"
echo "     python cantautore.py --tema 'amore perduto' --voce mia_voce"
echo ""
echo "  4. GENERA UN ALBUM:"
echo "     python cantautore.py --album 10 --nome-album 'Il Mio Album'"
echo ""
echo "  5. LISTA VOCI DISPONIBILI:"
echo "     python manage_voices.py list"
echo ""
echo "============================================================"
