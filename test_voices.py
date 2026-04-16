"""
Test A/B di reference voices con focus sulla pronuncia italiana della R.
Genera 5 voci diverse e le testa su Seed-VC con una frase ricca di R.
"""
import sys
import json
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import patch_torchaudio

import torch
import soundfile as sf
from rich.console import Console
from rich.table import Table
sys.path.insert(0, str(Path(__file__).parent / "models" / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "models" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from config import MODELS_DIR, GEMINI_API_KEY

console = Console()

# Frasi con molte R per testare la pronuncia
TEST_TEXTS = [
    "Ricordo il rumore del treno che attraversava la campagna romana, "
    "portando con se il profumo della primavera e il ricordo "
    "di un amore perduto tra le rose rosse del giardino.",

    "Il treno corre rapido verso Roma, il rumore delle rotaie "
    "risuona come un ritornello, ricordandomi le sere d'estate "
    "trascorse a guardare il tramonto rosso sul mare.",
]

# 5 diverse istruzioni per la voce — focus sulla R
VOICE_CONFIGS = [
    {
        "name": "v1_standard",
        "instruct": (
            "Voce maschile italiana, baritono caldo, tono intimo e emotivo. "
            "Pronuncia italiana nativa perfetta."
        ),
    },
    {
        "name": "v2_romano",
        "instruct": (
            "Voce maschile italiana con accento romano, baritono caldo e profondo. "
            "Pronuncia la R come un vero romano, vibrante e rotolata. "
            "Dizione chiara, ritmo naturale del parlato italiano."
        ),
    },
    {
        "name": "v3_toscano",
        "instruct": (
            "Voce maschile italiana con accento toscano, timbro caldo e morbido. "
            "Pronuncia perfetta della R italiana, mai gutturale. "
            "Parlata fluida e melodica come un cantautore toscano."
        ),
    },
    {
        "name": "v4_cantautore",
        "instruct": (
            "Voce di cantautore italiano professionista, come Fabrizio De André "
            "o Francesco De Gregori. Baritono espressivo con vibrato naturale. "
            "Pronuncia italiana impeccabile con R vibrante alveolare. "
            "Tono poetico e narrativo."
        ),
    },
    {
        "name": "v5_roca_italiana",
        "instruct": (
            "Voce maschile italiana leggermente roca e calda, come un cantautore "
            "che ha vissuto molto. Pronuncia ogni parola con cura, "
            "la R è sempre vibrante e italiana, mai straniera. "
            "Dizione perfetta dell'italiano standard, tono emotivo e profondo."
        ),
    },
]


def generate_test_voices():
    """Genera 5 reference voices e le salva."""
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torchaudio

    model_dir = str(MODELS_DIR / "CosyVoice" / "pretrained_models" / "CosyVoice2-0.5B")
    console.print("[yellow]Caricamento CosyVoice2...[/yellow]")
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False)

    output_dir = Path(__file__).parent / "voice" / "test_voices"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Usa la voce attuale come prompt_wav (trimmata a 25s per il limite di CosyVoice)
    current_voice = Path(__file__).parent / "voice" / "artist_voice.wav"
    trimmed_voice = output_dir / "prompt_trimmed.wav"
    if current_voice.exists():
        data, sr = sf.read(str(current_voice))
        max_samples = int(25 * sr)
        sf.write(str(trimmed_voice), data[:max_samples], sr)
        prompt_wav = str(trimmed_voice)
        console.print(f"  [dim]Prompt wav trimmato a 25s[/dim]")
    else:
        prompt_wav = None

    for config in VOICE_CONFIGS:
        name = config["name"]
        instruct = config["instruct"]
        console.print(f"\n[cyan]Generazione {name}...[/cyan]")
        console.print(f"  [dim]{instruct[:80]}...[/dim]")

        all_audio = []
        for text in TEST_TEXTS:
            try:
                if prompt_wav:
                    # instruct2: usa prompt_wav per il timbro base
                    for chunk in model.inference_instruct2(
                        tts_text=text,
                        instruct_text=instruct,
                        prompt_wav=prompt_wav,
                        stream=False,
                    ):
                        all_audio.append(chunk["tts_speech"])
                else:
                    # instruct: usa speaker ID integrato
                    for chunk in model.inference_instruct(
                        tts_text=text,
                        spk_id='',
                        instruct_text=instruct,
                        stream=False,
                    ):
                        all_audio.append(chunk["tts_speech"])
            except Exception as e:
                console.print(f"  [yellow]Errore su testo: {e}[/yellow]")
                continue

        if not all_audio:
            console.print(f"  [red]Nessun audio generato per {name}[/red]")
            continue

        full_audio = torch.cat(all_audio, dim=-1)
        if full_audio.dim() == 1:
            full_audio = full_audio.unsqueeze(0)

        out_path = output_dir / f"{name}.wav"
        torchaudio.save(str(out_path), full_audio, 22050)
        duration = full_audio.shape[-1] / 22050
        console.print(f"  [green]✓ {name}: {duration:.1f}s[/green]")

    return output_dir


def score_voices_with_gemini(voices_dir: Path):
    """Valuta ogni voce con Gemini per pronuncia italiana."""
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    results = []

    for wav_file in sorted(voices_dir.glob("*.wav")):
        name = wav_file.stem
        console.print(f"\n[cyan]Valutazione {name}...[/cyan]")

        with open(wav_file, 'rb') as f:
            audio_data = f.read()

        prompt = '''Ascolta questa voce italiana parlata. Valuta SOLO la pronuncia.
Focus specifico sulla lettera R: deve essere vibrante alveolare italiana, NON gutturale/straniera.
Rispondi in JSON:
{"pronunciation_overall": X, "r_quality": X, "native_likeness": X, "accent": "descrizione accento", "notes": "breve commento"}
Punteggi 1-10. 10 = madrelingua italiano perfetto.'''

        try:
            for model_name in ['gemini-2.5-pro', 'gemini-2.5-flash']:
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
            scores["name"] = name
            results.append(scores)
            console.print(f"  R: {scores.get('r_quality', '?')}/10, "
                         f"Native: {scores.get('native_likeness', '?')}/10, "
                         f"Accent: {scores.get('accent', '?')}")
        except Exception as e:
            console.print(f"  [red]Errore: {e}[/red]")
            results.append({"name": name, "r_quality": 0, "native_likeness": 0})

    # Tabella risultati
    table = Table(title="Test Voci - Pronuncia Italiana")
    table.add_column("Voce", style="cyan")
    table.add_column("R quality", justify="center")
    table.add_column("Native", justify="center")
    table.add_column("Overall", justify="center")
    table.add_column("Accento")

    for r in sorted(results, key=lambda x: x.get("r_quality", 0), reverse=True):
        table.add_row(
            r["name"],
            f"{r.get('r_quality', '?')}/10",
            f"{r.get('native_likeness', '?')}/10",
            f"{r.get('pronunciation_overall', '?')}/10",
            r.get("accent", "?")[:40]
        )
    console.print(table)

    # Salva risultati
    results_file = voices_dir / "scores.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Trova la migliore
    best = max(results, key=lambda x: x.get("r_quality", 0) + x.get("native_likeness", 0))
    console.print(f"\n[bold green]Migliore: {best['name']} "
                  f"(R: {best.get('r_quality')}, Native: {best.get('native_likeness')})[/bold green]")

    return best["name"]


if __name__ == "__main__":
    console.print("[bold magenta]Test A/B Voci - Focus R Italiana[/bold magenta]\n")

    voices_dir = generate_test_voices()
    best_name = score_voices_with_gemini(voices_dir)

    console.print(f"\n[bold]Per usare la voce migliore:[/bold]")
    console.print(f"  cp voice/test_voices/{best_name}.wav voice/artist_voice.wav")
