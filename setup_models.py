"""
Setup: scarica e installa tutti i modelli AI necessari.
Esegui una volta sola: python setup_models.py
"""
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run(cmd, cwd=None, check=True):
    """Esegui un comando e mostra l'output."""
    console.print(f"[dim]$ {cmd}[/dim]")
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=False, text=True
    )
    if check and result.returncode != 0:
        console.print(f"[red]Errore nel comando: {cmd}[/red]")
        sys.exit(1)
    return result


def setup_cosyvoice():
    """Scarica e installa CosyVoice per la generazione della voce."""
    cosyvoice_dir = MODELS_DIR / "CosyVoice"
    if cosyvoice_dir.exists():
        console.print("[green]✓ CosyVoice già presente[/green]")
        return

    console.print(Panel("Installazione CosyVoice...", style="blue"))
    run(f"git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git {cosyvoice_dir}")
    run(f"{sys.executable} -m pip install -r requirements.txt", cwd=cosyvoice_dir)

    # Scarica modello pre-trained
    console.print("[yellow]Scaricamento modello CosyVoice2-0.5B...[/yellow]")
    run(
        f"{sys.executable} -c \""
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('FunAudioLLM/CosyVoice2-0.5B', "
        f"local_dir='{cosyvoice_dir}/pretrained_models/CosyVoice2-0.5B')\"",
    )
    console.print("[green]✓ CosyVoice installato[/green]")


def setup_seed_vc():
    """Scarica e installa Seed-VC per la voice conversion."""
    seed_vc_dir = MODELS_DIR / "seed-vc"
    if seed_vc_dir.exists():
        console.print("[green]✓ Seed-VC già presente[/green]")
        return

    console.print(Panel("Installazione Seed-VC...", style="blue"))
    run(f"git clone https://github.com/Plachtaa/seed-vc.git {seed_vc_dir}")
    run(f"{sys.executable} -m pip install -r requirements.txt", cwd=seed_vc_dir)

    # I checkpoint vengono scaricati automaticamente al primo inference
    console.print("[green]✓ Seed-VC installato (checkpoint scaricati al primo uso)[/green]")


def setup_demucs():
    """Verifica che Demucs sia installato."""
    try:
        import demucs
        console.print("[green]✓ Demucs già installato[/green]")
    except ImportError:
        console.print(Panel("Installazione Demucs...", style="blue"))
        run(f"{sys.executable} -m pip install demucs")
        console.print("[green]✓ Demucs installato[/green]")


def setup_genai():
    """Verifica che google-genai sia installato."""
    try:
        import google.genai
        console.print("[green]✓ Google GenAI già installato[/green]")
    except ImportError:
        console.print(Panel("Installazione Google GenAI...", style="blue"))
        run(f"{sys.executable} -m pip install google-genai")
        console.print("[green]✓ Google GenAI installato[/green]")


def verify_gpu():
    """Verifica le GPU disponibili."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / 1e9
                console.print(f"[green]✓ GPU {i}: {name} ({mem:.0f} GB)[/green]")
        else:
            console.print("[red]✗ Nessuna GPU CUDA trovata![/red]")
            sys.exit(1)
    except ImportError:
        console.print("[red]✗ PyTorch non installato![/red]")
        sys.exit(1)


def main():
    console.print(Panel(
        "[bold]Setup Cantautore Digitale[/bold]\n"
        "Installazione di tutti i modelli AI necessari",
        style="magenta"
    ))

    verify_gpu()
    setup_genai()
    setup_demucs()
    setup_cosyvoice()
    setup_seed_vc()

    console.print("\n")
    console.print(Panel(
        "[bold green]Setup completato![/bold green]\n\n"
        "Prossimi passi:\n"
        "1. Genera la voce dell'artista:\n"
        "   [cyan]python generate_voice.py[/cyan]\n\n"
        "2. Genera una canzone:\n"
        "   [cyan]python cantautore.py --tema 'nostalgia di un viaggio in treno'[/cyan]",
        style="green"
    ))


if __name__ == "__main__":
    main()
