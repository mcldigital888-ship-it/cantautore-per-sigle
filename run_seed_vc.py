"""
Wrapper per Seed-VC inference che applica la patch torchaudio.
Esegui come: python run_seed_vc.py --source ... --target ... --output ...
"""
import sys
import os
import runpy
from pathlib import Path

# Applica patch torchaudio PRIMA di qualsiasi import di seed-vc
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))
import patch_torchaudio  # noqa: F401

# Imposta directory di lavoro su seed-vc (per i path relativi dei checkpoint)
seed_vc_dir = project_dir / "models" / "seed-vc"
os.chdir(seed_vc_dir)
sys.path.insert(0, str(seed_vc_dir))

# Esegui inference.py di seed-vc come se fosse __main__
runpy.run_path(str(seed_vc_dir / "inference.py"), run_name="__main__")
