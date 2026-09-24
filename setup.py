"""
Hunyuan3D 2.1 Cloud — extension setup script.
Modly volá:  python setup.py <json_args>  (nebo pozičně, viz oficiální rozšíření).
Vytvoří venv a nainstaluje jen lehké závislosti (žádný torch, žádná GPU).
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(python_exe: str, ext_dir: Path, **_ignored) -> None:
    venv = ext_dir / "venv"
    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)
    print("[setup] Installing dependencies …")
    pip(venv, "install", "--upgrade", "pip")
    pip(venv, "install", "gradio_client", "huggingface_hub", "Pillow", "numpy", "trimesh")
    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(python_exe=args["python_exe"], ext_dir=Path(args["ext_dir"]))
    elif len(sys.argv) >= 3:
        setup(python_exe=sys.argv[1], ext_dir=Path(sys.argv[2]))
    else:
        print("Usage: python setup.py '{\"python_exe\":\"...\",\"ext_dir\":\"...\"}'")
        sys.exit(1)
