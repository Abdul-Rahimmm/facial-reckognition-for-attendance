"""
First-run setup and launcher for the attendance app.

This script intentionally uses only the Python standard library so it can run
before the project dependencies are installed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parent
MINIMAL_VENV = ROOT / ".venv"
FULL_VENV = ROOT / ".venv-full"
SUPPORTED_FULL_PYTHONS = ((3, 11), (3, 10))


def log(message: str) -> None:
    print(f"[setup] {message}")


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    log(" ".join(command))
    return subprocess.run(command, cwd=ROOT, check=check)


def capture_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def python_version(python_command: list[str]) -> Optional[tuple[int, int, int]]:
    result = capture_command([*python_command, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"])
    if result.returncode != 0:
        return None
    try:
        return tuple(int(part) for part in result.stdout.strip().split("."))
    except ValueError:
        return None


def python_supports_full(python_command: list[str]) -> bool:
    version = python_version(python_command)
    return bool(version and version[:2] in SUPPORTED_FULL_PYTHONS)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def module_status(python: Path, module: str) -> tuple[bool, str]:
    result = capture_command([str(python), "-c", f"import {module}; print('ok')"])
    if result.returncode == 0:
        return True, "ok"
    detail = (result.stderr or result.stdout).strip().splitlines()
    return False, detail[-1] if detail else "not available"


def choose_python(profile: str) -> list[str]:
    if profile == "minimal":
        return [sys.executable]

    if python_supports_full([sys.executable]):
        return [sys.executable]

    for candidate in ("python3.11", "python3.10", "py -3.11", "py -3.10"):
        parts = candidate.split()
        executable = shutil.which(parts[0])
        if not executable:
            continue
        command = [executable, *parts[1:]]
        if python_supports_full(command):
            return command

    log("Python 3.10/3.11 was not found. Full face recognition will not be installed on this Python.")
    log("Continuing with a safe lightweight-compatible venv.")
    return [sys.executable]


def create_venv(base_python: list[str], venv_dir: Path, recreate: bool = False) -> Path:
    if recreate and venv_dir.exists():
        log(f"Removing unhealthy venv at {venv_dir}")
        shutil.rmtree(venv_dir)
    if not venv_python(venv_dir).exists():
        log(f"Creating virtual environment at {venv_dir.name}")
        run_command([*base_python, "-m", "venv", str(venv_dir)])
    return venv_python(venv_dir)


def install_requirements(python: Path, profile: str) -> None:
    requirements = "requirements-full.txt" if profile == "full" else "requirements-minimal.txt"
    run_command([str(python), "-m", "pip", "install", "-r", requirements])


def cli_ready(python: Path) -> bool:
    return all(module_status(python, module)[0] for module in ("cv2", "numpy", "PIL"))


def gui_ready(python: Path) -> bool:
    return cli_ready(python) and module_status(python, "tkinter")[0]


def print_module_report(python: Path) -> None:
    log(f"Python: {python}")
    for module in ("cv2", "numpy", "PIL", "tkinter", "face_recognition"):
        ok, detail = module_status(python, module)
        print(f"  {module}: {'ok' if ok else detail}")


def homebrew_formula_for_python(python_command: list[str]) -> Optional[str]:
    version = python_version(python_command)
    if not version:
        return None
    return f"python-tk@{version[0]}.{version[1]}"


def maybe_repair_tk_macos(base_python: list[str], assume_yes: bool) -> None:
    if platform.system() != "Darwin":
        return

    brew = shutil.which("brew")
    if not brew:
        log("Homebrew not found. Install a Python build with Tk support, or use CLI mode.")
        return

    formula = homebrew_formula_for_python(base_python)
    if not formula:
        return

    log(f"Tkinter is missing. Homebrew may fix it with: brew install {formula}")
    if not assume_yes:
        if not sys.stdin.isatty():
            log("Skipping Homebrew install because this is not an interactive terminal.")
            return
        answer = input("Install the matching Homebrew Tk package now? [y/N]: ").strip().lower()
        if answer != "y":
            log("Skipping Tk repair. GUI may be unavailable; CLI mode will still work.")
            return

    run_command([brew, "install", formula], check=False)


def print_os_tk_hint() -> None:
    system = platform.system()
    if system == "Darwin":
        print("Tkinter is an OS/Python build dependency on macOS.")
        print("- Homebrew: brew install python-tk@3.13  # adjust version to match Python")
        print("- Alternative: install Python from https://www.python.org/downloads/macos/")
    elif system == "Linux":
        print("Install the Tk package for your distro, then recreate the venv.")
        print("- Debian/Ubuntu: sudo apt install python3-tk")
        print("- Fedora: sudo dnf install python3-tkinter")
        print("- Arch: sudo pacman -S tk")
    elif system == "Windows":
        print("Use the official Python installer and enable Tcl/Tk support, then recreate the venv.")


def ensure_environment(profile: str, assume_yes: bool, repair_tk: bool) -> Path:
    venv_dir = FULL_VENV if profile == "full" else MINIMAL_VENV
    base_python = choose_python(profile)
    python = create_venv(base_python, venv_dir)

    if not cli_ready(python):
        install_requirements(python, profile)

    if not module_status(python, "tkinter")[0] and repair_tk:
        maybe_repair_tk_macos(base_python, assume_yes)
        if not module_status(python, "tkinter")[0]:
            print_os_tk_hint()

    return python


def run_check(python: Path) -> int:
    result = run_command([str(python), "main.py", "--check"], check=False)
    return result.returncode


def run_app(python: Path) -> int:
    mode = "gui" if gui_ready(python) else "cli"
    if mode == "cli":
        log("GUI is not available in this environment; launching CLI mode.")
    return run_command([str(python), "main.py", "--mode", mode], check=False).returncode


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up and launch the Face Attendance app")
    parser.add_argument("--profile", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--run", action="store_true", help="Launch the app after setup")
    parser.add_argument("--check-only", action="store_true", help="Report environment status without installing")
    parser.add_argument("--yes", action="store_true", help="Allow setup to run supported OS package installs without prompting")
    parser.add_argument("--no-repair-tk", action="store_true", help="Do not attempt Homebrew Tk repair")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    venv_dir = FULL_VENV if args.profile == "full" else MINIMAL_VENV
    existing_python = venv_python(venv_dir)

    if args.check_only:
        if not existing_python.exists():
            log(f"{venv_dir.name} does not exist yet. Run python setup_app.py --profile {args.profile} --run")
            return 1
        print_module_report(existing_python)
        return 0 if cli_ready(existing_python) else 1

    python = ensure_environment(args.profile, args.yes, not args.no_repair_tk)
    print_module_report(python)
    check_code = run_check(python)
    if args.run:
        return run_app(python)
    return check_code


if __name__ == "__main__":
    raise SystemExit(main())
