# Face Recognition Attendance System

A local-first attendance register built with Python, SQLite, Tkinter, and optional automatic face recognition.

The app now has two setup paths:

- **Lightweight mode**: camera preview, student records, manual attendance, logs, and CSV export.
- **Full mode**: everything in lightweight mode plus automatic face recognition with `face-recognition`.

All data stays on the device in SQLite. No network service is required at runtime.

## Quick Start

The easiest first run is the project bootstrapper:

```bash
python setup_app.py --run
```

On macOS/Linux you can also run:

```bash
./run.sh
```

On Windows:

```bat
run.bat
```

The bootstrapper creates or reuses the right virtual environment, installs dependencies, checks the setup, and starts GUI mode when possible. If Tkinter GUI support is missing, it launches CLI mode instead and prints the exact fix.

### Lightweight Setup

Use this on weak devices or machines where `dlib` is hard to install:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-minimal.txt
.venv/bin/python main.py --check
.venv/bin/python main.py --mode cli
```

Lightweight mode supports manual attendance:

```bash
.venv/bin/python main.py --mode cli
```

Then use:

- `register` to add a student.
- `people` to view registered staff/students.
- `delete` to remove a registration and its attendance records.
- `manual` to mark attendance by enrollment number.
- `logs` to view attendance.
- `export` to export CSV.

The GUI can also register people with an optional staff/student ID, class or department, and training photos. Use **Manage people** to view registered staff/students and delete a registration when needed.

### Full Automatic Face Recognition Setup

Use this with **Python 3.10 or 3.11**. The `face-recognition` package depends on `dlib`, and `dlib` commonly fails to build on Python 3.12+ and Python 3.13 with a long CMake error.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-full.txt
.venv/bin/python main.py --check
.venv/bin/python main.py
```

On Python 3.12+, `requirements-full.txt` intentionally skips `face-recognition` so installation does not fail. The app will still run in lightweight/manual mode. If full setup is painful on a device, stay on `requirements-minimal.txt`; the app remains useful through manual attendance.

## Run Commands

```bash
python main.py
python main.py --mode gui
python main.py --mode cli
python main.py --database data/attendance.db
python main.py --check
python setup_app.py --check-only
```

If `python main.py` says `No module named '_tkinter'`, that Python build does not include Tk GUI support. The app will fall back to CLI mode automatically. To use the GUI, install or select a Python build with Tk support, then recreate the venv. On macOS, the python.org installer usually includes Tk support; Homebrew Python may require separate Tk setup.

For full automatic recognition through the bootstrapper:

```bash
python setup_app.py --profile full --run
```

This uses `.venv-full` and looks for Python 3.10/3.11 before installing the full dependency set.

## Useful Environment Options

```bash
ATTENDANCE_BACKEND=minimal
ATTENDANCE_LOW_POWER=true
ATTENDANCE_CAMERA_INDEX=0
ATTENDANCE_CAMERA_WIDTH=640
ATTENDANCE_CAMERA_HEIGHT=480
ATTENDANCE_CAMERA_FPS=15
ATTENDANCE_FRAME_SCALE=0.5
ATTENDANCE_RECOGNITION_INTERVAL=5
ATTENDANCE_DUPLICATE_LOG_WINDOW=300
ATTENDANCE_TOLERANCE=0.6
ATTENDANCE_MIN_CONFIDENCE=0.5
ATTENDANCE_DB_PATH=data/attendance.db
```

Low-power defaults reduce camera FPS, resize frames before recognition, and run recognition every few frames.

## What The App Stores

- `students`: names, enrollment numbers, optional class/department labels, and selected photo paths.
- `face_embeddings`: optional face vectors for automatic recognition.
- `attendance_logs`: timestamped attendance records.

New embeddings are stored as JSON bytes instead of pickle. Existing legacy pickle embeddings are still readable for backward compatibility.

## Troubleshooting

- Run `python main.py --check` first. It lists missing dependencies, Python compatibility, and camera status.
- If GUI mode is unavailable with `No module named '_tkinter'`, run `python main.py --mode cli` or use a Python build that includes Tkinter.
- If you see a `Failed building wheel for dlib` error, you are using a Python version that is too new for the automatic-recognition dependency. Use Python 3.10/3.11 for full mode or reinstall with `requirements-minimal.txt`.
- If the camera fails, close other camera apps and try `ATTENDANCE_CAMERA_INDEX=1`.
- If full recognition dependencies fail to install, use `requirements-minimal.txt` and CLI manual attendance.
- On low-end hardware, keep `ATTENDANCE_LOW_POWER=true`, `ATTENDANCE_FRAME_SCALE=0.5`, and `ATTENDANCE_RECOGNITION_INTERVAL=5`.

## Development

Run the test suite:

```bash
python test_system.py
```

The current source of truth is the top-level implementation:

- `main.py`
- `gui.py`
- `db.py`
- `recognition.py`
- `config.py`
- `test_system.py`
