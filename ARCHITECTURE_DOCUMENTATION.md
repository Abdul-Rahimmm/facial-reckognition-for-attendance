# Architecture Documentation

The project is intentionally consolidated into one small top-level Python app. The older package-style implementation was removed because it duplicated the active code and used incompatible database assumptions.

## Active Modules

- `main.py`: CLI, GUI launch, setup checks, and application orchestration.
- `gui.py`: Tkinter interface for registration, camera display, logs, and CSV export.
- `db.py`: SQLite storage for students, embeddings, and attendance logs.
- `recognition.py`: optional automatic face recognition and lightweight fallback behavior.
- `config.py`: environment configuration and setup diagnostics.
- `test_system.py`: focused regression tests for setup, database behavior, duplicate suppression, and minimal-mode safety.

## Runtime Modes

### Lightweight Mode

Installed with `requirements-minimal.txt`.

Supports:

- SQLite database
- Tkinter GUI
- Camera preview when OpenCV is available
- Manual student enrollment
- Manual attendance by enrollment number
- Logs and CSV export

### Full Mode

Installed with `requirements-full.txt`.

Adds:

- `face-recognition` automatic face detection and encoding
- cached known face encodings
- frame skipping and downscaled detection for weaker devices
- automatic attendance logging with duplicate suppression

The automatic-recognition dependency stack is only installed on Python 3.10 and 3.11. On Python 3.12+, `requirements-full.txt` skips `face-recognition` so users do not hit `dlib` CMake wheel build failures.

## Data Model

SQLite remains the only database.

- `students(id, name, enrollment_number, created_at)`
- `face_embeddings(id, student_id, embedding, encoding_format, created_at)`
- `attendance_logs(id, student_id, timestamp, session, confidence)`

New embeddings are stored as JSON bytes. Legacy pickle embeddings remain readable so existing local databases do not need to be deleted.

## Performance Choices

- Known face encodings are averaged and cached when students load.
- Recognition runs every `ATTENDANCE_RECOGNITION_INTERVAL` frames.
- Detection can run on scaled frames using `ATTENDANCE_FRAME_SCALE`.
- SQLite indexes cover enrollment lookup, embedding lookup, and attendance history queries.
- Duplicate attendance logs are suppressed for `ATTENDANCE_DUPLICATE_LOG_WINDOW` seconds.

## Setup Diagnostics

Run:

```bash
python main.py --check
```

The check reports minimal readiness, full automatic recognition readiness, missing packages, camera status, and current environment-derived configuration.

For first-run setup and launch, use:

```bash
python setup_app.py --run
```

The bootstrapper owns venv creation, dependency installation, Tkinter repair hints, and GUI-to-CLI fallback. Full automatic recognition uses `.venv-full` and only chooses Python 3.10/3.11 when available.
