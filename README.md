# Face Recognition — Installation, Dependencies & Usage

Overview
--------
This repository implements a webcam-based face recognition workflow using DeepFace and OpenCV with built-in anti-spoofing protection. The system uses DeepFace's advanced anti-spoofing analysis module to detect and prevent fake face attacks (photos, videos, masks).

Included files
--------------
- `face_db.py` — create/append a face database (multi-embedding per person)
- `live_face_recognition.py` — run live recognition against the DB
- `config.json` — runtime settings (thresholds, detector backends, etc.)
- `run_face_recognition.bat` — simple launcher (optional)
- `DB_USAGE.md` — details about DB format and behavior
- `face_db.json` — (user data; recommended to keep out of git)

Supported OS and Python
-----------------------
- Tested on Windows 10/11
- Python 3.8 — 3.11 recommended

High-level dependencies
-----------------------
Core Python packages the project uses:
- deepface
- tensorflow or tensorflow-cpu (DeepFace requires TensorFlow)
- opencv-python
- numpy
- tqdm

Optional packages (helpful for development):
- matplotlib
- pandas

Recommended pinned versions (example)
-------------------------------------
These versions are suggestions for a stable environment:
- tensorflow==2.11.0 (or `tensorflow-cpu==2.11.0` for CPU-only)
- deepface==0.0.75
- opencv-python==4.6.0.66
- numpy==1.23.5
- tqdm==4.64.1

Quick setup (Windows)
---------------------
Open PowerShell or Command Prompt in the project root.

1. Create and activate a virtual environment:

   python -m venv venv
   venv\Scripts\activate

2. Upgrade pip and install core packages:

   python -m pip install --upgrade pip
   pip install "tensorflow-cpu"  # or `tensorflow` if you need GPU support and have drivers
   pip install deepface opencv-python numpy tqdm

3. (Optional) Save installed packages to a requirements file:

   pip freeze > requirements.txt

Alternative: install pinned packages from a requirements file (if provided):

   pip install -r requirements.txt

Notes about TensorFlow and GPU
------------------------------
- For GPU acceleration you must install a TensorFlow wheel compatible with your CUDA/cuDNN drivers. Follow the official TensorFlow installation docs for correct driver and version alignment.
- For most users, `tensorflow-cpu` is easier and avoids GPU driver issues.

Git and face_db.json
---------------------
- `face_db.json` contains your saved face embeddings. Add it to `.gitignore` to avoid committing sensitive data.
- If `face_db.json` is already tracked by Git (committed earlier), adding it to `.gitignore` will not remove it from the repository. To stop tracking the file while keeping it locally:

  git rm --cached face_db.json
  git commit -m "Stop tracking face_db.json"
  git push

Using the scripts
-----------------
1. Create or append to the face database

   python face_db.py

   - The script will suggest a default name `personN` where `N` is the next unused index (it detects existing `person1`, `person2`, etc., and proposes the next number).
   - You can accept the suggested default (press Enter) to create a new `personN` entry.
   - If you type a name that already exists, the script prompts to (a)ppend embeddings, (o)verwrite them, or (s)kip.
   - The DB format is JSON and stores multiple embedding vectors per person, for better accuracy.

2. Start live recognition

   python live_face_recognition.py

   - The script loads `face_db.json` and runs frame-by-frame recognition from your webcam.
   - Use `config.json` to tune detection backends, similarity thresholds, and confidence filters.

Tips and behavior notes
-----------------------
- The face capture window attempts to stay on top using OpenCV's `WND_PROP_TOPMOST`. Some OpenCV builds or window managers may not support that property — the code handles it safely but behavior can vary.
- `face_db.py` will append new embeddings to an existing person by default (if you choose "append"). If you prefer to always create a `personN` entry, accept the suggested default or use unique names.
- Back up your `face_db.json` before making bulk changes.

Troubleshooting
---------------
- Camera not detected: ensure no other application is using the webcam. Try different device index: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`.
- DeepFace errors: ensure TensorFlow is installed and its version is compatible with the DeepFace version. Check the full traceback for missing dependencies.
- Window top-most not working: try updating `opencv-python` or use a separate OS-level tool/window manager to pin the window.

Security & privacy
------------------
- Embeddings are stored locally in `face_db.json`. Treat this file as sensitive — do not commit it to public repositories.
- Consider encrypting or storing the DB in a secure location if needed.

Further improvements (suggestions)
---------------------------------
- Add a CLI option to `face_db.py` to run non-interactively (provide name and mode via args).
- Provide an optional encrypted DB backend or password-protected export/import.
- Add unit tests and a CI pipeline for reproducibility.

Need help?
----------
If you want, I can:
- produce a `requirements.txt` with pinned versions based on your environment,
- add a PowerShell installer script to automate setup,
- add CLI flags to `face_db.py` for headless use.

