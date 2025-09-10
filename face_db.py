import os
import json
import numpy as np
from deepface import DeepFace
import cv2
import logging
from tqdm import tqdm
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config.json: {e}. Using default settings.")
        return {
            "detection": {
                "backends": ["retinaface", "mediapipe", "mtcnn", "opencv"],
                "similarity_threshold": 0.45,
                "face_confidence_min": 0.7
            }
        }

def get_face_embedding(img_path, model_name="ArcFace", detector_backend="retinaface", augment=False):
    """Get face embeddings from an image path"""
    embeddings = []
    config = load_config()
    backends = config["detection"]["backends"]

    try:
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True
        )
        if reps:
            embeddings.extend([rep["embedding"] for rep in reps])
    except Exception as e:
        logger.warning(f"Failed with {detector_backend} detector: {e}")
        for backend in backends:
            if backend == detector_backend:
                continue
            try:
                logger.info(f"Trying with {backend} detector")
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True
                )
                if reps:
                    embeddings.extend([rep["embedding"] for rep in reps])
                    break
            except Exception as e2:
                logger.warning(f"Failed with {backend} detector: {e2}")
    return embeddings

def capture_face_from_webcam_auto(temp_filename="captured_face.jpg"):
    """Automatically capture face from webcam when detected"""
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    temp_path = None
    frames_with_face = 0

    win_name = "Auto Face Capture (press 'q' to quit)"

    # Create a named window and attempt to make it always on top.
    # Some OpenCV builds/platforms support WND_PROP_TOPMOST; wrap in try/except for safety.
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        # If the property isn't supported, continue without crashing.
        pass

    logger.info("Opening webcam... will auto-capture when a face is detected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangle for visualization
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Re-apply the topmost property each frame — some window managers reset it.
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        cv2.imshow(win_name, frame)

        if len(faces) > 0:
            frames_with_face += 1
        else:
            frames_with_face = 0

        # Capture after face appears consistently for ~10 frames
        if frames_with_face >= 10:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            temp_path = temp_filename
            cv2.imwrite(temp_path, face_crop)
            logger.info(f"Auto-captured face saved to {temp_path}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # Destroy only the named window we created.
    try:
        cv2.destroyWindow(win_name)
    except Exception:
        cv2.destroyAllWindows()
    return temp_path

def _load_existing_db(output_file):
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                db = json.load(f)
            if isinstance(db, dict):
                logger.info(f"Loaded existing DB from {output_file} ({len(db)} people)")
                return db
            else:
                logger.warning(f"{output_file} does not contain a dict - starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load {output_file}: {e}")
    return {}

def _next_person_index(db):
    max_idx = 0
    for k in db.keys():
        m = re.match(r"person(\d+)$", k, re.IGNORECASE)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                pass
    return max_idx + 1

def create_face_database_from_webcam(output_file="face_db.json",
                                     model_name="ArcFace",
                                     detector_backend="retinaface"):
    """Create or append to face database by capturing face(s) from webcam.

    - If output_file exists, it is loaded and new persons are appended.
    - Default name suggested is person{N} where N is next available index.
    - If provided name already exists, user can choose to append embeddings, overwrite or skip.
    """
    db = _load_existing_db(output_file)
    next_idx = _next_person_index(db)

    while True:
        default_name = f"person{next_idx}"
        prompt = f"Enter person's name (press Enter for '{default_name}', type 'exit' to finish): "
        name_in = input(prompt).strip()
        if name_in.lower() == "exit":
            break
        name = name_in if name_in else default_name

        # If name already exists, ask what to do
        if name in db:
            while True:
                choice = input(f"'{name}' exists. (a)ppend, (o)verwrite, (s)kip [a]: ").strip().lower()
                if choice == "" or choice == "a":
                    action = "append"
                    break
                if choice == "o":
                    action = "overwrite"
                    break
                if choice == "s":
                    action = "skip"
                    break
                print("Invalid choice. Enter 'a', 'o' or 's'.")
            if action == "skip":
                logger.info(f"Skipping {name}")
                continue
        else:
            action = "create"

        temp_img = capture_face_from_webcam_auto()
        if not temp_img:
            logger.warning("No image captured. Skipping this person.")
            continue

        embeddings = get_face_embedding(temp_img, model_name, detector_backend, augment=True)
        if embeddings:
            if action == "overwrite":
                db[name] = {"embeddings": embeddings}
                logger.info(f"Overwrote embeddings for {name} ({len(embeddings)} vectors)")
            elif action == "append":
                existing = db.get(name, {}).get("embeddings", [])
                if not isinstance(existing, list):
                    existing = []
                existing.extend(embeddings)
                db[name] = {"embeddings": existing}
                logger.info(f"Appended {len(embeddings)} embeddings to {name} (total now {len(existing)})")
            else:  # create
                db[name] = {"embeddings": embeddings}
                logger.info(f"Added {len(embeddings)} embeddings for new {name}")

            # If we used the default personN name, increment next index to avoid collision
            m = re.match(r"person(\d+)$", name, re.IGNORECASE)
            if m:
                try:
                    used_idx = int(m.group(1))
                    if used_idx >= next_idx:
                        next_idx = used_idx + 1
                except Exception:
                    next_idx += 1
            else:
                # non-default name used, still increment to keep personN sequence free
                next_idx += 1
        else:
            logger.warning(f"No face embeddings extracted for {name}")

        # delete temporary image
        try:
            if temp_img and os.path.exists(temp_img):
                os.remove(temp_img)
        except Exception:
            pass

    if db:
        try:
            with open(output_file, "w") as f:
                json.dump(db, f, indent=2)
            logger.info(f"Face database saved to {output_file} with {len(db)} people")
        except Exception as e:
            logger.error(f"Failed to save {output_file}: {e}")
    else:
        logger.error("No faces were added to the database")

if __name__ == "__main__":
    create_face_database_from_webcam()
