import os
import json
import numpy as np
from deepface import DeepFace
import cv2
import logging
from tqdm import tqdm

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

        cv2.imshow("Auto Face Capture (press 'q' to quit)", frame)

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
    cv2.destroyAllWindows()
    return temp_path

def create_face_database_from_webcam(output_file="face_db.json",
                                     model_name="ArcFace",
                                     detector_backend="retinaface"):
    """Create face database by capturing face(s) from webcam"""
    db = {}

    while True:
        name = input("Enter person's name (or press Enter to finish): ").strip()
        if not name:
            break

        temp_img = capture_face_from_webcam_auto()
        if not temp_img:
            logger.warning("No image captured. Skipping this person.")
            continue

        embeddings = get_face_embedding(temp_img, model_name, detector_backend, augment=True)
        if embeddings:
            db[name] = {"embeddings": embeddings}
            logger.info(f"Added {len(embeddings)} embeddings for {name}")
        else:
            logger.warning(f"No face embeddings extracted for {name}")

        # delete temporary image
        if os.path.exists(temp_img):
            os.remove(temp_img)

    if db:
        with open(output_file, "w") as f:
            json.dump(db, f)
        logger.info(f"Face database saved to {output_file} with {len(db)} people")
    else:
        logger.error("No faces were added to the database")

if __name__ == "__main__":
    create_face_database_from_webcam()
