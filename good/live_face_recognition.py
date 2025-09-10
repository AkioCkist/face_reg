import cv2
import json
import numpy as np
import threading
from queue import Queue
from collections import deque, Counter
from deepface import DeepFace

# ---------------------------
# Load stored embeddings (local DB)
# ---------------------------
with open("face_db.json", "r") as f:
    db = json.load(f)

embeddings_db = {name: np.array(data["embedding"]) for name, data in db.items()}

# ---------------------------
# Configuration: detection backends and thresholds
# ---------------------------
# Ordered list of detector backends to try (will fall back if a backend raises)
DETECTOR_BACKENDS = ["retinaface", "mediapipe", "mtcnn", "opencv"]
# similarity distance threshold (1 - cosine). Lower = stricter. Default was 0.6, make stricter.
SIMILARITY_THRESHOLD = 0.45
# Minimum face detection confidence (if detector provides it). Range approx 0..1
FACE_CONFIDENCE_MIN = 0.6
# Temporal smoothing: number of recent result-sets to keep and minimum votes to confirm a label
SMOOTHING_WINDOW = 5
MIN_VOTES = 3

# ---------------------------
# Load ArcFace model once
# ---------------------------
print("[INFO] Loading ArcFace model...")
model = DeepFace.build_model("ArcFace")
print("[INFO] Model loaded.")

# ---------------------------
# Worker thread for recognition
# ---------------------------
def recognition_worker(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        try:
            # use DeepFace.represent (functions module may not be exported in this DeepFace version)
            reps = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="opencv",  # fast detector
                enforce_detection=False,
                align=True,
                max_faces=1,
            )

            results = []
            for rep in reps:
                embedding = np.array(rep["embedding"])
                facial_area = rep["facial_area"]

                # Compare with DB embeddings
                best_match = None
                best_score = 1e6
                for name, db_emb in embeddings_db.items():
                    cos_sim = np.dot(embedding, db_emb) / (
                        np.linalg.norm(embedding) * np.linalg.norm(db_emb)
                    )
                    distance = 1 - cos_sim
                    if distance < best_score:
                        best_score = distance
                        best_match = name

                if best_score < 0.6:
                    label = f"{best_match} ({best_score:.2f})"
                else:
                    label = f"Unknown ({best_score:.2f})"

                results.append((facial_area, label))

            result_queue.put(results)

        except Exception:
            result_queue.put([])

# ---------------------------
# Webcam loop
# ---------------------------
cap = cv2.VideoCapture(0)
frame_queue = Queue(maxsize=1)
result_queue = Queue()

# Start recognition thread
threading.Thread(target=recognition_worker, args=(frame_queue, result_queue), daemon=True).start()

frame_count = 0
results = []

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1
    if frame_count % 5 == 0:  # only process 1 in 5 frames
        if not frame_queue.full():
            frame_queue.put(frame.copy())

    # Draw last recognition results
    while not result_queue.empty():
        results = result_queue.get()

    for (facial_area, label) in results:
        x, y, w, h = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"],
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)
