import cv2
import json
import numpy as np
import threading
from queue import Queue
from collections import deque, Counter
from deepface import DeepFace
import time
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Anti-spoofing functions
# ---------------------------
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
            },
            "anti_spoofing": {
                "enabled": True,
                "texture_variance_threshold": 100,
                "edge_density_threshold": 0.05,
                "color_variance_threshold": 200,
                "motion_threshold": 5.0
            }
        }

def check_anti_spoofing_live(frame, face_region, previous_frame=None):
    """DeepFace built-in anti-spoofing analysis for live recognition"""
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    
    if not anti_spoof_config.get("enabled", True):
        return True, "Anti-spoofing disabled", 1.0
    
    x, y, w, h = face_region
    # Ensure coordinates are within frame bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)
    
    if w <= 0 or h <= 0:
        return False, "Invalid face region", 0.0
    
    try:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Save temporary face image for DeepFace analysis
        temp_face_path = "temp_face_live_antispoofing.jpg"
        cv2.imwrite(temp_face_path, face_roi)
        
        # Use DeepFace's built-in anti-spoofing
        result = DeepFace.extract_faces(
            img_path=temp_face_path,
            anti_spoofing=True,
            enforce_detection=False
        )
        
        # Clean up temporary file
        if os.path.exists(temp_face_path):
            os.remove(temp_face_path)
        
        if result and len(result) > 0:
            # DeepFace returns a list of dictionaries with face info
            face_info = result[0]
            if 'is_real' in face_info:
                is_real = face_info['is_real']
                confidence = face_info.get('antispoof_score', 0.5)
                
                if is_real:
                    return True, f"Real face (conf: {confidence:.3f})", confidence
                else:
                    return False, f"Fake face (conf: {confidence:.3f})", confidence
            else:
                # Fallback if anti-spoofing data not available
                return True, "Anti-spoofing data not available", 0.5
        else:
            return False, "No face detected for analysis", 0.0
            
    except Exception as e:
        logger.warning(f"DeepFace anti-spoofing failed: {e}")
        # Fallback to basic texture analysis
        return check_anti_spoofing_fallback(frame, face_region)

def check_anti_spoofing_fallback(frame, face_region):
    """Fallback anti-spoofing checks if DeepFace fails"""
    try:
        x, y, w, h = face_region
        face_roi = frame[y:y+h, x:x+w]
        
        # Basic texture analysis
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray_face)
        
        if texture_variance < 100:
            return False, f"Low texture: {texture_variance:.1f}", 0.3
        
        return True, f"Fallback OK: {texture_variance:.1f}", 0.7
        
    except Exception as e:
        logger.warning(f"Fallback anti-spoofing failed: {e}")
        return True, "Check failed, assuming real", 0.5

# ---------------------------
# Load stored embeddings (local DB)
# ---------------------------
with open("face_db.json", "r") as f:
    db = json.load(f)

# New DB format has multiple embeddings per person
embeddings_db = {}
for name, data in db.items():
    if "embeddings" in data:
        # New format with multiple embeddings per person
        embeddings_db[name] = [np.array(emb) for emb in data["embeddings"]]
    elif "embedding" in data:
        # Old format with single embedding per person
        embeddings_db[name] = [np.array(data["embedding"])]

# ---------------------------
# Configuration: load from config.json
# ---------------------------
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Extract configuration values with defaults
    DETECTOR_BACKENDS = config.get("detection", {}).get("backends", ["retinaface", "mediapipe", "mtcnn", "opencv"])
    SIMILARITY_THRESHOLD = config.get("detection", {}).get("similarity_threshold", 0.45)
    FACE_CONFIDENCE_MIN = config.get("detection", {}).get("face_confidence_min", 0.6)
    
    # Temporal smoothing
    SMOOTHING_WINDOW = config.get("smoothing", {}).get("window_size", 5)
    MIN_VOTES = config.get("smoothing", {}).get("min_votes", 3)
    
    # Performance settings
    DETECTION_SIZE = tuple(config.get("performance", {}).get("detection_size", [320, 240]))
    DISPLAY_SIZE = tuple(config.get("performance", {}).get("display_size", [640, 480]))
    FRAME_SKIP = config.get("performance", {}).get("frame_skip", 3)
    
    # Display settings
    SHOW_FPS = config.get("display", {}).get("show_fps", True)
    SHOW_BACKEND = config.get("display", {}).get("show_backend", True)
    SHOW_PROCESSING_TIME = config.get("display", {}).get("show_processing_time", True)
    
except Exception as e:
    print(f"[WARNING] Failed to load config.json: {e}. Using default settings.")
    # Default configuration if config.json is not available
    DETECTOR_BACKENDS = ["retinaface", "mediapipe", "mtcnn", "opencv"]
    SIMILARITY_THRESHOLD = 0.45
    FACE_CONFIDENCE_MIN = 0.6
    SMOOTHING_WINDOW = 5
    MIN_VOTES = 3
    DETECTION_SIZE = (320, 240)
    DISPLAY_SIZE = (640, 480)
    FRAME_SKIP = 3
    SHOW_FPS = True
    SHOW_BACKEND = True
    SHOW_PROCESSING_TIME = True

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
    previous_frame = None
    
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            break

        frame = frame_data
        
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

                # Perform anti-spoofing check
                x, y, w, h = (
                    facial_area["x"],
                    facial_area["y"],
                    facial_area["w"],
                    facial_area["h"],
                )
                
                is_live, spoof_reason, live_score = check_anti_spoofing_live(
                    frame, (x, y, w, h), previous_frame
                )
                
                # Only process if face passes anti-spoofing
                if is_live:
                    # Compare with DB embeddings
                    best_match = None
                    best_score = 1e6
                    
                    for name, db_embeddings in embeddings_db.items():
                        # Compare with each embedding for this person
                        for db_emb in db_embeddings:
                            cos_sim = np.dot(embedding, db_emb) / (
                                np.linalg.norm(embedding) * np.linalg.norm(db_emb)
                            )
                            distance = 1 - cos_sim
                            if distance < best_score:
                                best_score = distance
                                best_match = name

                    if best_score < SIMILARITY_THRESHOLD:
                        label = f"{best_match} ({best_score:.2f}) [LIVE:{live_score:.2f}]"
                    else:
                        label = f"Unknown ({best_score:.2f}) [LIVE:{live_score:.2f}]"
                else:
                    # Face detected but failed anti-spoofing
                    label = f"SPOOF DETECTED: {spoof_reason}"

                results.append((facial_area, label))

            result_queue.put(results)
            previous_frame = frame.copy()

        except Exception:
            result_queue.put([])
            
    previous_frame = None

# ---------------------------
# Webcam loop
# ---------------------------
cap = cv2.VideoCapture(0)
frame_queue = Queue(maxsize=1)
result_queue = Queue()

# Start recognition thread
threading.Thread(target=recognition_worker, args=(frame_queue, result_queue), daemon=True).start()

# For FPS calculation
frame_count = 0
fps = 0
fps_time = cv2.getTickCount()

# For temporal smoothing
face_history = {}  # Maps face regions to a history of identifications
recent_results = []  # Store recent results for rendering

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam")
        break

    # Resize for display
    display_frame = cv2.resize(frame, DISPLAY_SIZE)
    
    # Process only every Nth frame for speed
    frame_count += 1
    if frame_count % FRAME_SKIP == 0:
        # Resize to smaller size for faster detection
        detection_frame = cv2.resize(frame, DETECTION_SIZE)
        
        if not frame_queue.full():
            frame_queue.put(detection_frame.copy())

    # Calculate and display FPS
    if SHOW_FPS and frame_count % 30 == 0:
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - fps_time) / cv2.getTickFrequency()
        fps = 30 / elapsed_time
        fps_time = current_time

    # Get latest recognition results
    while not result_queue.empty():
        new_results = result_queue.get()
        if new_results:
            recent_results = new_results

    # Apply temporal smoothing
    smoothed_results = []
    for (facial_area, label) in recent_results:
        # Create a region key (quantized to handle slight movement)
        x, y, w, h = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"],
        )
        # Quantize face location to handle small movements
        region_key = (x//10, y//10, w//10, h//10)
        
        # Add to history
        if region_key not in face_history:
            face_history[region_key] = deque(maxlen=SMOOTHING_WINDOW)
        
        # Extract person name from label
        person = label.split(' ')[0] if label != "Unknown" else "Unknown"
        face_history[region_key].append(person)
        
        # Count occurrences
        counter = Counter(face_history[region_key])
        most_common = counter.most_common(1)[0]
        
        # Use the most common if it has enough votes
        if most_common[1] >= MIN_VOTES:
            smoothed_name = most_common[0]
            # Extract confidence from label (handle different formats)
            confidence = 0.0
            if '(' in label and ')' in label:
                try:
                    conf_part = label.split('(')[1].split(')')[0]
                    # Handle "conf: 0.507" format
                    if 'conf:' in conf_part:
                        confidence = float(conf_part.split('conf:')[1].strip())
                    # Handle "0.507" format
                    else:
                        confidence = float(conf_part)
                except (ValueError, IndexError):
                    confidence = 0.0
            smoothed_label = f"{smoothed_name} ({confidence:.2f})"
            smoothed_results.append((facial_area, smoothed_label))
        else:
            smoothed_results.append((facial_area, label))
    
    # Clean up old face regions that haven't been seen recently
    if frame_count % 30 == 0:
        current_regions = set(region_key for facial_area, _ in recent_results 
                             for region_key in [(facial_area["x"]//10, 
                                               facial_area["y"]//10, 
                                               facial_area["w"]//10, 
                                               facial_area["h"]//10)])
        keys_to_remove = []
        for key in face_history:
            if key not in current_regions:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del face_history[key]

    # Draw results on the frame
    for (facial_area, label) in smoothed_results:
        x, y, w, h = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"],
        )
        # Scale coordinates to display size
        scale_x = DISPLAY_SIZE[0] / DETECTION_SIZE[0]
        scale_y = DISPLAY_SIZE[1] / DETECTION_SIZE[1]
        
        # Compute display coordinates
        disp_x = int(x * scale_x)
        disp_y = int(y * scale_y)
        disp_w = int(w * scale_x)
        disp_h = int(h * scale_y)
        
        # Set color based on recognition and anti-spoofing
        if "SPOOF DETECTED" in label:
            color = (0, 0, 255)  # Red for spoof
        elif "Unknown" not in label and "LIVE:" in label:
            color = (0, 255, 0)  # Green for known live face
        elif "Unknown" in label and "LIVE:" in label:
            color = (0, 255, 255)  # Yellow for unknown live face
        else:
            color = (128, 128, 128)  # Gray for uncertain
        
        # Draw rectangle and label
        cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), color, 2)
        cv2.putText(display_frame, label, (disp_x, disp_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add FPS counter
    if SHOW_FPS:
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Live Face Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
print("[INFO] Cleaning up...")
cap.release()
cv2.destroyAllWindows()
frame_queue.put(None)  # Signal worker thread to exit
