import os
import json
import numpy as np
from deepface import DeepFace
import cv2
import logging
from tqdm import tqdm
import re
from setup_logging import setup_logging
from persistence.repository import FaceRepository

# Set up logging to file and console
logger, log_file_path = setup_logging("face_db", logging.INFO)
logger.info(f"Face database logging started. Log file: {log_file_path}")

# Initialize repository for DB persistence
repo = FaceRepository("face_db.json")

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config/config.json", "r") as f:
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
                "eye_aspect_ratio_threshold": 0.25,
                "blink_frames_threshold": 3,
                "motion_threshold": 5.0,
                "texture_variance_threshold": 100
            }
        }

def check_anti_spoofing_deepface(frame, face_region):
    """DeepFace built-in anti-spoofing analysis"""
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    
    if not anti_spoof_config.get("enabled", True):
        return True, "Anti-spoofing disabled", 1.0
    
    try:
        x, y, w, h = face_region
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return False, "Invalid face region", 0.0
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Save temporary face image for DeepFace analysis
        temp_face_path = "temp_face_antispoofing.jpg"
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
                    return True, f"Real face detected (confidence: {confidence:.3f})", confidence
                else:
                    return False, f"Fake face detected (confidence: {confidence:.3f})", confidence
            else:
                # Fallback if anti-spoofing data not available
                return True, "Anti-spoofing data not available, assuming real", 0.5
        else:
            return False, "No face detected for anti-spoofing analysis", 0.0
            
    except Exception as e:
        logger.warning(f"DeepFace anti-spoofing failed: {e}")
        # Fallback to basic checks
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
            return False, f"Low texture variance: {texture_variance:.1f}", 0.3
        
        return True, f"Fallback check passed (texture: {texture_variance:.1f})", 0.7
        
    except Exception as e:
        logger.warning(f"Fallback anti-spoofing failed: {e}")
        return True, "Anti-spoofing check failed, assuming real", 0.5

def detect_eye_blink(frame, face_cascade):
    """Detect eye blinks for liveness detection"""
    try:
        # Load eye cascade classifier
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # If we detect 2 eyes, it's more likely a real face
            if len(eyes) >= 2:
                return True, len(eyes)
        
        return False, 0
    except Exception as e:
        logger.warning(f"Eye detection failed: {e}")
        return True, 0  # Assume live if detection fails

def get_face_embedding(img_path, model_name="ArcFace", detector_backend="retinaface", augment=False):
    """Get face embeddings from an image path with anti-spoofing verification"""
    embeddings = []
    config = load_config()
    backends = config["detection"]["backends"]

    try:
        # First, verify the image is not a spoof using DeepFace's anti-spoofing
        spoof_check = DeepFace.extract_faces(
            img_path=img_path,
            anti_spoofing=True,
            enforce_detection=False
        )
        
        # Check if any faces pass anti-spoofing
        real_faces_detected = False
        if spoof_check:
            for face_info in spoof_check:
                if face_info.get('is_real', True):  # Default to True if not available
                    real_faces_detected = True
                    break
        
        if not real_faces_detected:
            logger.warning(f"Anti-spoofing check failed for {img_path} - potential fake image")
            return embeddings  # Return empty list for fake images
        
        # If anti-spoofing passes, extract embeddings
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True
        )
        if reps:
            embeddings.extend([rep["embedding"] for rep in reps])
            logger.info(f"Successfully extracted {len(embeddings)} embeddings from verified real face")
            
    except Exception as e:
        logger.warning(f"Failed with {detector_backend} detector: {e}")
        for backend in backends:
            if backend == detector_backend:
                continue
            try:
                logger.info(f"Trying with {backend} detector")
                # Try anti-spoofing check with fallback backend
                try:
                    spoof_check = DeepFace.extract_faces(
                        img_path=img_path,
                        anti_spoofing=True,
                        enforce_detection=False
                    )
                    
                    real_faces_detected = False
                    if spoof_check:
                        for face_info in spoof_check:
                            if face_info.get('is_real', True):
                                real_faces_detected = True
                                break
                    
                    if not real_faces_detected:
                        logger.warning(f"Anti-spoofing failed with {backend} - skipping")
                        continue
                        
                except Exception:
                    logger.warning(f"Anti-spoofing check failed with {backend}, proceeding without verification")
                
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
    liveness_checks = 0
    required_liveness_checks = 5  # Require multiple successful anti-spoofing checks

    win_name = "Auto Face Capture (press 'q' to quit)"

    # Create a named window and attempt to make it always on top.
    # Some OpenCV builds/platforms support WND_PROP_TOPMOST; wrap in try/except for safety.
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        # If the property isn't supported, continue without crashing.
        pass

    logger.info("Opening webcam... will auto-capture when a live face is detected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw rectangle for visualization
        for (x, y, w, h) in faces:
            # Perform anti-spoofing checks
            is_live, spoof_reason, confidence = check_anti_spoofing_deepface(frame, (x, y, w, h))
            
            if is_live:
                # Check for eye detection as additional liveness indicator
                has_eyes, eye_count = detect_eye_blink(frame, face_cascade)
                
                if has_eyes:
                    liveness_checks += 1
                    color = (0, 255, 0)  # Green for live face
                    status_text = f"Live face detected ({liveness_checks}/{required_liveness_checks})"
                else:
                    color = (0, 255, 255)  # Yellow for questionable
                    status_text = f"Face detected but no eyes found"
            else:
                liveness_checks = 0  # Reset if spoofing detected
                color = (0, 0, 255)  # Red for potential spoof
                status_text = f"Potential spoof: {spoof_reason}"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
            liveness_checks = 0  # Reset liveness checks if no face

        # Capture after face appears consistently for ~10 frames AND passes liveness checks
        if frames_with_face >= 10 and liveness_checks >= required_liveness_checks:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            temp_path = temp_filename
            cv2.imwrite(temp_path, face_crop)
            logger.info(f"Auto-captured live face saved to {temp_path}")
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
    # Use repository to load and convert to legacy format {name: {"embeddings": [...]}}
    try:
        raw = repo.load()
        out = {}
        for name, embeddings in raw.items():
            out[name] = {"embeddings": [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]}
        logger.info(f"Loaded existing DB via repository ({len(out)} people)")
        return out
    except Exception as e:
        logger.warning(f"Failed to load DB via repository: {e}")
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

def create_face_database_from_webcam(model_name="ArcFace", detector_backend="retinaface"):
    """Capture face → extract embedding → save (id, embedding) into DB"""
    while True:
        account_id = input("Enter account ID (or type 'exit' to quit): ").strip()
        if account_id.lower() == "exit":
            break
        if not account_id:
            print("⚠️  Account ID cannot be empty")
            continue

        temp_img = capture_face_from_webcam_auto()
        if not temp_img:
            logger.warning("No image captured. Skipping this person.")
            continue

        embeddings = get_face_embedding(temp_img, model_name, detector_backend, augment=True)
        if embeddings:
            # Use first embedding
            try:
                # If database layer expects JSON-serializable embedding
                from database.db import insert_embedding
                insert_embedding(account_id, embeddings[0])   # ✅ DB insert
                logger.info(f"✅ Saved embedding for account {account_id} into DB")
                print(f"✅ Saved embedding for account {account_id} into DB")
            except Exception as e:
                logger.error(f"Failed to save embedding for {account_id}: {e}")
                print(f"❌ Failed to save embedding for {account_id}: {e}")
        else:
            logger.warning(f"No embeddings extracted for {account_id}")

        if temp_img and os.path.exists(temp_img):
            try:
                os.remove(temp_img)
            except Exception:
                pass

if __name__ == "__main__":
    create_face_database_from_webcam()
