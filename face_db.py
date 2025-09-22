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

def analyze_lighting_conditions(face_roi, config):
    """Analyze lighting conditions and determine compensation factors"""
    try:
        # Get lighting compensation settings
        lighting_comp = config.get("anti_spoofing", {}).get("lighting_compensation", {})
        enabled = lighting_comp.get("enabled", True)
        
        if not enabled:
            return 1.0, "normal"
            
        # Convert to grayscale for brightness analysis
        if len(face_roi.shape) > 2:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
            
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        min_brightness = lighting_comp.get("min_brightness", 30)
        max_brightness = lighting_comp.get("max_brightness", 220)
        
        # Determine lighting condition
        if avg_brightness < min_brightness:
            light_condition = "low"
            # Calculate adjustment factor - more lenient in very dark conditions
            factor = lighting_comp.get("low_light_texture_factor", 0.7)
            factor = max(0.5, factor * (avg_brightness / min_brightness))
            
        elif avg_brightness > max_brightness:
            light_condition = "bright"
            # Calculate adjustment factor - stricter in very bright conditions
            factor = lighting_comp.get("bright_light_texture_factor", 1.3)
            
        else:
            # Normal lighting
            light_condition = "normal"
            # Linear interpolation between low and bright factors
            norm_brightness = (avg_brightness - min_brightness) / (max_brightness - min_brightness)
            low_factor = lighting_comp.get("low_light_texture_factor", 0.7)
            bright_factor = lighting_comp.get("bright_light_texture_factor", 1.3)
            factor = low_factor + norm_brightness * (bright_factor - low_factor)
            
        logger.debug(f"Lighting condition: {light_condition}, brightness: {avg_brightness:.1f}, adjustment factor: {factor:.2f}")
        return factor, light_condition
        
    except Exception as e:
        logger.warning(f"Lighting analysis failed: {e}")
        return 1.0, "normal"  # Default to no adjustment

def check_anti_spoofing_deepface(frame, face_region):
    """DeepFace built-in anti-spoofing analysis"""
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    
    # Check for adaptive mode
    mode = anti_spoof_config.get("mode", "normal")
    is_adaptive = mode == "adaptive"
    
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
        
        # Analyze lighting conditions if in adaptive mode
        lighting_factor = 1.0
        light_condition = "normal"
        if is_adaptive:
            lighting_factor, light_condition = analyze_lighting_conditions(face_roi, config)
        
        # Save temporary face image for DeepFace analysis
        temp_face_path = "temp_face_antispoofing.jpg"
        
        # In low light, try enhancing the image before analysis
        if is_adaptive and light_condition == "low":
            # Apply histogram equalization to improve contrast
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            enhanced_roi = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            
            # Blend original with enhanced version to maintain some color information
            alpha = 0.7  # Weight for enhanced image
            blended_roi = cv2.addWeighted(face_roi, 1-alpha, enhanced_roi, alpha, 0)
            cv2.imwrite(temp_face_path, blended_roi)
            logger.debug("Applied low-light enhancement for anti-spoofing analysis")
        else:
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
                
                # Apply lighting compensation to confidence threshold in adaptive mode
                min_confidence = 0.5
                if is_adaptive:
                    if light_condition == "low":
                        # Be more lenient in low light
                        min_confidence = 0.4
                    elif light_condition == "bright":
                        # Be more strict in bright light
                        min_confidence = 0.6
                
                if is_real or (is_adaptive and confidence >= min_confidence):
                    return True, f"Real face detected (confidence: {confidence:.3f}, light: {light_condition})", confidence
                else:
                    return False, f"Fake face detected (confidence: {confidence:.3f}, light: {light_condition})", confidence
            else:
                # Fallback if anti-spoofing data not available
                return check_anti_spoofing_fallback(frame, face_region)
        else:
            return False, "No face detected for anti-spoofing analysis", 0.0
            
    except Exception as e:
        logger.warning(f"DeepFace anti-spoofing failed: {e}")
        # Fallback to basic checks
        return check_anti_spoofing_fallback(frame, face_region)

def check_anti_spoofing_fallback(frame, face_region):
    """Fallback anti-spoofing checks if DeepFace fails"""
    try:
        config = load_config()
        anti_spoof_config = config.get("anti_spoofing", {})
        mode = anti_spoof_config.get("mode", "normal")
        is_adaptive = mode == "adaptive"
        
        x, y, w, h = face_region
        face_roi = frame[y:y+h, x:x+w]
        
        # Analyze lighting conditions if in adaptive mode
        lighting_factor = 1.0
        light_condition = "normal"
        if is_adaptive:
            lighting_factor, light_condition = analyze_lighting_conditions(face_roi, config)
        
        # Basic texture analysis
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        texture_variance = np.var(gray_face)
        
        # Get texture threshold from config and adjust it based on lighting conditions
        base_texture_thresh = anti_spoof_config.get('texture_variance_threshold', 80)
        
        if is_adaptive:
            # Apply lighting compensation
            adjusted_thresh = base_texture_thresh * lighting_factor
            logger.debug(f"Adjusted texture threshold: {base_texture_thresh} → {adjusted_thresh} (factor: {lighting_factor})")
        else:
            adjusted_thresh = base_texture_thresh
        
        # If lighting is too dark, normalize the face for better texture analysis
        if is_adaptive and light_condition == "low":
            # Apply histogram equalization to improve contrast in low light
            equalized_face = cv2.equalizeHist(gray_face)
            equalized_variance = np.var(equalized_face)
            
            # If equalization helps, use the improved texture variance
            if equalized_variance > texture_variance:
                logger.debug(f"Using equalized texture variance: {texture_variance:.1f} → {equalized_variance:.1f}")
                texture_variance = equalized_variance
        
        # Check texture variance against adjusted threshold
        if texture_variance < adjusted_thresh:
            if is_adaptive and light_condition == "low":
                # In low light with adaptive mode, check if it's extremely low
                if texture_variance < adjusted_thresh * 0.5:
                    return False, f"Very low texture variance: {texture_variance:.1f} (low light)", 0.2
                else:
                    # If it's marginal in low light, give benefit of doubt
                    return True, f"Low light condition - acceptable texture: {texture_variance:.1f}", 0.6
            else:
                return False, f"Low texture variance: {texture_variance:.1f}", 0.3
        
        # Additional lighting-aware confidence calculation
        if is_adaptive:
            if light_condition == "low":
                confidence = 0.6  # Lower confidence in low light
            elif light_condition == "bright":
                confidence = 0.8  # Higher confidence in good lighting
            else:
                confidence = 0.7  # Normal lighting
        else:
            confidence = 0.7
            
        return True, f"Fallback check passed (texture: {texture_variance:.1f}, light: {light_condition})", confidence
        
    except Exception as e:
        logger.warning(f"Fallback anti-spoofing failed: {e}")
        # In adaptive mode with an error, give benefit of doubt
        if is_adaptive:
            return True, "Anti-spoofing check failed in adaptive mode, assuming real", 0.5
        else:
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
    
    # Check for adaptive mode
    anti_spoof_config = config.get("anti_spoofing", {})
    mode = anti_spoof_config.get("mode", "normal")
    is_adaptive = mode == "adaptive"

    try:
        # First, read the image for analysis and lighting evaluation
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to read image file: {img_path}")
            return embeddings
            
        # First, analyze lighting conditions if in adaptive mode
        lighting_factor = 1.0
        light_condition = "normal"
        if is_adaptive:
            # Find face region for lighting analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = img[y:y+h, x:x+w]
                lighting_factor, light_condition = analyze_lighting_conditions(face_roi, config)
                logger.info(f"Detected lighting condition: {light_condition} (factor: {lighting_factor:.2f})")
                
                # In very low light, try to enhance the image before verification
                if light_condition == "low" and lighting_factor < 0.7:
                    logger.info("Applying enhancement for low light condition")
                    # Apply histogram equalization to improve contrast
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    equalized = cv2.equalizeHist(face_gray)
                    enhanced_roi = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                    
                    # Replace face area with enhanced version
                    img[y:y+h, x:x+w] = enhanced_roi
                    
                    # Save enhanced version
                    enhanced_path = "enhanced_" + os.path.basename(img_path)
                    cv2.imwrite(enhanced_path, img)
                    img_path = enhanced_path
                    logger.info(f"Saved enhanced image for processing: {enhanced_path}")
        
        # Verify the image is not a spoof using DeepFace's anti-spoofing
        spoof_check = DeepFace.extract_faces(
            img_path=img_path,
            anti_spoofing=True,
            enforce_detection=False
        )
        
        # Check if any faces pass anti-spoofing
        real_faces_detected = False
        if spoof_check:
            for face_info in spoof_check:
                # Apply adaptive confidence threshold based on lighting
                min_confidence = 0.5
                if is_adaptive:
                    if light_condition == "low":
                        min_confidence = 0.4
                    elif light_condition == "bright":
                        min_confidence = 0.6
                
                is_real = face_info.get('is_real', False)
                confidence = face_info.get('antispoof_score', 0.0)
                
                if is_real or (is_adaptive and confidence >= min_confidence):
                    logger.info(f"Face passed anti-spoofing (confidence: {confidence:.3f}, threshold: {min_confidence:.2f})")
                    real_faces_detected = True
                    break
        
        if not real_faces_detected:
            # Try fallback anti-spoofing check
            if is_adaptive and light_condition == "low":
                logger.info("Low light detected - using adaptive fallback check instead of rejecting")
                # For database creation in low light, use our enhanced check
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    is_live, reason, conf = check_anti_spoofing_fallback(img, (x, y, w, h))
                    if is_live:
                        logger.info(f"Face passed adaptive fallback check: {reason}")
                        real_faces_detected = True
            
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
                            # Apply adaptive thresholds here too
                            min_confidence = 0.5
                            if is_adaptive:
                                if light_condition == "low":
                                    min_confidence = 0.4
                                elif light_condition == "bright":
                                    min_confidence = 0.6
                            
                            is_real = face_info.get('is_real', False)
                            confidence = face_info.get('antispoof_score', 0.0)
                            
                            if is_real or (is_adaptive and confidence >= min_confidence):
                                real_faces_detected = True
                                break
                    
                    if not real_faces_detected:
                        logger.warning(f"Anti-spoofing failed with {backend} - skipping")
                        continue
                        
                except Exception as e1:
                    logger.warning(f"Anti-spoofing check failed with {backend}: {e1}, proceeding without verification")
                
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
    
    # Clean up any temp enhanced image
    if img_path.startswith("enhanced_") and os.path.exists(img_path):
        try:
            os.remove(img_path)
        except Exception as e:
            logger.warning(f"Failed to clean up enhanced image: {e}")
                
    return embeddings

def capture_face_from_webcam_auto(temp_filename="captured_face.jpg"):
    """Automatically capture face from webcam when detected"""
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    temp_path = None
    frames_with_face = 0
    liveness_checks = 0
    required_liveness_checks = 5  # Require multiple successful anti-spoofing checks
    
    # Load configuration for adaptive settings
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    mode = anti_spoof_config.get("mode", "normal")
    is_adaptive = mode == "adaptive"
    
    # Track lighting conditions for display
    current_lighting = "normal"
    brightness_value = 0

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
        
        # Analyze lighting conditions if in adaptive mode
        if is_adaptive and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            lighting_factor, current_lighting = analyze_lighting_conditions(face_roi, config)
            brightness_value = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
            
            # Apply enhancement for display if in low light
            if current_lighting == "low":
                # Apply histogram equalization to the display frame for better visibility
                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(face_gray)
                enhanced_roi = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                
                # Use enhanced version for display only
                display_frame = frame.copy()
                display_frame[y:y+h, x:x+w] = enhanced_roi
            else:
                display_frame = frame
        else:
            display_frame = frame

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
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add lighting info if in adaptive mode
            if is_adaptive:
                light_color = (255, 255, 0)  # Yellow for lighting info
                light_text = f"Lighting: {current_lighting} (brightness: {brightness_value:.1f})"
                cv2.putText(display_frame, light_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)

        # Re-apply the topmost property each frame — some window managers reset it.
        try:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        cv2.imshow(win_name, display_frame)

        if len(faces) > 0:
            frames_with_face += 1
        else:
            frames_with_face = 0
            liveness_checks = 0  # Reset liveness checks if no face

        # Capture after face appears consistently for ~10 frames AND passes liveness checks
        if frames_with_face >= 10 and liveness_checks >= required_liveness_checks:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            
            # Save the original face crop
            temp_path = temp_filename
            cv2.imwrite(temp_path, face_crop)
            logger.info(f"Auto-captured live face saved to {temp_path}")
            
            # If in low light, also save an enhanced version
            if is_adaptive and current_lighting == "low":
                # Apply enhancement
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                equalized = cv2.equalizeHist(face_gray)
                enhanced_roi = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                
                # Save enhanced version in case it's needed
                enhanced_path = "enhanced_" + temp_filename
                cv2.imwrite(enhanced_path, enhanced_roi)
                logger.info(f"Enhanced face image saved to {enhanced_path} (low light)")
            
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
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    mode = anti_spoof_config.get("mode", "normal")
    is_adaptive = mode == "adaptive"

    while True:
        account_id = input("Enter account ID (or type 'exit' to quit): ").strip()
        if account_id.lower() == "exit":
            break
        if not account_id:
            print("⚠️  Account ID cannot be empty")
            continue
        
        # Check if ID already exists in database
        try:
            # First try using the repository
            existing_data = repo.load()
            id_exists = account_id in existing_data
            
            # If not found in repository, try database layer directly
            if not id_exists:
                try:
                    from database.db import check_id_exists
                    id_exists = check_id_exists(account_id)
                except (ImportError, AttributeError):
                    # If database module doesn't have this function, use what we have
                    pass
                    
            if id_exists:
                print(f"⚠️  Account ID '{account_id}' already exists in the database.")
                action = ""
                while action.lower() not in ["override", "skip", "exit"]:
                    action = input("Choose an action [override/skip/exit]: ").strip().lower()
                
                if action == "skip":
                    print(f"Skipping ID '{account_id}'")
                    continue
                elif action == "exit":
                    break
                elif action == "override":
                    print(f"Will override existing embedding for '{account_id}'")
                    # Continue with capture process for override
            
        except Exception as e:
            logger.warning(f"Error checking existing ID: {e}")
            # Continue anyway to avoid blocking registration due to DB errors

        temp_img = capture_face_from_webcam_auto()
        if not temp_img:
            logger.warning("No image captured. Skipping this person.")
            continue

        print("Processing face image and extracting features...")
        embeddings = get_face_embedding(temp_img, model_name, detector_backend, augment=True)
        
        # If no embeddings and we're in adaptive mode, try with the enhanced image if it exists
        if not embeddings and is_adaptive:
            enhanced_path = "enhanced_" + temp_img
            if os.path.exists(enhanced_path):
                logger.info(f"Attempting with enhanced image: {enhanced_path}")
                print("Low light detected - trying with enhanced image...")
                embeddings = get_face_embedding(enhanced_path, model_name, detector_backend, augment=True)
                
                # Clean up enhanced image
                try:
                    os.remove(enhanced_path)
                except Exception:
                    pass
        
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
            print(f"❌ No valid face features could be extracted. Please try again in better lighting conditions.")

        if temp_img and os.path.exists(temp_img):
            try:
                os.remove(temp_img)
            except Exception:
                pass

if __name__ == "__main__":
    create_face_database_from_webcam()

