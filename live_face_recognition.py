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
from setup_logging import setup_logging
from persistence.repository import FaceRepository

# Set up logging to file and console
logger, log_file_path = setup_logging("live_recognition", logging.INFO)
logger.info(f"Live face recognition logging started. Log file: {log_file_path}")

# ---------------------------
# Global variables for incremental learning
# ---------------------------
last_update_time = {}  # Track last update time per person
update_cooldown = 5.0  # Minimum seconds between updates
successful_recognitions = {}  # Track successful recognitions for learning
embedding_quality_history = {}  # Track embedding quality over time
suspicious_activity = {}  # Track potential mismatches

# Identity confirmation variables
identity_confirmed = False  # Flag to track if identity is confirmed
confirmed_identity = None   # Store the confirmed identity
confirmation_count = {}     # Track confirmation counts per person

# ---------------------------
# Anti-spoofing functions
# ---------------------------
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
                "mode": "adaptive",
                "texture_variance_threshold": 40,
                "edge_density_threshold": 0.02,
                "color_variance_threshold": 100,
                "motion_threshold": 3.0,
                "confidence_boost_real_face": 0.25,
                "lighting_compensation": {
                    "enabled": True,
                    "min_brightness": 10,
                    "max_brightness": 350,
                    "adjust_thresholds": True,
                    "low_light_texture_factor": 0.3,
                    "bright_light_texture_factor": 2.0,
                    "brightness_variance_threshold": 25
                }
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

def check_anti_spoofing_live(frame, face_region, previous_frame=None):
    """DeepFace built-in anti-spoofing analysis for live recognition"""
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    
    # Check for adaptive mode
    mode = anti_spoof_config.get("mode", "normal")
    is_adaptive = mode == "adaptive"
    
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
        
        # Analyze lighting conditions if in adaptive mode
        lighting_factor = 1.0
        light_condition = "normal"
        if is_adaptive:
            lighting_factor, light_condition = analyze_lighting_conditions(face_roi, config)
            
        # Check for motion if previous frame exists
        if previous_frame is not None and is_adaptive:
            # Adjust motion threshold based on lighting
            motion_threshold = anti_spoof_config.get("motion_threshold", 5.0)
            if light_condition == "low":
                # Lower expectations for motion in low light (more noise, less detail)
                motion_threshold *= 0.7
            
            try:
                prev_roi = previous_frame[y:y+h, x:x+w]
                if prev_roi.shape == face_roi.shape:
                    # Simple motion detection
                    diff = cv2.absdiff(prev_roi, face_roi)
                    motion_score = np.mean(diff)
                    
                    if motion_score < motion_threshold and light_condition != "low":
                        logger.debug(f"Low motion detected: {motion_score:.1f} < {motion_threshold}")
                        # In low light, don't reject purely on motion - it's unreliable
                        if light_condition != "low":
                            return False, f"Low motion: {motion_score:.1f}", 0.3
            except Exception as e:
                logger.warning(f"Motion check failed: {e}")
        
        # Save temporary face image for DeepFace analysis
        temp_face_path = "temp_face_live_antispoofing.jpg"
        
        # In low light, try enhancing the image before analysis
        if is_adaptive and light_condition == "low":
            # Apply histogram equalization to improve contrast
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            enhanced_roi = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            
            # Blend original with enhanced version to maintain some color
            alpha = 0.7  # Weight for enhanced image
            blended_roi = cv2.addWeighted(face_roi, 1-alpha, enhanced_roi, alpha, 0)
            cv2.imwrite(temp_face_path, blended_roi)
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
                        # Be very lenient in low light
                        min_confidence = 0.3
                    elif light_condition == "bright":
                        # Still be lenient in bright light
                        min_confidence = 0.45
                
                # Additional boost for real faces in challenging conditions
                confidence_boost = anti_spoof_config.get('confidence_boost_real_face', 0.25)
                if is_real and is_adaptive:
                    confidence += confidence_boost
                
                if is_real or (is_adaptive and confidence >= min_confidence):
                    if light_condition == "low":
                        return True, f"Real face (low light, conf: {confidence:.3f})", confidence
                    else:
                        return True, f"Real face (conf: {confidence:.3f})", confidence
                else:
                    # Even if marked as fake, give benefit of doubt in adaptive mode
                    if is_adaptive and confidence >= 0.25:
                        return True, f"Benefit of doubt - challenging conditions (conf: {confidence:.3f})", confidence
                    return False, f"Fake face (conf: {confidence:.3f})", confidence
            else:
                # Fallback if anti-spoofing data not available
                return check_anti_spoofing_fallback(frame, face_region)
        else:
            return False, "No face detected for analysis", 0.0
            
    except Exception as e:
        logger.warning(f"DeepFace anti-spoofing failed: {e}")
        # Fallback to basic texture analysis
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
                # In low light with adaptive mode, be very lenient
                if texture_variance < adjusted_thresh * 0.3:
                    return False, f"Extremely low texture: {texture_variance:.1f} (low light)", 0.3
                else:
                    # If it's marginal in low light, give strong benefit of doubt
                    return True, f"Low light - texture acceptable: {texture_variance:.1f}", 0.7
            elif is_adaptive and light_condition == "bright":
                # In bright light, also be more forgiving
                return True, f"Bright light - texture acceptable: {texture_variance:.1f}", 0.6
            else:
                # Even in normal conditions, be more lenient
                if texture_variance > adjusted_thresh * 0.6:
                    return True, f"Marginal texture but acceptable: {texture_variance:.1f}", 0.5
                return False, f"Low texture: {texture_variance:.1f}", 0.4
        
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
            
        return True, f"Fallback OK: {texture_variance:.1f}", confidence
        
    except Exception as e:
        logger.warning(f"Fallback anti-spoofing failed: {e}")
        # In adaptive mode with an error, give benefit of doubt
        if is_adaptive:
            return True, "Check failed in adaptive mode, assuming real", 0.5
        else:
            return True, "Check failed, assuming real", 0.5

# ---------------------------
# Incremental Learning Functions
# ---------------------------
def calculate_embedding_distance(emb1, emb2):
    """Calculate cosine distance between two embeddings"""
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return 1 - cos_sim

def is_valid_embedding_update(new_embedding, existing_embeddings, threshold=None):
    """Check if new embedding is valid for incremental learning"""
    if threshold is None:
        threshold = OUTLIER_THRESHOLD
        
    if not existing_embeddings:
        return True
    
    # Calculate average distance to existing embeddings
    distances = [calculate_embedding_distance(new_embedding, emb) for emb in existing_embeddings]
    avg_distance = np.mean(distances)
    
    # If too far from existing embeddings, it might be an outlier
    if avg_distance > threshold:
        logger.warning(f"New embedding distance {avg_distance:.3f} exceeds threshold {threshold}")
        return False
    
    return True

def update_embedding_weighted(old_embeddings, new_embedding, alpha=None, max_embeddings=None):
    """Update embeddings using weighted averaging with memory management"""
    if alpha is None:
        alpha = WEIGHTED_ALPHA
    if max_embeddings is None:
        max_embeddings = MAX_EMBEDDINGS
        
    if not old_embeddings:
        return [new_embedding]
    
    # Validate new embedding
    if not is_valid_embedding_update(new_embedding, old_embeddings):
        logger.info("New embedding rejected as outlier")
        return old_embeddings
    
    # Add new embedding to the list
    updated_embeddings = old_embeddings.copy()
    updated_embeddings.append(new_embedding)
    
    # Keep only the most recent embeddings
    if len(updated_embeddings) > max_embeddings:
        updated_embeddings = updated_embeddings[-max_embeddings:]
    
    logger.info(f"Added new embedding. Total embeddings: {len(updated_embeddings)}")
    return updated_embeddings

def save_updated_database(embeddings_db, filename="face_db.json"):
    """Save updated embeddings database to file"""
    try:
        # Use repository to save (it converts numpy arrays)
        repo.save(embeddings_db)
        logger.info(f"Database updated and saved via repository: {repo.path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save database via repository: {e}")
        return False

def should_update_embedding(person_name, confidence_score, min_confidence=None):
    """Determine if we should update embedding based on conditions"""
    if not LEARNING_ENABLED:
        return False
        
    if min_confidence is None:
        min_confidence = MIN_LEARNING_CONFIDENCE
        
    current_time = time.time()
    
    # Check confidence threshold
    if confidence_score < min_confidence:
        return False
    
    # Check cooldown period
    if person_name in last_update_time:
        time_since_last = current_time - last_update_time[person_name]
        if time_since_last < UPDATE_COOLDOWN:
            return False
    
    # Track successful recognitions
    if person_name not in successful_recognitions:
        successful_recognitions[person_name] = 0
    
    successful_recognitions[person_name] += 1
    
    # Update every N successful recognitions or after cooldown
    if successful_recognitions[person_name] % UPDATE_FREQUENCY == 0:
        last_update_time[person_name] = current_time
        return True
    
    return False

# ---------------------------
# Anti-Mismatch Protection Functions
# ---------------------------
def detect_embedding_drift(person_name, new_embedding, existing_embeddings):
    """Detect if new embedding shows significant drift from person's profile"""
    if not existing_embeddings:
        return False, 0.0
    
    # Calculate distances to all existing embeddings
    distances = [calculate_embedding_distance(new_embedding, emb) for emb in existing_embeddings]
    
    # Statistical analysis
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    std_distance = np.std(distances)
    
    # Define drift threshold (configurable)
    drift_threshold = DRIFT_DETECTION_THRESHOLD
    
    # Check for significant drift
    is_drift = (min_distance > drift_threshold) or (avg_distance > drift_threshold * 1.5)
    
    logger.debug(f"Drift analysis for {person_name}: avg={avg_distance:.3f}, min={min_distance:.3f}, std={std_distance:.3f}")
    
    return is_drift, min_distance

def track_suspicious_activity(person_name, embedding, confidence_score):
    """Track patterns that might indicate false matches"""
    current_time = time.time()
    
    if person_name not in suspicious_activity:
        suspicious_activity[person_name] = {
            'recent_embeddings': [],
            'confidence_history': [],
            'timestamps': [],
            'drift_events': 0,
            'last_drift_time': 0
        }
    
    activity = suspicious_activity[person_name]
    
    # Add current data
    activity['recent_embeddings'].append(embedding)
    activity['confidence_history'].append(confidence_score)
    activity['timestamps'].append(current_time)
    
    # Keep only recent data (last 10 recognitions)
    max_history = 10
    if len(activity['recent_embeddings']) > max_history:
        activity['recent_embeddings'] = activity['recent_embeddings'][-max_history:]
        activity['confidence_history'] = activity['confidence_history'][-max_history:]
        activity['timestamps'] = activity['timestamps'][-max_history:]
    
    # Analyze patterns
    if len(activity['confidence_history']) >= 5:
        # Check for declining confidence trend
        recent_confidences = activity['confidence_history'][-5:]
        confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
        
        # Check for high variance in embeddings
        if len(activity['recent_embeddings']) >= 3:
            embedding_matrix = np.array(activity['recent_embeddings'][-3:])
            pairwise_distances = []
            for i in range(len(embedding_matrix)):
                for j in range(i+1, len(embedding_matrix)):
                    dist = calculate_embedding_distance(embedding_matrix[i], embedding_matrix[j])
                    pairwise_distances.append(dist)
            
            embedding_variance = np.var(pairwise_distances) if pairwise_distances else 0
            
            # Flag suspicious patterns
            if confidence_trend < -0.05 or embedding_variance > EMBEDDING_VARIANCE_THRESHOLD:
                logger.warning(f"Suspicious pattern detected for {person_name}: "
                             f"confidence_trend={confidence_trend:.3f}, "
                             f"embedding_variance={embedding_variance:.3f}")
                return True
    
    return False

def validate_embedding_consistency(person_name, new_embedding, existing_embeddings):
    """Multi-layer validation before accepting new embedding"""
    
    # Layer 1: Drift detection
    is_drift, min_distance = detect_embedding_drift(person_name, new_embedding, existing_embeddings)
    if is_drift:
        logger.warning(f"Embedding drift detected for {person_name} (min_distance: {min_distance:.3f})")
        return False, f"Drift detected (distance: {min_distance:.3f})"
    
    # Layer 2: Suspicious activity check
    is_suspicious = track_suspicious_activity(person_name, new_embedding, 1.0)
    if is_suspicious:
        logger.warning(f"Suspicious activity pattern for {person_name}")
        return False, "Suspicious activity pattern"
    
    # Layer 3: Embedding quality check
    if not is_valid_embedding_update(new_embedding, existing_embeddings):
        return False, "Failed quality check"
    
    return True, "Validation passed"

# ---------------------------
# Identity Confirmation Functions
# ---------------------------
def check_identity_confirmation(person_name, confidence_score, live_score):
    """Check if identity should be confirmed and trigger program exit"""
    global identity_confirmed, confirmed_identity, confirmation_count
    
    # Get confirmation settings from config
    config = load_config()
    confirmation_config = config.get("identity_confirmation", {})
    
    min_confirmations = confirmation_config.get("min_confirmations", 3)
    min_confidence = confirmation_config.get("min_confidence_for_confirmation", 0.9)
    min_live_score = confirmation_config.get("min_live_score_for_confirmation", 0.8)
    
    # Check if this recognition meets confirmation criteria
    if (confidence_score >= min_confidence and 
        live_score >= min_live_score and 
        person_name != "Unknown"):
        
        # Initialize confirmation count for this person
        if person_name not in confirmation_count:
            confirmation_count[person_name] = 0
        
        confirmation_count[person_name] += 1
        
        logger.info(f"Identity confirmation progress for {person_name}: "
                   f"{confirmation_count[person_name]}/{min_confirmations} "
                   f"(conf: {confidence_score:.3f}, live: {live_score:.3f})")
        
        # Check if we have enough confirmations
        if confirmation_count[person_name] >= min_confirmations:
            identity_confirmed = True
            confirmed_identity = person_name
            
            logger.info(f"✅ IDENTITY CONFIRMED: {person_name}")
            logger.info(f"   Confirmations: {confirmation_count[person_name]}")
            logger.info(f"   Final confidence: {confidence_score:.3f}")
            logger.info(f"   Final live score: {live_score:.3f}")
            
            return True
    else:
        # Reset confirmation count if criteria not met
        if person_name in confirmation_count:
            confirmation_count[person_name] = 0
            logger.debug(f"Reset confirmation count for {person_name} - criteria not met")
    
    return False

def display_confirmation_status(display_frame):
    """Display identity confirmation status on the frame"""
    if identity_confirmed:
        # Show confirmation success - program will exit immediately
        cv2.putText(display_frame, f"IDENTITY CONFIRMED: {confirmed_identity}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # Show confirmation progress for any person being tracked
        y_offset = 60
        for person_name, count in confirmation_count.items():
            if count > 0:
                config = load_config()
                min_confirmations = config.get("identity_confirmation", {}).get("min_confirmations", 3)
                cv2.putText(display_frame, f"{person_name}: {count}/{min_confirmations} confirmations", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25

# ---------------------------
# Persistence: use repository for DB load/save
# ---------------------------

from persistence.repository import FaceRepository
import json
import numpy as np

# Initialize repository (creates directory if needed) and load DB
repo = FaceRepository("face_db.json")

def load_embeddings_from_database():
    """Try to load embeddings from SQL DB (database.db.get_all_embeddings).
    Convert returned values to the legacy format: {id: [np.array(...), ...]}
    """
    try:
        from database import db as sql_db
        # Ensure table exists (safe; no-op if already present)
        try:
            sql_db.ensure_table_exists()
        except Exception:
            # continue even if ensure_table_exists isn't available
            pass

        rows = sql_db.get_all_embeddings()
        if not rows:
            logger.info("No rows returned from SQL DB.")
            return {}

        out = {}
        for id_, emb in rows.items():
            try:
                # emb may be:
                # - a single list of floats -> treat as one embedding
                # - a list of lists -> treat as multiple embeddings
                # - a JSON string -> parse then handle above
                if isinstance(emb, str):
                    parsed = json.loads(emb)
                else:
                    parsed = emb

                if isinstance(parsed, list) and parsed and isinstance(parsed[0], (list, tuple)):
                    out[id_] = [np.array(e, dtype=float) for e in parsed]
                elif isinstance(parsed, list):
                    out[id_] = [np.array(parsed, dtype=float)]
                else:
                    # Unknown format; skip
                    logger.warning(f"Skipping DB entry {id_}: unexpected embedding format")
            except Exception as e:
                logger.warning(f"Failed to parse embedding for {id_} from DB: {e}")
        logger.info(f"Loaded {len(out)} entries from SQL DB.")
        return out

    except Exception as e:
        logger.warning(f"Could not load embeddings from SQL DB: {e}")
        return {}

# Try DB first, then fallback to repository file
embeddings_db = load_embeddings_from_database()
if not embeddings_db:
    try:
        embeddings_db = repo.load()
        logger.info(f"Loaded embeddings from repository file ({repo.path}): {len(embeddings_db)} people")
    except Exception as e:
        embeddings_db = {}
        logger.error(f"Failed to load embeddings from repository: {e}")
else:
    logger.info("Using embeddings loaded from SQL database.")

# ---------------------------
# Configuration: load from config.json
# ---------------------------
try:
    with open("config/config.json", "r") as f:
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
    
    # Incremental learning settings
    LEARNING_ENABLED = config.get("incremental_learning", {}).get("enabled", True)
    UPDATE_COOLDOWN = config.get("incremental_learning", {}).get("update_cooldown", 5.0)
    MIN_LEARNING_CONFIDENCE = config.get("incremental_learning", {}).get("min_confidence", 0.8)
    MIN_LIVENESS_SCORE = config.get("incremental_learning", {}).get("min_liveness_score", 0.7)
    UPDATE_FREQUENCY = config.get("incremental_learning", {}).get("update_frequency", 5)
    MAX_EMBEDDINGS = config.get("incremental_learning", {}).get("max_embeddings_per_person", 20)
    OUTLIER_THRESHOLD = config.get("incremental_learning", {}).get("outlier_threshold", 0.3)
    WEIGHTED_ALPHA = config.get("incremental_learning", {}).get("weighted_alpha", 0.8)
    
    # Anti-mismatch protection settings
    DRIFT_DETECTION_THRESHOLD = config.get("incremental_learning", {}).get("drift_detection_threshold", 0.4)
    EMBEDDING_VARIANCE_THRESHOLD = config.get("incremental_learning", {}).get("embedding_variance_threshold", 0.1)
    MISMATCH_DETECTION_ENABLED = config.get("incremental_learning", {}).get("mismatch_detection_enabled", True)
    
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
    
    # Default incremental learning settings
    LEARNING_ENABLED = True
    UPDATE_COOLDOWN = 5.0
    MIN_LEARNING_CONFIDENCE = 0.8
    MIN_LIVENESS_SCORE = 0.7
    UPDATE_FREQUENCY = 5
    MAX_EMBEDDINGS = 20
    OUTLIER_THRESHOLD = 0.3
    WEIGHTED_ALPHA = 0.8
    DRIFT_DETECTION_THRESHOLD = 0.4
    EMBEDDING_VARIANCE_THRESHOLD = 0.1
    MISMATCH_DETECTION_ENABLED = True

# Announce incremental learning status (print + logger)
print(f"[INFO] INCREMENTAL_LEARNING enabled: {LEARNING_ENABLED}")
logger.info(f"INCREMENTAL_LEARNING enabled: {LEARNING_ENABLED}")

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
                        confidence_score = 1 - best_score  # Convert distance to confidence
                        label = f"{best_match} ({best_score:.2f}) [LIVE:{live_score:.2f}]"
                        
                        # Check for identity confirmation
                        if check_identity_confirmation(best_match, confidence_score, live_score):
                            # Identity confirmed - signal to exit immediately
                            logger.info("Identity confirmed - triggering immediate exit")
                            result_queue.put("IDENTITY_CONFIRMED")
                            return  # Exit the worker thread immediately
                        
                        # Incremental Learning: Only update if face is LIVE and is a KNOWN person
                        # Additional conditions: face must pass anti-spoofing AND be recognized
                        if (is_live and live_score > MIN_LIVENESS_SCORE and 
                            best_match != "Unknown" and 
                            should_update_embedding(best_match, confidence_score)):
                            
                            # ANTI-MISMATCH PROTECTION: Multiple validation layers
                            validation_passed, validation_reason = validate_embedding_consistency(
                                best_match, embedding, embeddings_db[best_match]
                            )
                            
                            if validation_passed:
                                logger.info(f"Updating embeddings for {best_match} (conf: {confidence_score:.3f}, live_score: {live_score:.3f})")
                                print(f"[INCREMENTAL_LEARNING] Updating embeddings for {best_match} (conf: {confidence_score:.3f}, live_score: {live_score:.3f})")
                                
                                # Update embeddings using weighted averaging
                                updated_embeddings = update_embedding_weighted(
                                    embeddings_db[best_match], 
                                    embedding
                                )
                                
                                # Update in-memory database
                                embeddings_db[best_match] = updated_embeddings
                                
                                # Save to file synchronously to ensure persistence before exit
                                save_success = save_updated_database(embeddings_db)
                                if save_success:
                                    logger.info(f"Synchronous save completed for {best_match}")
                                    print(f"[INCREMENTAL_LEARNING] Saved updated embeddings for {best_match}")
                                else:
                                    logger.error(f"Synchronous save failed for {best_match}")
                                
                                label += " [LEARNING]"
                                
                                # Track this learning event for monitoring
                                logger.info(f"Successfully updated {best_match}: {len(updated_embeddings)} total embeddings")
                                print(f"[INCREMENTAL_LEARNING] Successfully updated {best_match}: {len(updated_embeddings)} embeddings")
                                
                            else:
                                # Validation failed - potential mismatch detected
                                logger.warning(f"MISMATCH PREVENTION: Blocked embedding update for {best_match}: {validation_reason}")
                                label += f" [BLOCKED: {validation_reason}]"
                                
                                # Alert about potential security issue
                                if MISMATCH_DETECTION_ENABLED:
                                    logger.error(f"⚠️  SECURITY ALERT: Potential face mismatch detected for {best_match}")
                                    logger.error(f"   Reason: {validation_reason}")
                                    logger.error(f"   Confidence: {confidence_score:.3f}, Live Score: {live_score:.3f}")
                            

                        elif not is_live:
                            logger.debug(f"Skipping embedding update for {best_match}: face not live (score: {live_score:.3f})")
                        elif live_score <= MIN_LIVENESS_SCORE:
                            logger.debug(f"Skipping embedding update for {best_match}: low liveness score ({live_score:.3f} <= {MIN_LIVENESS_SCORE})")
                            
                    else:
                        label = f"Unknown ({best_score:.2f}) [LIVE:{live_score:.2f}]"
                        # Never update embeddings for unknown faces
                        logger.debug(f"Skipping embedding update: unknown person (score: {best_score:.3f})")
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
# Create a named window and attempt to make it always-on-top (platform-dependent)
window_name = "Live Face Recognition"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
try:
    # OpenCV provides setWindowProperty for some builds; try to set topmost
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
except Exception:
    logger.debug("Could not set window topmost via OpenCV; window may not be always on top")

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
        if new_results == "IDENTITY_CONFIRMED":
            # Identity confirmed - exit immediately
            logger.info("Identity confirmation received - exiting immediately")
            print(f"[SUCCESS] Identity confirmed: {confirmed_identity}")
            break
        elif new_results:
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

    # Display identity confirmation status
    display_confirmation_status(display_frame)

    cv2.imshow(window_name, display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or (identity_confirmed and confirmed_identity):
        break

# Cleanup resources
cap.release()
cv2.destroyAllWindows()



