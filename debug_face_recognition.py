#!/usr/bin/env python3
"""Debug script to test face detection and recognition components"""

import cv2
import numpy as np
from deepface import DeepFace
import json
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from setup_logging import setup_logging

logger, _ = setup_logging("debug_face_recognition", logging.INFO)

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config/config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}

def test_face_detection():
    """Test basic face detection"""
    print("=== Testing Face Detection ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Cannot capture frame")
        return False
    
    print(f"✅ Camera working - Frame size: {frame.shape}")
    
    # Test different detection backends
    backends = ["opencv", "retinaface", "mediapipe", "mtcnn"]
    
    for backend in backends:
        try:
            print(f"\n--- Testing {backend} backend ---")
            
            # Try face detection
            reps = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend=backend,
                enforce_detection=False,
                align=True,
                max_faces=3,
            )
            
            print(f"✅ {backend}: Detected {len(reps)} faces")
            
            for i, rep in enumerate(reps):
                facial_area = rep["facial_area"]
                confidence = rep.get("face_confidence", "N/A")
                print(f"   Face {i+1}: {facial_area}, confidence: {confidence}")
                
        except Exception as e:
            print(f"❌ {backend}: Failed - {e}")
    
    return True

def test_anti_spoofing():
    """Test anti-spoofing configuration"""
    print("\n=== Testing Anti-Spoofing Configuration ===")
    
    config = load_config()
    anti_spoof_config = config.get("anti_spoofing", {})
    
    print(f"Anti-spoofing enabled: {anti_spoof_config.get('enabled', True)}")
    print(f"Mode: {anti_spoof_config.get('mode', 'normal')}")
    
    # Test the anti-spoofing function
    try:
        from live_face_recognition import check_anti_spoofing_live
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Create a dummy face region
            h, w = frame.shape[:2]
            face_region = (w//4, h//4, w//2, h//2)  # Center region
            
            is_live, reason, score = check_anti_spoofing_live(frame, face_region)
            print(f"Anti-spoofing result: is_live={is_live}, reason='{reason}', score={score}")
        else:
            print("❌ Cannot test anti-spoofing - no camera frame")
            
    except Exception as e:
        print(f"❌ Anti-spoofing test failed: {e}")

def test_database_loading():
    """Test database loading"""
    print("\n=== Testing Database Loading ===")
    
    try:
        from live_face_recognition import load_embeddings_from_database
        from persistence.repository import FaceRepository
        
        # Test SQL database
        embeddings_db = load_embeddings_from_database()
        print(f"SQL Database: {len(embeddings_db)} entries loaded")
        
        # Test JSON repository
        repo = FaceRepository("face_db.json")
        json_db = repo.load()
        print(f"JSON Repository: {len(json_db)} entries loaded")
        
        if embeddings_db:
            sample_key = list(embeddings_db.keys())[0]
            sample_embeddings = embeddings_db[sample_key]
            print(f"Sample entry '{sample_key}': {len(sample_embeddings)} embeddings")
        
    except Exception as e:
        print(f"❌ Database loading failed: {e}")

def test_complete_recognition():
    """Test complete recognition pipeline"""
    print("\n=== Testing Complete Recognition Pipeline ===")
    
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot capture frame for testing")
            return
        
        print("🔍 Testing face detection...")
        reps = DeepFace.represent(
            img_path=frame,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            max_faces=1,
        )
        
        print(f"✅ Detected {len(reps)} faces")
        
        if reps:
            rep = reps[0]
            embedding = np.array(rep["embedding"])
            facial_area = rep["facial_area"]
            
            print(f"Face region: {facial_area}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding sample: {embedding[:5]}")
            
            # Test anti-spoofing
            from live_face_recognition import check_anti_spoofing_live
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            is_live, reason, score = check_anti_spoofing_live(frame, (x, y, w, h))
            
            print(f"Anti-spoofing: is_live={is_live}, reason='{reason}', score={score}")
            
            if is_live:
                print("✅ Face passed anti-spoofing - would proceed to recognition")
            else:
                print("❌ Face failed anti-spoofing - would be skipped")
        
    except Exception as e:
        print(f"❌ Complete recognition test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import logging
    
    print("🔍 Face Recognition Debug Tool")
    print("=" * 50)
    
    test_face_detection()
    test_anti_spoofing()
    test_database_loading()
    test_complete_recognition()
    
    print("\n" + "=" * 50)
    print("Debug complete!")