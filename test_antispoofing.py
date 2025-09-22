#!/usr/bin/env python3
"""
Test script to verify improved anti-spoofing performance in various lighting conditions
"""

import cv2
import numpy as np
import json
import logging
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_db import check_anti_spoofing_deepface, analyze_lighting_conditions, load_config
from setup_logging import setup_logging

# Set up logging
logger, _ = setup_logging("antispoofing_test", logging.INFO)

def create_test_conditions():
    """Create test frames with different lighting conditions"""
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return None
    
    test_frames = {}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    conditions = {
        'normal': 'Normal lighting (press SPACE when ready)',
        'dark': 'Low lighting condition (press SPACE when ready)',  
        'bright': 'Bright lighting condition (press SPACE when ready)'
    }
    
    for condition, instruction in conditions.items():
        print(f"\n{instruction}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display current frame
            display_frame = frame.copy()
            cv2.putText(display_frame, instruction, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture, ESC to skip", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Anti-spoofing Test Setup', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    test_frames[condition] = {
                        'frame': frame.copy(),
                        'face_region': faces[0]  # Use first detected face
                    }
                    logger.info(f"Captured {condition} lighting condition")
                    break
                else:
                    print("No face detected, try again...")
            elif key == 27:  # ESC to skip
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return test_frames

def test_lighting_analysis(test_frames):
    """Test the lighting analysis function"""
    config = load_config()
    
    print("\n" + "="*60)
    print("LIGHTING ANALYSIS TEST")
    print("="*60)
    
    for condition, data in test_frames.items():
        frame = data['frame']
        x, y, w, h = data['face_region']
        face_roi = frame[y:y+h, x:x+w]
        
        factor, detected_condition = analyze_lighting_conditions(face_roi, config)
        
        # Calculate actual brightness stats
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        variance = np.std(gray)
        
        print(f"\n{condition.upper()} LIGHTING:")
        print(f"  Expected: {condition}")
        print(f"  Detected: {detected_condition}")
        print(f"  Brightness: {brightness:.1f}")
        print(f"  Variance: {variance:.1f}")
        print(f"  Adjustment Factor: {factor:.3f}")

def test_anti_spoofing(test_frames):
    """Test anti-spoofing with different lighting conditions"""
    print("\n" + "="*60)
    print("ANTI-SPOOFING TEST")
    print("="*60)
    
    for condition, data in test_frames.items():
        frame = data['frame']
        face_region = data['face_region']
        
        print(f"\n{condition.upper()} LIGHTING:")
        
        # Test anti-spoofing
        is_real, reason, confidence = check_anti_spoofing_deepface(frame, face_region)
        
        print(f"  Result: {'✓ REAL' if is_real else '✗ FAKE'}")
        print(f"  Reason: {reason}")
        print(f"  Confidence: {confidence:.3f}")
        
        # Color code the result
        color = (0, 255, 0) if is_real else (0, 0, 255)
        status = "REAL FACE" if is_real else "FAKE FACE"
        
        # Display result frame
        result_frame = frame.copy()
        x, y, w, h = face_region
        cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_frame, f"{condition.upper()}: {status}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result_frame, f"Conf: {confidence:.3f}", (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Show frame for a few seconds
        cv2.imshow(f'Anti-spoofing Result - {condition}', result_frame)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()

def print_config_summary():
    """Print current configuration summary"""
    config = load_config()
    anti_spoof = config.get("anti_spoofing", {})
    
    print("="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Anti-spoofing enabled: {anti_spoof.get('enabled', True)}")
    print(f"Mode: {anti_spoof.get('mode', 'normal')}")
    print(f"Texture threshold: {anti_spoof.get('texture_variance_threshold', 80)}")
    
    lighting = anti_spoof.get("lighting_compensation", {})
    if lighting.get("enabled", True):
        print("\nLighting Compensation:")
        print(f"  Min brightness: {lighting.get('min_brightness', 20)}")
        print(f"  Max brightness: {lighting.get('max_brightness', 235)}")
        print(f"  Low light factor: {lighting.get('low_light_texture_factor', 0.5)}")
        print(f"  Bright light factor: {lighting.get('bright_light_texture_factor', 1.5)}")

def main():
    print("Face Recognition Anti-spoofing Test")
    print("This script will test the improved anti-spoofing system")
    print("in different lighting conditions.")
    
    print_config_summary()
    
    # Create test conditions
    test_frames = create_test_conditions()
    
    if not test_frames:
        print("Failed to capture test frames")
        return
    
    if len(test_frames) == 0:
        print("No test frames captured")
        return
    
    # Run tests
    test_lighting_analysis(test_frames)
    test_anti_spoofing(test_frames)
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    print("Check the results above to see if the anti-spoofing system")
    print("is working better in poor lighting conditions.")
    
    # Summary
    print(f"\nTested {len(test_frames)} lighting conditions:")
    for condition in test_frames.keys():
        print(f"  - {condition}")

if __name__ == "__main__":
    main()