#!/usr/bin/env python3
"""
Quick test to verify live face recognition improvements
"""

import cv2
import json
import logging
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from live_face_recognition import check_anti_spoofing_live, load_config, analyze_lighting_conditions
from setup_logging import setup_logging

# Set up logging
logger, _ = setup_logging("live_test", logging.DEBUG)

def test_live_config():
    """Test that the live recognition system loads the correct config"""
    print("Testing Live Recognition Configuration...")
    
    config = load_config()
    anti_spoof = config.get("anti_spoofing", {})
    
    print(f"Anti-spoofing enabled: {anti_spoof.get('enabled', True)}")
    print(f"Mode: {anti_spoof.get('mode', 'normal')}")
    print(f"Texture threshold: {anti_spoof.get('texture_variance_threshold', 80)}")
    
    lighting = anti_spoof.get("lighting_compensation", {})
    if lighting:
        print(f"Lighting compensation enabled: {lighting.get('enabled', False)}")
        print(f"Min brightness: {lighting.get('min_brightness', 30)}")
        print(f"Max brightness: {lighting.get('max_brightness', 220)}")
        print(f"Low light factor: {lighting.get('low_light_texture_factor', 0.7)}")
    else:
        print("⚠️  WARNING: No lighting compensation configured!")
    
    return config

def test_quick_face_detection():
    """Quick test with webcam to verify anti-spoofing"""
    print("\nTesting Live Face Detection...")
    print("This will capture a few frames and test anti-spoofing")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    test_count = 0
    max_tests = 5
    
    print(f"Testing {max_tests} frames...")
    
    while test_count < max_tests:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            test_count += 1
            x, y, w, h = faces[0]
            
            # Test anti-spoofing
            is_live, reason, confidence = check_anti_spoofing_live(frame, (x, y, w, h))
            
            # Test lighting analysis
            face_roi = frame[y:y+h, x:x+w]
            config = load_config()
            factor, light_condition = analyze_lighting_conditions(face_roi, config)
            
            print(f"Frame {test_count}: {'✅ LIVE' if is_live else '❌ SPOOF'} - {reason}")
            print(f"  Lighting: {light_condition} (factor: {factor:.3f})")
            print(f"  Confidence: {confidence:.3f}")
            print()
            
        cv2.waitKey(100)  # Small delay
    
    cap.release()
    print("Live face detection test completed.")

def main():
    print("Live Face Recognition System Test")
    print("=" * 50)
    
    # Test 1: Configuration
    config = test_live_config()
    
    # Test 2: Live detection
    input("\nPress Enter to start live face detection test...")
    test_quick_face_detection()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nIf you see:")
    print("✅ Adaptive mode enabled")
    print("✅ Lighting compensation configured") 
    print("✅ Live faces being detected properly")
    print("\nThen the improvements are working!")

if __name__ == "__main__":
    main()