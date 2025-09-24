#!/usr/bin/env python3
"""Test script for auto-exit face recognition"""

import subprocess
import time
import json
import os
import sys

def test_auto_exit():
    """Test the auto-exit functionality"""
    print("🔍 Testing Auto-Exit Face Recognition")
    print("=" * 50)
    
    # Remove any existing result file
    result_file = "recognition_result.txt"
    if os.path.exists(result_file):
        os.remove(result_file)
        print("Removed existing result file")
    
    print("\n📹 Starting live face recognition...")
    print("👤 Please position your face in front of the camera")
    print("⏰ The program should auto-exit when your face is recognized")
    print("🛑 Press Ctrl+C to stop manually if needed\n")
    
    try:
        # Start the live recognition process
        process = subprocess.Popen(
            [sys.executable, "live_face_recognition.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        while True:
            # Check if process has ended
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print("✅ Process ended automatically")
                if stdout:
                    print("STDOUT:", stdout[-500:])  # Last 500 chars
                break
            
            # Check for timeout
            if time.time() - start_time > timeout:
                print("⏰ Timeout reached - terminating process")
                process.terminate()
                process.wait()
                break
            
            time.sleep(0.5)
        
        # Check if result file was created
        if os.path.exists(result_file):
            print(f"\n📄 Reading result from {result_file}:")
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            print(json.dumps(result, indent=2))
            
            if result.get("success"):
                print("✅ Recognition successful!")
                print(f"   Person: {result.get('person_name', 'Unknown')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                print(f"   Live Score: {result.get('live_score', 0):.3f}")
                print(f"   Timestamp: {result.get('timestamp', 'N/A')}")
            else:
                print("❌ Recognition failed")
        else:
            print("❌ No result file found")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        if process.poll() is None:
            process.terminate()
            process.wait()
    except Exception as e:
        print(f"❌ Error: {e}")
        if process.poll() is None:
            process.terminate()
            process.wait()
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    test_auto_exit()