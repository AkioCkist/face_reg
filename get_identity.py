#!/usr/bin/env python3
"""
Identity Recognition Wrapper
This script runs the face recognition system and returns the confirmed identity.
"""

import subprocess
import sys
import os
from setup_logging import setup_logging

# Set up logging
logger, _ = setup_logging("identity_wrapper")

def run_face_recognition():
    """Run face recognition and return the confirmed identity"""
    
    try:
        # Run the face recognition system
        logger.info("Starting face recognition system...")
        
        # Execute the live recognition script
        result = subprocess.run([
            sys.executable, "live_face_recognition.py"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        # Check if the process completed successfully
        if result.returncode == 0:
            logger.info("Face recognition completed successfully")
            
            # Try to read the result from the output file
            try:
                with open("recognition_result.txt", "r") as f:
                    confirmed_name = f.read().strip()
                
                if confirmed_name and confirmed_name != "NONE":
                    logger.info(f"Identity confirmed: {confirmed_name}")
                    return confirmed_name
                else:
                    logger.warning("No identity was confirmed")
                    return None
                    
            except FileNotFoundError:
                logger.error("Recognition result file not found")
                return None
            except Exception as e:
                logger.error(f"Error reading result file: {e}")
                return None
        else:
            logger.error(f"Face recognition failed with return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error running face recognition: {e}")
        return None

def get_confirmed_identity():
    """Main function to get confirmed identity"""
    
    print("🔍 Starting Identity Recognition System...")
    print("=" * 50)
    
    # Run face recognition
    confirmed_name = run_face_recognition()
    
    if confirmed_name:
        print(f"✅ IDENTITY CONFIRMED: {confirmed_name}")
        print(f"📋 Result: {confirmed_name}")
        return confirmed_name
    else:
        print("❌ IDENTITY NOT CONFIRMED")
        print("📋 Result: None")
        return None

if __name__ == "__main__":
    # Example usage
    identity = get_confirmed_identity()
    
    if identity:
        print(f"\n🎯 You can now use the confirmed identity: '{identity}'")
        print("\n💡 Integration examples:")
        print(f"   - Door access for: {identity}")
        print(f"   - Login as: {identity}")
        print(f"   - Attendance logged for: {identity}")
    else:
        print("\n⚠️  No identity confirmed - access denied")
    
    # For automation/scripting, you can use the return value
    sys.exit(0 if identity else 1)