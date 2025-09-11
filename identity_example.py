"""
Example: How to get and use the confirmed identity
This demonstrates different ways to retrieve the confirmed person's name.
"""

import os
import sys
import subprocess
from datetime import datetime

def method1_read_result_file():
    """Method 1: Read from recognition_result.txt after running the system"""
    
    print("Method 1: Using result file")
    print("-" * 30)
    
    # Run the face recognition system
    print("🔍 Starting face recognition...")
    os.system("python live_face_recognition.py")
    
    # Read the result
    try:
        with open("recognition_result.txt", "r") as f:
            result = f.read().strip()
        
        if result and result != "NONE":
            print(f"✅ Identity confirmed: {result}")
            return result
        else:
            print("❌ No identity confirmed")
            return None
    except FileNotFoundError:
        print("❌ Result file not found")
        return None

def method2_capture_output():
    """Method 2: Capture output directly from the script"""
    
    print("Method 2: Capturing output")
    print("-" * 30)
    
    try:
        # Run and capture output
        result = subprocess.run([
            sys.executable, "live_face_recognition.py"
        ], capture_output=True, text=True)
        
        # Look for confirmation in output
        lines = result.stdout.split('\n')
        for line in lines:
            if "CONFIRMED:" in line:
                name = line.split("CONFIRMED:")[1].strip()
                print(f"✅ Identity confirmed: {name}")
                return name
        
        print("❌ No identity confirmed")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def method3_using_wrapper():
    """Method 3: Using the wrapper function"""
    
    print("Method 3: Using wrapper function")
    print("-" * 30)
    
    # Import and use the wrapper
    from get_identity import get_confirmed_identity
    
    identity = get_confirmed_identity()
    
    if identity:
        print(f"✅ Identity confirmed: {identity}")
        return identity
    else:
        print("❌ No identity confirmed")
        return None

def use_confirmed_identity(name):
    """Example of how to use the confirmed identity"""
    
    if not name:
        print("⚠️  Cannot proceed - no identity confirmed")
        return
    
    print(f"\n🎯 Using confirmed identity: {name}")
    print("=" * 40)
    
    # Example 1: Access control
    print(f"🚪 Granting access to: {name}")
    
    # Example 2: Logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"📝 Logging entry: {name} at {timestamp}")
    
    # Example 3: Personalized greeting
    print(f"👋 Welcome back, {name}!")
    
    # Example 4: Save to attendance log
    try:
        with open("attendance_log.txt", "a") as f:
            f.write(f"{timestamp} - {name}\n")
        print(f"📊 Attendance logged for {name}")
    except Exception as e:
        print(f"❌ Failed to log attendance: {e}")
    
    # Example 5: Trigger other systems
    print(f"🔧 You can now trigger other systems for {name}:")
    print(f"   - Send notification to {name}")
    print(f"   - Load {name}'s preferences")
    print(f"   - Grant {name} access to resources")

def main():
    """Main demonstration"""
    
    print("🔍 Face Recognition Identity Retrieval Demo")
    print("=" * 50)
    
    print("\nChoose a method to get identity:")
    print("1. Read from result file")
    print("2. Capture output directly")
    print("3. Use wrapper function")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            identity = method1_read_result_file()
        elif choice == "2":
            identity = method2_capture_output()
        elif choice == "3":
            identity = method3_using_wrapper()
        else:
            print("❌ Invalid choice")
            return
        
        # Use the confirmed identity
        use_confirmed_identity(identity)
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo cancelled by user")
    except Exception as e:
        print(f"❌ Error in demo: {e}")

if __name__ == "__main__":
    main()