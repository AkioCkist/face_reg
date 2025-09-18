"""
Test scripts for Face Recognition API
Tests all endpoints with sample data
"""

import requests
import json
import base64
import cv2
import numpy as np
from io import BytesIO
import asyncio
import websockets
import time

# API Configuration
API_BASE_URL = "http://localhost:8000/api"
WS_BASE_URL = "ws://localhost:8000/api/ws"

def create_sample_image():
    """Create a sample face image for testing"""
    # Create a simple test image with basic face-like features
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128  # Gray background
    
    # Draw a simple face
    cv2.circle(img, (100, 100), 80, (200, 180, 160), -1)  # Face
    cv2.circle(img, (75, 80), 8, (0, 0, 0), -1)   # Left eye
    cv2.circle(img, (125, 80), 8, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (100, 110), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    return img

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def test_health_check():
    """Test the health check endpoint"""
    print("Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_get_config():
    """Test the configuration endpoint"""
    print("\nTesting Get Config...")
    try:
        response = requests.get(f"{API_BASE_URL}/config")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Config test failed: {e}")
        return False

def test_register_face():
    """Test face registration endpoint"""
    print("\nTesting Face Registration...")
    
    # Create sample image
    sample_image = create_sample_image()
    image_base64 = image_to_base64(sample_image)
    
    # Registration data
    register_data = {
        "account_id": "test_user_001",
        "image_base64": image_base64,
        "override": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/faces/register",
            json=register_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200 and result.get("success", False)
        
    except Exception as e:
        print(f"Registration test failed: {e}")
        return False

def test_recognize_face():
    """Test face recognition endpoint"""
    print("\nTesting Face Recognition...")
    
    # Create sample image (same as registered)
    sample_image = create_sample_image()
    image_base64 = image_to_base64(sample_image)
    
    # Recognition data
    recognize_data = {
        "image_base64": image_base64,
        "similarity_threshold": 0.45
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/faces/recognize",
            json=recognize_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200 and result.get("success", False)
        
    except Exception as e:
        print(f"Recognition test failed: {e}")
        return False

def test_list_faces():
    """Test list faces endpoint"""
    print("\nTesting List Faces...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/faces")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200 and result.get("success", False)
        
    except Exception as e:
        print(f"List faces test failed: {e}")
        return False

def test_delete_face():
    """Test delete face endpoint"""
    print("\nTesting Delete Face...")
    
    try:
        response = requests.delete(f"{API_BASE_URL}/faces/test_user_001")
        print(f"Status: {response.status_code}")
        
        if response.status_code in [200, 404]:  # 404 is OK if face doesn't exist
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Delete face test failed: {e}")
        return False

async def test_websocket():
    """Test WebSocket real-time recognition"""
    print("\nTesting WebSocket Connection...")
    
    try:
        uri = f"{WS_BASE_URL}/recognize/test_client"
        
        async with websockets.connect(uri) as websocket:
            print("WebSocket connected successfully")
            
            # Send ping
            await websocket.send(json.dumps({
                "type": "ping"
            }))
            
            # Wait for pong
            response = await websocket.recv()
            pong_data = json.loads(response)
            print(f"Ping/Pong test: {pong_data}")
            
            # Send status request
            await websocket.send(json.dumps({
                "type": "status"
            }))
            
            # Wait for status
            response = await websocket.recv()
            status_data = json.loads(response)
            print(f"Status: {json.dumps(status_data, indent=2)}")
            
            # Send sample frame
            sample_image = create_sample_image()
            image_base64 = image_to_base64(sample_image)
            
            await websocket.send(json.dumps({
                "type": "frame",
                "frame": f"data:image/jpeg;base64,{image_base64}",
                "similarity_threshold": 0.45
            }))
            
            # Wait for recognition result
            response = await websocket.recv()
            recognition_data = json.loads(response)
            print(f"Recognition result: {json.dumps(recognition_data, indent=2)}")
            
            print("WebSocket test completed successfully")
            return True
            
    except Exception as e:
        print(f"WebSocket test failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("="*60)
    print("FACE RECOGNITION API TEST SUITE")
    print("="*60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Get Config", test_get_config),
        ("Register Face", test_register_face),
        ("Recognize Face", test_recognize_face),
        ("List Faces", test_list_faces),
        ("Delete Face", test_delete_face),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print(f"{'='*40}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Test WebSocket separately
    print(f"\n{'='*40}")
    print("Running: WebSocket Test")
    print(f"{'='*40}")
    
    try:
        results["WebSocket"] = asyncio.run(test_websocket())
    except Exception as e:
        print(f"WebSocket test crashed: {e}")
        results["WebSocket"] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API server and database connection.")
    
    return results

if __name__ == "__main__":
    print("Face Recognition API Test Suite")
    print("Make sure the API server is running on http://localhost:8000")
    print()
    
    input("Press Enter to start tests...")
    
    results = run_all_tests()
    
    print(f"\nTest completed. Results: {results}")