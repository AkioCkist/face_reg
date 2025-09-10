"""
Performance benchmark for face recognition improvements
"""
import cv2
import numpy as np
import time
from deepface import DeepFace

def benchmark_detection_backends():
    """Test different detection backends on a sample image"""
    
    # Create a test image (or use an existing one)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    backends = ["opencv", "mediapipe", "mtcnn", "retinaface"]
    results = {}
    
    print("Benchmarking detection backends...")
    print("-" * 50)
    
    for backend in backends:
        try:
            start_time = time.time()
            
            # Test 5 iterations
            for _ in range(5):
                reps = DeepFace.represent(
                    img_path=test_image,
                    model_name="ArcFace", 
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True,
                    max_faces=1
                )
            
            avg_time = (time.time() - start_time) / 5
            results[backend] = {
                "avg_time": avg_time,
                "status": "✓ Available"
            }
            
        except Exception as e:
            results[backend] = {
                "avg_time": float('inf'),
                "status": f"✗ Error: {str(e)[:50]}..."
            }
    
    # Display results
    print(f"{'Backend':<12} {'Status':<20} {'Avg Time (s)':<15}")
    print("-" * 50)
    
    for backend, data in results.items():
        time_str = f"{data['avg_time']:.3f}" if data['avg_time'] != float('inf') else "N/A"
        print(f"{backend:<12} {data['status']:<20} {time_str:<15}")
    
    return results

def benchmark_resolution_impact():
    """Test performance impact of different resolutions"""
    
    resolutions = [
        (160, 120),
        (320, 240), 
        (640, 480),
        (1280, 720)
    ]
    
    print("\nBenchmarking resolution impact...")
    print("-" * 40)
    
    for width, height in resolutions:
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        try:
            start_time = time.time()
            
            # Test 3 iterations
            for _ in range(3):
                reps = DeepFace.represent(
                    img_path=test_image,
                    model_name="ArcFace",
                    detector_backend="opencv",  # Use fastest available
                    enforce_detection=False,
                    align=True,
                    max_faces=1
                )
            
            avg_time = (time.time() - start_time) / 3
            pixels = width * height
            
            print(f"{width}x{height:<8} ({pixels:>7} px): {avg_time:.3f}s")
            
        except Exception as e:
            print(f"{width}x{height:<8}: Error - {str(e)[:30]}...")

if __name__ == "__main__":
    print("Face Recognition Performance Benchmark")
    print("=" * 50)
    
    # Test detection backends
    backend_results = benchmark_detection_backends()
    
    # Test resolution impact  
    benchmark_resolution_impact()
    
    # Recommendations
    print("\nRecommendations:")
    print("-" * 20)
    
    available_backends = [b for b, data in backend_results.items() 
                         if "Available" in data["status"]]
    
    if available_backends:
        fastest_backend = min(available_backends, 
                            key=lambda b: backend_results[b]["avg_time"])
        print(f"• Fastest available backend: {fastest_backend}")
    
    print("• Optimal detection resolution: 320x240")
    print("• Consider installing missing backends for better accuracy:")
    
    missing = [b for b, data in backend_results.items() 
              if "Error" in data["status"]]
    for backend in missing:
        if backend == "mediapipe":
            print("  pip install mediapipe")
        elif backend == "retinaface":
            print("  pip install retinaface")
        elif backend == "mtcnn":
            print("  pip install mtcnn")
