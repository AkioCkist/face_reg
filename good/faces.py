import time
import os
import numpy as np

# Optional: limit TF log spam if TF exists
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Try to detect GPU support via torch and tensorflow
gpu_backend = None
try:
    import torch
    if torch.cuda.is_available():
        gpu_backend = "torch"
        print("PyTorch GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("PyTorch installed but CUDA not available.")
except Exception as e:
    print("PyTorch not available:", e)

if gpu_backend is None:
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            gpu_backend = "tensorflow"
            gpus = tf.config.list_physical_devices("GPU")
            print("TensorFlow GPU detected:", gpus)
            # Optionally enable memory growth to avoid allocating all memory
            for g in gpus:
                try:
                    tf.config.experimental.set_memory_growth(g, True)
                except Exception:
                    pass
        else:
            print("TensorFlow installed but no GPU detected.")
    except Exception as e:
        print("TensorFlow not available:", e)

if gpu_backend is None:
    print("No supported GPU backend detected (PyTorch/TF). DeepFace will fallback to CPU.")

# Now run DeepFace verify with ArcFace
from deepface import DeepFace

img1 = "person1.jpg"   # replace with your path
img2 = "person4.jpg"   # replace with your path

# Pre-build model to ensure it loads once (and to see where it's loaded from)
print("\nBuilding ArcFace model (this may download weights the first time)...")
t0 = time.time()
model = DeepFace.build_model("ArcFace")  # returns loaded model object
t1 = time.time()
print(f"Model built in {t1-t0:.2f}s. Backend: {gpu_backend or 'cpu'}")

# Now run verify and time it
print("\nRunning DeepFace.verify() ...")
t0 = time.time()
result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name="ArcFace", enforce_detection=True, detector_backend="retinaface")
t1 = time.time()
print(f"Verify done in {t1-t0:.2f}s")
print("Result:")
print(result)

# Quick GPU verification (if torch)
if gpu_backend == "torch":
    import torch
    print("\nPyTorch CUDA summary:")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        # show approximate memory usage (if supported)
        try:
            print("Allocated (MB):", torch.cuda.memory_allocated() / 1024**2)
            print("Reserved  (MB):", torch.cuda.memory_reserved() / 1024**2)
        except Exception:
            pass
else:
    print("\nNo PyTorch GPU backend in use.")

print("\nIf you want higher throughput for webcam use, run recognition in a separate worker thread and pre-load the model once.")
