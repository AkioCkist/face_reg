# 🧑‍💻 Real-time Face Recognition System  

This repository implements a **real-time face recognition system** with **incremental learning** and **anti-spoofing protection**.  
It is designed for practical security applications, ensuring **high accuracy**, **robust spoof detection**, and **continuous learning** over time.  

---

## 🎯 System Overview  
- **Real-time face recognition** using DeepFace ArcFace model  
- **Anti-spoofing protection** against photo, video, and mask attacks  
- **Incremental learning** to adapt and improve accuracy continuously  
- **Multi-layer mismatch prevention** for reliable identity verification  
- **Configurable settings** via JSON  
- **Detailed monitoring and logging**  

---

## 🏗️ System Architecture

### 1. **Core Data Structures**
```python
embeddings_db = {
    "person_name": [embedding1, embedding2, ...]  # Store multiple embeddings per person
}
````

### 2. **Multi-threaded Processing**

* **Main Thread**: Handles camera input and display
* **Worker Thread**: Performs asynchronous face recognition

---

## 🔄 Main Workflow

### **Step 1: Initialization**

```python
# Load face database
embeddings_db = repo.load()
# Load ArcFace model
model = DeepFace.build_model("ArcFace")
```

### **Step 2: Frame Collection**

* Camera capture → Resize for faster processing
* Process every **3rd frame (FRAME\_SKIP)** to optimize performance

### **Step 3: Face Recognition**

```python
def recognition_worker():
    # 1. Feature extraction using ArcFace
    reps = DeepFace.represent(img_path=frame, model_name="ArcFace")
    
    # 2. Anti-spoofing check
    is_live, spoof_reason, live_score = check_anti_spoofing_live()
    
    # 3. Compare against database
    # 4. Incremental learning if conditions are met
```

---

## 🛡️ Anti-Spoofing Security

### **Primary Method: DeepFace anti-spoofing**

```python
result = DeepFace.extract_faces(
    img_path=temp_face_path,
    anti_spoofing=True  # Enable anti-spoofing
)
```

### **Fallback Methods:**

* **Texture variance analysis**
* **Edge density check**
* **Color variance analysis**

---

## 🧠 Incremental Learning

### **Conditions for updating embeddings:**

```python
def should_update_embedding(person_name, confidence_score):
    # 1. Confidence > minimum threshold (0.8)
    # 2. Live score > liveness threshold (0.7)
    # 3. Cooldown period passed (5 seconds)
    # 4. Enough successful recognitions accumulated
```

### **Mismatch Protection:**

```python
def validate_embedding_consistency():
    # Layer 1: Drift detection – detect abnormal embedding deviation
    # Layer 2: Suspicious activity – track unusual recognition patterns
    # Layer 3: Quality check – ensure embedding quality is sufficient
```

---

## 🎯 Identity Verification

### **Multi-layer Confirmation Mechanism:**

```python
def check_identity_confirmation():
    # Require at least 3 confirmations
    # Confidence >= 0.9
    # Live score >= 0.8
    # → Automatically finalize when confirmed
```

---

## 📊 Optimization Features

### **1. Temporal Smoothing**

* Use a **sliding window (5 frames)** to stabilize results
* **Voting mechanism**: choose the most frequent result

### **2. Performance Optimization**

* **Frame skipping** (process every 3rd frame)
* **Multi-threading** (separate capture and recognition)
* **Frame resizing** for faster processing

### **3. Monitoring & Logging**

```python
logger.info("Tracking all important events")
# - Recognition results
# - Learning updates  
# - Security alerts
# - Performance metrics
```

---

## 🔧 Flexible Configuration

The system loads runtime settings from `config.json`:

```json
{
    "detection": {"similarity_threshold": 0.45},
    "anti_spoofing": {"enabled": true},
    "incremental_learning": {"enabled": true},
    "identity_confirmation": {"min_confirmations": 3}
}
```

---

## 💾 Persistent Storage

```python
# Using repository pattern
repo = FaceRepository("face_db.json")
repo.save(embeddings_db)  # Auto-save after each learning update
```

---

## 🎮 User Interface

* **Visual indicators:**

  * 🟢 Green: Real face recognized
  * 🔴 Red: Spoof detected
  * 🟡 Yellow: Real face but unknown identity

* **Displayed information:**

  * FPS counter
  * Confidence score
  * Live score
  * Identity confirmation status

---

## 🚀 Key Advantages  
- **High Security**: Multi-layer anti-spoofing + mismatch detection  
- **Adaptive Learning**: Improves accuracy over time automatically  
- **Optimized Performance**: Multi-threading + frame skipping  
- **Stable & Robust**: Temporal smoothing reduces noise  
- **Comprehensive Monitoring**: Logging for debug & audit  

---

## 📂 Repository Structure  
- `face_db.py` — create/append a face database (multi-embedding per person)  
- `live_face_recognition.py` — run live recognition with incremental learning  
- `config/config.json` — runtime settings (thresholds, detector backends, learning parameters)  
- `setup_logging.py` — logging configuration module  
- `view_logs.py` — utility to view log files  
- `run_face_recognition.bat` — optional Windows launcher  
- `DB_USAGE.md` — details about DB format and behavior  
- `INCREMENTAL_LEARNING.md` — detailed explanation of adaptive learning  
- `face_db.json` — user data (keep private, excluded from git)  
- `logs/` — directory containing log files  

---

## 🖥️ Supported OS & Python  
- Tested on **Windows 10/11**  
- Recommended: **Python 3.8 – 3.11**  

---

## 📦 Dependencies  
Core packages:  
- `deepface`  
- `tensorflow` or `tensorflow-cpu`  
- `opencv-python`  
- `numpy`  
- `tqdm`  

Optional (for development):  
- `matplotlib`  
- `pandas`  

---

## ⚙️ Installation  

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install "tensorflow-cpu" deepface opencv-python numpy tqdm

# 3. (Optional) Save environment
pip freeze > requirements.txt
````

For GPU support, install the correct TensorFlow version compatible with your CUDA/cuDNN drivers.

---

## ▶️ Usage

### 1. Create or update the face database

```bash
python face_db.py
```

* Default new person naming: `personN`
* Supports **append**, **overwrite**, or **skip** existing entries
* Database format: JSON with multiple embeddings per person

### 2. Start live recognition

```bash
python live_face_recognition.py
```

* Loads `face_db.json`
* Uses `config.json` for thresholds, backends, and learning parameters

### 3. View logs

```bash
python view_logs.py
```

* Lists log files with timestamps
* View recognition history and debug information

---

## 🔒 Security & Privacy

* `face_db.json` stores sensitive embeddings — **do not commit** to public repos.
* Add it to `.gitignore`.
* For extra protection, consider encryption or secure storage.

---

## 🛠️ Troubleshooting

* **Camera not detected**: check device index (`cv2.VideoCapture(0)` → `(1)` etc.)
* **DeepFace errors**: verify TensorFlow compatibility
* **Window top-most issues**: depends on OpenCV build/OS window manager

---


## 📄 License

This project is licensed under the MIT License 

*Built with ❤️ by Akio Ckist*
