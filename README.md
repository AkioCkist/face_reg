# Enhanced Face Recognition System

## Improvements Made

### Recognition Accuracy Improvements
1. **Multi-Backend Detection System**: Tries multiple detection backends in order:
   - RetinaFace (most accurate)
   - MediaPipe (fast and robust)
   - MTCNN (balanced)
   - OpenCV (fallback)

2. **Stricter Similarity Threshold**: Reduced from 0.6 to 0.35 for more precise matching

3. **Face Confidence Filtering**: Only processes faces with confidence > 0.7

4. **Normalized Embeddings**: Better cosine similarity calculation using normalized vectors

5. **Temporal Smoothing**: Uses voting across 7 frames with minimum 4 votes for stable recognition

### Performance Improvements
1. **Dual Resolution Processing**:
   - Detection: 320x240 (faster processing)
   - Display: 640x480 (better visualization)

2. **Optimized Frame Processing**:
   - Process every 3rd frame instead of every 5th
   - Larger queue buffer (maxsize=2)
   - Camera buffer optimization

3. **Backend Caching**: Remembers successful detection backend for faster subsequent frames

4. **Enhanced Threading**: Better queue management and error handling

5. **Real-time Performance Monitoring**: Shows FPS, backend used, and processing time

### Visual Enhancements
- Color-coded rectangles (green for known, red for unknown)
- Confidence-based line thickness
- Better label visibility with background
- Real-time performance stats overlay

## Configuration
Adjust these constants in the script for your needs:

```python
SIMILARITY_THRESHOLD = 0.35      # Lower = stricter matching
FACE_CONFIDENCE_MIN = 0.7        # Minimum face detection confidence
SMOOTHING_WINDOW = 7             # Frames for temporal smoothing
MIN_VOTES = 4                    # Minimum votes to confirm identity
FRAME_SKIP = 3                   # Process every Nth frame
```

## Usage
1. Ensure your face database (`face_db.json`) is ready
2. Run: `python live_face_recognition.py`
3. Press 'q' to quit

## Dependencies
- opencv-python
- deepface
- numpy
- tf-keras (for RetinaFace backend)
- mediapipe (for MediaPipe backend)

Install missing backends for better accuracy:
```bash
pip install mediapipe retinaface tf-keras
```
