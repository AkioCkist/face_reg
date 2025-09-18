# Face Recognition REST API Documentation

## Overview

This REST API provides endpoints for face registration, recognition, and database management. It's designed to integrate seamlessly with Next.js frontends and supports both traditional HTTP requests and WebSocket connections for real-time recognition.

## 🚀 Getting Started

### Installation

1. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

2. Start the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## 📡 Endpoints

### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "Face Recognition API",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Get Configuration
```http
GET /api/config
```
**Response:**
```json
{
  "success": true,
  "config": {
    "detection": {
      "backends": ["retinaface", "mediapipe", "mtcnn", "opencv"],
      "similarity_threshold": 0.45,
      "face_confidence_min": 0.7
    },
    "anti_spoofing_enabled": true
  }
}
```

### Register Face
```http
POST /api/faces/register
```
**Request Body:**
```json
{
  "account_id": "user123",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "override": false
}
```
**Response:**
```json
{
  "success": true,
  "message": "Face successfully registered for account user123",
  "account_id": "user123",
  "embeddings_count": 1,
  "anti_spoofing_passed": true,
  "confidence": 1.0
}
```

### Recognize Face
```http
POST /api/faces/recognize
```
**Request Body:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "similarity_threshold": 0.45
}
```
**Response:**
```json
{
  "success": true,
  "recognized": true,
  "account_id": "user123",
  "confidence": 0.892,
  "distance": 0.108,
  "anti_spoofing_passed": true,
  "anti_spoofing_confidence": 0.95,
  "message": "Face recognized as user123"
}
```

### List All Faces
```http
GET /api/faces
```
**Response:**
```json
{
  "success": true,
  "faces": [
    {
      "account_id": "user123",
      "embeddings_count": 1,
      "registered_date": null
    }
  ],
  "total_count": 1
}
```

### Delete Face
```http
DELETE /api/faces/{account_id}
```
**Response:**
```json
{
  "success": true,
  "message": "Face data for account 'user123' has been deleted",
  "account_id": "user123"
}
```

## 🔌 WebSocket API

### Connect to WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/recognize/client123');
```

### Message Types

#### Ping/Pong (Keepalive)
```json
// Send
{"type": "ping"}

// Receive
{"type": "pong", "timestamp": "2024-01-15T10:30:00Z"}
```

#### Send Frame for Recognition
```json
// Send
{
  "type": "frame",
  "frame": "data:image/jpeg;base64,/9j/4AAQ...",
  "similarity_threshold": 0.45
}

// Receive
{
  "type": "recognition_result",
  "data": {
    "success": true,
    "faces_detected": 1,
    "faces": [
      {
        "bbox": {"x": 100, "y": 50, "w": 150, "h": 150},
        "anti_spoofing": {
          "passed": true,
          "confidence": 0.92,
          "reason": "Real face detected"
        },
        "recognition": {
          "recognized": true,
          "account_id": "user123",
          "confidence": 0.89,
          "distance": 0.11
        }
      }
    ],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Get Status
```json
// Send
{"type": "status"}

// Receive
{
  "type": "status",
  "data": {
    "client_id": "client123",
    "connected_at": "2024-01-15T10:25:00Z",
    "frames_processed": 156,
    "last_recognition": {
      "account_id": "user123",
      "confidence": 0.89,
      "timestamp": "2024-01-15T10:29:45Z"
    },
    "active_connections": 3
  }
}
```

## 🌐 Next.js Integration Examples

### Registration Component
```javascript
// components/FaceRegistration.js
import { useState } from 'react';

export default function FaceRegistration() {
  const [accountId, setAccountId] = useState('');
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageCapture = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const registerFace = async () => {
    if (!accountId || !image) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/faces/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          account_id: accountId,
          image_base64: image,
          override: false
        })
      });
      
      const result = await response.json();
      if (result.success) {
        alert('Face registered successfully!');
      } else {
        alert(`Registration failed: ${result.message}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl mb-4">Register Face</h2>
      <input
        type="text"
        placeholder="Account ID"
        value={accountId}
        onChange={(e) => setAccountId(e.target.value)}
        className="w-full p-2 mb-4 border rounded"
      />
      <input
        type="file"
        accept="image/*"
        onChange={handleImageCapture}
        className="w-full p-2 mb-4 border rounded"
      />
      {image && (
        <img src={image} alt="Preview" className="w-32 h-32 object-cover mb-4" />
      )}
      <button
        onClick={registerFace}
        disabled={loading || !accountId || !image}
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Registering...' : 'Register Face'}
      </button>
    </div>
  );
}
```

### Recognition Component
```javascript
// components/FaceRecognition.js
import { useState } from 'react';

export default function FaceRecognition() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageCapture = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const recognizeFace = async () => {
    if (!image) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/faces/recognize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: image,
          similarity_threshold: 0.45
        })
      });
      
      const result = await response.json();
      setResult(result);
    } catch (error) {
      setResult({ success: false, message: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl mb-4">Face Recognition</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageCapture}
        className="w-full p-2 mb-4 border rounded"
      />
      {image && (
        <img src={image} alt="Preview" className="w-32 h-32 object-cover mb-4" />
      )}
      <button
        onClick={recognizeFace}
        disabled={loading || !image}
        className="bg-green-500 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {loading ? 'Recognizing...' : 'Recognize Face'}
      </button>
      
      {result && (
        <div className="mt-4 p-4 border rounded">
          <h3 className="font-bold">Result:</h3>
          {result.success ? (
            <div>
              {result.recognized ? (
                <div className="text-green-600">
                  ✅ Recognized as: {result.account_id}
                  <br />
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                  <br />
                  Anti-spoofing: {result.anti_spoofing_passed ? 'Passed' : 'Failed'}
                </div>
              ) : (
                <div className="text-yellow-600">
                  ⚠️ Face not recognized
                  <br />
                  Anti-spoofing: {result.anti_spoofing_passed ? 'Passed' : 'Failed'}
                </div>
              )}
            </div>
          ) : (
            <div className="text-red-600">❌ Error: {result.message}</div>
          )}
        </div>
      )}
    </div>
  );
}
```

### Real-time Recognition with WebSocket
```javascript
// components/LiveRecognition.js
import { useState, useEffect, useRef } from 'react';

export default function LiveRecognition() {
  const [ws, setWs] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [results, setResults] = useState([]);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const websocket = new WebSocket('ws://localhost:8000/api/ws/recognize');
    
    websocket.onopen = () => {
      setStatus('Connected');
      setWs(websocket);
    };
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'recognition_result') {
        setResults(prev => [data.data, ...prev.slice(0, 9)]); // Keep last 10 results
      }
    };
    
    websocket.onclose = () => {
      setStatus('Disconnected');
    };
    
    websocket.onerror = () => {
      setStatus('Error');
    };

    return () => websocket.close();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      
      // Start sending frames
      const sendFrame = () => {
        if (videoRef.current && canvasRef.current && ws) {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          ctx.drawImage(videoRef.current, 0, 0);
          
          const frameData = canvas.toDataURL('image/jpeg', 0.8);
          
          ws.send(JSON.stringify({
            type: 'frame',
            frame: frameData,
            similarity_threshold: 0.45
          }));
        }
      };
      
      setInterval(sendFrame, 1000); // Send frame every second
    } catch (error) {
      console.error('Camera error:', error);
    }
  };

  return (
    <div className="p-4">
      <h2 className="text-xl mb-4">Live Face Recognition</h2>
      <div className="mb-4">
        Status: <span className={status === 'Connected' ? 'text-green-600' : 'text-red-600'}>
          {status}
        </span>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <video ref={videoRef} autoPlay className="w-full border rounded" />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          <button 
            onClick={startCamera}
            className="mt-2 bg-blue-500 text-white px-4 py-2 rounded"
          >
            Start Camera
          </button>
        </div>
        
        <div>
          <h3 className="font-bold mb-2">Recent Results:</h3>
          <div className="h-64 overflow-y-auto space-y-2">
            {results.map((result, index) => (
              <div key={index} className="p-2 border rounded text-sm">
                {result.faces_detected > 0 ? (
                  result.faces.map((face, faceIndex) => (
                    <div key={faceIndex}>
                      {face.recognition?.recognized ? (
                        <div className="text-green-600">
                          ✅ {face.recognition.account_id} 
                          ({(face.recognition.confidence * 100).toFixed(1)}%)
                        </div>
                      ) : (
                        <div className="text-yellow-600">❓ Unknown face</div>
                      )}
                      <div className="text-xs">
                        Spoof: {face.anti_spoofing.passed ? '✅' : '❌'}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-gray-500">No faces detected</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```env
# Database Configuration
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=face_reg

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Custom Configuration
Modify `config/config.json`:
```json
{
  "detection": {
    "similarity_threshold": 0.45,
    "face_confidence_min": 0.7
  },
  "anti_spoofing": {
    "enabled": true,
    "mode": "adaptive"
  }
}
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
```

### Production Settings
```python
# For production, modify main.py:
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    log_level="warning",  # Reduce logging
    reload=False,
    workers=4  # Multiple workers for better performance
)
```

## 🔐 Security Considerations

1. **CORS Configuration**: Update allowed origins for production
2. **API Keys**: Consider adding authentication middleware
3. **Rate Limiting**: Implement rate limiting for endpoints
4. **Input Validation**: Validate image size and format
5. **HTTPS**: Use HTTPS in production

## 📊 Error Codes

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (account/face not found)
- `422`: Validation Error
- `500`: Internal Server Error

## 🔍 Testing

Run the test suite:
```bash
python test_api.py
```

The test script will validate all endpoints and provide a comprehensive report.

---

**Note**: This API is designed to work seamlessly with your existing face recognition system while providing modern REST and WebSocket interfaces for frontend integration.