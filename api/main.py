"""
Face Recognition REST API
Provides endpoints for registering faces, recognizing faces, and managing the face database.
Compatible with Next.js frontend integration.
"""

import os
import sys
import json
import base64
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import custom modules
try:
    from setup_logging import setup_logging
    from face_db import get_face_embedding, check_anti_spoofing_deepface
    from persistence.repository import FaceRepository
    from database import db as sql_db
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Parent directory: {parent_dir}")
    print("Available files in parent directory:")
    try:
        for file in os.listdir(parent_dir):
            print(f"  {file}")
    except:
        pass
    raise

# Set up logging
logger, log_file_path = setup_logging("face_api", logging.INFO)
logger.info(f"Face Recognition API started. Log file: {log_file_path}")

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="REST API for face registration, recognition, and database management",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize database and repository
repo = FaceRepository("face_db.json")

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config/config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config.json: {e}")
        return {
            "detection": {
                "backends": ["retinaface", "mediapipe", "mtcnn", "opencv"],
                "similarity_threshold": 0.45,
                "face_confidence_min": 0.7
            },
            "anti_spoofing": {"enabled": True}
        }

# Pydantic models for API requests/responses
class RegisterFaceRequest(BaseModel):
    account_id: str = Field(..., description="Unique identifier for the person")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    use_webcam: Optional[bool] = Field(False, description="Whether to use webcam for capture")
    override: Optional[bool] = Field(False, description="Override existing face data")

class RegisterFaceResponse(BaseModel):
    success: bool
    message: str
    account_id: str
    embeddings_count: Optional[int] = None
    anti_spoofing_passed: Optional[bool] = None
    confidence: Optional[float] = None

class RecognizeFaceRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data")
    similarity_threshold: Optional[float] = Field(0.45, description="Recognition threshold")

class RecognizeFaceResponse(BaseModel):
    success: bool
    recognized: bool
    account_id: Optional[str] = None
    confidence: Optional[float] = None
    distance: Optional[float] = None
    anti_spoofing_passed: bool
    anti_spoofing_confidence: Optional[float] = None
    message: str

class FaceInfo(BaseModel):
    account_id: str
    embeddings_count: int
    registered_date: Optional[str] = None

class ListFacesResponse(BaseModel):
    success: bool
    faces: List[FaceInfo]
    total_count: int

class DeleteFaceResponse(BaseModel):
    success: bool
    message: str
    account_id: str

# Utility functions
def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image data")
            
        return image
        
    except Exception as e:
        logger.error(f"Failed to convert base64 to image: {e}")
        raise ValueError(f"Invalid image data: {e}")

def image_to_temp_file(image: np.ndarray, prefix: str = "api_temp") -> str:
    """Save OpenCV image to temporary file and return path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = f"{prefix}_{timestamp}.jpg"
    
    success = cv2.imwrite(temp_path, image)
    if not success:
        raise ValueError("Failed to save temporary image file")
    
    return temp_path

def cleanup_temp_file(file_path: str):
    """Safely remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

def compare_embeddings(target_embedding: List[float], db_embeddings: Dict[str, List[List[float]]], threshold: float = 0.45):
    """Compare target embedding against database embeddings"""
    best_match = None
    best_distance = float('inf')
    
    target_array = np.array(target_embedding)
    
    for account_id, embeddings_list in db_embeddings.items():
        for embedding in embeddings_list:
            db_array = np.array(embedding)
            
            # Calculate cosine distance
            cos_sim = np.dot(target_array, db_array) / (
                np.linalg.norm(target_array) * np.linalg.norm(db_array)
            )
            distance = 1 - cos_sim
            
            if distance < best_distance:
                best_distance = distance
                best_match = account_id
    
    if best_distance < threshold:
        confidence = 1 - best_distance
        return best_match, confidence, best_distance
    else:
        return None, 0.0, best_distance

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Face Recognition API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/config")
async def get_config():
    """Get current API configuration"""
    config = load_config()
    return {
        "success": True,
        "config": {
            "detection": config.get("detection", {}),
            "anti_spoofing_enabled": config.get("anti_spoofing", {}).get("enabled", True)
        }
    }

@app.post("/api/faces/register", response_model=RegisterFaceResponse)
async def register_face(request: RegisterFaceRequest):
    """Register a new face in the database"""
    temp_file = None
    
    try:
        logger.info(f"Face registration request for account_id: {request.account_id}")
        
        # Validate account_id
        if not request.account_id.strip():
            raise HTTPException(status_code=400, detail="Account ID cannot be empty")
        
        account_id = request.account_id.strip()
        
        # Check if account already exists
        try:
            existing_faces = sql_db.get_all_embeddings()
            if account_id in existing_faces and not request.override:
                return RegisterFaceResponse(
                    success=False,
                    message=f"Account ID '{account_id}' already exists. Set override=True to replace.",
                    account_id=account_id
                )
        except Exception as e:
            logger.warning(f"Could not check existing faces: {e}")
        
        # Handle webcam capture (placeholder - would need frontend integration)
        if request.use_webcam:
            return RegisterFaceResponse(
                success=False,
                message="Webcam capture not supported via API. Please use image_base64 parameter.",
                account_id=account_id
            )
        
        # Validate image data
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="Either image_base64 or use_webcam must be provided")
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        temp_file = image_to_temp_file(image, "register")
        
        # Extract face embeddings with anti-spoofing
        logger.info(f"Extracting face embeddings for {account_id}")
        embeddings = get_face_embedding(temp_file, model_name="ArcFace", detector_backend="retinaface")
        
        if not embeddings:
            return RegisterFaceResponse(
                success=False,
                message="No valid face detected or anti-spoofing failed. Please provide a clear, live face image.",
                account_id=account_id,
                anti_spoofing_passed=False
            )
        
        # Save to database
        try:
            sql_db.ensure_table_exists()
            sql_db.insert_embedding(account_id, embeddings[0])
            
            logger.info(f"Successfully registered face for {account_id}")
            
            return RegisterFaceResponse(
                success=True,
                message=f"Face successfully registered for account {account_id}",
                account_id=account_id,
                embeddings_count=len(embeddings),
                anti_spoofing_passed=True,
                confidence=1.0
            )
            
        except Exception as e:
            logger.error(f"Failed to save embedding for {account_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face registration error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)

@app.post("/api/faces/recognize", response_model=RecognizeFaceResponse)
async def recognize_face(request: RecognizeFaceRequest):
    """Recognize a face from the provided image"""
    temp_file = None
    
    try:
        logger.info("Face recognition request received")
        
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        temp_file = image_to_temp_file(image, "recognize")
        
        # Perform anti-spoofing check first
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        anti_spoofing_passed = False
        anti_spoofing_confidence = 0.0
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            is_live, spoof_reason, live_confidence = check_anti_spoofing_deepface(image, (x, y, w, h))
            anti_spoofing_passed = is_live
            anti_spoofing_confidence = live_confidence
        
        if not anti_spoofing_passed:
            return RecognizeFaceResponse(
                success=True,
                recognized=False,
                anti_spoofing_passed=False,
                anti_spoofing_confidence=anti_spoofing_confidence,
                message="Face detection failed anti-spoofing verification"
            )
        
        # Extract embeddings
        embeddings = get_face_embedding(temp_file, model_name="ArcFace", detector_backend="retinaface")
        
        if not embeddings:
            return RecognizeFaceResponse(
                success=True,
                recognized=False,
                anti_spoofing_passed=anti_spoofing_passed,
                anti_spoofing_confidence=anti_spoofing_confidence,
                message="No face detected in the image"
            )
        
        # Load database and compare
        try:
            db_embeddings = sql_db.get_all_embeddings()
            
            if not db_embeddings:
                return RecognizeFaceResponse(
                    success=True,
                    recognized=False,
                    anti_spoofing_passed=anti_spoofing_passed,
                    anti_spoofing_confidence=anti_spoofing_confidence,
                    message="No faces registered in the database"
                )
            
            # Convert database format for comparison
            formatted_db = {}
            for acc_id, emb in db_embeddings.items():
                if isinstance(emb, list) and emb:
                    formatted_db[acc_id] = [emb] if not isinstance(emb[0], list) else emb
            
            # Compare embeddings
            match_id, confidence, distance = compare_embeddings(
                embeddings[0], 
                formatted_db, 
                request.similarity_threshold
            )
            
            if match_id:
                logger.info(f"Face recognized as {match_id} with confidence {confidence:.3f}")
                return RecognizeFaceResponse(
                    success=True,
                    recognized=True,
                    account_id=match_id,
                    confidence=confidence,
                    distance=distance,
                    anti_spoofing_passed=anti_spoofing_passed,
                    anti_spoofing_confidence=anti_spoofing_confidence,
                    message=f"Face recognized as {match_id}"
                )
            else:
                return RecognizeFaceResponse(
                    success=True,
                    recognized=False,
                    distance=distance,
                    anti_spoofing_passed=anti_spoofing_passed,
                    anti_spoofing_confidence=anti_spoofing_confidence,
                    message="Face not recognized - no matching identity found"
                )
                
        except Exception as e:
            logger.error(f"Database error during recognition: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face recognition error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)

@app.get("/api/faces", response_model=ListFacesResponse)
async def list_faces():
    """List all registered faces"""
    try:
        logger.info("Listing all registered faces")
        
        # Get faces from database
        db_embeddings = sql_db.get_all_embeddings()
        
        faces = []
        for account_id, embedding_data in db_embeddings.items():
            if isinstance(embedding_data, list):
                embeddings_count = len(embedding_data) if isinstance(embedding_data[0], list) else 1
            else:
                embeddings_count = 1
            
            faces.append(FaceInfo(
                account_id=account_id,
                embeddings_count=embeddings_count,
                registered_date=None  # Could add timestamp to database schema
            ))
        
        return ListFacesResponse(
            success=True,
            faces=faces,
            total_count=len(faces)
        )
        
    except Exception as e:
        logger.error(f"Error listing faces: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.delete("/api/faces/{account_id}", response_model=DeleteFaceResponse)
async def delete_face(account_id: str):
    """Delete a registered face"""
    try:
        logger.info(f"Deleting face for account_id: {account_id}")
        
        # Check if account exists
        existing_faces = sql_db.get_all_embeddings()
        if account_id not in existing_faces:
            raise HTTPException(status_code=404, detail=f"Account ID '{account_id}' not found")
        
        # Delete from database
        sql_db.delete_embedding(account_id)
        
        logger.info(f"Successfully deleted face for {account_id}")
        
        return DeleteFaceResponse(
            success=True,
            message=f"Face data for account '{account_id}' has been deleted",
            account_id=account_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc)
        }
    )

# WebSocket endpoint for real-time recognition
@app.websocket("/api/ws/recognize/{client_id}")
async def websocket_recognition_endpoint(websocket: WebSocket, client_id: str = "default"):
    """WebSocket endpoint for real-time face recognition"""
    try:
        from websocket_handler import websocket_endpoint
        await websocket_endpoint(websocket, client_id)
    except ImportError:
        logger.error("WebSocket handler not available")
        await websocket.close()

@app.websocket("/api/ws/recognize")
async def websocket_recognition_endpoint_no_id(websocket: WebSocket):
    """WebSocket endpoint for real-time face recognition (no client ID)"""
    try:
        from websocket_handler import websocket_endpoint
        await websocket_endpoint(websocket)
    except ImportError:
        logger.error("WebSocket handler not available")
        await websocket.close()

if __name__ == "__main__":
    # Initialize database
    try:
        sql_db.ensure_table_exists()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Start the server
    logger.info("Starting Face Recognition API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )