"""
WebSocket endpoints for real-time face recognition
Handles live camera feeds and streaming recognition results
"""

import json
import asyncio
import base64
import cv2
import numpy as np
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

from setup_logging import setup_logging
from face_db import get_face_embedding, check_anti_spoofing_deepface
from database import db as sql_db

# Set up logging
logger, _ = setup_logging("websocket_api")

class ConnectionManager:
    """Manages WebSocket connections for real-time face recognition"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "client_id": client_id or f"client_{len(self.active_connections)}",
            "connected_at": datetime.now().isoformat(),
            "frames_processed": 0,
            "last_recognition": None
        }
        logger.info(f"New WebSocket connection: {self.connection_info[websocket]['client_id']}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            client_info = self.connection_info.get(websocket, {})
            self.active_connections.remove(websocket)
            del self.connection_info[websocket]
            logger.info(f"WebSocket disconnected: {client_info.get('client_id', 'unknown')}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

# Global connection manager
manager = ConnectionManager()

def process_frame_for_recognition(frame_base64: str, similarity_threshold: float = 0.45):
    """Process single frame for face recognition"""
    temp_file = None
    
    try:
        # Convert base64 to image
        if frame_base64.startswith('data:image'):
            frame_base64 = frame_base64.split(',')[1]
        
        image_data = base64.b64decode(frame_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                "success": False,
                "message": "Invalid frame data"
            }
        
        # Save temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_file = f"ws_temp_{timestamp}.jpg"
        cv2.imwrite(temp_file, image)
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return {
                "success": True,
                "faces_detected": 0,
                "faces": [],
                "message": "No faces detected"
            }
        
        results = []
        
        for (x, y, w, h) in faces:
            # Anti-spoofing check
            is_live, spoof_reason, live_confidence = check_anti_spoofing_deepface(image, (x, y, w, h))
            
            face_result = {
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "anti_spoofing": {
                    "passed": bool(is_live),
                    "confidence": float(live_confidence),
                    "reason": spoof_reason
                },
                "recognition": None
            }
            
            # Only proceed with recognition if anti-spoofing passes
            if is_live:
                try:
                    # Extract embeddings
                    embeddings = get_face_embedding(temp_file, model_name="ArcFace", detector_backend="retinaface")
                    
                    if embeddings:
                        # Load database and compare
                        db_embeddings = sql_db.get_all_embeddings()
                        
                        if db_embeddings:
                            # Find best match
                            best_match = None
                            best_distance = float('inf')
                            target_array = np.array(embeddings[0])
                            
                            for account_id, db_embedding in db_embeddings.items():
                                if isinstance(db_embedding, list):
                                    db_array = np.array(db_embedding)
                                    
                                    # Calculate cosine distance
                                    cos_sim = np.dot(target_array, db_array) / (
                                        np.linalg.norm(target_array) * np.linalg.norm(db_array)
                                    )
                                    distance = 1 - cos_sim
                                    
                                    if distance < best_distance:
                                        best_distance = distance
                                        best_match = account_id
                            
                            if best_distance < similarity_threshold:
                                face_result["recognition"] = {
                                    "recognized": True,
                                    "account_id": best_match,
                                    "confidence": float(1 - best_distance),
                                    "distance": float(best_distance)
                                }
                            else:
                                face_result["recognition"] = {
                                    "recognized": False,
                                    "confidence": 0.0,
                                    "distance": float(best_distance),
                                    "message": "No match found"
                                }
                        else:
                            face_result["recognition"] = {
                                "recognized": False,
                                "message": "No faces in database"
                            }
                    else:
                        face_result["recognition"] = {
                            "recognized": False,
                            "message": "Could not extract face features"
                        }
                        
                except Exception as e:
                    logger.error(f"Recognition error: {e}")
                    face_result["recognition"] = {
                        "recognized": False,
                        "error": str(e)
                    }
            
            results.append(face_result)
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "faces": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return {
            "success": False,
            "message": f"Processing error: {str(e)}"
        }
    
    finally:
        # Clean up temp file
        if temp_file:
            try:
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

async def handle_websocket_message(websocket: WebSocket, data: dict):
    """Handle incoming WebSocket messages"""
    
    message_type = data.get("type")
    client_info = manager.connection_info.get(websocket, {})
    
    if message_type == "ping":
        # Handle ping/pong for keepalive
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
    elif message_type == "frame":
        # Process frame for recognition
        frame_data = data.get("frame")
        similarity_threshold = data.get("similarity_threshold", 0.45)
        
        if not frame_data:
            await manager.send_personal_message({
                "type": "error",
                "message": "No frame data provided"
            }, websocket)
            return
        
        # Process frame
        result = process_frame_for_recognition(frame_data, similarity_threshold)
        
        # Update connection stats
        client_info["frames_processed"] += 1
        if result.get("faces"):
            for face in result["faces"]:
                if face.get("recognition", {}).get("recognized"):
                    client_info["last_recognition"] = {
                        "account_id": face["recognition"]["account_id"],
                        "confidence": face["recognition"]["confidence"],
                        "timestamp": datetime.now().isoformat()
                    }
        
        # Send result back
        await manager.send_personal_message({
            "type": "recognition_result",
            "data": result
        }, websocket)
        
    elif message_type == "config":
        # Update client configuration
        config_data = data.get("config", {})
        client_info.update(config_data)
        
        await manager.send_personal_message({
            "type": "config_updated",
            "message": "Configuration updated successfully"
        }, websocket)
        
    elif message_type == "status":
        # Send connection status
        await manager.send_personal_message({
            "type": "status",
            "data": {
                "client_id": client_info.get("client_id"),
                "connected_at": client_info.get("connected_at"),
                "frames_processed": client_info.get("frames_processed", 0),
                "last_recognition": client_info.get("last_recognition"),
                "active_connections": len(manager.active_connections)
            }
        }, websocket)
        
    else:
        # Unknown message type
        await manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }, websocket)

# WebSocket endpoint will be added to main FastAPI app
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """Main WebSocket endpoint for real-time face recognition"""
    
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "message": "Connected to Face Recognition WebSocket",
            "client_id": manager.connection_info[websocket]["client_id"]
        }, websocket)
        
        # Message handling loop
        while True:
            try:
                # Receive message
                raw_message = await websocket.receive_text()
                message_data = json.loads(raw_message)
                
                # Handle message
                await handle_websocket_message(websocket, message_data)
                
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except json.JSONDecodeError as e:
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Invalid JSON format: {str(e)}"
                }, websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                }, websocket)
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}")
    finally:
        manager.disconnect(websocket)

# Export manager and endpoint function for main app
__all__ = ["manager", "websocket_endpoint"]