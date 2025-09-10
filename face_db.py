import os
import json
import numpy as np
from deepface import DeepFace
import cv2
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config.json: {e}. Using default settings.")
        return {
            "detection": {
                "backends": ["retinaface", "mediapipe", "mtcnn", "opencv"],
                "similarity_threshold": 0.45,
                "face_confidence_min": 0.7
            }
        }

def get_face_embedding(img_path, model_name="ArcFace", detector_backend="retinaface", augment=False):
    """
    Get face embeddings with optional data augmentation
    
    Args:
        img_path: Path to image file
        model_name: Face recognition model name
        detector_backend: Face detection backend
        augment: Whether to use data augmentation
        
    Returns:
        List of embeddings
    """
    embeddings = []
    
    # Try different detector backends if the first one fails
    config = load_config()
    backends = config["detection"]["backends"]
    
    # First try with the preferred backend
    try:
        reps = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True,
            align=True
        )
        if reps:
            embeddings.extend([rep["embedding"] for rep in reps])
    except Exception as e:
        logger.warning(f"Failed with {detector_backend} detector: {e}")
        
        # Try with fallback detectors
        for backend in backends:
            if backend == detector_backend:
                continue
            
            try:
                logger.info(f"Trying with {backend} detector")
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True
                )
                if reps:
                    embeddings.extend([rep["embedding"] for rep in reps])
                    break
            except Exception as e2:
                logger.warning(f"Failed with {backend} detector: {e2}")
    
    # Perform data augmentation if requested
    if augment and embeddings:
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            return embeddings
            
        # Create augmented versions (slight rotations and shifts)
        augmented_images = []
        
        # Slight rotations
        for angle in [-5, 5]:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
            augmented_images.append(rotated)
        
        # Slight brightness variations
        for factor in [0.9, 1.1]:
            brightened = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            augmented_images.append(brightened)
            
        # Get embeddings for augmented images
        for i, aug_img in enumerate(augmented_images):
            # Save temporary file
            temp_path = f"temp_aug_{i}.jpg"
            cv2.imwrite(temp_path, aug_img)
            
            try:
                aug_reps = DeepFace.represent(
                    img_path=temp_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                    align=True
                )
                if aug_reps:
                    embeddings.extend([rep["embedding"] for rep in aug_reps])
            except Exception as e:
                logger.warning(f"Failed to get embedding for augmented image {i}: {e}")
            
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return embeddings

def create_face_database(person_image_map, output_file="face_db.json", model_name="ArcFace", 
                         detector_backend="retinaface", use_augmentation=True):
    """
    Create a face database from a dictionary mapping person names to image paths
    
    Args:
        person_image_map: Dictionary mapping person names to image paths (str or list of str)
        output_file: Path to output JSON file
        model_name: Face recognition model name
        detector_backend: Face detection backend
        use_augmentation: Whether to use data augmentation
    """
    db = {}
    
    logger.info(f"Creating face database using {model_name} model and {detector_backend} detector")
    logger.info(f"Data augmentation: {'enabled' if use_augmentation else 'disabled'}")
    
    for person_name, image_paths in tqdm(person_image_map.items(), desc="Processing people"):
        # Convert single image path to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        person_embeddings = []
        
        for img_path in image_paths:
            logger.info(f"Processing {img_path} for {person_name}")
            embeddings = get_face_embedding(img_path, model_name, detector_backend, use_augmentation)
            
            if embeddings:
                person_embeddings.extend(embeddings)
                logger.info(f"Got {len(embeddings)} embeddings from {img_path}")
            else:
                logger.warning(f"No faces found in {img_path}")
        
        if person_embeddings:
            db[person_name] = {"embeddings": person_embeddings}
            logger.info(f"Added {len(person_embeddings)} embeddings for {person_name}")
        else:
            logger.error(f"Could not find any faces for {person_name}. Skipping.")
    
    # Save the database
    with open(output_file, "w") as f:
        json.dump(db, f)
    
    logger.info(f"Face database saved to {output_file} with {len(db)} people")
    return db

if __name__ == "__main__":
    # Detect available person*.jpg files in the current directory
    person_files = [f for f in os.listdir(".") if f.startswith("person") and f.endswith((".jpg", ".jpeg", ".png"))]
    
    if not person_files:
        logger.error("No person*.jpg files found in the current directory!")
        exit(1)
    
    # Create a mapping of person names to image files
    # You can customize this mapping if you have specific names for each person
    person_map = {}
    
    # Option 1: Use numeric IDs from filenames (person1.jpg → Person 1)
    for file in person_files:
        person_id = file.split(".")[0]  # Remove extension
        name = f"Person {person_id.replace('person', '')}"
        person_map[name] = file
    
    # Option 2: Prompt for names
    use_custom_names = input("Do you want to provide custom names for each person? (y/n): ").lower() == 'y'
    
    if use_custom_names:
        person_map = {}
        for file in person_files:
            name = input(f"Enter name for {file}: ")
            if name:
                person_map[name] = file
            else:
                person_id = file.split(".")[0]
                person_map[f"Person {person_id.replace('person', '')}"] = file
    
    # Create the face database
    model_name = "ArcFace"  # Best accuracy
    detector_backend = "retinaface"  # Best detection accuracy
    use_augmentation = True  # Enable data augmentation for better accuracy
    
    create_face_database(
        person_map,
        output_file="face_db.json",
        model_name=model_name,
        detector_backend=detector_backend,
        use_augmentation=use_augmentation
    )
