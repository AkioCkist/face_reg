from deepface import DeepFace
import json

embedding = DeepFace.represent(img_path="person1.jpg", model_name="ArcFace")[0]["embedding"]

db = {"Akio": {"embedding": embedding}}
with open("face_db.json", "w") as f:
    json.dump(db, f)
