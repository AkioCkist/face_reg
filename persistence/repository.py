import os
import json
import numpy as np


class FaceRepository:
    """Simple JSON-backed repository for face embeddings.

    Stores a mapping: name -> { "embeddings": [[...], [...]] }
    This class provides load/save and a helper to add embeddings with trimming.
    """

    def __init__(self, path="face_db.json"):
        self.path = path
        parent = os.path.dirname(self.path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def load(self):
        if not os.path.exists(self.path):
            return {}

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out = {}
        for name, entry in data.items():
            if isinstance(entry, dict) and "embeddings" in entry:
                out[name] = [np.array(e) for e in entry.get("embeddings", [])]
            elif isinstance(entry, dict) and "embedding" in entry:
                out[name] = [np.array(entry.get("embedding"))]
            else:
                out[name] = []

        return out

    def save(self, db_dict):
        # Expect db_dict: name -> list of numpy arrays (or lists)
        out = {}
        for name, embeddings in db_dict.items():
            out[name] = {
                "embeddings": [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]
            }

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    def add_embedding(self, name, embedding, max_embeddings=20):
        db = self.load()
        lst = db.get(name, [])
        lst.append(np.array(embedding))
        if len(lst) > max_embeddings:
            lst = lst[-max_embeddings:]
        db[name] = lst
        self.save(db)
        return db[name]
