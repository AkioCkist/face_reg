#!/usr/bin/env python3
"""Batch register images in a folder into the face database.

Usage:
  python scripts/batch_register.py --dir ./images --model ArcFace --backend retinaface

For each image file in the directory, the script will extract a face embedding using
the project's `get_face_embedding` function and store the first embedding found using
the database layer `database.db.insert_embedding`. If the DB layer isn't available,
it will fall back to the JSON repository `persistence.repository.FaceRepository`.

The account id used for each embedding is the filename without extension.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Make project root importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from face_db import get_face_embedding
from setup_logging import setup_logging

logger, _ = setup_logging("batch_register", logging.INFO)


def find_image_files(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    p = argparse.ArgumentParser(description="Batch register images into face DB")
    p.add_argument("--dir", required=True, help="Directory with images to register")
    p.add_argument("--model", default="ArcFace", help="DeepFace model name to use")
    p.add_argument("--backend", default="retinaface", help="detector backend for DeepFace")
    p.add_argument("--override", action="store_true", help="Override existing DB entries")
    args = p.parse_args()

    folder = Path(args.dir)
    if not folder.exists() or not folder.is_dir():
        logger.error(f"Provided folder does not exist: {folder}")
        sys.exit(1)

    # Try to import DB layer; fall back to repo
    use_sql = False
    try:
        from database import db as sql_db
        use_sql = True
    except Exception as e:
        logger.info(f"SQL DB layer not available, will use JSON repo: {e}")
        sql_db = None

    from persistence.repository import FaceRepository
    repo = FaceRepository("face_db.json")

    files = list(find_image_files(folder))
    logger.info(f"Found {len(files)} images in {folder}")

    for img_path in files:
        account_id = img_path.stem
        logger.info(f"Processing {img_path.name} -> id={account_id}")

        # If not overriding, check existence
        if not args.override:
            try:
                if use_sql:
                    existing = sql_db.get_embedding(account_id)
                    if existing:
                        logger.info(f"Skipping {account_id}: already in SQL DB (use --override to replace)")
                        continue
                else:
                    repo_db = repo.load()
                    if account_id in repo_db:
                        logger.info(f"Skipping {account_id}: already in JSON repo (use --override to replace)")
                        continue
            except Exception as e:
                logger.warning(f"Existence check failed for {account_id}: {e}")

        embeddings = get_face_embedding(str(img_path), model_name=args.model, detector_backend=args.backend, augment=True)
        if not embeddings:
            logger.warning(f"No embeddings extracted for {img_path.name}")
            continue

        emb = embeddings[0]

        if use_sql:
            try:
                sql_db.ensure_table_exists()
                
                # Ensure the account exists before inserting embedding
                account_created = sql_db.ensure_account_exists(account_id, name=account_id, role="user")
                if account_created:
                    logger.info(f"Created new account for {account_id}")
                
                sql_db.insert_embedding(account_id, emb)
                logger.info(f"Inserted embedding for {account_id} into SQL DB")
            except Exception as e:
                logger.error(f"Failed to insert into SQL DB for {account_id}: {e}")
                # Try repo fallback
                try:
                    repo.add_embedding(account_id, emb.tolist() if hasattr(emb, "tolist") else emb)
                    logger.info(f"Inserted embedding for {account_id} into JSON repo (fallback)")
                except Exception as e2:
                    logger.error(f"Fallback repo insert failed for {account_id}: {e2}")
        else:
            try:
                repo.add_embedding(account_id, emb.tolist() if hasattr(emb, "tolist") else emb)
                logger.info(f"Inserted embedding for {account_id} into JSON repo")
            except Exception as e:
                logger.error(f"Failed to insert into JSON repo for {account_id}: {e}")


if __name__ == "__main__":
    main()
