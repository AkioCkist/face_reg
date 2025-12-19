#!/usr/bin/env python3
"""Debug script to check what's in the database"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import db
import json

print("=" * 60)
print("Checking all embeddings in database...")
print("=" * 60)

# Get raw embeddings
embeddings = db.get_all_embeddings()

print(f"\nTotal accounts: {len(embeddings)}")

for account_id, embedding_data in embeddings.items():
    print(f"\nAccount ID: {account_id}")
    print(f"  Type: {type(embedding_data)}")
    print(f"  Value: {embedding_data}")
    
    if embedding_data is None:
        print(f"  ⚠️  EMBEDDING IS NULL!")
    elif isinstance(embedding_data, list):
        print(f"  Length: {len(embedding_data)}")
        if embedding_data:
            print(f"  First element type: {type(embedding_data[0])}")
            print(f"  First 3 elements: {embedding_data[:3]}")
    else:
        print(f"  ⚠️  UNEXPECTED TYPE!")

# Also try to get the raw database result
print("\n" + "=" * 60)
print("Raw database query result:")
print("=" * 60)

from sqlalchemy import select
stmt = select(db.face_embeddings.c.id, db.face_embeddings.c.embedding).order_by(db.face_embeddings.c.id)
with db.engine.connect() as conn:
    res = conn.execute(stmt)
    for row in res:
        account_id, embedding_raw = row
        print(f"\nAccount ID: {account_id}")
        print(f"  Raw type: {type(embedding_raw)}")
        print(f"  Raw value (first 100 chars): {str(embedding_raw)[:100]}")
        parsed = db._parse_vector_result(embedding_raw)
        print(f"  Parsed type: {type(parsed)}")
        print(f"  Parsed value: {parsed}")
