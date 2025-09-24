#!/usr/bin/env python3
"""Check the actual database schema to understand foreign key constraints."""

import sys
from pathlib import Path

# Make project root importable
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from database.db import engine
from sqlalchemy import text

def check_schema():
    """Check the actual database schema"""
    print("=== Checking Database Schema ===")
    
    # Check what tables exist
    print("\n1. Existing tables:")
    query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            tables = [row[0] for row in result]
            for table in tables:
                print(f"  - {table}")
            
            # Check if account table exists
            if 'account' in tables:
                print("\n2. Account table structure:")
                account_query = text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'account'
                    ORDER BY ordinal_position;
                """)
                result = conn.execute(account_query)
                for row in result:
                    print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
            else:
                print("\n2. Account table does not exist!")
            
            # Check face_embeddings table structure
            print("\n3. Face_embeddings table structure:")
            fe_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'face_embeddings'
                ORDER BY ordinal_position;
            """)
            result = conn.execute(fe_query)
            for row in result:
                print(f"  - {row[0]}: {row[1]} ({'NULL' if row[2] == 'YES' else 'NOT NULL'})")
            
            # Check foreign key constraints
            print("\n4. Foreign key constraints:")
            fk_query = text("""
                SELECT
                    tc.constraint_name,
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                        AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_name = 'face_embeddings';
            """)
            result = conn.execute(fk_query)
            constraints = list(result)
            if constraints:
                for row in constraints:
                    print(f"  - {row[0]}: {row[1]}.{row[2]} -> {row[3]}.{row[4]}")
            else:
                print("  - No foreign key constraints found")
                
    except Exception as e:
        print(f"Error checking schema: {e}")

if __name__ == "__main__":
    check_schema()