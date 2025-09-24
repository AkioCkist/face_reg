#!/usr/bin/env python3
"""Check what accounts exist in the database."""

import sys
from pathlib import Path

# Make project root importable
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from database.db import engine
from sqlalchemy import text

def check_accounts():
    """Check existing accounts in the database"""
    print("=== Checking Existing Accounts ===")
    
    try:
        with engine.connect() as conn:
            # Count accounts
            count_query = text("SELECT COUNT(*) FROM account;")
            result = conn.execute(count_query)
            count = result.fetchone()[0]
            print(f"\nTotal accounts in database: {count}")
            
            if count > 0:
                # Show first 10 accounts
                accounts_query = text("SELECT id, name, role FROM account ORDER BY id LIMIT 10;")
                result = conn.execute(accounts_query)
                print("\nFirst 10 accounts:")
                for row in result:
                    print(f"  - {row[0]}: {row[1]} ({row[2]})")
                
                if count > 10:
                    print(f"  ... and {count - 10} more")
            else:
                print("No accounts found in database!")
                
    except Exception as e:
        print(f"Error checking accounts: {e}")

if __name__ == "__main__":
    check_accounts()