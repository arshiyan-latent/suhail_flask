#!/usr/bin/env python3
"""
Database migration script to add meeting_summary column to transcript table
"""

import sqlite3
import sys
import os

def migrate_database():
    """Add meeting_summary column to transcript table if it doesn't exist"""
    db_path = 'instance/users.db'
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(transcript)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'meeting_summary' not in columns:
            print("Adding meeting_summary column to transcript table...")
            cursor.execute("ALTER TABLE transcript ADD COLUMN meeting_summary TEXT")
            conn.commit()
            print("Successfully added meeting_summary column")
        else:
            print("meeting_summary column already exists")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if migrate_database():
        print("Migration completed successfully")
        sys.exit(0)
    else:
        print("Migration failed")
        sys.exit(1)
