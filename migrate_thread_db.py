#!/usr/bin/env python3

from app import app, db
from models import MeetingThreadMessage

def migrate_thread_table():
    """Create the meeting_thread_messages table"""
    with app.app_context():
        try:
            # Create the table
            db.create_all()
            print("✅ Meeting thread messages table created successfully!")
        except Exception as e:
            print(f"❌ Error creating table: {e}")

if __name__ == '__main__':
    migrate_thread_table()
