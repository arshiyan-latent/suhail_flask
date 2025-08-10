from typing import List, Dict
from datetime import datetime
from models import db, TeamNotification, NotificationRead

def get_unread_notifications(user_id: int = None) -> List[Dict]:
    """
    Get all unread team notifications for a specific user.
    If user_id is None, returns all unread notifications.
    """
    try:
        print(f"Getting unread notifications for user {user_id}")
        
        # Start with active notifications
        query = db.session.query(TeamNotification).filter(
            TeamNotification.is_active == True
        )
        print(f"Found {query.count()} total active notifications")
        
        # If user_id provided, exclude notifications that user has read
        if user_id is not None:
            read_notifications = db.session.query(NotificationRead.notification_id).filter(
                NotificationRead.user_id == user_id
            )
            print(f"User has read {read_notifications.count()} notifications")
            query = query.filter(~TeamNotification.id.in_(read_notifications))
        
        # Order by priority and timestamp
        notifications = query.order_by(
            # Priority ordering: Internal -> External -> General
            db.case(
                {'Internal Announcement': 1, 'External Broadcast For Clients': 2, 'General Notes': 3},
                value=TeamNotification.priority
            ),
            TeamNotification.timestamp.desc()
        ).all()
        
        # Convert to dictionary format
        return [{
            'id': n.id,
            'message': n.message,
            'priority': n.priority,
            'timestamp': n.timestamp.isoformat() if isinstance(n.timestamp, datetime) else n.timestamp
        } for n in notifications]
        
    except Exception as e:
        print(f"Error getting notifications: {e}")
        return []

def format_notifications_for_prompt(user_id: int = None) -> str:
    """Format unread notifications for inclusion in agent prompt"""
    notifications = get_unread_notifications(user_id)
    if not notifications:
        print("No unread notifications found")
        return "No unread team notifications at the moment. How can I assist you further today?"
        
    print(f"Formatting {len(notifications)} notifications")
    notification_text = "\n🔔 **Unread Team Messages:**\n\n"
    for n in notifications:
        priority_icon = "🔴" if n['priority'] == "Internal Announcement" else "🟡" if n['priority'] == "External Broadcast For Clients" else "🟢"
        notification_text += f"{priority_icon} {n['message']}\n"
    print(f"Final formatted text: {notification_text}")
    return notification_text
