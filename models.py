# models.py

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import uuid




# ✅ This line ensures db can be imported in app.py
db = SQLAlchemy()

# ✅ User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(1000), nullable=False)
    role = db.Column(db.String(20), nullable=False) 
    manager_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    # Relationships
    # Self-referential relationship for manager-agent hierarchy
    manager = db.relationship('User', remote_side=[id], backref='managed_users')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def manager_name(self):
        """Get the manager's username if exists"""
        return self.manager.username if self.manager else None

    def get_managed_agents(self):
        """Get all agents managed by this user (for managers)"""
        return User.query.filter_by(manager_id=self.id, role='salesagent').all()

    def get_managed_users(self):
        """Get all users managed by this user (agents and managers)"""
        return User.query.filter_by(manager_id=self.id).all()

    @staticmethod
    def get_managers():
        """Get all users with manager or smeleader role"""
        return User.query.filter(User.role.in_(['manager', 'smeleader'])).all()


class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(100), default='Untitled Chat')
    created_at = db.Column(db.DateTime, default=datetime.now())
    client_name = db.Column(db.String(100), nullable=True)  # Simple client name for now, can be extended later into a full Client model if needed
    
class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('chat_sessions.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(10), nullable=False)  
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship('ChatSession', backref=db.backref('messages', lazy=True))

class ClientSummary(db.Model):
    __tablename__ = 'client_summaries'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    client_name = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    message_count = db.Column(db.Integer, default=0)  # Track number of messages since last summary
    
    user = db.relationship('User', backref=db.backref('client_summaries', lazy=True))

class TeamNotification(db.Model):
    __tablename__ = 'team_notifications'
    id = db.Column(db.Integer, primary_key=True)
    manager_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    priority = db.Column(db.String(50), default='General Notes')  # 'Internal Announcement', 'External Broadcast For Clients', 'General Notes'

    # Manager relationship
    manager = db.relationship('User', backref=db.backref('notifications_sent', lazy=True))

class NotificationRead(db.Model):
    __tablename__ = 'notification_reads'
    id = db.Column(db.Integer, primary_key=True)
    notification_id = db.Column(db.Integer, db.ForeignKey('team_notifications.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    read_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    notification = db.relationship('TeamNotification')
    user = db.relationship('User', backref=db.backref('notification_reads', lazy=True))

class Transcript(db.Model):
    __tablename__ = 'transcript'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chat_id = db.Column(db.String, db.ForeignKey('chat_sessions.id'), nullable=True)
    title = db.Column(db.String(255), nullable=False, default='Untitled Meeting')
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now())

    # NEW (optional)
    file_path = db.Column(db.String, nullable=True)
    speakers_count = db.Column(db.Integer, nullable=True)
    language = db.Column(db.String(16), nullable=True)

    user = db.relationship('User', backref='transcripts')
    chat = db.relationship('ChatSession', backref='transcripts')