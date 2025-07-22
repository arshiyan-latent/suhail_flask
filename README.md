# suhail_flask

Must Include your OPENAI_API_KEY and FLASK_SECRET_KEY

Curreny summary of application:

1. Core Application Files
- app.py - Main Flask Application

Purpose: Main entry point and web server configuration
Key Features:
User authentication and authorization system
Role-based access control (admin, manager, regular users)
Chat interface for AI interactions
User management for admins
API endpoints for chat functionality
Key Routes:

/ - Home/chat interface (login required)
/login & /register - Authentication
/admin & /admin/users - Admin-only user management
/v1/chat/salesrep - AI chat endpoint
/v1/chat/newchat - Create new chat sessions

models.py - Database Models

Purpose: Defines the database structure using SQLAlchemy
Models:
User: Stores user credentials, roles, and authentication
ChatSession: Tracks individual chat conversations


2. AI Agent System (agents folder)
agent.py - Main AI Supervisor

Purpose: Core AI assistant named "Suhail" for health insurance sales guidance
Capabilities:
Pre-sales meeting preparation
In-meeting live guidance
Post-sales recommendations
Multilingual support (English/Arabic)
Uses OpenAI GPT-4 for conversations
agent.py - Insurance Package Tool

Purpose: Specialized agent for insurance package information
Features:
Provides details about insurance packages (Basic, Bronze, Silver, Gold, Platinum, Diamond)
Coverage limits, co-payments, maternity benefits
Network classes and eligibility criteria

3. Frontend Templates (templates folder)
chat.html - Main Chat Interface

Modern, responsive chat UI with Tailwind CSS
Sidebar with recent chats and role-based navigation
Real-time messaging interface
Role-specific dashboards (admin/manager access)
login.html - Authentication Page

Styled login form with animations
Secure user authentication
register.html - User Registration

New user signup with role assignment
admin_users.html - Admin Dashboard

User management interface for administrators
Add, edit, delete users
Role management


🗄️ Database Structure
The application uses two SQLite databases:

1. users.db - User & Chat Data

users.db - User & Chat Data

2. suhail_database.db - AI Conversation Memory