# app.py
from webbrowser import get
from agents.agent import llm, get_supervisor_for_user
from agents.summary.summary_agent import extract_transcript, generate_summary
from agents.manager_agent import create_manager_agent
from agents.general_agent import supervisor_agent_general
from dotenv import load_dotenv
import os, tempfile, time, json, subprocess
# import logging
from langchain_openai import ChatOpenAI
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import send_file
from openai import OpenAI
client = OpenAI()

# Set logging level to suppress langgraph debug messages
# logging.getLogger('langgraph').setLevel(logging.ERROR)


load_dotenv()

import io
from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, ChatSession, ChatMessage, ClientSummary, TeamNotification, NotificationRead, Transcript
from datetime import datetime
import uuid
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas  # kept (fallback)
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
except Exception:
    A4 = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

def _wrap_lines(line, max_width, canv, font_name="Helvetica", font_size=10):
    """Yield wrapped substrings that fit max_width."""
    words = line.split(" ")
    out, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(test, font_name, font_size) <= max_width:
            cur = test
        else:
            if cur:
                out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out

def write_transcript_pdf(pdf_path: str, title: str, pretty_text: str, meta: dict | None = None):
    c = canvas.Canvas(pdf_path, pagesize=LETTER)
    width, height = LETTER
    lm = rm = 0.75 * inch
    tm = bm = 0.75 * inch
    y = height - tm

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(lm, y, title)
    y -= 0.28 * inch

    # Meta (optional)
    c.setFont("Helvetica", 9)
    if meta:
        for k, v in meta.items():
            c.drawString(lm, y, f"{k}: {v}")
            y -= 12
        y -= 6

    # Body
    c.setFont("Helvetica", 10)
    maxw = width - lm - rm
    line_h = 14
    for line in pretty_text.splitlines():
        chunks = _wrap_lines(line, maxw, c)
        for chunk in chunks:
            if y <= bm:
                c.showPage()
                y = height - tm
                c.setFont("Helvetica", 10)
            c.drawString(lm, y, chunk)
            y -= line_h
        # paragraph spacing
        if not chunks:
            y -= line_h
    c.save()


# Initialize manager_agent as None, will be created when needed
manager_agent = None



def get_manager_agent():
    global manager_agent
    if manager_agent is None:
        # Create manager agent within app context
        with app.app_context():
            manager_agent = create_manager_agent(llm=llm).compile()
    return manager_agent

# --- helper: optional diarization via pyannote (set HUGGINGFACE_TOKEN) ---
_DIAR_PIPELINE = None  # cache

def _to_wav_mono16k(src_path: str) -> str:
    """Return a temp WAV (mono, 16 kHz) converted from any input using ffmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ac", "1",        # mono
        "-ar", "16000",    # 16 kHz
        "-f", "wav", tmp_wav.name
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return tmp_wav.name
    except Exception as e:
        # If ffmpeg missing or conversion failed
        try:
            os.unlink(tmp_wav.name)
        except Exception:
            pass
        print("[diarize] ffmpeg conversion failed:", repr(e))
        return None

def diarize_file(audio_path: str):
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("[diarize] HUGGINGFACE_TOKEN not set; skipping diarization.")
        return None

    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        print("[diarize] pyannote import failed:", repr(e))
        return None

    # Convert to WAV mono/16k so libsndfile can read it
    wav_path = _to_wav_mono16k(audio_path)
    if not wav_path:
        return None

    global _DIAR_PIPELINE
    try:
        if _DIAR_PIPELINE is None:
            _DIAR_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )

        result = _DIAR_PIPELINE(wav_path)
        segs = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segs.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            })
        return segs
    except Exception as e:
        print("[diarize] failed:", repr(e))
        return None
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

# --- helper: map transcript segments -> speakers by time overlap ---
def assign_speakers(transcript_segments, diar_segments):
    if not diar_segments:
        return [{"speaker": "Speaker 1", **s} for s in transcript_segments]

    def overlap(a_start, a_end, b_start, b_end):
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    labeled = []
    for seg in transcript_segments:
        best = None
        best_ov = 0.0
        for d in diar_segments:
            ov = overlap(seg["start"], seg["end"], d["start"], d["end"])
            if ov > best_ov:
                best_ov = ov
                best = d
        speaker = (best["speaker"] if best else "Speaker 1")
        labeled.append({"speaker": speaker, **seg})
    return labeled


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Role-based access control decorator
def role_required(role):
    def wrapper(f):
        @login_required
        def decorated_view(*args, **kwargs):
            if current_user.role != role:
                flash('Access denied: You do not have permission to access this page.')
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        decorated_view.__name__ = f.__name__
        return decorated_view
    return wrapper

# Management roles access decorator (manager and SME leader)
def management_required(f):
    @login_required
    def decorated_view(*args, **kwargs):
        if current_user.role not in ['manager', 'smeleader']:
            flash('Access denied: You do not have permission to access this page.')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    decorated_view.__name__ = f.__name__
    return decorated_view

# Management roles access decorator (manager and SME leader)

def get_clients_for_user(user_id):
    """Helper function """
    clients = db.session.query(ChatSession.client_name).filter(
        ChatSession.user_id == user_id,
        ChatSession.client_name.isnot(None)
    ).distinct().all()
    return [{'name': client[0]} for client in clients if client[0]]


@app.route('/')
@login_required
def home():
    clients = get_clients_for_user(current_user.id)
    recent_chats = (
    db.session.query(ChatSession.id, ChatSession.title)
    .join(ChatMessage, ChatSession.id == ChatMessage.session_id)
    .filter(
        ChatSession.user_id == current_user.id,
        ChatSession.client_name.is_(None)  # Only include general chats, not client chats
    )
    .group_by(ChatSession.id)
    .order_by(db.func.max(ChatMessage.timestamp).desc())
    .all()
    )
    
    # Get sales agents if current user is a manager or SME leader
    sales_agents = None
    if current_user.role in ['manager', 'smeleader']:
        sales_agents = []
        for agent in User.query.filter_by(role='salesagent').all():
            summaries = ClientSummary.query.filter_by(user_id=agent.id).all()
            sales_agents.append({
                'id': agent.id,
                'username': agent.username,
                'summaries': [{'client_name': s.client_name, 'summary': s.summary} for s in summaries]
            })
    
    return render_template('chat.html', username=current_user.username, role=current_user.role,
                         user_id=current_user.id, clients=clients, recent_chats=recent_chats,
                         sales_agents=sales_agents)

@app.route('/admin')
@role_required('admin')
def admin():
    return f"Hello Admin {current_user.username}, this is a protected admin page."

from dashboard.stats import get_sales_agents_client_stats, get_total_clients, get_dashboard_summary, get_seller_productivity, get_predictions_data

@app.route('/api/team-message', methods=['POST'])
@management_required
def create_team_message():
    data = request.json
    message = data.get('message')
    priority = data.get('priority', 'General Notes')
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    if priority not in ['Internal Announcement', 'External Broadcast For Clients', 'General Notes']:
        priority = 'General Notes'  # Default fallback
        
    notification = TeamNotification(
        manager_id=current_user.id,
        message=message,
        priority=priority
    )
    db.session.add(notification)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Notification sent successfully'})

@app.route('/api/notifications/unread', methods=['GET'])
@login_required
def get_unread_notifications_endpoint():
    from agents.notification_helper import get_unread_notifications
    # print(f"Fetching unread notifications for user {current_user.id}")
    
    notifications = get_unread_notifications(current_user.id)
    # print(f"Found {len(notifications)} unread notifications")
    
    return jsonify(notifications)

@app.route('/api/notifications/mark-read', methods=['POST'])
@login_required
def mark_notification_read():
    notification_id = request.json.get('notification_id')
    if not notification_id:
        return jsonify({'error': 'Notification ID required'}), 400
        
    read_record = NotificationRead(
        notification_id=notification_id,
        user_id=current_user.id
    )
    db.session.add(read_record)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/manager/dashboard')
@management_required
def manager_dashboard():
    agent_stats = get_sales_agents_client_stats()
    dashboard_summary = get_dashboard_summary()
    productivity_data = get_seller_productivity()
    predictions_data = get_predictions_data()
    return render_template('manager_dashboard.html', 
                         agent_stats=agent_stats,
                         dashboard_summary=dashboard_summary,
                         productivity_data=productivity_data,
                         predictions_data=predictions_data)

@app.route('/smeleader/dashboard')
@role_required('smeleader')
def smeleader_dashboard():
    # SME Leaders get access to manager stats plus manager oversight
    agent_stats = get_sales_agents_client_stats()
    dashboard_summary = get_dashboard_summary()
    productivity_data = get_seller_productivity()
    predictions_data = get_predictions_data()
    
    # Get all managers for SME Leader oversight
    managers = User.query.filter_by(role='manager').all()
    
    return render_template('smeleader_dashboard.html', 
                         agent_stats=agent_stats,
                         dashboard_summary=dashboard_summary,
                         productivity_data=productivity_data,
                         predictions_data=predictions_data,
                         managers=managers)

@app.route('/manager/team')
@management_required
def get_team_members():
    sales_agents = User.query.filter_by(role='salesagent').all()
    return render_template('team_members.html', team_members=sales_agents)

@app.route('/manager/agent-summary/<int:agent_id>', methods=['POST'])
@management_required
def create_agent_summary_chat(agent_id):
    agent = User.query.get(agent_id)
    if not agent:
        return jsonify({'error': 'Agent not found'}), 404
    
    # Get all client summaries for this agent
    summaries = ClientSummary.query.filter_by(user_id=agent_id).all()
    
    # Format the summary message
    message = f"In summary, Agent {agent.username} is working on {len(summaries)} active clients.\n\nHere are the updates:\n"
    
    for summary in summaries:
        message += f"- {summary.client_name}: {summary.summary}\n"
    
    # Check if a chat with this title already exists
    chat_title = f"Summary: {agent.username}'s Clients"
    existing_chat = ChatSession.query.filter_by(
        user_id=current_user.id,
        title=chat_title
    ).order_by(ChatSession.created_at.desc()).first()
    
    if existing_chat:
        # Use existing chat
        chat_id = existing_chat.id
    else:
        # Create a new chat session
        chat_id = str(uuid.uuid4())
        new_chat = ChatSession(
            id=chat_id,
            user_id=current_user.id,
            title=chat_title,
        )
        db.session.add(new_chat)
    
    # Add the summary message as a bot message
    bot_msg = ChatMessage(
        session_id=chat_id,
        user_id=current_user.id,
        message=message,
        sender='bot'
    )
    db.session.add(bot_msg)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'chat_id': chat_id,
        'message': message,
        'isExisting': existing_chat is not None
    })

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']

        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))

        new_user = User(username=username, role=role, manager_id = str(uuid.uuid4()) if role in ['manager', 'smeleader'] else None)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))



@app.route('/admin/users', methods=['GET', 'POST'])
@role_required('admin')
def manage_users():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add':
            username = request.form['username']
            password = request.form['password']
            role = request.form['role']
            if not User.query.filter_by(username=username).first():
                new_user = User(username=username, role=role)
                new_user.set_password(password)
                db.session.add(new_user)
                db.session.commit()
        elif action == 'update':
            user_id = request.form['user_id']
            user = db.session.get(User, user_id)
            if user:
                user.role = request.form['role']
                if request.form['password']:
                    user.set_password(request.form['password'])
                db.session.commit()
        elif action == 'delete':
            user_id = request.form['user_id']
            user = db.session.get(User, user_id)
            if user and user.username != 'admin':  # don't delete default admin
                db.session.delete(user)
                db.session.commit()

    users = User.query.all()
    return render_template('admin_users.html', users=users)


@app.route("/v1/chat/agent", methods=['POST'])
@login_required
def agent_chat():
    data = request.json
    chat_id = data['chat_id']
    user_message = data['message']

    # Save user message
    user_msg = ChatMessage(
        session_id=chat_id,
        user_id=current_user.id,
        message=user_message,
        sender='user'
    )
    db.session.add(user_msg)

    # Get the chat session to check if it's a client chat
    chat_session = ChatSession.query.get(chat_id)
    
    # Choose agent based on context and role
    if current_user.role in ['manager', 'smeleader']:
        # Get or create manager agent with current dashboard data
        agent = get_manager_agent()
        
        # Get conversation history for context
        chat_history = ChatMessage.query.filter_by(session_id=chat_id).order_by(ChatMessage.timestamp).all()
        messages = []
        
        # Include first message (summary) and user's new message
        if chat_history:
            first_message = chat_history[0]
            if first_message.sender == 'bot':
                messages.append({"role": "system", "content": f"Previous context - Agent Summary: {first_message.message}"})
        
        messages.append({"role": "user", "content": user_message})
        
        # Run the bot with context
        result = agent.invoke({
            "messages": messages
        }, config={"configurable": {"thread_id": chat_id}})
    elif chat_session and chat_session.client_name:  # client-specific chat
        # Create a new supervisor agent specifically for this client chat
        agent = get_supervisor_for_user(user_id=str(current_user.id))
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        }, config={"configurable": {"thread_id": chat_id}})
    else:  # general chat for sales agents and other roles
        agent = supervisor_agent_general
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        }, config={"configurable": {"thread_id": chat_id}})

    ai_response = result['messages'][-1].content

    # Save bot response
    bot_msg = ChatMessage(
        session_id=chat_id,
        user_id=current_user.id,
        message=ai_response,
        sender='bot'
    )
    db.session.add(bot_msg)

    # Update client summary counter if this is a client chat
    if chat_session and chat_session.client_name:
        # Count total messages in this client chat
        message_count = ChatMessage.query.filter_by(session_id=chat_id).count()
        
        # Get or create client summary
        client_summary = ClientSummary.query.filter_by(
            user_id=current_user.id,
            client_name=chat_session.client_name
        ).first()
        
        # Every 5 messages, update the summary
        if message_count % 5 == 0:
            # Get recent messages
            messages = ChatMessage.query.filter_by(
                session_id=chat_id
            ).order_by(ChatMessage.timestamp.desc()).limit(10).all()
            
            # Generate summary
            transcript = extract_transcript(messages)
            summary = generate_summary(transcript)
            
            if client_summary:
                client_summary.summary = summary
                client_summary.last_updated = datetime.utcnow()
            else:
                client_summary = ClientSummary(
                    user_id=current_user.id,
                    client_name=chat_session.client_name,
                    summary=summary
                )
                db.session.add(client_summary)

    db.session.commit()
    return jsonify(ai_response)



@app.route("/v1/chat/newchat",methods=['POST'])
@login_required
def new_chat_id():
    user_id = int(request.json['user_id'])
    client_name = request.json.get('client_name')  # Optional client name
    title = request.json.get('title', 'Untitled Chat')  # Default title if not provided
    agent_id = request.json.get('agent_id')  # For agent summaries
    
    chat_id = str(uuid.uuid4())
    new_chat = ChatSession(id=chat_id, user_id=current_user.id, title=title, client_name=client_name)
    db.session.add(new_chat)

    # If this is an agent summary request
    if agent_id and current_user.role in ['manager', 'smeleader']:
        agent = User.query.get(agent_id)
        if agent:
            # Get all client summaries for this agent
            summaries = ClientSummary.query.filter_by(user_id=agent_id).all()
            
            # Format the summary message
            message = f"In summary, Agent {agent.username} is working on {len(summaries)} active clients.\n\nHere are the updates:\n"
            for summary in summaries:
                message += f"- {summary.client_name}: {summary.summary}\n"
            
            # Add the summary as a bot message
            bot_msg = ChatMessage(
                session_id=chat_id,
                user_id=current_user.id,
                message=message,
                sender='bot'
            )
            db.session.add(bot_msg)
    
    db.session.commit()
    return jsonify({'id': chat_id, 'title': title})

# Create client endpoint
@app.route('/v1/clients', methods=['POST'])
@login_required
def create_client():
    client_name = request.json.get('name', '').strip()
    if not client_name:
        return jsonify({'error': 'Client name required'}), 400
    
    # This saves the client in the database for future reference
    chat_id = uuid.uuid4()
    new_chat = ChatSession(
        id=str(chat_id), 
        user_id=current_user.id, 
        client_name=client_name,
        title=f"Chat with {client_name}"
    )
    db.session.add(new_chat)
    db.session.commit()
    
    return jsonify({'name': client_name, 'success': True, 'chat_id': str(chat_id)})

# Get clients for current user
@app.route('/v1/clients', methods=['GET'])
@login_required
def get_clients():
    clients = db.session.query(ChatSession.client_name).filter(
        ChatSession.user_id == current_user.id,
        ChatSession.client_name.isnot(None)
    ).distinct().all()
    
    client_list = [{'name': client[0]} for client in clients if client[0]]
    return jsonify(client_list)

# Find existing chat for client
@app.route('/v1/clients/<client_name>/chat', methods=['GET'])
@login_required
def find_client_chat(client_name):
    existing_chat = ChatSession.query.filter_by(
        user_id=current_user.id,
        client_name=client_name
    ).first()
    
    if existing_chat:
        print("Chat exists")
        return jsonify({'chat_id': existing_chat.id, 'exists': True})
    else:
        print("Chat does not exist")
        return jsonify({'exists': False})

# Delete client (deletes all chats for that client)
@app.route('/v1/clients/<client_name>', methods=['DELETE'])
@login_required
def delete_client(client_name):
    # Delete all chat sessions for this client
    deleted_count = ChatSession.query.filter_by(
        user_id=current_user.id,
        client_name=client_name
    ).delete()
    
    db.session.commit()
    return jsonify({'success': True, 'deleted_chats': deleted_count})

# Edit client name (updates all chats for that client)
@app.route('/v1/clients/<old_name>/rename', methods=['PUT'])
@login_required
def rename_client(old_name):
    new_name = request.json.get('new_name', '').strip()
    if not new_name:
        return jsonify({'error': 'New name required'}), 400
    
    # Update all chat sessions for this client
    updated_count = ChatSession.query.filter_by(
        user_id=current_user.id,
        client_name=old_name
    ).update({'client_name': new_name})
    
    if updated_count == 0:
        return jsonify({'error': 'Client not found'}), 404
    
    db.session.commit()
    return jsonify({'success': True, 'updated_chats': updated_count, 'new_name': new_name})
        
@app.route("/v1/chat/sessions", methods=['GET'])
@login_required
def get_user_chats():
    sessions = ChatSession.query.filter_by(user_id=current_user.id).order_by(ChatSession.created_at.desc()).all()
    return jsonify([
        {'id': s.id, 'title': s.title, 'created_at': s.created_at.isoformat()}
        for s in sessions
    ])

@app.route('/v1/chat/loadchat/<chat_id>', methods=['GET'])
@login_required
def load_chat(chat_id):
    session = ChatSession.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not session:
        return jsonify({'error': 'Chat session not found'}), 404

    messages = ChatMessage.query.filter_by(session_id=chat_id).order_by(ChatMessage.timestamp).all()
    
    return jsonify([
        {
            'content': m.message,
            'role': 'user' if m.sender == 'user' else 'bot',
            'timestamp': m.timestamp.isoformat()
        }
        for m in messages
    ])

@app.route('/v1/chat/renamechat', methods=['POST'])
@login_required
def rename_chat():
    data = request.get_json()
    chat_id = data.get('chat_id')
    new_title = data.get('new_title')

    chat = ChatSession.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    chat.title = new_title
    db.session.commit()
    return jsonify({'success': True})
    print("Rename request received:", data)
    print("Chat found:", chat is not None)


@app.route('/v1/chat/deletechat', methods=['POST'])
@login_required
def delete_chat():
    data = request.get_json()
    chat_id = data.get('chat_id')
    
    if not chat_id:
        return jsonify({'error': 'Chat ID is required'}), 400
    
    # Find the chat session
    chat = ChatSession.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Delete all messages for this chat
    ChatMessage.query.filter_by(session_id=chat_id).delete()
    
    # Delete the chat session
    db.session.delete(chat)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Chat deleted successfully'})

@app.route('/v1/chat/summary', methods=['POST'])
@login_required
def get_chat_summary():
    data = request.get_json()
    chat_id = data.get('chat_id')
    
    if not chat_id:
        return jsonify({'error': 'Chat ID is required'}), 400
    
    # Verify the chat belongs to the current user
    chat = ChatSession.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    try:
        # Get all messages for this chat
        messages = ChatMessage.query.filter_by(session_id=chat_id).order_by(ChatMessage.timestamp).all()
        
        if not messages:
            return jsonify({'error': 'No messages found in this chat'}), 404
        
        # Extract transcript
        transcript = extract_transcript(messages)

        # Generate summary using the LLM directly
        summary = generate_summary(transcript)
        
        return jsonify({'summary': summary, 'success': True})
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return jsonify({'error': 'Failed to generate summary'}), 500

@app.route('/api/transcribe', methods=['POST'])
@login_required
def transcribe_audio():
    try:
        title = request.form.get('title', 'Live Meeting')

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        filename = secure_filename(audio_file.filename or 'meeting.webm')

        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        # Choose model & response format
        model_name = os.getenv("STT_MODEL", "whisper-1")
        use_verbose = ("whisper" in model_name)
        resp_format = "verbose_json" if use_verbose else "json"

        # OpenAI call
        with open(tmp_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model_name,
                file=f,
                response_format=resp_format
            )

        # Parse via dict
        rd = resp.model_dump() if hasattr(resp, "model_dump") else {}
        segments = []
        if use_verbose and isinstance(rd.get("segments"), list):
            for s in rd["segments"]:
                segments.append({
                    "start": float(s.get("start") or 0.0),
                    "end": float(s.get("end") or 0.0),
                    "text": (s.get("text") or "").strip()
                })
        else:
            text = (rd.get("text") or "").strip()
            segments = [{"start": 0.0, "end": 0.0, "text": text}]

        # Diarization only if timestamps exist
        if use_verbose:
            diar_segments = diarize_file(tmp_path)  # returns None if disabled
            labeled = assign_speakers(segments, diar_segments)
        else:
            labeled = [{"speaker": "Speaker 1", **s} for s in segments]

        # Normalize speakers
        speaker_map, next_idx = {}, 1
        for it in labeled:
            key = it["speaker"]
            if key not in speaker_map:
                speaker_map[key] = f"Speaker {next_idx}"
                next_idx += 1
            it["speaker"] = speaker_map[key]
        num_speakers = max(1, len(speaker_map))

        # Pretty text
        def hhmmss(t):
            t = max(0.0, float(t or 0.0))
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        lines = [f"[{hhmmss(seg['start'])}–{hhmmss(seg['end'])}] {seg['speaker']}: {seg['text']}"
                 for seg in labeled]
        pretty_text = "\n".join(lines).strip()

        # Save PDF
        base_dir = Path("uploads") / "transcripts" / str(current_user.id)
        base_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip() or "meeting"
        pdf_path = base_dir / f"{stamp}_{safe_title}.pdf"

        write_transcript_pdf(
            str(pdf_path),
            title=title,
            pretty_text=pretty_text,
            meta={"Model": model_name, "Speakers": str(num_speakers)}
        )

        # DB row (store preview text & file path)
        transcript = Transcript(
            user_id=current_user.id,
            chat_id=None,
            title=title,
            text=pretty_text  # used for sidebar preview
        )
        try:
            transcript.file_path = str(pdf_path)
            transcript.speakers_count = num_speakers
            transcript.language = rd.get("language")
        except Exception:
            pass

        db.session.add(transcript)
        db.session.commit()

        # Cleanup temp
        try: os.remove(tmp_path)
        except Exception: pass

        return jsonify({
            "success": True,
            "transcript_id": transcript.id,
            "title": title,
            "file_url": url_for('download_transcript', transcript_id=transcript.id),
            "speakers": num_speakers
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    
@app.route('/v1/transcripts', methods=['GET'])
@login_required
def list_transcripts():
    rows = Transcript.query.filter_by(user_id=current_user.id).order_by(Transcript.created_at.desc()).all()
    out = []
    for t in rows:
        out.append({
            "id": t.id,
            "title": t.title,
            "created_at": t.created_at.isoformat(),
            "preview": (t.text[:200] + '...') if len(t.text) > 200 else t.text,
            "file_url": url_for('download_transcript', transcript_id=t.id)
        })
    return jsonify(out)

@app.route('/v1/transcripts/<int:transcript_id>/download', methods=['GET'])
@login_required
def download_transcript(transcript_id):
    t = Transcript.query.filter_by(id=transcript_id, user_id=current_user.id).first()
    if not t or not t.file_path or not os.path.exists(t.file_path):
        return jsonify({"error": "Not found"}), 404
    return send_file(t.file_path, as_attachment=True, download_name=os.path.basename(t.file_path))

@app.route('/v1/transcripts/<int:transcript_id>', methods=['DELETE'])
@login_required
def delete_transcript(transcript_id):
    t = Transcript.query.filter_by(id=transcript_id, user_id=current_user.id).first()
    if not t:
        return jsonify({"error": "Not found"}), 404
    # remove file
    try:
        if t.file_path and os.path.exists(t.file_path):
            os.remove(t.file_path)
    except Exception as e:
        print("file delete warning:", e)
    db.session.delete(t)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/v1/chat/report', methods=['POST'])
@login_required
def generate_chat_report():
    data = request.get_json()
    chat_id = data.get('chat_id') if data else None
    output_format = (data.get('format') if data else 'markdown') or 'markdown'
    if not chat_id:
        return jsonify({'error': 'Chat ID is required'}), 400

    # Verify ownership
    chat_session = ChatSession.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat_session:
        return jsonify({'error': 'Chat not found'}), 404

    messages = ChatMessage.query.filter_by(session_id=chat_id).order_by(ChatMessage.timestamp).all()
    if not messages:
        return jsonify({'error': 'No messages in chat'}), 404

    # Find the most relevant bot message containing benchmark content
    target_bot_message = None
    for msg in reversed(messages):
        if msg.sender == 'bot' and (
            'Benchmark Comparison Table' in msg.message or
            'Side-by-side Comparison Table' in msg.message or
            'How to Pitch It' in msg.message
        ):
            target_bot_message = msg.message
            break
    if not target_bot_message:
        # fallback to last bot message
        for msg in reversed(messages):
            if msg.sender == 'bot':
                target_bot_message = msg.message
                break

    if not target_bot_message:
        target_bot_message = 'No bot analytical response available.'
    generated_at = datetime.utcnow().isoformat() + 'Z'
    client_name = chat_session.client_name or 'General'
    if output_format.lower() == 'pdf' and A4 is not None:
        # Enhanced PDF rendering using platypus (tables + headings) with fallback
        buffer = io.BytesIO()
        # Attempt rich layout; fallback to plain text if imports failed
        try:
            styles = getSampleStyleSheet()
            # Custom tweaks
            styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=9, leading=11))
            heading_map = {
                1: styles['Heading1'],
                2: styles['Heading2'],
                3: styles['Heading3']
            }
            story = []
            # Title block
            story.append(Paragraph('Chat Report', styles['Heading1']))
            meta_html = f"<b>Client:</b> {client_name}<br/><b>Chat ID:</b> {chat_id}<br/><b>Generated (UTC):</b> {generated_at}".replace('\n',' ')
            story.append(Paragraph(meta_html, styles['Small']))
            story.append(Spacer(1, 8))
            story.append(Paragraph('Analytical Output', styles['Heading2']))
            story.append(Spacer(1, 4))
            # Build lines list from analytical markdown
            lines = target_bot_message.splitlines()
            i = 0
            while i < len(lines):
                raw_line = lines[i]
                line = raw_line.rstrip('\n')
                stripped = line.lstrip()
                # Utility to clean markdown emphases in cells
                import re as _cell_re
                def _clean_cell(txt: str):
                    txt = txt.strip()
                    # Remove surrounding ** ** or __ __
                    txt = _cell_re.sub(r'^\*\*(.+?)\*\*$', r'\1', txt)
                    txt = _cell_re.sub(r'^__(.+?)__$', r'\1', txt)
                    return txt
                # Pseudo-table (bold tokens without pipes) detection (allow leading spaces)
                if stripped.startswith('**') and stripped.count('**') >= 4 and '|' not in stripped:
                    import re
                    header_tokens = re.findall(r'\*\*(.+?)\*\*', stripped)
                    header_tokens = [_clean_cell(h) for h in header_tokens]
                    if header_tokens:
                        i += 1
                        row_data = []
                        while i < len(lines):
                            row_line = lines[i].rstrip('\n')
                            row_stripped = row_line.lstrip()
                            if not (row_stripped.startswith('**') and row_stripped.count('**') >= 2):
                                break
                            cells = re.findall(r'\*\*(.+?)\*\*', row_stripped)
                            tail = row_stripped.split('**')[-1].strip()
                            if tail:
                                tail_parts = tail.split()
                                cells.extend(tail_parts)
                            if len(cells) < len(header_tokens):
                                cells += [''] * (len(header_tokens) - len(cells))
                            # Clean each cell
                            cells = [_clean_cell(c) for c in cells]
                            row_data.append(cells[:len(header_tokens)])
                            i += 1
                        if row_data:
                            tbl_rows = [header_tokens] + row_data
                            tbl = Table(tbl_rows, hAlign='LEFT')
                            tbl_style = TableStyle([
                                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#222222')),
                                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0,0), (-1,0), 9),
                                ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
                                ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
                                ('FONTSIZE', (0,1), (-1,-1), 8),
                                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor('#eeeeee')])
                            ])
                            tbl.setStyle(tbl_style)
                            story.append(tbl)
                            story.append(Spacer(1, 10))
                            continue
                # Skip trivial
                if not stripped:
                    story.append(Spacer(1, 6))
                    i += 1
                    continue
                # Headings
                if stripped.startswith('#'):
                    hashes = len(stripped) - len(stripped.lstrip('#'))
                    text = stripped[hashes:].strip()
                    level = min(hashes,3)
                    story.append(Paragraph(text, heading_map.get(level, styles['Heading3'])))
                    i += 1
                    continue
                # Table detection (pipe tables)
                def is_table_line(l):
                    return '|' in l and l.count('|') >= 2
                if is_table_line(stripped):
                    table_block = []
                    while i < len(lines) and is_table_line(lines[i].strip()):
                        table_block.append(lines[i].strip())
                        i += 1
                    raw_rows = [r.strip('|') for r in table_block]
                    rows = []
                    for r in raw_rows:
                        cells = [c.strip() for c in r.split('|')]
                        if all(set(c.replace('-',''))==set() for c in cells):
                            continue
                        cells = [_clean_cell(c) for c in cells]
                        rows.append(cells)
                    if rows:
                        tbl = Table(rows, hAlign='LEFT')
                        tbl_style = TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#333333')),
                            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0,0), (-1,0), 9),
                            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f5f5f5')),
                            ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
                            ('FONTSIZE', (0,1), (-1,-1), 8),
                            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor('#eeeeee')])
                        ])
                        tbl.setStyle(tbl_style)
                        story.append(tbl)
                        story.append(Spacer(1, 10))
                    continue
                # Bullet list
                if stripped.startswith(('*','-')) and len(stripped.split())>1:
                    bullets = []
                    while i < len(lines) and lines[i].lstrip().startswith(('*','-')):
                        bullets.append(lines[i].lstrip()[1:].strip())
                        i += 1
                    bullet_html = '<br/>'.join([f'• {b}' for b in bullets])
                    story.append(Paragraph(bullet_html, styles['Small']))
                    story.append(Spacer(1,6))
                    continue
                # Paragraph fallback with proper bold conversion
                import re as _re
                def boldify(s):
                    s2 = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
                    s2 = _re.sub(r'^\*\*(.+?)\*\*$', r'<b>\1</b>', s2)
                    return s2.replace('**','')
                story.append(Paragraph(boldify(stripped), styles['BodyText']))
                story.append(Spacer(1,4))
                i += 1
            story.append(Spacer(1,12))
            story.append(Paragraph('<i>End of Report</i>', styles['Small']))
            doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
            doc.build(story)
            buffer.seek(0)
            return send_file(
                buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'report_{chat_id}.pdf'
            )
        except Exception:
            # Fallback to previous plain text canvas approach
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            margin = 20 * mm
            text_obj = c.beginText(margin, height - margin)
            text_obj.setFont('Helvetica-Bold', 14)
            text_obj.textLine('Chat Report')
            text_obj.setFont('Helvetica', 10)
            meta_lines = [
                f'Client: {client_name}', f'Chat ID: {chat_id}', f'Generated (UTC): {generated_at}', '', 'Analytical Output:', ''
            ]
            for line in meta_lines:
                text_obj.textLine(line)
            import textwrap
            wrap_width = 100
            for raw_line in target_bot_message.splitlines():
                if not raw_line.strip():
                    text_obj.textLine('')
                    continue
                for wrapped in textwrap.wrap(raw_line, wrap_width):
                    if text_obj.getY() < 40 * mm:
                        c.drawText(text_obj)
                        c.showPage()
                        text_obj = c.beginText(margin, height - margin)
                        text_obj.setFont('Helvetica', 10)
                    text_obj.textLine(wrapped)
            c.drawText(text_obj)
            c.showPage()
            c.save()
            buffer.seek(0)
            return send_file(
                buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'report_{chat_id}.pdf'
            )
    # default markdown
    report_md = f"""# Chat Report\n\nClient: {client_name}\nChat ID: {chat_id}\nGenerated At (UTC): {generated_at}\n\n## Analytical Output\n\n{target_bot_message}\n\n---\n*End of Report*\n"""
    md_bytes = report_md.encode('utf-8')
    return send_file(
        io.BytesIO(md_bytes),
        mimetype='text/markdown; charset=utf-8',
        as_attachment=True,
        download_name=f'report_{chat_id}.md'
    )

# SME Leader Dashboard API Routes
@app.route('/api/sme/gwp-growth-trend', methods=['GET'])
@role_required('smeleader')
def get_gwp_growth_trend():
    """Get GWP Growth Trend data from Excel file for D3.js chart"""
    try:
        # Read the kpi_monthly sheet from sme.xlsx
        df = pd.read_excel('sme.xlsx', sheet_name='kpi_monthly')
        
        # Extract the Month and GWP Monthly columns
        data = []
        for index, row in df.iterrows():
            if pd.notna(row['Month']) and pd.notna(row['1.2a GWP Monthly M SAR']):
                month_str = str(row['Month']).strip()
                
                data.append({
                    'month': month_str,  # Keep full format like "Jul 2024"
                    'gwp_value': float(row['1.2a GWP Monthly M SAR'])
                })
        
        return jsonify(data)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return jsonify({'error': 'SME Excel file not found'}), 404
    except Exception as e:
        print(f"Error in get_gwp_growth_trend: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load GWP data: {str(e)}'}), 500

@app.route('/api/sme/funnel-size-trend', methods=['GET'])
@role_required('smeleader')
def get_funnel_size_trend():
    """Get Funnel Size Trend data from Excel file for D3.js line chart"""
    try:
        # Read the kpi_monthly sheet from sme.xlsx
        df = pd.read_excel('sme.xlsx', sheet_name='kpi_monthly')
        
        # Extract the Month and Funnel Size Count columns
        data = []
        for index, row in df.iterrows():
            if pd.notna(row['Month']) and pd.notna(row['3.1 Funnel Size Count']):
                month_str = str(row['Month']).strip()
                
                data.append({
                    'month': month_str,  # Keep full format like "Jul 2024"
                    'funnel_count': float(row['3.1 Funnel Size Count'])
                })
        
        return jsonify(data)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return jsonify({'error': 'SME Excel file not found'}), 404
    except Exception as e:
        print(f"Error in get_funnel_size_trend: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load Funnel Size data: {str(e)}'}), 500

@app.route('/api/sme/funnel-coverage', methods=['GET'])
@role_required('smeleader')
def get_funnel_coverage():
    """Get Funnel Coverage data from Excel file - monthly coverage percentages"""
    try:
        # Read the kpi_monthly sheet from sme.xlsx
        df = pd.read_excel('sme.xlsx', sheet_name='kpi_monthly')
        
        # Extract the Month and Funnel Coverage Pct columns
        data = []
        for index, row in df.iterrows():
            if pd.notna(row['Month']) and pd.notna(row['3.2 Funnel Coverage Pct']):
                month_str = str(row['Month']).strip()
                
                data.append({
                    'month': month_str,  # Keep full format like "Jul 2024"
                    'coverage_pct': float(row['3.2 Funnel Coverage Pct'])
                })
        
        return jsonify(data)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return jsonify({'error': 'SME Excel file not found'}), 404
    except Exception as e:
        print(f"Error in get_funnel_coverage: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load Funnel Coverage data: {str(e)}'}), 500

@app.route('/api/sme/budget-fit-analysis', methods=['GET'])
@role_required('smeleader')
def get_budget_fit_analysis():
    """Get Budget Fit Analysis data from Excel file - averaged values across all months"""
    try:
        # Read the suhail_signals sheet from sme.xlsx
        df = pd.read_excel('sme.xlsx', sheet_name='suhail_signals')
        
        # Check if required columns exist
        required_columns = ['3.4.1_Budget_Fit_Pct', '3.4.2_Proposal_Diversity_Pct', '3.4.3_Competitor_WinRate_Pct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing columns in suhail_signals sheet: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return jsonify([])
        
        # Calculate averages across all months for each metric
        budget_fit_avg = df['3.4.1_Budget_Fit_Pct'].dropna().mean()
        proposal_diversity_avg = df['3.4.2_Proposal_Diversity_Pct'].dropna().mean()
        competitor_winrate_avg = df['3.4.3_Competitor_WinRate_Pct'].dropna().mean()
        
        # Create the response data
        result = [
            {
                'metric': 'Budget Fit Analysis',
                'percentage': round(budget_fit_avg, 1) if not pd.isna(budget_fit_avg) else 0
            },
            {
                'metric': 'Proposal Diversity',
                'percentage': round(proposal_diversity_avg, 1) if not pd.isna(proposal_diversity_avg) else 0
            },
            {
                'metric': 'Competitor Win/Loss',
                'percentage': round(competitor_winrate_avg, 1) if not pd.isna(competitor_winrate_avg) else 0
            }
        ]
        
        print(f"Returning budget fit analysis data: {result}")
        return jsonify(result)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return jsonify({'error': 'SME Excel file not found'}), 404
    except Exception as e:
        print(f"Error in get_budget_fit_analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load Budget Fit Analysis data: {str(e)}'}), 500

@app.route('/api/sme/renewal-heatmap', methods=['GET'])
@role_required('smeleader')
def get_renewal_heatmap():
    """Get Renewal Heat Map data from Excel file for D3.js heatmap visualization"""
    try:
        # Read the heatmap_monthly sheet from sme.xlsx
        df = pd.read_excel('sme.xlsx', sheet_name='heatmap_monthly')
        
        # Process the data for heatmap
        data = []
        for index, row in df.iterrows():
            if pd.notna(row['Month']) and pd.notna(row['Bucket_Days']) and pd.notna(row['2.2_Renewal_Value_M_SAR']):
                data.append({
                    'month': str(row['Month']).strip(),
                    'bucket_days': int(row['Bucket_Days']),
                    'renewal_value': float(row['2.2_Renewal_Value_M_SAR'])
                })
        
        return jsonify(data)
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        return jsonify({'error': 'SME Excel file not found'}), 404
    except Exception as e:
        print(f"Error in get_renewal_heatmap: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load heatmap data: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        # Check if we need to add manager_id column
        
        inspector = db.inspect(db.engine)
        existing_columns = [column['name'] for column in inspector.get_columns('user')]
        
        if 'manager_id' not in existing_columns:
            # Add the manager_id column to existing database
            with db.engine.connect() as conn:
                conn.execute(db.text("ALTER TABLE user ADD COLUMN manager_id INTEGER"))
                conn.commit()
                print("Added manager_id column to user table")
        db.create_all()
    app.run(host='0.0.0.0', port=5002, debug=True)
