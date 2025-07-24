# app.py
from webbrowser import get
from agents.agent import supervisor_agent, llm
from agents.summary.summary_agent import extract_transcript, generate_summary
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI


load_dotenv()


from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, ChatSession, ChatMessage
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

#db_password = os.getenv('POSTGRE_PASSWORD')
#DATABASE_URL=f'postgresql://postgres:{db_password}@db.gvuxfdhghxrxlyjnlfly.supabase.co:6543/postgres'
#print(DATABASE_URL)
#app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

#helper function to get clients

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
    return render_template('chat.html', username=current_user.username, role=current_user.role,user_id = current_user.id, clients=clients, recent_chats=recent_chats)

@app.route('/admin')
@role_required('admin')
def admin():
    return f"Hello Admin {current_user.username}, this is a protected admin page."

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

        new_user = User(username=username, role=role, manager_id = str(uuid.uuid4()) if role =='manager' else None)
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


@app.route("/v1/chat/salesrep",methods=['POST'])
@login_required
def sales_agent():
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

    # Run the bot
    result = supervisor_agent.invoke({
        "messages": [{"role": "user", "content": user_message}]
    }, config={"configurable": {"thread_id": str(chat_id)}})

    ai_response = result['messages'][-1].content

    # Save bot response
    bot_msg = ChatMessage(
        session_id=chat_id,
        user_id=current_user.id,
        message=ai_response,
        sender='bot'
    )
    db.session.add(bot_msg)

    db.session.commit()
    return jsonify(ai_response)


@app.route("/v1/chat/newchat",methods=['POST'])
@login_required
def new_chat_id():
    user_id = int(request.json['user_id'])
    client_name = request.json.get('client_name')  # Optional client name
    title = request.json.get('title', 'Untitled Chat')  # Default title if not provided
    chat_id = uuid.uuid4()
    
    new_chat = ChatSession(id=str(chat_id), user_id=current_user.id, title = title, client_name=client_name)
    db.session.add(new_chat)
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
    chat_id = str(uuid.uuid4())
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
    app.run(debug=True)