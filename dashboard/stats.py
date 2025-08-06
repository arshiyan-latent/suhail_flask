from models import User, ChatSession, db
from sqlalchemy import func

def get_total_sellers():
    """Get the total number of sales agents"""
    return User.query.filter_by(role='salesagent').count()

def get_dashboard_summary():
    """
    Get summary statistics for the dashboard header
    Returns a dictionary containing high-level metrics
    """
    return {
        'total_sellers': get_total_sellers(),
        'total_accounts': get_total_clients(),
        'total_business': 1500000,  # Placeholder value in SAR
        'target_achievement': 75.5,  # Placeholder percentage
    }

def get_sales_agents_client_stats():
    """
    Get statistics about clients for each sales agent
    Returns a list of dictionaries containing:
    - agent_name
    - client_count
    - client_list
    """
    sales_agents = User.query.filter_by(role='salesagent').all()
    stats = []
    
    for agent in sales_agents:
        # Get distinct client names for this agent
        clients = db.session.query(ChatSession.client_name)\
            .filter(
                ChatSession.user_id == agent.id,
                ChatSession.client_name.isnot(None),
                ChatSession.client_name != ''  # Exclude empty strings
            )\
            .distinct()\
            .all()
        
        # Convert list of tuples to list of names, filtering out any None or empty strings
        client_names = [client[0] for client in clients if client[0] and client[0].strip()]
        
        stats.append({
            'agent_name': agent.username,
            'client_count': len(client_names),
            'client_list': client_names
        })
    
    return stats

def get_total_clients():
    """Get the total number of unique clients across all sales agents"""
    # Get all unique client names that are not None and not empty
    clients = db.session.query(func.distinct(ChatSession.client_name))\
        .join(User, ChatSession.user_id == User.id)\
        .filter(
            User.role == 'salesagent',
            ChatSession.client_name.isnot(None),
            ChatSession.client_name != ''  # Exclude empty strings
        )\
        .all()
    
    # Filter out any remaining empty strings or whitespace-only strings
    valid_clients = [client[0] for client in clients if client[0] and client[0].strip()]
    return len(valid_clients)

def get_predictions_data():
    """
    Get end of year predictions and opportunities data
    Returns a dictionary containing predictions and opportunities
    """
    return {
        'year_end_prediction': 4500000,  # Placeholder value in SAR
        'projected_closure': 3200000,    # Placeholder value in SAR
        'at_risk_deals': [
            {'client': 'Client A', 'value': 750000, 'risk_factor': 'High'},
            {'client': 'Client B', 'value': 500000, 'risk_factor': 'Medium'},
            {'client': 'Client C', 'value': 250000, 'risk_factor': 'High'}
        ],
        'top_opportunities': [
            {'client': 'Prospect X', 'potential': 1200000, 'probability': '80%'},
            {'client': 'Prospect Y', 'potential': 800000, 'probability': '65%'},
            {'client': 'Prospect Z', 'potential': 600000, 'probability': '75%'}
        ]
    }

def get_seller_productivity():
    """
    Get productivity metrics for all sales agents
    Returns a list of dictionaries containing seller performance data
    """
    sales_agents = User.query.filter_by(role='salesagent').all()
    productivity_data = []
    
    risk_levels = ['Low', 'Medium', 'High']  # Placeholder risk levels
    recommendations = [
        'Increase client engagement',
        'Focus on high-value prospects',
        'Schedule follow-up meetings',
        'Review pipeline strategy'
    ]  # Placeholder recommendations
    
    import random  # For generating placeholder data
    
    for agent in sales_agents:
        # Get actual client count
        assigned_clients = db.session.query(func.count(func.distinct(ChatSession.client_name)))\
            .filter(
                ChatSession.user_id == agent.id,
                ChatSession.client_name.isnot(None),
                ChatSession.client_name != ''
            ).scalar() or 0
            
        # Placeholder data - will be replaced with real data later
        closed_accounts = random.randint(1, assigned_clients) if assigned_clients > 0 else 0
        quota_achievement = random.uniform(50, 120)
        risk_level = random.choice(risk_levels)
        potential_close = random.randint(100000, 1000000)
        
        productivity_data.append({
            'seller_name': agent.username,
            'accounts_assigned': assigned_clients,
            'accounts_closed': closed_accounts,
            'quota_achievement': quota_achievement,
            'risk_level': risk_level,
            'potential_close': potential_close,
            'ai_recommendation': random.choice(recommendations)
        })
    
    return productivity_data
