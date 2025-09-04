from models import User, ChatSession, db
from sqlalchemy import func

def get_dashboard_summary(manager_id=None):
    """
    Get summary statistics for the dashboard header
    If manager_id is provided, return stats for that manager's team only
    Returns a dictionary containing high-level metrics
    """
    return {
        'total_sellers': get_total_sellers(manager_id),
        'total_accounts': get_total_clients(manager_id),
        'total_business': 1500000,  # Placeholder value in SAR
        'target_achievement': 75.5,  # Placeholder percentage
    }

def get_total_sellers(manager_id=None):
    """Get the total number of sales agents for a manager or all agents"""
    if manager_id:
        return User.query.filter_by(role='salesagent', manager_id=manager_id).count()
    else:
        return User.query.filter_by(role='salesagent').count()

def get_sales_agents_client_stats(manager_id=None):
    """
    Get statistics about clients for each sales agent
    If manager_id is provided, only return stats for agents managed by that manager
    Returns a list of dictionaries containing:
    - agent_name
    - client_count
    - client_list
    """
    if manager_id:
        # Get only agents managed by this specific manager
        sales_agents = User.query.filter_by(role='salesagent', manager_id=manager_id).all()
    else:
        # Get all sales agents (for SME Leaders)
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

def get_total_clients(manager_id=None):
    """Get the total number of unique clients across all sales agents or manager's team"""
    if manager_id:
        # Get unique clients for agents managed by this manager
        clients = db.session.query(func.distinct(ChatSession.client_name))\
            .join(User, ChatSession.user_id == User.id)\
            .filter(
                User.role == 'salesagent',
                User.manager_id == manager_id,
                ChatSession.client_name.isnot(None),
                ChatSession.client_name != ''  # Exclude empty strings
            )\
            .all()
    else:
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

def get_seller_productivity(manager_id=None):
    """
    Get productivity metrics for sales agents
    If manager_id is provided, return stats for that manager's team only
    Returns a list of dictionaries containing detailed seller performance data
    """
    if manager_id:
        sales_agents = User.query.filter_by(role='salesagent', manager_id=manager_id).all()
    else:
        sales_agents = User.query.filter_by(role='salesagent').all()
    
    productivity_data = []
    
    insights = [
        'High engagement with prospects, consistent follow-ups',
        'Strong focus on high-value opportunities, needs more client engagement',
        'Excellent deal closure rate, could improve initial engagement',
        'Good pipeline management, needs focus on deal conversion',
        'Active in client outreach, needs support in deal closure',
        'Strong relationship building, pipeline needs growth'
    ]
    
    import random
    
    for agent in sales_agents:
        # Get actual engaged clients count
        engaged_clients = db.session.query(func.count(func.distinct(ChatSession.client_name)))\
            .filter(
                ChatSession.user_id == agent.id,
                ChatSession.client_name.isnot(None),
                ChatSession.client_name != ''
            ).scalar() or 0
            
        # Generate realistic placeholder data
        closed_deals = random.randint(0, max(1, engaged_clients // 2))
        win_probability = random.uniform(0.30, 0.85)  # 30% to 85%
        win_ratio = random.uniform(0.10, 0.85)
        pipeline_value = random.randint(1000000, 5000000)  # 1M to 5M SAR
        engaged_opportunities = random.randint(max(1, engaged_clients // 2), engaged_clients + 3)
        
        productivity_data.append({
            'seller_name': agent.username,
            'engaged_clients': engaged_clients,
            'win_probability': f"{win_probability * 100:.1f}%",
            'closed_deals': closed_deals,
            'win_ratio': f"{win_ratio * 100:.1f}%",
            'sales_pipeline': pipeline_value,
            'engaged_opportunities': engaged_opportunities,
            'general_insights': random.choice(insights)
        })
    
    return productivity_data
