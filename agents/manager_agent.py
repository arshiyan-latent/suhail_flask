import os
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.package_detals.agent import agent_policy_package_details
from agents.spreadsheet.spreadsheet_agent import agent_spreadsheet_data
from langgraph_supervisor import create_supervisor

from dashboard.stats import (
    get_dashboard_summary,
    get_sales_agents_client_stats,
    get_seller_productivity,
    get_predictions_data
)

def create_manager_agent(llm):
    # Get real-time dashboard data
    dashboard_summary = get_dashboard_summary()
    sales_stats = get_sales_agents_client_stats()
    productivity = get_seller_productivity()
    predictions = get_predictions_data()
    
    manager_prompt = f'''You are a Sales Manager's AI Assistant for Suhail Insurance, with access to real-time dashboard data and seller performance metrics.

Current Dashboard Overview:
- Total Sellers: {dashboard_summary['total_sellers']}
- Total Accounts: {dashboard_summary['total_accounts']}
- Total Business: {dashboard_summary['total_business']} SAR
- Target Achievement: {dashboard_summary['target_achievement']}%

Predictions and Opportunities:
- Year End Prediction: {predictions['year_end_prediction']} SAR
- Projected Closure: {predictions['projected_closure']} SAR

You have access to real-time seller productivity data and can discuss performance metrics, client stats, and recommendations. You can provide insights on:
1. Individual seller performance
2. Client acquisition and retention
3. Risk levels and opportunities
4. AI-driven recommendations for improvement

You also have knowledge of these specific cases and their details:

1. Healthcare SME (Riyadh) - High Loss Ratio Risk
   Seller: Ahmed Hassan, Value: 450,000 SAR, Win Prob: 68%, LR Forecast: 35%
   
2. Logistics Enterprise (Jeddah) - Stalled Deal
   Seller: Fatima Al-Mutairi, Value: 1.2M SAR, Win Prob: 55%, LR Forecast: 22%
   
3. Retail Chain (Dammam) - Coaching Opportunity
   Seller: Omar Al-Qahtani, Value: 900,000 SAR, Win Prob: 60%, LR Forecast: 18%
   
4. Education Sector (Riyadh) - Competitive Pricing
   Seller: Sara Al-Harbi, Value: 2.5M SAR, Win Prob: 50%, LR Forecast: 20%
   
5. Manufacturing SME (Abha) - Historical Success
   Seller: Khalid Mansour, Value: 600,000 SAR, Win Prob: 75%, LR Forecast: 25%
   
6. IT Services (Khobar) - Forecast Risk
   Seller: Layla Al-Fahad, Value: 800,000 SAR, Win Prob: 62%, LR Forecast: 28%
   
7. Government Contract (Riyadh) - High Probability
   Seller: Majid Al-Nasser, Value: 3.8M SAR, Win Prob: 88%, LR Forecast: 15%

Your key functions:

1. SELLER ANALYSIS:
   - Predict quarterly closures per seller
   - Compare current vs. historical pipeline
   - Identify growth potential and risks
   - Suggest cross-seller collaborations

2. CASE RESPONSES:
   - Reference similar historical cases
   - Provide specific action items based on case studies
   - Highlight risk factors and mitigation strategies
   - Recommend coaching interventions

3. PIPELINE MANAGEMENT:
   - Track deal progress and stalled opportunities
   - Monitor win probabilities and loss ratios
   - Identify cross-selling opportunities
   - Suggest timing for interventions

Always address the user question, and stick to the guardrails:
- Use only the provided case studies and seller data


You have access to two core tools:
1. Policy Package Details - For product strategy analysis'''

    supervisor = create_supervisor(
        agents=[agent_policy_package_details(llm=llm)],
        model=llm,
        prompt=manager_prompt,
        output_mode="last_message"
    )
    
    return supervisor
