from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import pandas as pd
import os

def load_historical_data():
    """Load the historical data with proper error handling"""
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # Go up 3 levels from spreadsheet/
        file_path = os.path.join(project_root, 'agents', 'spreadsheet', 'sheets', 'test_historical.xlsx')

        print(f"Resolved path: {file_path}")

        if os.path.exists(file_path):
            return pd.read_excel(file_path)
        else:
            print(f"Warning: {file_path} not found")
            return pd.DataFrame()  # Return empty DataFrame if file not found
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_historical_data()

# Below is the rules covered till 10th row in demo features

'''
Some rules for the spreadsheet agent:
- Valid Range = Benchmark lives ± 15%
- True Cost per Life = Budget ÷ Target LR; Target Price = Budget + (5% × Budget)
- If Target Price > Budget, downgrade package
- Expected LR = Package Avg Claims per Life ÷ Recommended Price per Life
- Sales probability model: Final Probability = (0.6 × Logistic Output) + (0.4 × Similarity Score) × Adjustment
- If Expected LR ≤ Target LR, Final Probability × 1.05
- Sale Probability Penalty if Required Price Exceeds Budget: Penalty Factor = 1 - (Variance %); Final Probability × Penalty Factor
'''

@tool
def assess_new_offer(
    region: str,
    lives: int,
    budget_per_life: float,
    target_lr: float,
    package: str,
    historical_claims_per_life: str,
    logistic_output: float = 0.65,
    similarity_score: float = 0.75,
    adjustment: float = 1.1,
) -> str:
    """Evaluate the feasibility and likelihood of success for a new insurance offer using historical benchmarks and strict sales logic."""
    
    if df.empty:
        return "❌ No historical data available. Please ensure the spreadsheet is properly loaded."
    
    # Print available columns for debugging
    # print(f"Available columns: {list(df.columns)}")
    
    # Try to find the right columns (case-insensitive matching)
    region_col = 'Contract Region'  # Exact match from spreadsheet
    package_col = 'Product Package'  # Exact match from spreadsheet  
    lives_col = 'Earned Exposure'    # Exact match from spreadsheet
    lr_col = 'Loss Ratio'           # Exact match from spreadsheet
    premium_col = 'Average Premium'  # Exact match from spreadsheet
    claims_col = 'Claims'           # Exact match from spreadsheet
    
    # Verify columns exist
    missing_cols = []
    for col_name, col_var in [('Contract Region', region_col), ('Product Package', package_col), 
                              ('Earned Exposure', lives_col), ('Loss Ratio', lr_col),
                              ('Average Premium', premium_col), ('Claims', claims_col)]:
        if col_var not in df.columns:
            missing_cols.append(col_name)
    
    if missing_cols:
        return f"❌ Missing required columns: {', '.join(missing_cols)}. Available columns: {list(df.columns)}"
    
    # Filter data (handle package names with prefixes like "D. Gold Package")
    try:
        region_matches = df[df[region_col].str.lower() == region.lower()]
        # For packages, check if the package name is contained in the full package name
        package_matches = region_matches[region_matches[package_col].str.lower().str.contains(package.lower(), na=False)]
        matches = package_matches
    except Exception as e:
        return f"❌ Error filtering data: {str(e)}"

    if matches.empty:
        available_regions = df[region_col].unique()
        available_packages = df[package_col].unique()
        return f"❌ No historical data for {region}/{package} combination.\nAvailable regions: {available_regions}\nAvailable packages: {available_packages}"

    # --- Historical Benchmarks ---
    try:
        benchmark_lives = matches[lives_col].mean()
        avg_loss_ratio = matches[lr_col].mean() if lr_col else None
        avg_premium = matches[premium_col].mean() if premium_col else None
        avg_claims_per_life = (matches[claims_col].sum() / matches[lives_col].sum()) if claims_col else None
    except Exception as e:
        return f"❌ Error calculating benchmarks: {str(e)}"

    # --- Input-derived Values ---
    offered_budget = lives * budget_per_life
    true_cost = offered_budget / target_lr
    target_price = offered_budget * 1.05

    # --- Rule 1: Valid Range Check ---
    valid_range = abs(lives - benchmark_lives) / benchmark_lives <= 0.15

    # --- Expected LR Calculation ---
    expected_lr = None
    if historical_claims_per_life.lower() != "i don’t know":
        try:
            claims_per_life = float(historical_claims_per_life)
            expected_lr = claims_per_life / budget_per_life
        except:
            expected_lr = None

    # --- Sale Probability Calculation ---
    base_prob = (0.6 * logistic_output) + (0.4 * similarity_score) * adjustment
    sale_probability = base_prob

    if expected_lr is not None and expected_lr <= target_lr:
        sale_probability *= 1.05  # Rule 6

    if target_price > offered_budget:
        penalty = 1 - ((target_price - offered_budget) / offered_budget)
        sale_probability *= penalty  # Rule 7

    # --- Result Summary ---
    result = f"--- 📊 Benchmark Comparison for {package} in {region} ---\n"
    result += f"• Benchmark Lives: {benchmark_lives:.0f}\n"
    result += f"• Valid Range (±15%): {'✅ Yes' if valid_range else '❌ No'}\n"
    
    if avg_loss_ratio is not None:
        result += f"• Avg Loss Ratio: {avg_loss_ratio:.2f}\n"
    if avg_premium is not None:
        result += f"• Avg Premium per Life: {avg_premium:.2f} SAR\n"
    if avg_claims_per_life is not None:
        result += f"• Avg Claims per Life: {avg_claims_per_life:.2f} SAR\n"

    result += f"\n--- 💡 Offer Evaluation ---\n"
    result += f"• Lives Offered: {lives}\n"
    result += f"• Budget per Life: {budget_per_life:.2f} SAR\n"
    result += f"• Offered Budget: {offered_budget:.2f} SAR\n"
    result += f"• Target Price (+5%): {target_price:.2f} SAR\n"
    result += f"• True Cost per Life (Budget ÷ Target LR): {true_cost:.2f} SAR\n"

    if avg_premium and budget_per_life < avg_premium:
        result += "⚠️ Budget per life is below historical average.\n"

    if expected_lr is not None:
        result += f"• Expected LR (Claims ÷ Price): {expected_lr:.2f}\n"
        if expected_lr > target_lr:
            result += "❌ Expected LR exceeds Target LR — risk of unprofitable contract.\n"
        else:
            result += "✅ Expected LR is within acceptable target range.\n"
    else:
        result += "ℹ️ Historical claims per life not provided — cannot compute expected LR.\n"

    if target_price > offered_budget:
        result += "📉 Target price exceeds budget — recommend **downgrading package**.\n"

    result += f"\n--- 🧠 Sales Forecast ---\n"
    result += f"• Final Sale Probability: {sale_probability:.2%} (after adjustments)\n"

    return {
        "benchmark_lives": round(benchmark_lives, 2),
        "valid_range": valid_range,
        "avg_loss_ratio": round(avg_loss_ratio, 4) if avg_loss_ratio else None,
        "avg_premium": round(avg_premium, 2) if avg_premium else None,
        "avg_claims_per_life": round(avg_claims_per_life, 2) if avg_claims_per_life else None,
        "offered_lives": lives,
        "budget_per_life": budget_per_life,
        "offered_budget": round(offered_budget, 2),
        "target_price": round(target_price, 2),
        "true_cost_per_life": round(true_cost, 2),
        "expected_lr": round(expected_lr, 4) if expected_lr is not None else None,
        "recommend_downgrade": target_price > offered_budget,
        "final_probability": round(sale_probability, 4),
        "used_claims_fallback": historical_claims_per_life.lower() == "i don’t know"
    }


def agent_spreadsheet_data(llm):
    """Create a spreadsheet data agent using langgraph"""
    return create_react_agent(
        model=llm,
        prompt="""You are an intelligent insurance data analyst with access to historical insurance data.
        
        Your role:
        1. When users ask questions involving offer assessment, use the assess_new_offer tool
        2. Always reference actual data from the spreadsheet when providing numerical answers
        3. Be proactive - if someone mentions premiums, regions, packages, claims, etc., check the data
        4. Provide context and insights based on the historical data
        5. Help evaluate new business opportunities using historical benchmarks
        
        Use assess_new_offer for evaluating new insurance offers with all the required parameters.
        """,
        name="spreadsheet_data_assistant", 
        tools=[assess_new_offer]
    )