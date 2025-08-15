from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import pandas as pd
import os

def load_historical_data():
    """Load the historical data with proper error handling"""

    file_path = os.path.join(os.getcwd(), 'agents', 'spreadsheet', 'sheets', 'test_historical.xlsx')

    print(f"Resolved path: {file_path}")

    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    else:
        print(f"Warning: {file_path} not found")
        return pd.DataFrame()  # Return empty DataFrame if file not found
    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     return pd.DataFrame()

df = load_historical_data()

# Below is the rules covered till 10th row in demo features..
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

# Package hierarchy from highest to lowest
PACKAGE_HIERARCHY = {
    "diamond": 5,
    "gold": 4,
    "silver": 3,
    "bronze": 2,
    "basic": 1
}

def get_alternative_package(current_package: str) -> str:
    """Get the next lower package in the hierarchy"""
    current_level = PACKAGE_HIERARCHY.get(current_package.lower())
    if not current_level or current_level == 1:  # If current is basic or unknown
        return None
    
    # Find the package one level down
    for pkg, level in PACKAGE_HIERARCHY.items():
        if level == current_level - 1:
            return pkg.capitalize()
    return None

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
) -> dict:
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
    offered_budget = lives * budget_per_life  # Total budget for all lives
    offered_budget_per_life = budget_per_life  # Budget per life
    # true_cost = budget_per_life / target_lr  # Calculate per life first
    # target_price = true_cost * 1.05  # Add 5% contingency to the true cost (per life)

    # --- Rule 1: Valid Range Check ---
    valid_range = abs(lives - benchmark_lives) / benchmark_lives <= 0.15

    # --- Expected LR Calculation ---
    expected_lr = None
    claims_input = str(historical_claims_per_life).strip().lower().replace("’", "'")
    if claims_input != "i don't know":
        try:
            claims_per_life = float(claims_input)
            # expected_lr = claims_per_life / budget_per_life
        except Exception:
            #expected_lr = None
            claims_per_life = None

    if claims_per_life is None:
        claims_per_life = avg_claims_per_life  # package/region-specific

    true_cost = None
    target_price = None
    expected_lr = None
    if (claims_per_life is not None) and (target_lr > 0):
        true_cost = claims_per_life / target_lr
        target_price = true_cost * 1.05
        expected_lr = claims_per_life / target_price  # compare to recommended price
    # --- Sale Probability Calculation ---
    base_prob = (0.6 * logistic_output) + (0.4 * similarity_score) * adjustment
    sale_probability = base_prob

    if expected_lr is not None and expected_lr <= target_lr:
        sale_probability *= 1.05  # Rule 6

    # Use per-life budget for variance/penalty
    if target_price > offered_budget_per_life:
        variance = (target_price - offered_budget_per_life) / max(offered_budget_per_life, 1e-9)
        penalty = max(0.0, 1 - variance)
        sale_probability *= penalty  # Rule 7
  
    # Calculate alternative package if current package exceeds budget
    alternative_results = None
    fallback_results = None
    
    print(f"\n=== Package Comparison Debug ===")
    print(f"Current package: {package}")
    print(f"Target price per life: {target_price:.2f} SAR")
    print(f"Budget per life: {offered_budget_per_life:.2f} SAR")
    print(f"Total target price: {target_price * lives:,.2f} SAR")
    print(f"Total offered budget: {offered_budget:,.2f} SAR")
    print(f"Over budget? {target_price > offered_budget_per_life}")
    
    if target_price > offered_budget_per_life:
        alt_package = get_alternative_package(package)
        print(f"Alternative package suggested: {alt_package}")
        if alt_package:
            try:
                alt_matches = df[df[package_col].str.lower().str.contains(alt_package.lower(), na=False)]
                if not alt_matches.empty:
                    alt_claims_per_life = (alt_matches[claims_col].sum() / alt_matches[lives_col].sum()) if claims_col else None
                    alt_true_cost = alt_target_price = alt_expected_lr = None
                    if (alt_claims_per_life is not None) and (target_lr > 0):
                        alt_true_cost    = alt_claims_per_life / target_lr
                        alt_target_price = alt_true_cost * 1.05
                        alt_expected_lr  = alt_claims_per_life / alt_target_price

                    alt_sale_prob = base_prob
                    if alt_expected_lr and alt_expected_lr <= target_lr:
                        alt_sale_prob *= 1.05
                    if alt_target_price > offered_budget_per_life:
                        variance = (alt_target_price - offered_budget_per_life) / max(offered_budget_per_life, 1e-9)
                        penalty = max(0.0, 1 - variance)
                        alt_sale_prob *= penalty
                    
                    alternative_results = {
                        "package": alt_package,
                        "price_per_life": alt_target_price,
                        "fits_budget": alt_target_price <= offered_budget_per_life,
                        "expected_lr": alt_expected_lr,
                        "sale_probability": alt_sale_prob
                    }
                    print(f"\n=== Alternative Package Details ===")
                    print(f"Package: {alt_package}")
                    print(f"Price per life: {alt_target_price:.2f} SAR")
                    print(f"Fits budget? {alt_target_price <= offered_budget_per_life}")
                    
                    # If alternative is still over budget, calculate Basic package as fallback
                    if alt_target_price > offered_budget_per_life and alt_package.lower() != "basic":
                        basic_matches = df[df[package_col].str.lower().str.contains("basic", na=False)]
                        if not basic_matches.empty:
                            basic_claims_per_life = (basic_matches[claims_col].sum() / basic_matches[lives_col].sum()) if claims_col else None
                            basic_true_cost = basic_target_price = basic_expected_lr = None
                            if (basic_claims_per_life is not None) and (target_lr > 0):
                                basic_true_cost    = basic_claims_per_life / target_lr
                                basic_target_price = basic_true_cost * 1.05
                                basic_expected_lr  = basic_claims_per_life / basic_target_price

                            basic_sale_prob = base_prob
                            if basic_expected_lr and basic_expected_lr <= target_lr:
                                basic_sale_prob *= 1.05
                            
                            fallback_results = {
                                "package": "Basic",
                                "price_per_life": basic_target_price,
                                "fits_budget": basic_target_price <= offered_budget_per_life,
                                "expected_lr": basic_expected_lr,
                                "sale_probability": basic_sale_prob
                            }
                            print(f"\n=== Fallback Package Details ===")
                            print(f"Basic package price per life: {basic_target_price:.2f} SAR")
                            print(f"Fits budget? {basic_target_price <= offered_budget_per_life}")
            except Exception as e:
                print(f"Error calculating alternative package: {str(e)}")
    # --- Result Summary ---
    print("\n=== Generating Comparison Table ===")
    print(f"Number of packages to display: {1 + bool(alternative_results) + bool(fallback_results)}")
    
    result = "--- 📊 Package Comparison Table ---\n"
    result += f"{'Package':<15} {'Price/Life':<15} {'Fits Budget':<15} {'Exp. LR':<15} {'Sale Prob.':<15}\n"
    result += f"{'-' * 75}\n"
    
    # Current package
    result += f"{package:<15} {target_price:,.2f} SAR     {'✅' if target_price <= offered_budget_per_life else '❌':<15} "
    result += f"{expected_lr*100:.1f}%{' '*10 if expected_lr else 'N/A':<15} {sale_probability*100:.1f}%\n"
    
    # Alternative package
    if alternative_results:
        result += f"{alternative_results['package']:<15} {alternative_results['price_per_life']:,.2f} SAR     "
        result += f"{'✅' if alternative_results['fits_budget'] else '❌':<15} "
        result += f"{alternative_results['expected_lr']*100:.1f}%{' '*10 if alternative_results['expected_lr'] else 'N/A':<15} "
        result += f"{alternative_results['sale_probability']*100:.1f}%\n"
    
    # Fallback package
    if fallback_results:
        result += f"{fallback_results['package']:<15} {fallback_results['price_per_life']:,.2f} SAR     "
        result += f"{'✅' if fallback_results['fits_budget'] else '❌':<15} "
        result += f"{fallback_results['expected_lr']*100:.1f}%{' '*10 if fallback_results['expected_lr'] else 'N/A':<15} "
        result += f"{fallback_results['sale_probability']*100:.1f}%\n"
    
    result += f"\n--- 📈 Additional Information ---\n"
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

    # Use per-life comparison for message as well
    if target_price > offered_budget_per_life:
        result += "📉 Target price exceeds budget — recommend **downgrading package**.\n"

    result += f"\n--- 🧠 Sales Forecast ---\n"
    result += f"• Final Sale Probability: {sale_probability:.2%} (after adjustments)\n" 
    
    response = {
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
        # Use per-life recommendation
        "recommend_downgrade": target_price > offered_budget_per_life,
        "final_probability": round(sale_probability, 4),
        # Normalize fallback detection
        "used_claims_fallback": claims_input == "i don't know",
        # "comparison_table": result,
        "packages": [
            {
                "name": package,
                "price_per_life": round(target_price, 2),
                "fits_budget": target_price <= offered_budget_per_life,
                "expected_lr": round(expected_lr, 4) if expected_lr is not None else None,
                "sale_probability": round(sale_probability, 4)
            }
        ]
    }
   
    if alternative_results:
        response["packages"].append({
            "name": alternative_results["package"],
            "price_per_life": round(alternative_results["price_per_life"], 2),
            "fits_budget": alternative_results["fits_budget"],
            "expected_lr": round(alternative_results["expected_lr"], 4) if alternative_results["expected_lr"] is not None else None,
            "sale_probability": round(alternative_results["sale_probability"], 4)
        })

    if fallback_results:
        response["packages"].append({
            "name": fallback_results["package"],
            "price_per_life": round(fallback_results["price_per_life"], 2),
            "fits_budget": fallback_results["fits_budget"],
            "expected_lr": round(fallback_results["expected_lr"], 4) if fallback_results["expected_lr"] is not None else None,
            "sale_probability": round(fallback_results["sale_probability"], 4)
        })
    
    return response


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
        
        When presenting multiple package options:
        - Always show all available packages in the comparison
        - Include the requested package even if over budget
        - Show the next lower package as an alternative
        - Show Basic package as a fallback when needed
        
        Use assess_new_offer for evaluating new insurance offers with all the required parameters.
        """,
        name="spreadsheet_data_assistant", 
        tools=[assess_new_offer]
    )
