import pandas as pd

import os


df = pd.read_excel('agents/spreadsheet/sheets/test_historical.xlsx')

#fin average earned exposure of all basic packages in central region
def average_earned_exposure(region, package_type):
    """
    Calculate the average earned exposure for a specific region and package type.
    
    Args:
        region (str): The region to filter by.
        package_type (str): The type of package to filter by.
        
    Returns:
        float: The average earned exposure, or None if no data is found.
    """
    filtered_df = df[(df['Contract Region'] == region) & (df['Product Package'] == package_type)]
    
    if not filtered_df.empty:
        return filtered_df['Earned Exposure'].mean()
    
    return None

def average_budget_per_life(region, package_type):
    """
    Calculate the average budget per life for a specific region and package type.
    
    Args:
        region (str): The region to filter by.
        package_type (str): The type of package to filter by.
        
    Returns:
        float: The average budget per life, or None if no data is found.
    """
    filtered_df = df[(df['Contract Region'] == region) & (df['Product Package'] == package_type)]
    
    if not filtered_df.empty:
        return filtered_df['Average Premium'].mean()
    
    return None


#test the function
if __name__ == "__main__":
    region = 'Central'
    package_type = 'A. Basic Package'
    
    average_exposure = average_earned_exposure(region, package_type)
    
    if average_exposure is not None:
        print(f"The average earned exposure for {package_type} packages in {region} region is: {average_exposure}")
    else:
        print(f"No data found for {package_type} packages in {region} region.")

    average_budget = average_budget_per_life(region, package_type)

    if average_budget is not None:
        print(f"The average budget per life for {package_type} packages in {region} region is: {average_budget}")
    else:
        print(f"No data found for {package_type} packages in {region} region.")