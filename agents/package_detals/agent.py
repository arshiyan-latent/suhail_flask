from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, Literal



class Policy(BaseModel):
    package_type: Literal["basic", "bronze", "silver", "gold","platinum"] = Field(
        default="basic",
        description="The name of the policy package. Must be one of the predefined insurance policy package."
    ) 



@tool("get_policy_package_details",args_schema=Policy)
def get_policy_package_details(package_type:str):
    """" Provide details about a specific policy package"""
    insurance_data = {
            "insurance_packages": [
                {
                    "package_type": "Basic",
                    "eligibility": "All employees",
                    "annual_limit": "SAR 500,000",
                    "maternity_limit": "Not covered",
                    "room_type": "General ward",
                    "network_class": "Ministry approved only",
                    "pre_existing_conditions": "Covered after 12 months",
                    "maternity": "Not covered",
                    "dental": "Not covered",
                    "optical": "Not covered",
                    "co_payment": {
                        "outpatient": "20%",
                        "inpatient": "0%",
                        "medications": "20%"
                    }
                },
                {
                    "package_type": "Bronze",
                    "eligibility": "All employees + family",
                    "annual_limit": "SAR 600,000",
                    "maternity_limit": "SAR 5,000",
                    "room_type": "Shared room",
                    "network_class": "Class C",
                    "pre_existing_conditions": "Covered after 12 months",
                    "maternity": "SAR 5,000",
                    "dental": "Emergency only",
                    "optical": "Not covered",
                    "co_payment": {
                        "outpatient": "15%",
                        "inpatient": "0%",
                        "medications": "15%"
                    }
                },
                {
                    "package_type": "Silver",
                    "eligibility": "All employees + family",
                    "annual_limit": "SAR 1,000,000",
                    "maternity_limit": "SAR 10,000",
                    "room_type": "Shared room",
                    "network_class": "Class B",
                    "pre_existing_conditions": "Covered",
                    "maternity": "SAR 10,000",
                    "dental": "Covered up to SAR 2,000",
                    "optical": "SAR 1,000 every 2 years",
                    "co_payment": {
                        "outpatient": "10%",
                        "inpatient": "0%",
                        "medications": "10%"
                    }
                },
                {
                    "package_type": "Gold",
                    "eligibility": "Senior management + family",
                    "annual_limit": "SAR 1,500,000",
                    "maternity_limit": "SAR 15,000",
                    "room_type": "Private room",
                    "network_class": "Class A",
                    "pre_existing_conditions": "Covered",
                    "maternity": "SAR 15,000",
                    "dental": "Covered up to SAR 3,000",
                    "optical": "SAR 1,500 every 2 years",
                    "co_payment": {
                        "outpatient": "10%",
                        "inpatient": "0%",
                        "medications": "10%"
                    }
                },
                {
                    "package_type": "Diamond",
                    "eligibility": "Top executives + family",
                    "annual_limit": "SAR 2,000,000",
                    "maternity_limit": "SAR 20,000",
                    "room_type": "VIP suite",
                    "network_class": "VIP",
                    "pre_existing_conditions": "Covered",
                    "maternity": "SAR 20,000",
                    "dental": "Covered up to SAR 5,000",
                    "optical": "SAR 2,000 every year",
                    "co_payment": {
                        "outpatient": "0%",
                        "inpatient": "0%",
                        "medications": "0%"
                    }
                }
            ]
        }
    filtered = [
            package for package in insurance_data["insurance_packages"]
            if package["package_type"].lower() == package_type.lower()
        ]
    return filtered



def agent_policy_package_details(llm):
    return create_react_agent(
        model = llm,
        prompt = "You are helpful agent that provide information about a specific insurance policy package",
        name = "policy_detail_assistant",
        tools=[get_policy_package_details]
    )







