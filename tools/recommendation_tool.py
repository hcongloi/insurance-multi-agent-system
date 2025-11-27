# tools/recommendation_tools.py
from typing import Dict, Any, List
from langchain.tools import tool
import json

@tool
def generate_insurance_recommendations(customer_profile_json: str, available_products_kb: str) -> str:
    """
    Generates insurance product recommendations based on a customer's profile (JSON string) and available product knowledge.
    The customer_profile_json should be a JSON string containing customer details (e.g., '{"id": "CUST001", "name": "John Smith", ...}').
    available_products_kb should be a string containing information about insurance products from the knowledge base.
    """
    try:
        customer_profile = json.loads(customer_profile_json)
    except json.JSONDecodeError:
        return "Invalid customer profile JSON format provided for recommendations."
    
    if not customer_profile:
        return "No customer profile provided to generate recommendations."
    if not available_products_kb:
        return "No product knowledge base information provided for recommendations."

    recommendations = []
    customer_name = customer_profile.get("name", "customer")
    customer_policies_raw = customer_profile.get("policies", [])
    customer_policy_types = [p["type"] for p in customer_policies_raw]
    
    recommendations.append(f"Based on {customer_name}'s profile:")
    
    if customer_policy_types:
        recommendations.append(f"- Currently holds: {', '.join(customer_policy_types)}")
    else:
        recommendations.append("- Does not currently hold any active policies with us.")

    recommendations.append("\nPotential recommendations:")
    
    if "Auto Insurance" not in customer_policy_types:
        if "auto insurance" in available_products_kb.lower():
            recommendations.append("- Auto Insurance: To protect against financial loss in case of accidents.")
    if "Home Insurance" not in customer_policy_types and customer_profile.get("address"):
        if "home insurance" in available_products_kb.lower():
            recommendations.append("- Home Insurance: Essential for property owners to protect their residence and belongings.")
    if "Life Insurance" not in customer_policy_types:
        if "life insurance" in available_products_kb.lower():
            recommendations.append("- Life Insurance: To provide financial security for loved ones in the future.")
    if "Health Insurance" not in customer_policy_types:
        if "health insurance" in available_products_kb.lower():
            recommendations.append("- Health Insurance: For covering medical expenses and ensuring access to quality healthcare.")
            
    if len(recommendations) <= 3:
        recommendations.append("- You might be interested in exploring our general range of insurance products such as Health, Life, or Travel insurance.")
        recommendations.append("  For more detailed information, please specify your interests.")
        
    return "\n".join(recommendations)

if __name__ == "__main__":
    print("--- Testing Recommendation Tool ---")
    
    mock_customer_john_smith = {
        "id": "CUST001",
        "name": "John Smith",
        "email": "john@example.com",
        "phone": "+1-555-1234",
        "address": "123 Main St, Anytown, USA",
        "policies": [
            {"policy_id": "AUTO-001", "type": "Auto Insurance", "status": "Active", "premium": 1200}
        ],
        "history": "Claimed fender bender in 2022."
    }
    
    mock_customer_emily_brown = {
        "id": "CUST005",
        "name": "Emily Brown",
        "email": "emily@example.com",
        "phone": "+1-555-7890",
        "address": "222 Birch Rd, Bigcity, USA",
        "policies": [],
        "history": "Inquired about pet insurance but did not purchase."
    }

    mock_kb_content = """
    # Auto Insurance
    ... details about auto insurance ...
    # Home Insurance
    ... details about home insurance ...
    # Life Insurance
    ... details about life insurance ...
    # Health Insurance
    ... details about health insurance ...
    """
    
    print("\n--- John Smith (has Auto Insurance) ---")
    recommendations_john = generate_insurance_recommendations(json.dumps(mock_customer_john_smith), mock_kb_content)
    print(recommendations_john)
    
    print("\n--- Emily Brown (no policies) ---")
    recommendations_emily = generate_insurance_recommendations(json.dumps(mock_customer_emily_brown), mock_kb_content)
    print(recommendations_emily)

    print("\n--- John Smith, NO KB ---")
    recommendations_no_kb = generate_insurance_recommendations(json.dumps(mock_customer_john_smith), "")
    print(recommendations_no_kb)

    print("\n--- No Customer Profile ---")
    recommendations_no_customer = generate_insurance_recommendations("{}", mock_kb_content)
    print(recommendations_no_customer)