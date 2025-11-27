# tools/crm_tools.py
import json
import os
from typing import Dict, Any, List
from langchain.tools import tool

# Adjust paths to be relative to the project root, assuming tools/ is a subfolder
# of the project root. This ensures paths work when called from main.py or langgraph_workflow.py.
CUSTOMER_DB_PATH = "data/customers.json"
LEAD_DB_PATH = "data/leads.json"

def _load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads JSON data from a file."""
    try:
        # Construct absolute path to data files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        abs_data_path = os.path.join(project_root, file_path)

        if not os.path.exists(abs_data_path):
            print(f"❌ {file_path} not found at {abs_data_path}. Returning empty list.")
            return []
        with open(abs_data_path, 'r', encoding="utf-8") as f: # Explicitly set 'r' mode
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Warning: {file_path} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"⚠️ Warning: Error decoding JSON from {file_path}. Returning empty list.")
        return []
    except Exception as e:
        print(f"⚠️ An unexpected error occurred while loading {file_path}: {e}. Returning empty list.")
        return []

@tool
def get_customer_info(query: str) -> Dict[str, Any]:
    """
    Retrieves customer information from the CRM database based on ID, email, name, or policy ID.
    The query should contain a customer ID (e.g., 'CUST001'), email (e.g., 'john@example.com'),
    name (e.g., 'John Smith'), or policy ID (e.g., 'AUTO-001').
    Returns a dictionary of customer details if found, otherwise an empty dictionary.
    """
    customers = _load_json_data(CUSTOMER_DB_PATH)
    query_lower = query.strip().lower()

    for customer in customers:
        if customer.get("id", "").lower() == query_lower:
            return customer
        if customer.get("email", "").lower() == query_lower:
            return customer
        if customer.get("name", "").lower() == query_lower:
            return customer
        for policy in customer.get("policies", []):
            if policy.get("policy_id", "").lower() == query_lower:
                return customer
    return {}

@tool
def search_leads(criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Searches for leads in the lead database based on various criteria.
    Criteria should be a dictionary. Supported keys include:
    - 'score_min': minimum lead score (int)
    - 'interest': keyword in interest (str, case-insensitive)
    - 'area': geographic area (str, case-insensitive)
    - 'status': lead status (e.g., 'New', 'Contacted', 'Qualified') (str, case-insensitive)
    - 'name': part of the lead's name (str, case-insensitive)

    Example criteria: {{"score_min": 70, "interest": "auto", "status": "New"}}
    Returns a list of matching lead dictionaries. If no leads are found, returns an empty list.
    """
    leads = _load_json_data(LEAD_DB_PATH)
    matching_leads = []

    for lead in leads:
        match = True
        if 'score_min' in criteria and lead.get('score', 0) < criteria['score_min']:
            match = False
        if 'interest' in criteria and criteria['interest'].lower() not in lead.get('interest', '').lower():
            match = False
        if 'area' in criteria and criteria['area'].lower() != lead.get('area', '').lower():
            match = False
        if 'status' in criteria and criteria['status'].lower() != lead.get('status', '').lower():
            match = False
        if 'name' in criteria and criteria['name'].lower() not in lead.get('name', '').lower():
            match = False

        if match:
            matching_leads.append(lead)
    return matching_leads

if __name__ == "__main__":
    print("--- Testing get_customer_info tool directly ---")
    
    print("\nQuery: John Smith")
    result = get_customer_info("John Smith")
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    print("\n--- Testing search_leads tool directly ---")
    
    print("\nQuery: Find leads with score above 80 interested in auto insurance")
    result = search_leads({"score_min": 80, "interest": "auto"})
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    print("\nQuery: Show me qualified leads in Texas")
    result = search_leads({"status": "Qualified", "area": "Texas"})
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\nQuery: Find leads named 'John'")
    result = search_leads({"name": "John"})
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    print("\nQuery: Find leads with score below 50")
    result = search_leads({"score_min": 50})
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")