# tools/crm_tool.py
import json
import os
from typing import Dict, Any, List, Optional, Union
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

CUSTOMER_DB_PATH = "data/customers.json"
LEAD_DB_PATH = "data/leads.json"

def _load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads JSON data from a file."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        abs_data_path = os.path.join(project_root, file_path)

        if not os.path.exists(abs_data_path):
            print(f"❌ {file_path} not found at {abs_data_path}. Returning empty list.")
            return []
        with open(abs_data_path, 'r', encoding="utf-8") as f:
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


# ✅ NEW APPROACH: Single string parameter that we parse ourselves
@tool
def search_leads(criteria: str) -> List[Dict[str, Any]]:
    """
    Searches for leads in the lead database based on various criteria.
    
    Input should be a JSON string with search criteria. Supported keys:
    - score_min: minimum lead score (integer, e.g., 80)
    - interest: interest keyword (string, e.g., "auto", "life", "home")
    - area: geographic area (string, e.g., "Texas", "California")
    - status: lead status (string, e.g., "New", "Contacted", "Qualified", "Lost")
    - name: part of lead's name (string)
    
    Example inputs:
    - '{"status": "Qualified", "area": "Texas"}'
    - '{"score_min": 80, "interest": "auto"}'
    - '{"name": "John"}'
    
    Returns a list of matching lead dictionaries.
    """
    try:
        # Parse JSON string to dict
        if isinstance(criteria, str):
            criteria_dict = json.loads(criteria)
        elif isinstance(criteria, dict):
            criteria_dict = criteria
        else:
            print(f"⚠️ Invalid criteria type: {type(criteria)}. Expected string or dict.")
            return []
        
        print(f"🔍 Parsed search criteria: {criteria_dict}")
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Failed to parse criteria JSON: {e}")
        return []
    
    # Perform the actual search
    leads = _load_json_data(LEAD_DB_PATH)
    matching_leads = []

    for lead in leads:
        match = True
        
        # Check score_min
        if 'score_min' in criteria_dict:
            if lead.get('score', 0) < criteria_dict['score_min']:
                match = False
        
        # Check interest
        if 'interest' in criteria_dict:
            if criteria_dict['interest'].lower() not in lead.get('interest', '').lower():
                match = False
        
        # Check area
        if 'area' in criteria_dict:
            if criteria_dict['area'].lower() != lead.get('area', '').lower():
                match = False
        
        # Check status
        if 'status' in criteria_dict:
            if criteria_dict['status'].lower() != lead.get('status', '').lower():
                match = False
        
        # Check name
        if 'name' in criteria_dict:
            if criteria_dict['name'].lower() not in lead.get('name', '').lower():
                match = False

        if match:
            matching_leads.append(lead)
    
    print(f"✅ Found {len(matching_leads)} matching leads")
    return matching_leads


if __name__ == "__main__":
    print("--- Testing get_customer_info tool directly ---")
    
    print("\nQuery: John Smith")
    result = get_customer_info.invoke("John Smith")
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    print("\n--- Testing search_leads tool directly ---")
    
    print("\nTest 1: Find leads with score above 80 interested in auto insurance")
    result = search_leads.invoke('{"score_min": 80, "interest": "auto"}')
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    print("\nTest 2: Show me qualified leads in Texas")
    result = search_leads.invoke('{"status": "Qualified", "area": "Texas"}')
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\nTest 3: Find leads named 'John'")
    result = search_leads.invoke('{"name": "John"}')
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

    print("\nTest 4: Find leads with score >= 50")
    result = search_leads.invoke('{"score_min": 50}')
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    print("\nTest 5: Test with dict input (edge case)")
    result = search_leads.invoke({"status": "New", "area": "California"})
    print(f"Result: {json.dumps(result, indent=2, ensure_ascii=False)}")