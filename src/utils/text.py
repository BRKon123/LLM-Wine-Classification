from typing import Dict, List, Any

def format_dict_to_string(data: Dict[str, Any], keys: List[str] = None) -> str:
    """
    Convert a dictionary to a formatted string with keys and values on separate lines.
    
    Args:
        data: Dictionary containing the data
        keys: List of keys to include. If None, includes all keys
        
    Returns:
        Formatted string with keys and values on separate lines
    """
    if keys is None:
        keys = data.keys()
    
    # Filter out keys that don't exist in the data
    valid_keys = [k for k in keys if k in data]
    
    # Build the formatted string
    formatted_parts = [
        f"{key}:\n{str(data[key])}\n"
        for key in valid_keys
    ]
    
    return "\n".join(formatted_parts)
