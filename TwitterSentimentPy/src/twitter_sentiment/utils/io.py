import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional


def save_to_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: Indentation level for JSON formatting
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving JSON to: {file_path}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_from_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    print(f"Loading JSON from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_to_csv(data: List[Dict[str, Any]], file_path: Union[str, Path], 
                fieldnames: Optional[List[str]] = None) -> None:
    """
    Save a list of dictionaries to a CSV file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the CSV file
        fieldnames: List of field names for the CSV header. If None, the keys
                    of the first dictionary will be used.
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving CSV to: {file_path}")
    
    if not data:
        print("Warning: No data to save to CSV.")
        return
    
    # Use the keys of the first dictionary if fieldnames not provided
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Only include fields that are in fieldnames
            writer.writerow({k: v for k, v in row.items() if k in fieldnames})


def load_from_csv(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries representing rows in the CSV
    """
    file_path = Path(file_path)
    
    print(f"Loading CSV from: {file_path}")
    
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_text_file(text: str, file_path: Union[str, Path]) -> None:
    """
    Save text to a file.
    
    Args:
        text: Text content to save
        file_path: Path to save the text file
    """
    file_path = Path(file_path)
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def load_text_file(file_path: Union[str, Path]) -> str:
    """
    Load text from a file.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the text file
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()