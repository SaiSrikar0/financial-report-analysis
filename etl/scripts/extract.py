#import necessary libraries
import os
import json
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime

# Extractor Adapter Interface
class DataExtractor(ABC):
    @abstractmethod
    def extract(self):
        pass

# CSV Extractor Adapter
class CSVExtractor(DataExtractor):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def extract(self):
        df = pd.read_csv(self.file_path)
        return df.to_dict('records')

# Excel Extractor Adapter
class ExcelExtractor(DataExtractor):
    def __init__(self, file_path, sheet_name=0):
        self.file_path = file_path
        self.sheet_name = sheet_name
    
    def extract(self):
        df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        return df.to_dict('records')

# API Extractor Adapter (Template for future implementation)
class APIExtractor(DataExtractor):
    def __init__(self, api_url, api_key=None):
        self.api_url = api_url
        self.api_key = api_key
    
    def extract(self):
        # TODO: Implement API call logic
        # Example: requests.get(self.api_url, headers={'Authorization': f'Bearer {self.api_key}'})
        # For now, return sample data
        sample_data = [
            {"date": "2026-01-21", "ticker": "AAPL", "open_price": 150.0, "close_price": 155.0},
            {"date": "2026-01-21", "ticker": "GOOGL", "open_price": 2800.0, "close_price": 2850.0}
        ]
        return sample_data

#function to extract and save data
def extract_data(source_type='api', source_path=None):
    """
    Extract financial data using appropriate adapter and save as JSON
    
    Args:
        source_type: 'api', 'csv', or 'excel'
        source_path: path to file (for csv/excel) or API URL
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(data_dir, exist_ok = True)
    
    # Select appropriate extractor adapter
    if source_type == 'csv':
        extractor = CSVExtractor(source_path)
    elif source_type == 'excel':
        extractor = ExcelExtractor(source_path)
    elif source_type == 'api':
        extractor = APIExtractor(source_path or 'https://api.example.com/financial-data')
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    
    # Extract data using adapter
    data = extractor.extract()
    
    # Save as raw JSON
    raw_path = os.path.join(data_dir, 'financial_data_raw.json')
    with open(raw_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data extracted and saved to {raw_path}")
    print(f"Total records extracted: {len(data)}")
    return raw_path

if __name__ == "__main__":
    extract_data()
