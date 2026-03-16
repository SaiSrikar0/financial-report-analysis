"""ETL extract layer with adapter/factory pattern."""

import json
import os
from abc import ABC, abstractmethod

import pandas as pd


class BaseExtractor(ABC):
    """Abstract extractor interface."""

    @abstractmethod
    def extract(self, source: str) -> pd.DataFrame:
        raise NotImplementedError


class CSVExtractor(BaseExtractor):
    def extract(self, source: str) -> pd.DataFrame:
        return pd.read_csv(source)


class ExcelExtractor(BaseExtractor):
    def extract(self, source: str) -> pd.DataFrame:
        return pd.read_excel(source)


class JSONExtractor(BaseExtractor):
    def extract(self, source: str) -> pd.DataFrame:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


class ExtractorFactory:
    @staticmethod
    def get(source_type: str) -> BaseExtractor:
        source_type = source_type.lower().strip()
        if source_type == "csv":
            return CSVExtractor()
        if source_type == "excel":
            return ExcelExtractor()
        if source_type in {"json", "api"}:
            return JSONExtractor()
        raise ValueError(f"Unsupported source type: {source_type}")


def extract_data(source_type="api", source_path=None):
    """
    Extract financial data and save a normalized raw JSON file.

    Args:
        source_type: 'api', 'json', 'csv', or 'excel'
        source_path: source file path; for 'api' this expects a JSON dump path
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    default_raw = os.path.join(data_dir, "financial_data_raw.json")
    input_path = source_path or default_raw

    if source_type.lower() in {"api", "json"} and not os.path.exists(input_path):
        raise FileNotFoundError(
            "For source_type 'api'/'json', provide a local JSON path or generate data via data_retrieval/retrieve_api.py"
        )

    extractor = ExtractorFactory.get(source_type)
    df = extractor.extract(input_path)

    records = df.to_dict("records")
    with open(default_raw, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Data extracted and saved to {default_raw}")
    print(f"Total records extracted: {len(records)}")
    return default_raw


if __name__ == "__main__":
    extract_data(source_type="json")
