# Auto-generated Python classes from archetype
# Loads dummy data from CSV file for testing

import csv
import os
from typing import Any, Dict, List, Optional


def create_dataset(csv_file=None):
    """Create a dataset from CSV dummy data"""
    if csv_file is None:
        # Default to dummy CSV file
        csv_file = 'generated/data.csv'
    
    if not os.path.exists(csv_file):
        print(f'CSV file not found: {csv_file}')
        print('Run "epsilon archetypes <dataset_id>" to generate dummy data')
        return DatasetWrapper([])
    
    # Load CSV data
    records = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    print(f'Loaded {len(records)} dummy records from CSV')
    return DatasetWrapper(records)


class DatasetWrapper:
    """Wrapper for dataset records with easy access"""
    def __init__(self, records):
        self.records = records if isinstance(records, list) else [records]
    
    def __len__(self):
        return len(self.records)
    
    def __iter__(self):
        for record in self.records:
            yield Root(record)
    
    def __getitem__(self, index):
        return Root(self.records[index])
    
    @property
    def first(self):
        return Root(self.records[0]) if self.records else None


class Demographics:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def gender(self):
        # Try direct access first (JSON format)
        if 'gender' in self._data:
            return self._data['gender']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.gender'):
                return v
        return None

    @property
    def age(self):
        # Try direct access first (JSON format)
        if 'age' in self._data:
            return self._data['age']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.age'):
                return v
        return None


class Diabd_Diabetes_Risk:
    def __init__(self, data):
        self._data = data if data else {}


class Vitals:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def diastolic_bp(self):
        # Try direct access first (JSON format)
        if 'diastolic_bp' in self._data:
            return self._data['diastolic_bp']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.diastolic_bp'):
                return v
        return None

    @property
    def pulse_rate(self):
        # Try direct access first (JSON format)
        if 'pulse_rate' in self._data:
            return self._data['pulse_rate']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.pulse_rate'):
                return v
        return None

    @property
    def systolic_bp(self):
        # Try direct access first (JSON format)
        if 'systolic_bp' in self._data:
            return self._data['systolic_bp']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.systolic_bp'):
                return v
        return None


class Anthropometry:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def height(self):
        # Try direct access first (JSON format)
        if 'height' in self._data:
            return self._data['height']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.height'):
                return v
        return None

    @property
    def bmi(self):
        # Try direct access first (JSON format)
        if 'bmi' in self._data:
            return self._data['bmi']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.bmi'):
                return v
        return None

    @property
    def weight(self):
        # Try direct access first (JSON format)
        if 'weight' in self._data:
            return self._data['weight']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.weight'):
                return v
        return None


class Labs:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def glucose(self):
        # Try direct access first (JSON format)
        if 'glucose' in self._data:
            return self._data['glucose']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.glucose'):
                return v
        return None


class History:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def hypertensive(self):
        # Try direct access first (JSON format)
        if 'hypertensive' in self._data:
            return self._data['hypertensive']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.hypertensive'):
                return v
        return None

    @property
    def family_diabetes(self):
        # Try direct access first (JSON format)
        if 'family_diabetes' in self._data:
            return self._data['family_diabetes']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.family_diabetes'):
                return v
        return None

    @property
    def cardiovascular_disease(self):
        # Try direct access first (JSON format)
        if 'cardiovascular_disease' in self._data:
            return self._data['cardiovascular_disease']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.cardiovascular_disease'):
                return v
        return None

    @property
    def family_hypertension(self):
        # Try direct access first (JSON format)
        if 'family_hypertension' in self._data:
            return self._data['family_hypertension']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.family_hypertension'):
                return v
        return None

    @property
    def stroke(self):
        # Try direct access first (JSON format)
        if 'stroke' in self._data:
            return self._data['stroke']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.stroke'):
                return v
        return None


class Outcome:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def diabetic(self):
        # Try direct access first (JSON format)
        if 'diabetic' in self._data:
            return self._data['diabetic']
        # Try flattened access (CSV format)
        for k, v in self._data.items():
            if k.endswith('.diabetic'):
                return v
        return None


class Root:
    def __init__(self, data):
        self._data = data if data else {}

    @property
    def demographics(self):
        # Handle nested field access with dot notation
        if 'demographics' in self._data and isinstance(self._data['demographics'], dict):
            return Demographics(self._data['demographics'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'demographics.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return Demographics(flattened)

    @property
    def diabd_diabetes_risk(self):
        # Handle nested field access with dot notation
        if 'diabd_diabetes_risk' in self._data and isinstance(self._data['diabd_diabetes_risk'], dict):
            return Diabd_Diabetes_Risk(self._data['diabd_diabetes_risk'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'diabd_diabetes_risk.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return Diabd_Diabetes_Risk(flattened)

    @property
    def vitals(self):
        # Handle nested field access with dot notation
        if 'vitals' in self._data and isinstance(self._data['vitals'], dict):
            return Vitals(self._data['vitals'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'vitals.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return Vitals(flattened)

    @property
    def anthropometry(self):
        # Handle nested field access with dot notation
        if 'anthropometry' in self._data and isinstance(self._data['anthropometry'], dict):
            return Anthropometry(self._data['anthropometry'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'anthropometry.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return Anthropometry(flattened)

    @property
    def labs(self):
        # Handle nested field access with dot notation
        if 'labs' in self._data and isinstance(self._data['labs'], dict):
            return Labs(self._data['labs'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'labs.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return Labs(flattened)

    @property
    def history(self):
        # Handle nested field access with dot notation
        if 'history' in self._data and isinstance(self._data['history'], dict):
            return History(self._data['history'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'history.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return History(flattened)

    @property
    def outcome(self):
        # Handle nested field access with dot notation
        if 'outcome' in self._data and isinstance(self._data['outcome'], dict):
            return Outcome(self._data['outcome'])
        # Handle flattened CSV data
        flattened = {}
        prefix = 'outcome.'
        for k, v in self._data.items():
            if k.startswith(prefix):
                nested_key = k[len(prefix):]
                flattened[nested_key] = v
        return Outcome(flattened)
