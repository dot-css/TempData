"""
Patient records generator

Generates realistic patient data with demographic distributions and medical history correlations.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class PatientGenerator(BaseGenerator):
    """
    Generator for realistic patient records
    
    Creates patient datasets with demographic distributions, medical history
    correlations, realistic patient demographics, and health patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_demographic_distributions()
        self._setup_medical_distributions()
        self._setup_health_patterns()
        self._setup_insurance_patterns()
    
    def _setup_demographic_distributions(self):
        """Setup realistic demographic distributions for healthcare"""
        # Age distribution for healthcare (skewed toward n task 7.1
        data = {
            'patient_id': [f'PAT_{i:06d}' for i in range(rows)],
            'first_name': [self.faker.first_name() for _ in range(rows)],
            'last_name': [self.faker.last_name() for _ in range(rows)],
            'date_of_birth': [self.faker.date_of_birth() for _ in range(rows)],
            'gender': [self.faker.random_element(['M', 'F', 'Other']) for _ in range(rows)],
            'blood_type': [self.faker.random_element(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']) for _ in range(rows)]
        }
        
        return pd.DataFrame(data)