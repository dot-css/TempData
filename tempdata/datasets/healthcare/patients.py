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
        # Age distribution for healthcare (skewed toward older patients)
        self.age_distribution = {
            'pediatric': {'range': (0, 17), 'weight': 0.15},      # Children
            'young_adult': {'range': (18, 35), 'weight': 0.20},   # Young adults
            'middle_aged': {'range': (36, 55), 'weight': 0.25},   # Middle aged
            'senior': {'range': (56, 75), 'weight': 0.25},        # Seniors
            'elderly': {'range': (76, 95), 'weight': 0.15}        # Elderly
        }
        
        # Gender distribution (slightly more female patients)
        self.gender_distribution = {
            'M': 0.47,
            'F': 0.52,
            'Other': 0.01
        }
        
        # Blood type distribution (realistic frequencies)
        self.blood_type_distribution = {
            'O+': 0.374,
            'A+': 0.357,
            'B+': 0.085,
            'AB+': 0.034,
            'O-': 0.066,
            'A-': 0.063,
            'B-': 0.015,
            'AB-': 0.006
        }
        
        # Ethnicity distribution (can be adjusted per country)
        self.ethnicity_distribution = {
            'Caucasian': 0.60,
            'Hispanic/Latino': 0.18,
            'African American': 0.13,
            'Asian': 0.06,
            'Native American': 0.02,
            'Other': 0.01
        }
    
    def _setup_medical_distributions(self):
        """Setup medical condition distributions by age group"""
        # Common conditions by age group with realistic prevalence
        self.conditions_by_age = {
            'pediatric': {
                'Asthma': 0.08,
                'ADHD': 0.05,
                'Allergies': 0.12,
                'Autism Spectrum Disorder': 0.02,
                'None': 0.73
            },
            'young_adult': {
                'Anxiety': 0.15,
                'Depression': 0.12,
                'Asthma': 0.06,
                'Allergies': 0.10,
                'Migraine': 0.08,
                'None': 0.49
            },
            'middle_aged': {
                'Hypertension': 0.25,
                'Diabetes Type 2': 0.12,
                'High Cholesterol': 0.20,
                'Anxiety': 0.18,
                'Depression': 0.15,
                'Arthritis': 0.08,
                'None': 0.02
            },
            'senior': {
                'Hypertension': 0.45,
                'Diabetes Type 2': 0.22,
                'High Cholesterol': 0.35,
                'Arthritis': 0.25,
                'Heart Disease': 0.15,
                'Osteoporosis': 0.12,
                'None': 0.01
            },
            'elderly': {
                'Hypertension': 0.65,
                'Diabetes Type 2': 0.28,
                'Heart Disease': 0.30,
                'Arthritis': 0.40,
                'Osteoporosis': 0.25,
                'Dementia': 0.15,
                'COPD': 0.12,
                'None': 0.005
            }
        }
        
        # Medication patterns by condition
        self.medications_by_condition = {
            'Hypertension': ['Lisinopril', 'Amlodipine', 'Metoprolol', 'Losartan'],
            'Diabetes Type 2': ['Metformin', 'Insulin', 'Glipizide', 'Januvia'],
            'High Cholesterol': ['Atorvastatin', 'Simvastatin', 'Rosuvastatin'],
            'Anxiety': ['Sertraline', 'Alprazolam', 'Lorazepam', 'Escitalopram'],
            'Depression': ['Sertraline', 'Fluoxetine', 'Escitalopram', 'Bupropion'],
            'Asthma': ['Albuterol', 'Fluticasone', 'Montelukast', 'Budesonide'],
            'Arthritis': ['Ibuprofen', 'Naproxen', 'Methotrexate', 'Prednisone'],
            'Heart Disease': ['Aspirin', 'Metoprolol', 'Lisinopril', 'Clopidogrel'],
            'ADHD': ['Adderall', 'Ritalin', 'Concerta', 'Vyvanse'],
            'Migraine': ['Sumatriptan', 'Topiramate', 'Propranolol', 'Amitriptyline']
        }
    
    def _setup_health_patterns(self):
        """Setup health patterns and risk factors"""
        # BMI distribution by age group
        self.bmi_patterns = {
            'pediatric': {'mean': 18.5, 'std': 3.0, 'min': 12.0, 'max': 35.0},
            'young_adult': {'mean': 24.5, 'std': 4.5, 'min': 16.0, 'max': 45.0},
            'middle_aged': {'mean': 27.2, 'std': 5.2, 'min': 18.0, 'max': 50.0},
            'senior': {'mean': 28.1, 'std': 5.8, 'min': 18.0, 'max': 50.0},
            'elderly': {'mean': 26.8, 'std': 5.5, 'min': 16.0, 'max': 45.0}
        }
        
        # Smoking patterns by age group
        self.smoking_patterns = {
            'pediatric': 0.0,
            'young_adult': 0.15,
            'middle_aged': 0.18,
            'senior': 0.12,
            'elderly': 0.08
        }
        
        # Alcohol consumption patterns
        self.alcohol_patterns = {
            'pediatric': 0.0,
            'young_adult': 0.65,
            'middle_aged': 0.58,
            'senior': 0.45,
            'elderly': 0.30
        }
        
        # Emergency contact relationship distribution
        self.emergency_contact_relationships = {
            'Spouse': 0.35,
            'Parent': 0.25,
            'Child': 0.20,
            'Sibling': 0.10,
            'Friend': 0.05,
            'Other Relative': 0.05
        }
    
    def _setup_insurance_patterns(self):
        """Setup insurance and healthcare access patterns"""
        # Insurance type distribution
        self.insurance_distribution = {
            'Private': 0.55,
            'Medicare': 0.18,
            'Medicaid': 0.15,
            'Military/VA': 0.05,
            'Self-Pay': 0.07
        }
        
        # Primary care provider assignment (most patients have one)
        self.has_primary_care = 0.85
        
        # Preferred language distribution
        self.language_distribution = {
            'English': 0.78,
            'Spanish': 0.13,
            'Chinese': 0.03,
            'French': 0.02,
            'German': 0.01,
            'Other': 0.03
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate patient records dataset
        
        Args:
            rows: Number of patient records to generate
            **kwargs: Additional parameters (country, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated patient data with realistic patterns
        """
        country = kwargs.get('country', 'global')
        include_sensitive = kwargs.get('include_sensitive', False)  # For privacy compliance
        
        data = []
        
        for i in range(rows):
            # Generate demographic information
            age_group = self._select_age_group()
            age = self._generate_age_for_group(age_group)
            gender = self._select_gender()
            blood_type = self._select_blood_type()
            ethnicity = self._select_ethnicity()
            
            # Generate personal information
            if gender == 'M':
                first_name = self.faker.first_name_male()
            elif gender == 'F':
                first_name = self.faker.first_name_female()
            else:
                first_name = self.faker.first_name()
            
            last_name = self.faker.last_name()
            date_of_birth = self._calculate_birth_date(age)
            
            # Generate contact information
            phone = self.faker.phone_number()
            email = self._generate_email(first_name, last_name) if self.faker.random.random() < 0.75 else None
            
            # Generate address
            address = self._generate_address(country)
            
            # Generate medical information
            medical_conditions = self._generate_medical_conditions(age_group, gender)
            medications = self._generate_medications(medical_conditions)
            allergies = self._generate_allergies()
            
            # Generate health metrics
            health_metrics = self._generate_health_metrics(age_group, gender, medical_conditions)
            
            # Generate insurance and care information
            insurance_info = self._generate_insurance_info(age_group)
            primary_care_provider = self._generate_primary_care_provider()
            
            # Generate emergency contact
            emergency_contact = self._generate_emergency_contact(age_group, gender)
            
            # Generate visit history
            visit_history = self._generate_visit_history(age_group, medical_conditions)
            
            # Generate preferred language
            preferred_language = self._select_preferred_language()
            
            patient = {
                'patient_id': f'PAT_{i+1:06d}',
                'first_name': first_name,
                'last_name': last_name,
                'date_of_birth': date_of_birth,
                'age': age,
                'age_group': age_group,
                'gender': gender,
                'blood_type': blood_type,
                'ethnicity': ethnicity,
                'phone': phone,
                'email': email,
                'address_line1': address['street'],
                'city': address['city'],
                'state_province': address['state'],
                'postal_code': address['postal_code'],
                'country': address['country'],
                'preferred_language': preferred_language,
                'medical_conditions': ', '.join(medical_conditions) if medical_conditions else 'None',
                'current_medications': ', '.join(medications) if medications else 'None',
                'allergies': ', '.join(allergies) if allergies else 'None',
                'height_cm': health_metrics['height_cm'],
                'weight_kg': health_metrics['weight_kg'],
                'bmi': health_metrics['bmi'],
                'blood_pressure_systolic': health_metrics['bp_systolic'],
                'blood_pressure_diastolic': health_metrics['bp_diastolic'],
                'smoking_status': health_metrics['smoking_status'],
                'alcohol_consumption': health_metrics['alcohol_consumption'],
                'insurance_type': insurance_info['type'],
                'insurance_id': insurance_info['id'],
                'primary_care_provider': primary_care_provider,
                'emergency_contact_name': emergency_contact['name'],
                'emergency_contact_relationship': emergency_contact['relationship'],
                'emergency_contact_phone': emergency_contact['phone'],
                'last_visit_date': visit_history['last_visit'],
                'total_visits_last_year': visit_history['visits_last_year'],
                'registration_date': visit_history['registration_date']
            }
            
            # Add sensitive information only if requested (for privacy compliance)
            if include_sensitive:
                patient['ssn'] = self.faker.ssn()
                patient['medical_record_number'] = f'MRN{self.faker.random_int(100000, 999999)}'
            
            data.append(patient)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _select_age_group(self) -> str:
        """Select age group based on healthcare distribution"""
        choices = list(self.age_distribution.keys())
        weights = [self.age_distribution[group]['weight'] for group in choices]
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _generate_age_for_group(self, age_group: str) -> int:
        """Generate age within the specified group range"""
        min_age, max_age = self.age_distribution[age_group]['range']
        return self.faker.random_int(min_age, max_age)
    
    def _select_gender(self) -> str:
        """Select gender based on distribution"""
        choices = list(self.gender_distribution.keys())
        weights = list(self.gender_distribution.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _select_blood_type(self) -> str:
        """Select blood type based on realistic distribution"""
        choices = list(self.blood_type_distribution.keys())
        weights = list(self.blood_type_distribution.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _select_ethnicity(self) -> str:
        """Select ethnicity based on distribution"""
        choices = list(self.ethnicity_distribution.keys())
        weights = list(self.ethnicity_distribution.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _calculate_birth_date(self, age: int) -> date:
        """Calculate birth date based on age"""
        today = datetime.now().date()
        birth_year = today.year - age
        
        # Add some randomness to the birth date within the year
        start_date = date(birth_year, 1, 1)
        end_date = date(birth_year, 12, 31)
        
        return self.faker.date_between(start_date=start_date, end_date=end_date)
    
    def _generate_email(self, first_name: str, last_name: str) -> str:
        """Generate realistic email address"""
        patterns = [
            f"{first_name.lower()}.{last_name.lower()}",
            f"{first_name.lower()}{last_name.lower()}",
            f"{first_name[0].lower()}{last_name.lower()}",
            f"{first_name.lower()}{self.faker.random_int(1, 99)}"
        ]
        
        pattern = self.faker.random_element(patterns)
        domain = self.faker.random_element([
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'icloud.com', 'aol.com'
        ])
        
        return f"{pattern}@{domain}"
    
    def _generate_address(self, country: str) -> Dict[str, str]:
        """Generate address information"""
        return {
            'street': self.faker.street_address(),
            'city': self.faker.city(),
            'state': self.faker.state(),
            'postal_code': self.faker.postcode(),
            'country': country if country != 'global' else self.faker.country()
        }
    
    def _generate_medical_conditions(self, age_group: str, gender: str) -> List[str]:
        """Generate medical conditions based on age group and gender"""
        conditions = []
        condition_probs = self.conditions_by_age[age_group]
        
        # Each condition is evaluated independently
        for condition, probability in condition_probs.items():
            if condition != 'None' and self.faker.random.random() < probability:
                conditions.append(condition)
        
        # Gender-specific condition adjustments
        if gender == 'F':
            # Women have higher rates of certain conditions
            if age_group in ['middle_aged', 'senior', 'elderly']:
                if self.faker.random.random() < 0.15:
                    conditions.append('Osteoporosis')
                if self.faker.random.random() < 0.08:
                    conditions.append('Thyroid Disorder')
        
        # Remove duplicates and return
        return list(set(conditions))
    
    def _generate_medications(self, medical_conditions: List[str]) -> List[str]:
        """Generate medications based on medical conditions"""
        medications = []
        
        for condition in medical_conditions:
            if condition in self.medications_by_condition:
                # Each patient might be on 1-2 medications per condition
                condition_meds = self.medications_by_condition[condition]
                num_meds = self.faker.random_int(1, min(2, len(condition_meds)))
                selected_meds = self.faker.random.sample(condition_meds, num_meds)
                medications.extend(selected_meds)
        
        # Remove duplicates
        return list(set(medications))
    
    def _generate_allergies(self) -> List[str]:
        """Generate allergies with realistic distribution"""
        common_allergies = [
            'Penicillin', 'Sulfa drugs', 'Aspirin', 'Ibuprofen', 'Codeine',
            'Peanuts', 'Tree nuts', 'Shellfish', 'Eggs', 'Milk', 'Soy',
            'Pollen', 'Dust mites', 'Pet dander', 'Latex'
        ]
        
        allergies = []
        
        # 30% chance of having at least one allergy
        if self.faker.random.random() < 0.30:
            num_allergies = self.faker.random_int(1, 3)
            allergies = self.faker.random.sample(common_allergies, num_allergies)
        
        return allergies
    
    def _generate_health_metrics(self, age_group: str, gender: str, medical_conditions: List[str]) -> Dict[str, Any]:
        """Generate health metrics based on demographics and conditions"""
        # Generate height (gender-specific)
        if gender == 'M':
            height_cm = max(150, min(200, self.faker.random.gauss(175, 8)))
        elif gender == 'F':
            height_cm = max(140, min(185, self.faker.random.gauss(162, 7)))
        else:
            height_cm = max(145, min(190, self.faker.random.gauss(168, 10)))
        
        height_cm = round(height_cm, 1)
        
        # Generate BMI based on age group patterns
        bmi_pattern = self.bmi_patterns[age_group]
        bmi = max(bmi_pattern['min'], min(bmi_pattern['max'], 
                  self.faker.random.gauss(bmi_pattern['mean'], bmi_pattern['std'])))
        bmi = round(bmi, 1)
        
        # Calculate weight from BMI and height
        weight_kg = round((bmi * (height_cm / 100) ** 2), 1)
        
        # Generate blood pressure (affected by conditions)
        base_systolic = 120
        base_diastolic = 80
        
        if 'Hypertension' in medical_conditions:
            base_systolic += self.faker.random_int(20, 40)
            base_diastolic += self.faker.random_int(10, 20)
        
        # Age affects blood pressure
        age_adjustment = (self._generate_age_for_group(age_group) - 30) * 0.5
        bp_systolic = max(90, min(200, int(base_systolic + age_adjustment + self.faker.random.gauss(0, 10))))
        bp_diastolic = max(60, min(120, int(base_diastolic + age_adjustment * 0.5 + self.faker.random.gauss(0, 5))))
        
        # Generate smoking status
        smoking_prob = self.smoking_patterns[age_group]
        if self.faker.random.random() < smoking_prob:
            smoking_status = self.faker.random_element(['Current smoker', 'Former smoker'])
        else:
            smoking_status = 'Never smoked'
        
        # Generate alcohol consumption
        alcohol_prob = self.alcohol_patterns[age_group]
        if self.faker.random.random() < alcohol_prob:
            alcohol_consumption = self.faker.random_element([
                'Occasional', 'Moderate', 'Social drinker', 'Regular'
            ])
        else:
            alcohol_consumption = 'None'
        
        return {
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'bmi': bmi,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption
        }
    
    def _generate_insurance_info(self, age_group: str) -> Dict[str, str]:
        """Generate insurance information based on age group"""
        # Age affects insurance type distribution
        if age_group == 'elderly':
            # Most elderly have Medicare
            insurance_type = self.faker.random_element(['Medicare', 'Medicare', 'Medicare', 'Private'])
        elif age_group == 'pediatric':
            # Children usually on parents' insurance or Medicaid
            insurance_type = self.faker.random_element(['Private', 'Private', 'Medicaid'])
        else:
            # Use general distribution for other age groups
            choices = list(self.insurance_distribution.keys())
            weights = list(self.insurance_distribution.values())
            
            cumulative_weights = []
            total = 0
            for weight in weights:
                total += weight
                cumulative_weights.append(total)
            
            rand_val = self.faker.random.uniform(0, total)
            insurance_type = choices[-1]
            for i, cum_weight in enumerate(cumulative_weights):
                if rand_val <= cum_weight:
                    insurance_type = choices[i]
                    break
        
        # Generate insurance ID
        if insurance_type == 'Medicare':
            insurance_id = f"{self.faker.random_int(100, 999)}-{self.faker.random_int(10, 99)}-{self.faker.random_int(1000, 9999)}"
        elif insurance_type == 'Medicaid':
            insurance_id = f"MC{self.faker.random_int(10000000, 99999999)}"
        else:
            insurance_id = f"{self.faker.random_element(['BC', 'AE', 'CI', 'HU'])}{self.faker.random_int(100000000, 999999999)}"
        
        return {
            'type': insurance_type,
            'id': insurance_id
        }
    
    def _generate_primary_care_provider(self) -> str:
        """Generate primary care provider name"""
        if self.faker.random.random() < self.has_primary_care:
            return f"Dr. {self.faker.first_name()} {self.faker.last_name()}"
        else:
            return "Not assigned"
    
    def _generate_emergency_contact(self, age_group: str, gender: str) -> Dict[str, str]:
        """Generate emergency contact based on patient demographics"""
        # Relationship depends on age group
        if age_group == 'pediatric':
            relationship = self.faker.random_element(['Parent', 'Guardian'])
        elif age_group in ['young_adult', 'middle_aged']:
            relationship = self.faker.random_element(['Spouse', 'Parent', 'Sibling', 'Friend'])
        else:  # senior, elderly
            relationship = self.faker.random_element(['Spouse', 'Child', 'Sibling', 'Friend'])
        
        # Generate contact name
        if relationship in ['Spouse', 'Sibling']:
            # Same last name for spouse, random for sibling
            if relationship == 'Spouse':
                contact_name = f"{self.faker.first_name()} {self.faker.last_name()}"
            else:
                contact_name = f"{self.faker.first_name()} {self.faker.last_name()}"
        else:
            contact_name = f"{self.faker.first_name()} {self.faker.last_name()}"
        
        return {
            'name': contact_name,
            'relationship': relationship,
            'phone': self.faker.phone_number()
        }
    
    def _generate_visit_history(self, age_group: str, medical_conditions: List[str]) -> Dict[str, Any]:
        """Generate visit history based on age and conditions"""
        # Base visit frequency by age group
        base_visits = {
            'pediatric': 4,      # Children visit more often
            'young_adult': 2,    # Young adults visit less
            'middle_aged': 3,    # Middle aged moderate visits
            'senior': 5,         # Seniors visit more often
            'elderly': 7         # Elderly visit most often
        }
        
        # Adjust for medical conditions
        condition_multiplier = 1.0
        if len(medical_conditions) > 0:
            condition_multiplier = 1.0 + (len(medical_conditions) * 0.3)
        
        visits_last_year = max(1, int(base_visits[age_group] * condition_multiplier * 
                                     self.faker.random.uniform(0.5, 1.5)))
        
        # Generate last visit date (more recent for patients with conditions)
        if medical_conditions:
            days_ago = self.faker.random_int(1, 90)  # Within 3 months
        else:
            days_ago = self.faker.random_int(30, 365)  # Within a year
        
        last_visit = datetime.now().date() - timedelta(days=days_ago)
        
        # Generate registration date (when they became a patient)
        registration_days_ago = self.faker.random_int(365, 3650)  # 1-10 years ago
        registration_date = datetime.now().date() - timedelta(days=registration_days_ago)
        
        return {
            'last_visit': last_visit,
            'visits_last_year': visits_last_year,
            'registration_date': registration_date
        }
    
    def _select_preferred_language(self) -> str:
        """Select preferred language based on distribution"""
        choices = list(self.language_distribution.keys())
        weights = list(self.language_distribution.values())
        
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to patient data"""
        # Add risk scores based on conditions and demographics
        data['health_risk_score'] = data.apply(
            lambda row: self._calculate_health_risk_score(row), axis=1
        ).round(2)
        
        # Add care coordination score
        data['care_coordination_score'] = data.apply(
            lambda row: self._calculate_care_coordination_score(row), axis=1
        ).round(1)
        
        # Add patient satisfaction score
        data['satisfaction_score'] = data.apply(
            lambda row: self._calculate_patient_satisfaction(row), axis=1
        ).round(1)
        
        # Sort by registration date for chronological order
        data = data.sort_values('registration_date').reset_index(drop=True)
        
        return data
    
    def _calculate_health_risk_score(self, row) -> float:
        """Calculate health risk score (0-1, higher = more risk)"""
        base_risk = 0.1
        
        # Age factor
        age_risk = {
            'pediatric': 0.05,
            'young_adult': 0.05,
            'middle_aged': 0.15,
            'senior': 0.25,
            'elderly': 0.35
        }
        base_risk += age_risk.get(row['age_group'], 0.1)
        
        # Condition factor
        conditions = row['medical_conditions'].split(', ') if row['medical_conditions'] != 'None' else []
        high_risk_conditions = ['Heart Disease', 'Diabetes Type 2', 'COPD', 'Dementia']
        
        for condition in conditions:
            if condition in high_risk_conditions:
                base_risk += 0.15
            else:
                base_risk += 0.05
        
        # Lifestyle factors
        if row['smoking_status'] == 'Current smoker':
            base_risk += 0.1
        elif row['smoking_status'] == 'Former smoker':
            base_risk += 0.05
        
        if row['bmi'] > 30:  # Obesity
            base_risk += 0.08
        elif row['bmi'] > 25:  # Overweight
            base_risk += 0.03
        
        # Add some randomness
        base_risk += self.faker.random.gauss(0, 0.05)
        
        return max(0.0, min(1.0, base_risk))
    
    def _calculate_care_coordination_score(self, row) -> float:
        """Calculate care coordination score (1-5, higher = better coordination)"""
        base_score = 3.0
        
        # Having a primary care provider improves coordination
        if row['primary_care_provider'] != 'Not assigned':
            base_score += 0.5
        
        # Insurance type affects coordination
        if row['insurance_type'] in ['Private', 'Medicare']:
            base_score += 0.3
        elif row['insurance_type'] == 'Self-Pay':
            base_score -= 0.5
        
        # More visits can indicate better or worse coordination
        if row['total_visits_last_year'] > 8:
            base_score -= 0.2  # Too many visits might indicate poor coordination
        elif row['total_visits_last_year'] < 2:
            base_score -= 0.3  # Too few visits might indicate poor access
        
        # Add randomness
        base_score += self.faker.random.gauss(0, 0.4)
        
        return max(1.0, min(5.0, base_score))
    
    def _calculate_patient_satisfaction(self, row) -> float:
        """Calculate patient satisfaction score (1-5, higher = more satisfied)"""
        base_score = 3.5
        
        # Care coordination affects satisfaction
        if row['care_coordination_score'] > 4.0:
            base_score += 0.4
        elif row['care_coordination_score'] < 2.5:
            base_score -= 0.4
        
        # Insurance type affects satisfaction
        if row['insurance_type'] == 'Private':
            base_score += 0.2
        elif row['insurance_type'] == 'Self-Pay':
            base_score -= 0.3
        
        # Language barriers might affect satisfaction
        if row['preferred_language'] != 'English':
            base_score -= 0.1
        
        # Add randomness
        base_score += self.faker.random.gauss(0, 0.5)
        
        return max(1.0, min(5.0, base_score))