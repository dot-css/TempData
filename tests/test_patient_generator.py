"""
Unit tests for PatientGenerator

Tests healthcare data privacy, accuracy, and realistic patterns.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.healthcare.patients import PatientGenerator


class TestPatientGenerator:
    """Test suite for PatientGenerator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = PatientGenerator(self.seeder)
    
    def test_basic_generation(self):
        """Test basic patient data generation"""
        df = self.generator.generate(100)
        
        # Check basic structure
        assert len(df) == 100
        assert not df.empty
        
        # Check required columns exist
        required_columns = [
            'patient_id', 'first_name', 'last_name', 'date_of_birth', 'age',
            'gender', 'blood_type', 'medical_conditions', 'current_medications',
            'insurance_type', 'primary_care_provider'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_patient_id_uniqueness(self):
        """Test that patient IDs are unique"""
        df = self.generator.generate(500)
        
        # Check uniqueness
        assert df['patient_id'].nunique() == len(df)
        
        # Check format
        assert all(df['patient_id'].str.match(r'PAT_\d{6}'))
    
    def test_demographic_distributions(self):
        """Test realistic demographic distributions"""
        df = self.generator.generate(1000)
        
        # Test age distribution
        age_groups = df['age_group'].value_counts(normalize=True)
        
        # Should have reasonable distribution across age groups
        assert 0.10 <= age_groups.get('pediatric', 0) <= 0.20
        assert 0.15 <= age_groups.get('young_adult', 0) <= 0.25
        assert 0.20 <= age_groups.get('middle_aged', 0) <= 0.30
        assert 0.20 <= age_groups.get('senior', 0) <= 0.30
        assert 0.10 <= age_groups.get('elderly', 0) <= 0.20
        
        # Test gender distribution
        gender_dist = df['gender'].value_counts(normalize=True)
        assert 0.40 <= gender_dist.get('M', 0) <= 0.55
        assert 0.45 <= gender_dist.get('F', 0) <= 0.60
        assert gender_dist.get('Other', 0) <= 0.05
    
    def test_blood_type_distribution(self):
        """Test realistic blood type distribution"""
        df = self.generator.generate(1000)
        
        blood_type_dist = df['blood_type'].value_counts(normalize=True)
        
        # O+ should be most common
        assert blood_type_dist.get('O+', 0) > 0.30
        
        # A+ should be second most common
        assert blood_type_dist.get('A+', 0) > 0.30
        
        # AB- should be rarest
        assert blood_type_dist.get('AB-', 0) < 0.02
    
    def test_age_consistency(self):
        """Test age consistency with date of birth"""
        df = self.generator.generate(100)
        
        today = datetime.now().date()
        
        for _, row in df.iterrows():
            birth_date = row['date_of_birth']
            calculated_age = today.year - birth_date.year
            
            # Account for birthday not yet occurred this year
            if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
                calculated_age -= 1
            
            # Age should match within 1 year (due to random birth dates within year)
            assert abs(row['age'] - calculated_age) <= 1
    
    def test_medical_conditions_by_age(self):
        """Test medical conditions correlate with age groups"""
        df = self.generator.generate(1000)
        
        # Pediatric patients should have fewer conditions
        pediatric = df[df['age_group'] == 'pediatric']
        pediatric_with_conditions = pediatric[pediatric['medical_conditions'] != 'None']
        assert len(pediatric_with_conditions) / len(pediatric) < 0.40
        
        # Elderly patients should have more conditions
        elderly = df[df['age_group'] == 'elderly']
        elderly_with_conditions = elderly[elderly['medical_conditions'] != 'None']
        assert len(elderly_with_conditions) / len(elderly) > 0.80
        
        # Check for age-appropriate conditions
        hypertension_patients = df[df['medical_conditions'].str.contains('Hypertension', na=False)]
        # Most hypertension patients should be middle-aged or older
        older_hypertension = hypertension_patients[
            hypertension_patients['age_group'].isin(['middle_aged', 'senior', 'elderly'])
        ]
        assert len(older_hypertension) / len(hypertension_patients) > 0.80
    
    def test_medication_condition_correlation(self):
        """Test medications correlate with medical conditions"""
        df = self.generator.generate(500)
        
        # Patients with hypertension should have hypertension medications
        hypertension_patients = df[df['medical_conditions'].str.contains('Hypertension', na=False)]
        
        hypertension_meds = ['Lisinopril', 'Amlodipine', 'Metoprolol', 'Losartan']
        
        for _, patient in hypertension_patients.iterrows():
            medications = patient['current_medications']
            if medications != 'None':
                # Should have at least one hypertension medication
                has_hypertension_med = any(med in medications for med in hypertension_meds)
                assert has_hypertension_med, f"Hypertension patient without appropriate medication: {medications}"
    
    def test_health_metrics_realism(self):
        """Test health metrics are realistic"""
        df = self.generator.generate(500)
        
        # BMI should be reasonable
        assert df['bmi'].min() >= 12.0
        assert df['bmi'].max() <= 50.0
        assert 20.0 <= df['bmi'].mean() <= 30.0
        
        # Blood pressure should be reasonable
        assert df['blood_pressure_systolic'].min() >= 90
        assert df['blood_pressure_systolic'].max() <= 200
        assert df['blood_pressure_diastolic'].min() >= 60
        assert df['blood_pressure_diastolic'].max() <= 120
        
        # Height should be reasonable
        assert df['height_cm'].min() >= 140
        assert df['height_cm'].max() <= 200
        
        # Weight should be reasonable
        assert df['weight_kg'].min() >= 20
        assert df['weight_kg'].max() <= 200
    
    def test_insurance_age_correlation(self):
        """Test insurance types correlate with age groups"""
        df = self.generator.generate(1000)
        
        # Elderly should mostly have Medicare
        elderly = df[df['age_group'] == 'elderly']
        medicare_elderly = elderly[elderly['insurance_type'] == 'Medicare']
        assert len(medicare_elderly) / len(elderly) > 0.60
        
        # Pediatric should mostly have Private or Medicaid
        pediatric = df[df['age_group'] == 'pediatric']
        appropriate_insurance = pediatric[
            pediatric['insurance_type'].isin(['Private', 'Medicaid'])
        ]
        assert len(appropriate_insurance) / len(pediatric) > 0.80
    
    def test_visit_frequency_by_age_and_conditions(self):
        """Test visit frequency correlates with age and medical conditions"""
        df = self.generator.generate(500)
        
        # Patients with conditions should visit more frequently
        with_conditions = df[df['medical_conditions'] != 'None']
        without_conditions = df[df['medical_conditions'] == 'None']
        
        avg_visits_with_conditions = with_conditions['total_visits_last_year'].mean()
        avg_visits_without_conditions = without_conditions['total_visits_last_year'].mean()
        
        assert avg_visits_with_conditions > avg_visits_without_conditions
        
        # Elderly should visit more frequently than young adults
        elderly = df[df['age_group'] == 'elderly']
        young_adults = df[df['age_group'] == 'young_adult']
        
        assert elderly['total_visits_last_year'].mean() > young_adults['total_visits_last_year'].mean()
    
    def test_emergency_contact_relationships(self):
        """Test emergency contact relationships are age-appropriate"""
        df = self.generator.generate(500)
        
        # Pediatric patients should have Parent or Guardian
        pediatric = df[df['age_group'] == 'pediatric']
        appropriate_contacts = pediatric[
            pediatric['emergency_contact_relationship'].isin(['Parent', 'Guardian'])
        ]
        assert len(appropriate_contacts) / len(pediatric) == 1.0
        
        # Adult patients should have varied relationships
        adults = df[df['age_group'].isin(['young_adult', 'middle_aged', 'senior', 'elderly'])]
        relationship_variety = adults['emergency_contact_relationship'].nunique()
        assert relationship_variety >= 4  # Should have at least 4 different relationship types
    
    def test_privacy_compliance(self):
        """Test privacy compliance features"""
        # Test without sensitive data (default)
        df_no_sensitive = self.generator.generate(100)
        assert 'ssn' not in df_no_sensitive.columns
        assert 'medical_record_number' not in df_no_sensitive.columns
        
        # Test with sensitive data
        df_with_sensitive = self.generator.generate(100, include_sensitive=True)
        assert 'ssn' in df_with_sensitive.columns
        assert 'medical_record_number' in df_with_sensitive.columns
        
        # Check SSN format
        assert all(df_with_sensitive['ssn'].str.match(r'\d{3}-\d{2}-\d{4}'))
        
        # Check MRN format
        assert all(df_with_sensitive['medical_record_number'].str.match(r'MRN\d{6}'))
    
    def test_data_quality_scores(self):
        """Test data quality and risk scores are reasonable"""
        df = self.generator.generate(200)
        
        # Health risk scores should be between 0 and 1
        assert df['health_risk_score'].min() >= 0.0
        assert df['health_risk_score'].max() <= 1.0
        
        # Care coordination scores should be between 1 and 5
        assert df['care_coordination_score'].min() >= 1.0
        assert df['care_coordination_score'].max() <= 5.0
        
        # Satisfaction scores should be between 1 and 5
        assert df['satisfaction_score'].min() >= 1.0
        assert df['satisfaction_score'].max() <= 5.0
        
        # Patients with more conditions should have higher risk scores
        high_condition_patients = df[df['medical_conditions'].str.count(',') >= 2]
        low_condition_patients = df[df['medical_conditions'] == 'None']
        
        if len(high_condition_patients) > 0 and len(low_condition_patients) > 0:
            assert high_condition_patients['health_risk_score'].mean() > low_condition_patients['health_risk_score'].mean()
    
    def test_smoking_alcohol_patterns(self):
        """Test smoking and alcohol patterns by age group"""
        df = self.generator.generate(1000)
        
        # Pediatric patients should never smoke or drink
        pediatric = df[df['age_group'] == 'pediatric']
        assert all(pediatric['smoking_status'] == 'Never smoked')
        assert all(pediatric['alcohol_consumption'] == 'None')
        
        # Young adults should have higher rates of alcohol consumption
        young_adults = df[df['age_group'] == 'young_adult']
        alcohol_consumers = young_adults[young_adults['alcohol_consumption'] != 'None']
        assert len(alcohol_consumers) / len(young_adults) > 0.50
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed"""
        generator1 = PatientGenerator(MillisecondSeeder(fixed_seed=42))
        generator2 = PatientGenerator(MillisecondSeeder(fixed_seed=42))
        
        df1 = generator1.generate(50)
        df2 = generator2.generate(50)
        
        # Should generate identical data
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_country_parameter(self):
        """Test country parameter affects address generation"""
        df_us = self.generator.generate(50, country='united_states')
        df_global = self.generator.generate(50, country='global')
        
        # US should have consistent country
        assert all(df_us['country'] == 'united_states')
        
        # Global should have variety
        assert df_global['country'].nunique() > 1
    
    def test_data_validation(self):
        """Test data validation passes"""
        df = self.generator.generate(100)
        
        # Should pass basic validation
        assert self.generator._validate_data(df)
        
        # Check for no null values in critical fields
        critical_fields = ['patient_id', 'first_name', 'last_name', 'age', 'gender']
        for field in critical_fields:
            assert not df[field].isnull().any(), f"Null values found in {field}"
    
    def test_realistic_patterns_applied(self):
        """Test that realistic patterns are applied"""
        df = self.generator.generate(100)
        
        # Should be sorted by registration date
        assert df['registration_date'].is_monotonic_increasing
        
        # Should have calculated scores
        assert 'health_risk_score' in df.columns
        assert 'care_coordination_score' in df.columns
        assert 'satisfaction_score' in df.columns
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with minimal rows
        df_small = self.generator.generate(1)
        assert len(df_small) == 1
        
        # Test with larger dataset
        df_large = self.generator.generate(2000)
        assert len(df_large) == 2000
        
        # All generated data should be valid
        assert self.generator._validate_data(df_large)


if __name__ == '__main__':
    pytest.main([__file__])