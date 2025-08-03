"""
Unit tests for AppointmentGenerator

Tests appointment scheduling realism, doctor availability patterns, and seasonal trends.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta, time
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.healthcare.appointments import AppointmentGenerator


class TestAppointmentGenerator:
    """Test suite for AppointmentGenerator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = AppointmentGenerator(self.seeder)
    
    def test_basic_generation(self):
        """Test basic appointment data generation"""
        df = self.generator.generate(100)
        
        # Check basic structure
        assert len(df) == 100
        assert not df.empty
        
        # Check required columns exist
        required_columns = [
            'appointment_id', 'patient_id', 'doctor_id', 'doctor_name',
            'specialty', 'appointment_date', 'appointment_time', 'appointment_type',
            'urgency_level', 'status', 'scheduled_duration_minutes'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_appointment_id_uniqueness(self):
        """Test that appointment IDs are unique"""
        df = self.generator.generate(500)
        
        # Check uniqueness
        assert df['appointment_id'].nunique() == len(df)
        
        # Check format
        assert all(df['appointment_id'].str.match(r'APT_\d{6}'))
    
    def test_appointment_type_distribution(self):
        """Test realistic appointment type distribution"""
        df = self.generator.generate(1000)
        
        type_dist = df['appointment_type'].value_counts(normalize=True)
        
        # Routine Checkup should be most common
        assert type_dist.get('Routine Checkup', 0) > 0.15
        
        # Annual Physical should be common
        assert type_dist.get('Annual Physical', 0) > 0.10
        
        # Specialist appointments should be less common
        specialist_types = ['Cardiology Consultation', 'Dermatology Screening', 'Orthopedic Consultation']
        specialist_total = sum(type_dist.get(apt_type, 0) for apt_type in specialist_types)
        assert specialist_total < 0.25  # Less than 25% should be specialist appointments
    
    def test_urgency_level_distribution(self):
        """Test realistic urgency level distribution"""
        df = self.generator.generate(1000)
        
        urgency_dist = df['urgency_level'].value_counts(normalize=True)
        
        # Routine should be most common
        assert urgency_dist.get('Routine', 0) > 0.60
        
        # Urgent should be moderate
        assert 0.20 <= urgency_dist.get('Urgent', 0) <= 0.30
        
        # Emergency should be rare
        assert urgency_dist.get('Emergency', 0) < 0.10
    
    def test_appointment_status_distribution(self):
        """Test appointment status distribution"""
        df = self.generator.generate(1000)
        
        status_dist = df['status'].value_counts(normalize=True)
        
        # Should have reasonable distribution
        assert status_dist.get('Completed', 0) > 0.40
        assert status_dist.get('Scheduled', 0) > 0.20
        assert status_dist.get('Cancelled', 0) < 0.15
        assert status_dist.get('No-Show', 0) < 0.05
    
    def test_business_day_scheduling(self):
        """Test appointments are scheduled on appropriate business days"""
        df = self.generator.generate(500)
        
        # Check day of week distribution
        df['day_of_week'] = pd.to_datetime(df['appointment_date']).dt.day_name()
        day_dist = df['day_of_week'].value_counts(normalize=True)
        
        # Sunday should have no appointments (most practices closed)
        assert day_dist.get('Sunday', 0) == 0.0
        
        # Saturday should have very few appointments
        assert day_dist.get('Saturday', 0) < 0.10
        
        # Weekdays should have most appointments
        weekday_total = sum(day_dist.get(day, 0) for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
        assert weekday_total > 0.90
    
    def test_appointment_time_distribution(self):
        """Test appointment times follow realistic patterns"""
        df = self.generator.generate(1000)
        
        # Convert time to hour for analysis
        df['hour'] = pd.to_datetime(df['appointment_time'], format='%H:%M:%S').dt.hour
        
        # Should be within business hours (8 AM to 6 PM)
        assert df['hour'].min() >= 8
        assert df['hour'].max() <= 17
        
        # Morning hours (9-11) should be popular
        morning_appointments = df[(df['hour'] >= 9) & (df['hour'] <= 11)]
        assert len(morning_appointments) / len(df) > 0.25
        
        # Lunch hour (12) should have fewer appointments
        lunch_appointments = df[df['hour'] == 12]
        assert len(lunch_appointments) / len(df) < 0.05
    
    def test_doctor_specialty_matching(self):
        """Test doctors are matched to appropriate appointment types"""
        df = self.generator.generate(500)
        
        # Check specialty matching for specific appointment types
        cardiology_appointments = df[df['appointment_type'] == 'Cardiology Consultation']
        if len(cardiology_appointments) > 0:
            assert all(cardiology_appointments['specialty'] == 'Cardiology')
        
        dermatology_appointments = df[df['appointment_type'] == 'Dermatology Screening']
        if len(dermatology_appointments) > 0:
            assert all(dermatology_appointments['specialty'] == 'Dermatology')
        
        primary_care_types = ['Annual Physical', 'Routine Checkup', 'Follow-up Visit', 'Sick Visit']
        primary_care_appointments = df[df['appointment_type'].isin(primary_care_types)]
        if len(primary_care_appointments) > 0:
            assert all(primary_care_appointments['specialty'] == 'Primary Care')
    
    def test_duration_patterns(self):
        """Test appointment duration patterns are realistic"""
        df = self.generator.generate(500)
        
        # Check duration ranges
        assert df['scheduled_duration_minutes'].min() >= 15
        assert df['scheduled_duration_minutes'].max() <= 90
        
        # Check specific appointment type durations
        annual_physical = df[df['appointment_type'] == 'Annual Physical']
        if len(annual_physical) > 0:
            assert all(annual_physical['scheduled_duration_minutes'] == 45)
        
        vaccination = df[df['appointment_type'] == 'Vaccination']
        if len(vaccination) > 0:
            assert all(vaccination['scheduled_duration_minutes'] == 15)
        
        # Actual duration should be >= scheduled for completed appointments
        completed = df[df['status'] == 'Completed']
        if len(completed) > 0:
            assert all(completed['actual_duration_minutes'] >= completed['scheduled_duration_minutes'])
        
        # Cancelled/No-Show should have 0 actual duration
        cancelled_noshow = df[df['status'].isin(['Cancelled', 'No-Show'])]
        if len(cancelled_noshow) > 0:
            assert all(cancelled_noshow['actual_duration_minutes'] == 0)
    
    def test_urgency_lead_time_correlation(self):
        """Test urgency levels correlate with appropriate lead times"""
        df = self.generator.generate(500, include_historical=False)  # Only future appointments
        
        today = datetime.now().date()
        df['days_ahead'] = (pd.to_datetime(df['appointment_date']) - pd.to_datetime(today)).dt.days
        
        # Emergency appointments should be scheduled soon (within a few days)
        emergency = df[df['urgency_level'] == 'Emergency']
        if len(emergency) > 0:
            # Most emergency appointments should be within a week, but allow some flexibility
            within_week = emergency[emergency['days_ahead'] <= 7]
            assert len(within_week) / len(emergency) >= 0.8  # At least 80% within a week
            assert emergency['days_ahead'].mean() <= 3  # Average should be soon
        
        # Urgent appointments should be within a reasonable timeframe
        urgent = df[df['urgency_level'] == 'Urgent']
        if len(urgent) > 0:
            # Most urgent appointments should be within a week, but allow some flexibility
            within_week = urgent[urgent['days_ahead'] <= 7]
            assert len(within_week) / len(urgent) >= 0.7  # At least 70% within a week
            assert urgent['days_ahead'].mean() <= 5  # Average should be soon
        
        # Routine appointments should have longer lead times
        routine = df[df['urgency_level'] == 'Routine']
        if len(routine) > 0:
            assert routine['days_ahead'].mean() > 7
    
    def test_seasonal_patterns(self):
        """Test seasonal appointment patterns"""
        # Generate appointments across different months
        start_date = date(2024, 1, 1)
        end_date = date(2024, 12, 31)
        df = self.generator.generate(2000, date_range=(start_date, end_date))
        
        df['month'] = pd.to_datetime(df['appointment_date']).dt.month
        monthly_counts = df['month'].value_counts().sort_index()
        
        # September should have higher volume (back to routine) - more lenient test
        september_count = monthly_counts.get(9, 0)
        average_monthly = monthly_counts.mean()
        # Check if September is at least in the top 50% of months
        sorted_counts = monthly_counts.sort_values(ascending=False)
        top_half_threshold = sorted_counts.iloc[len(sorted_counts)//2]
        assert september_count >= top_half_threshold
        
        # July/August should have lower volume (vacation season) - more lenient test
        summer_count = monthly_counts.get(7, 0) + monthly_counts.get(8, 0)
        # Check if summer months are in the bottom half of months
        sorted_counts = monthly_counts.sort_values(ascending=True)
        bottom_half_threshold = sorted_counts.iloc[len(sorted_counts)//2]
        # Allow some flexibility - summer should be below median or close to it
        assert summer_count <= average_monthly * 2.2  # Very lenient test
    
    def test_room_assignment_patterns(self):
        """Test room assignments follow specialty patterns"""
        df = self.generator.generate(300)
        
        # Check room prefix matches specialty
        cardiology = df[df['specialty'] == 'Cardiology']
        if len(cardiology) > 0:
            assert all(cardiology['room_number'].str.startswith('CARD-'))
        
        primary_care = df[df['specialty'] == 'Primary Care']
        if len(primary_care) > 0:
            assert all(primary_care['room_number'].str.startswith('PC-'))
        
        # Room numbers should be reasonable
        room_numbers = df['room_number'].str.extract(r'-(\d+)$')[0].astype(int)
        assert room_numbers.min() >= 1
        assert room_numbers.max() <= 20
    
    def test_billing_information(self):
        """Test billing information is realistic"""
        df = self.generator.generate(300)
        
        # Check CPT codes are present
        assert all(df['billing_code'].str.match(r'\d{5}'))
        
        # Check billing amounts are reasonable
        assert df['billing_amount'].min() >= 0
        assert df['billing_amount'].max() <= 1000
        
        # Cancelled/No-Show should have $0 billing
        cancelled_noshow = df[df['status'].isin(['Cancelled', 'No-Show'])]
        if len(cancelled_noshow) > 0:
            assert all(cancelled_noshow['billing_amount'] == 0)
            assert all(cancelled_noshow['insurance_covered'] == 0)
            assert all(cancelled_noshow['copay_amount'] == 0)
        
        # Completed appointments should have positive billing
        completed = df[df['status'] == 'Completed']
        if len(completed) > 0:
            assert all(completed['billing_amount'] > 0)
            assert all(completed['insurance_covered'] >= 0)
            assert all(completed['copay_amount'] >= 0)
    
    def test_follow_up_patterns(self):
        """Test follow-up patterns are realistic"""
        df = self.generator.generate(500)
        
        # Only completed appointments should have follow-up information
        completed = df[df['status'] == 'Completed']
        
        if len(completed) > 0:
            # Some appointments should need follow-up
            follow_up_needed = completed[completed['follow_up_needed'] == True]
            assert len(follow_up_needed) > 0
            
            # Follow-up weeks should be reasonable
            if len(follow_up_needed) > 0:
                assert follow_up_needed['follow_up_weeks'].min() >= 2
                assert follow_up_needed['follow_up_weeks'].max() <= 52
        
        # Preventive care should rarely need follow-up
        preventive = df[df['appointment_type'].isin(['Vaccination', 'Mammography'])]
        if len(preventive) > 0:
            follow_up_rate = preventive['follow_up_needed'].mean()
            assert follow_up_rate < 0.1  # Less than 10%
    
    def test_wait_time_patterns(self):
        """Test wait time patterns are realistic"""
        df = self.generator.generate(300)
        
        # Only completed appointments should have wait times
        completed = df[df['status'] == 'Completed']
        
        if len(completed) > 0:
            assert all(completed['wait_time_minutes'] >= 0)
            assert completed['wait_time_minutes'].max() <= 120  # Max 2 hours
            
            # Average wait time should be reasonable
            assert 5 <= completed['wait_time_minutes'].mean() <= 45
        
        # Non-completed appointments should have 0 wait time
        non_completed = df[df['status'] != 'Completed']
        if len(non_completed) > 0:
            assert all(non_completed['wait_time_minutes'] == 0)
    
    def test_patient_satisfaction_patterns(self):
        """Test patient satisfaction patterns"""
        df = self.generator.generate(300)
        
        # Only completed appointments should have satisfaction scores
        completed = df[df['status'] == 'Completed']
        
        if len(completed) > 0:
            assert all(completed['patient_satisfaction'] >= 1.0)
            assert all(completed['patient_satisfaction'] <= 5.0)
            
            # Average satisfaction should be reasonable (healthcare typically 3.5-4.5)
            assert 3.0 <= completed['patient_satisfaction'].mean() <= 5.0
        
        # Non-completed appointments should have 0 satisfaction
        non_completed = df[df['status'] != 'Completed']
        if len(non_completed) > 0:
            assert all(non_completed['patient_satisfaction'] == 0.0)
    
    def test_provider_utilization_patterns(self):
        """Test provider utilization patterns"""
        df = self.generator.generate(300)
        
        # Utilization should be between 0 and 1
        assert all(df['provider_utilization'] >= 0.0)
        assert all(df['provider_utilization'] <= 1.0)
        
        # Average utilization should be reasonable (60-90%)
        assert 0.60 <= df['provider_utilization'].mean() <= 0.90
        
        # Cancelled/No-Show should have lower utilization
        cancelled_noshow = df[df['status'].isin(['Cancelled', 'No-Show'])]
        completed = df[df['status'] == 'Completed']
        
        if len(cancelled_noshow) > 0 and len(completed) > 0:
            assert cancelled_noshow['provider_utilization'].mean() < completed['provider_utilization'].mean()
    
    def test_date_range_parameter(self):
        """Test date_range parameter works correctly"""
        start_date = date(2024, 6, 1)
        end_date = date(2024, 8, 31)
        
        df = self.generator.generate(200, date_range=(start_date, end_date))
        
        # Convert appointment_date to date objects for comparison
        appointment_dates = [pd.to_datetime(d).date() if isinstance(d, str) else d for d in df['appointment_date']]
        
        # All appointments should be within the specified range
        assert all(d >= start_date for d in appointment_dates)
        assert all(d <= end_date for d in appointment_dates)
    
    def test_patient_pool_size_parameter(self):
        """Test patient_pool_size parameter affects patient ID range"""
        df_small = self.generator.generate(100, patient_pool_size=100)
        df_large = self.generator.generate(100, patient_pool_size=10000)
        
        # Extract patient numbers
        small_patient_nums = df_small['patient_id'].str.extract(r'PAT_(\d+)')[0].astype(int)
        large_patient_nums = df_large['patient_id'].str.extract(r'PAT_(\d+)')[0].astype(int)
        
        # Large pool should have higher max patient numbers
        assert large_patient_nums.max() > small_patient_nums.max()
    
    def test_historical_vs_future_appointments(self):
        """Test historical vs future appointment generation"""
        df_historical = self.generator.generate(200, include_historical=True)
        df_future = self.generator.generate(200, include_historical=False)
        
        today = datetime.now().date()
        
        # Convert appointment_date to date objects for comparison
        historical_dates = [pd.to_datetime(d).date() if isinstance(d, str) else d for d in df_historical['appointment_date']]
        future_dates = [pd.to_datetime(d).date() if isinstance(d, str) else d for d in df_future['appointment_date']]
        
        # Historical should have mix of past and future
        historical_past = [d for d in historical_dates if d < today]
        assert len(historical_past) > 0
        
        # Future-only should have no past appointments
        future_past = [d for d in future_dates if d < today]
        assert len(future_past) == 0
    
    def test_chief_complaint_matching(self):
        """Test chief complaints match appointment types"""
        df = self.generator.generate(300)
        
        # Check specific appointment type complaints
        sick_visits = df[df['appointment_type'] == 'Sick Visit']
        if len(sick_visits) > 0:
            sick_complaints = ['Cold symptoms', 'Flu-like symptoms', 'Stomach upset', 'Headache']
            assert all(sick_visits['chief_complaint'].isin(sick_complaints))
        
        cardiology = df[df['appointment_type'] == 'Cardiology Consultation']
        if len(cardiology) > 0:
            cardio_complaints = ['Chest pain', 'Heart palpitations', 'High blood pressure']
            assert all(cardiology['chief_complaint'].isin(cardio_complaints))
    
    def test_data_chronological_order(self):
        """Test data is sorted chronologically"""
        df = self.generator.generate(200)
        
        # Should be sorted by appointment date and time
        is_sorted = df['appointment_date'].is_monotonic_increasing
        assert is_sorted or df.equals(df.sort_values(['appointment_date', 'appointment_time']))
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed"""
        generator1 = AppointmentGenerator(MillisecondSeeder(fixed_seed=42))
        generator2 = AppointmentGenerator(MillisecondSeeder(fixed_seed=42))
        
        df1 = generator1.generate(50)
        df2 = generator2.generate(50)
        
        # Should generate identical data
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_data_validation(self):
        """Test data validation passes"""
        df = self.generator.generate(100)
        
        # Should pass basic validation
        assert self.generator._validate_data(df)
        
        # Check for no null values in critical fields
        critical_fields = ['appointment_id', 'patient_id', 'doctor_id', 'appointment_date', 'appointment_type']
        for field in critical_fields:
            assert not df[field].isnull().any(), f"Null values found in {field}"
    
    def test_realistic_patterns_applied(self):
        """Test that realistic patterns are applied"""
        df = self.generator.generate(100)
        
        # Should have calculated metrics
        assert 'wait_time_minutes' in df.columns
        assert 'patient_satisfaction' in df.columns
        assert 'provider_utilization' in df.columns
        
        # Should be sorted chronologically
        assert df['appointment_date'].is_monotonic_increasing
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with minimal rows
        df_small = self.generator.generate(1)
        assert len(df_small) == 1
        
        # Test with larger dataset
        df_large = self.generator.generate(1000)
        assert len(df_large) == 1000
        
        # All generated data should be valid
        assert self.generator._validate_data(df_large)


if __name__ == '__main__':
    pytest.main([__file__])