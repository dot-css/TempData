"""
Unit tests for CustomerGenerator

Tests demographic distributions, registration patterns, customer segmentation, and data quality.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.customers import CustomerGenerator


class TestCustomerGenerator:
    """Test suite for CustomerGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests using enhanced seeding"""
        return MillisecondSeeder(fixed_seed=987654321)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create CustomerGenerator instance"""
        return CustomerGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns exist
        required_columns = [
            'customer_id', 'first_name', 'last_name', 'email', 'phone',
            'gender', 'age', 'age_group', 'annual_income', 'income_bracket',
            'registration_date', 'registration_channel', 'lifecycle_stage',
            'segment', 'total_orders', 'lifetime_value', 'avg_order_value',
            'last_order_date', 'days_since_last_order', 'address_line1',
            'city', 'state_province', 'postal_code', 'country',
            'preferred_contact', 'marketing_opt_in', 'preferred_category'
        ]
        for col in required_columns:
            assert col in data.columns
    
    def test_customer_id_uniqueness(self, generator):
        """Test that customer IDs are unique"""
        data = generator.generate(500)
        assert data['customer_id'].nunique() == len(data)
        
        # Check format
        assert all(data['customer_id'].str.startswith('CUST_'))
        assert all(data['customer_id'].str.len() == 11)  # CUST_ + 6 digits
    
    def test_demographic_distributions(self, generator):
        """Test realistic demographic distributions"""
        data = generator.generate(1000)
        
        # Test age groups distribution
        age_group_counts = data['age_group'].value_counts(normalize=True)
        assert 'millennial' in age_group_counts.index
        assert 'gen_x' in age_group_counts.index
        assert 'gen_z' in age_group_counts.index
        assert 'boomer' in age_group_counts.index
        
        # Millennials should be the largest group (around 35%)
        assert age_group_counts['millennial'] > 0.25
        
        # Test gender distribution
        gender_counts = data['gender'].value_counts(normalize=True)
        assert abs(gender_counts['male'] - 0.48) < 0.1  # Should be around 48%
        assert abs(gender_counts['female'] - 0.50) < 0.1  # Should be around 50%
        
        # Test age ranges by group
        for age_group in ['gen_z', 'millennial', 'gen_x', 'boomer']:
            group_data = data[data['age_group'] == age_group]
            if len(group_data) > 0:
                if age_group == 'gen_z':
                    assert group_data['age'].min() >= 18
                    assert group_data['age'].max() <= 26
                elif age_group == 'millennial':
                    assert group_data['age'].min() >= 27
                    assert group_data['age'].max() <= 42
    
    def test_income_patterns(self, generator):
        """Test income patterns by age group"""
        data = generator.generate(1000)
        
        # Check income brackets exist
        income_brackets = data['income_bracket'].unique()
        assert 'low' in income_brackets
        assert 'medium' in income_brackets
        assert 'high' in income_brackets
        
        # Check income ranges
        assert all(data['annual_income'] >= 25000)
        assert all(data['annual_income'] <= 200000)
        
        # Gen X and Boomers should have higher average income than Gen Z
        gen_z_income = data[data['age_group'] == 'gen_z']['annual_income'].mean()
        gen_x_income = data[data['age_group'] == 'gen_x']['annual_income'].mean()
        
        if len(data[data['age_group'] == 'gen_z']) > 20 and len(data[data['age_group'] == 'gen_x']) > 20:
            assert gen_x_income > gen_z_income * 0.9  # Allow some variance
    
    def test_registration_patterns(self, generator):
        """Test registration patterns and channels"""
        data = generator.generate(500)
        
        # Test registration channels
        channels = data['registration_channel'].unique()
        expected_channels = ['organic_search', 'social_media', 'referral', 
                           'email_marketing', 'paid_ads', 'direct']
        for channel in expected_channels:
            assert channel in channels
        
        # Organic search should be the most common
        channel_counts = data['registration_channel'].value_counts()
        assert channel_counts.index[0] in ['organic_search', 'social_media']  # Top channels
        
        # Test registration dates are reasonable (now using 10 year range)
        assert all(data['registration_date'] >= date(2015, 1, 1))
        assert all(data['registration_date'] <= datetime.now().date())
    
    def test_lifecycle_stages(self, generator):
        """Test customer lifecycle stage distribution"""
        data = generator.generate(1000)
        
        # Test lifecycle stages exist
        stages = data['lifecycle_stage'].unique()
        expected_stages = ['new', 'active', 'returning', 'dormant', 'churned']
        for stage in expected_stages:
            assert stage in stages
        
        # Active should be the most common stage
        stage_counts = data['lifecycle_stage'].value_counts(normalize=True)
        assert stage_counts['active'] > 0.25  # Should be around 40%
        
        # Test relationship between lifecycle stage and days since last order
        new_customers = data[data['lifecycle_stage'] == 'new']
        churned_customers = data[data['lifecycle_stage'] == 'churned']
        
        if len(new_customers) > 5 and len(churned_customers) > 5:
            assert new_customers['days_since_last_order'].mean() < churned_customers['days_since_last_order'].mean()
    
    def test_customer_segmentation(self, generator):
        """Test customer segmentation logic"""
        data = generator.generate(1000)
        
        # Test segments exist
        segments = data['segment'].unique()
        expected_segments = ['VIP', 'Premium', 'Standard', 'Basic']
        for segment in expected_segments:
            assert segment in segments
        
        # Standard should be the most common segment
        segment_counts = data['segment'].value_counts(normalize=True)
        assert segment_counts['Standard'] > 0.2  # Should be the most common
        
        # VIP customers should have higher lifetime value
        vip_customers = data[data['segment'] == 'VIP']
        basic_customers = data[data['segment'] == 'Basic']
        
        if len(vip_customers) > 5 and len(basic_customers) > 5:
            assert vip_customers['lifetime_value'].mean() > basic_customers['lifetime_value'].mean()
    
    def test_activity_metrics(self, generator):
        """Test customer activity metrics"""
        data = generator.generate(500)
        
        # Test activity metrics are reasonable
        assert all(data['total_orders'] >= 1)
        assert all(data['lifetime_value'] >= 50)
        assert all(data['avg_order_value'] > 0)
        
        # Test relationship between orders and lifetime value
        correlation = data['total_orders'].corr(data['lifetime_value'])
        assert correlation > 0.5  # Should be positively correlated
        
        # Test avg_order_value calculation
        calculated_avg = data['lifetime_value'] / data['total_orders']
        assert all(abs(calculated_avg - data['avg_order_value']) < 0.01)
    
    def test_email_generation(self, generator):
        """Test email address generation"""
        data = generator.generate(200)
        
        # All emails should be valid format
        assert all(data['email'].str.contains('@'))
        assert all(data['email'].str.contains('.'))
        
        # Should use common domains
        domains = data['email'].str.split('@').str[1]
        common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        domain_counts = domains.value_counts()
        
        # At least some emails should use common domains
        common_domain_usage = sum(domain_counts[domain] for domain in common_domains if domain in domain_counts)
        assert common_domain_usage > len(data) * 0.5
    
    def test_preferences_by_demographics(self, generator):
        """Test preferences based on demographics"""
        data = generator.generate(500)
        
        # Test contact method preferences
        contact_methods = data['preferred_contact'].unique()
        expected_methods = ['email', 'phone', 'sms', 'app_notification', 'mail']
        for method in expected_methods:
            assert method in contact_methods
        
        # Gen Z should prefer digital methods more
        gen_z_data = data[data['age_group'] == 'gen_z']
        boomer_data = data[data['age_group'] == 'boomer']
        
        if len(gen_z_data) > 10 and len(boomer_data) > 10:
            gen_z_digital = len(gen_z_data[gen_z_data['preferred_contact'].isin(['sms', 'app_notification'])])
            boomer_digital = len(boomer_data[boomer_data['preferred_contact'].isin(['sms', 'app_notification'])])
            
            gen_z_digital_rate = gen_z_digital / len(gen_z_data)
            boomer_digital_rate = boomer_digital / len(boomer_data)
            
            assert gen_z_digital_rate >= boomer_digital_rate
        
        # Test marketing opt-in rate
        opt_in_rate = data['marketing_opt_in'].mean()
        assert 0.5 < opt_in_rate < 0.8  # Should be around 65%
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied"""
        data = generator.generate(200)
        
        # Check additional columns from realistic patterns
        pattern_columns = ['satisfaction_score', 'churn_risk_score', 'preferred_contact_time']
        for col in pattern_columns:
            assert col in data.columns
        
        # Check satisfaction scores
        assert all(data['satisfaction_score'] >= 1.0)
        assert all(data['satisfaction_score'] <= 5.0)
        
        # Check churn risk scores
        assert all(data['churn_risk_score'] >= 0.0)
        assert all(data['churn_risk_score'] <= 1.0)
        
        # VIP customers should have higher satisfaction
        vip_customers = data[data['segment'] == 'VIP']
        basic_customers = data[data['segment'] == 'Basic']
        
        if len(vip_customers) > 3 and len(basic_customers) > 3:
            assert vip_customers['satisfaction_score'].mean() > basic_customers['satisfaction_score'].mean()
    
    def test_chronological_ordering(self, generator):
        """Test that data is sorted chronologically"""
        data = generator.generate(100)
        
        # Check registration dates are in ascending order
        reg_dates = pd.to_datetime(data['registration_date'])
        assert all(reg_dates[i] <= reg_dates[i+1] for i in range(len(reg_dates)-1))
    
    def test_country_specific_generation(self, generator):
        """Test country-specific data generation"""
        us_data = generator.generate(50, country='united_states')
        global_data = generator.generate(50, country='global')
        
        # US data should have consistent country
        assert all(us_data['country'] == 'united_states')
        
        # Global data should have variety
        assert global_data['country'].nunique() > 1
    
    def test_date_range_generation(self, generator):
        """Test generation with specific date range"""
        start_date = date(2023, 1, 1)
        end_date = date(2023, 12, 31)
        
        data = generator.generate(100, date_range=(start_date, end_date))
        
        # All registration dates should be within range
        assert all(data['registration_date'] >= start_date)
        assert all(data['registration_date'] <= end_date)
    
    def test_data_relationships(self, generator):
        """Test relationships between data fields"""
        data = generator.generate(500)
        
        # Higher income should correlate with higher lifetime value
        income_ltv_corr = data['annual_income'].corr(data['lifetime_value'])
        assert income_ltv_corr > 0.15  # Should be positively correlated
        
        # Older customers should have higher lifetime value on average
        older_customers = data[data['age'] > 50]
        younger_customers = data[data['age'] < 30]
        
        if len(older_customers) > 20 and len(younger_customers) > 20:
            assert older_customers['lifetime_value'].mean() >= younger_customers['lifetime_value'].mean() * 0.8
    
    def test_reproducibility(self, seeder):
        """Test that same seed produces same results"""
        gen1 = CustomerGenerator(seeder)
        gen2 = CustomerGenerator(MillisecondSeeder(fixed_seed=987654321))
        
        data1 = gen1.generate(50)
        data2 = gen2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_metrics(self, generator):
        """Test overall data quality metrics"""
        data = generator.generate(1000)
        
        # No null values in critical fields
        critical_fields = ['customer_id', 'email', 'age', 'segment', 'lifetime_value']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable distribution of segments
        segment_dist = data['segment'].value_counts(normalize=True)
        assert len(segment_dist) >= 3  # At least 3 segments
        assert segment_dist.max() <= 0.8  # No single segment dominates too much
        
        # Email addresses should be unique (mostly)
        email_uniqueness = data['email'].nunique() / len(data)
        assert email_uniqueness > 0.95  # At least 95% unique emails
    
    def test_edge_cases(self, generator):
        """Test edge cases and error handling"""
        # Test with minimal rows
        data = generator.generate(1)
        assert len(data) == 1
        
        # Test with larger dataset
        data = generator.generate(2000)
        assert len(data) == 2000
        
        # Test with invalid country (should default gracefully)
        data = generator.generate(10, country='invalid_country')
        assert len(data) == 10
        # Should still have valid data
        assert all(data['customer_id'].str.startswith('CUST_'))
    
    def test_lifecycle_stage_consistency(self, generator):
        """Test consistency between lifecycle stage and related metrics"""
        data = generator.generate(500)
        
        # New customers should have recent registration dates
        new_customers = data[data['lifecycle_stage'] == 'new']
        if len(new_customers) > 5:
            days_since_reg = [(datetime.now().date() - reg_date).days for reg_date in new_customers['registration_date']]
            assert all(days <= 30 for days in days_since_reg)
        
        # Churned customers should have old registration dates
        churned_customers = data[data['lifecycle_stage'] == 'churned']
        if len(churned_customers) > 5:
            days_since_reg = [(datetime.now().date() - reg_date).days for reg_date in churned_customers['registration_date']]
            assert all(days >= 1826 for days in days_since_reg)  # At least 5 years
    
    def test_segment_criteria_consistency(self, generator):
        """Test that segment assignment follows the defined criteria"""
        data = generator.generate(1000)
        
        # VIP customers should meet VIP criteria
        vip_customers = data[data['segment'] == 'VIP']
        if len(vip_customers) > 0:
            assert all(vip_customers['lifetime_value'] >= 5000)
            assert all(vip_customers['total_orders'] >= 20)
        
        # Premium customers should meet Premium criteria
        premium_customers = data[data['segment'] == 'Premium']
        if len(premium_customers) > 0:
            assert all(premium_customers['lifetime_value'] >= 1500)
            assert all(premium_customers['total_orders'] >= 8)


if __name__ == '__main__':
    pytest.main([__file__])