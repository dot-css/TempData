"""
Unit tests for CRMGenerator

Tests realistic patterns, sales pipeline progression, interaction histories, and relationship integrity.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.crm import CRMGenerator


class TestCRMGenerator:
    """Test suite for CRMGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create CRMGenerator instance"""
        return CRMGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        assert not data.empty
    
    def test_required_columns(self, generator):
        """Test that all required columns are present"""
        data = generator.generate(50)
        
        required_columns = [
            # Contact fields
            'contact_id', 'first_name', 'last_name', 'full_name', 'email', 'phone',
            'job_title', 'department', 'seniority_level',
            # Account fields
            'account_id', 'company_name', 'industry', 'company_size', 'employee_count',
            'annual_revenue', 'website', 'address', 'city', 'state', 'country',
            # Opportunity fields
            'opportunity_id', 'opportunity_name', 'stage', 'probability', 'deal_value',
            'deal_category', 'created_date', 'expected_close_date', 'lead_source',
            'sales_rep', 'sales_rep_experience', 'product_interest', 'competitor',
            'budget_confirmed', 'decision_maker_identified',
            # Interaction fields
            'total_interactions', 'successful_interactions', 'response_rate',
            'last_interaction_date', 'last_interaction_type', 'days_since_last_interaction',
            'email_interactions', 'phone_interactions', 'meeting_interactions',
            'demo_interactions', 'next_follow_up_date',
            # Derived fields
            'days_in_pipeline', 'weighted_deal_value', 'lead_quality_score',
            'predicted_win_probability', 'sales_velocity', 'engagement_score'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_data_types(self, generator):
        """Test that data types are appropriate"""
        data = generator.generate(50)
        
        # String fields
        string_fields = ['contact_id', 'first_name', 'last_name', 'email', 'company_name', 'stage']
        for field in string_fields:
            assert data[field].dtype == 'object', f"{field} should be string type"
        
        # Numeric fields
        numeric_fields = ['deal_value', 'probability', 'employee_count', 'annual_revenue']
        for field in numeric_fields:
            assert pd.api.types.is_numeric_dtype(data[field]), f"{field} should be numeric"
        
        # Boolean fields
        boolean_fields = ['budget_confirmed', 'decision_maker_identified']
        for field in boolean_fields:
            assert data[field].dtype == 'bool', f"{field} should be boolean"
    
    def test_opportunity_stages(self, generator):
        """Test that opportunity stages are realistic"""
        data = generator.generate(200)
        
        valid_stages = ['lead', 'qualified', 'proposal', 'negotiation', 'closed_won', 'lost']
        assert data['stage'].isin(valid_stages).all(), "All stages should be valid"
        
        # Test stage probabilities are realistic
        for stage in data['stage'].unique():
            stage_data = data[data['stage'] == stage]
            expected_prob = generator.opportunity_stages[stage]['probability']
            
            # Allow some variance but should be close to expected
            actual_probs = stage_data['probability'].unique()
            assert len(actual_probs) == 1, f"Stage {stage} should have consistent probability"
            assert actual_probs[0] == expected_prob, f"Stage {stage} probability mismatch"
    
    def test_deal_value_ranges(self, generator):
        """Test that deal values are within realistic ranges"""
        data = generator.generate(200)
        
        # Test deal values are positive
        assert (data['deal_value'] > 0).all(), "All deal values should be positive"
        
        # Test deal categories match values
        for category in data['deal_category'].unique():
            category_data = data[data['deal_category'] == category]
            min_val, max_val = generator.deal_sizes[category]['range']
            
            # Allow some variance due to industry multipliers
            assert (category_data['deal_value'] >= min_val * 0.5).all(), f"{category} deals too small"
            assert (category_data['deal_value'] <= max_val * 2.0).all(), f"{category} deals too large"
    
    def test_interaction_patterns(self, generator):
        """Test that interaction patterns are realistic"""
        data = generator.generate(100)
        
        # Test interaction counts are non-negative
        interaction_fields = ['total_interactions', 'successful_interactions', 
                            'email_interactions', 'phone_interactions', 
                            'meeting_interactions', 'demo_interactions']
        
        for field in interaction_fields:
            assert (data[field] >= 0).all(), f"{field} should be non-negative"
        
        # Test successful interactions <= total interactions
        assert (data['successful_interactions'] <= data['total_interactions']).all(), \
            "Successful interactions should not exceed total"
        
        # Test response rate is between 0 and 1
        assert (data['response_rate'] >= 0).all(), "Response rate should be >= 0"
        assert (data['response_rate'] <= 1).all(), "Response rate should be <= 1"
    
    def test_sales_pipeline_progression(self, generator):
        """Test that sales pipeline progression is realistic"""
        data = generator.generate(200)
        
        # Test that advanced stages have higher probabilities
        stage_order = ['lead', 'qualified', 'proposal', 'negotiation', 'closed_won']
        
        for i in range(len(stage_order) - 1):
            current_stage = stage_order[i]
            next_stage = stage_order[i + 1]
            
            current_prob = generator.opportunity_stages[current_stage]['probability']
            next_prob = generator.opportunity_stages[next_stage]['probability']
            
            assert next_prob >= current_prob, f"{next_stage} should have higher probability than {current_stage}"
    
    def test_lead_quality_scoring(self, generator):
        """Test that lead quality scores are calculated correctly"""
        data = generator.generate(100)
        
        # Test lead quality scores are in valid range
        assert (data['lead_quality_score'] >= 1.0).all(), "Lead quality score should be >= 1"
        assert (data['lead_quality_score'] <= 10.0).all(), "Lead quality score should be <= 10"
        
        # Test that referral leads generally have higher scores
        referral_data = data[data['lead_source'] == 'referral']
        cold_outreach_data = data[data['lead_source'] == 'cold_outreach']
        
        if len(referral_data) > 0 and len(cold_outreach_data) > 0:
            avg_referral_score = referral_data['lead_quality_score'].mean()
            avg_cold_score = cold_outreach_data['lead_quality_score'].mean()
            assert avg_referral_score > avg_cold_score, "Referral leads should have higher quality scores"
    
    def test_win_probability_calculation(self, generator):
        """Test that win probabilities are calculated realistically"""
        data = generator.generate(100)
        
        # Test win probabilities are in valid range
        assert (data['predicted_win_probability'] >= 0.0).all(), "Win probability should be >= 0"
        assert (data['predicted_win_probability'] <= 1.0).all(), "Win probability should be <= 1"
        
        # Test that closed_won opportunities have high win probability
        closed_won_data = data[data['stage'] == 'closed_won']
        if len(closed_won_data) > 0:
            assert (closed_won_data['predicted_win_probability'] > 0.8).all(), \
                "Closed won deals should have high win probability"
    
    def test_engagement_scoring(self, generator):
        """Test that engagement scores reflect interaction patterns"""
        data = generator.generate(100)
        
        # Test engagement scores are in valid range
        assert (data['engagement_score'] >= 0.0).all(), "Engagement score should be >= 0"
        assert (data['engagement_score'] <= 10.0).all(), "Engagement score should be <= 10"
        
        # Test that opportunities with more interactions have higher engagement
        high_interaction = data[data['total_interactions'] > data['total_interactions'].median()]
        low_interaction = data[data['total_interactions'] <= data['total_interactions'].median()]
        
        if len(high_interaction) > 0 and len(low_interaction) > 0:
            avg_high_engagement = high_interaction['engagement_score'].mean()
            avg_low_engagement = low_interaction['engagement_score'].mean()
            assert avg_high_engagement > avg_low_engagement, \
                "High interaction opportunities should have higher engagement scores"
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = generator.generate(
            rows=100,
            time_series=True,
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        # Check that timestamp column exists
        assert 'timestamp' in data.columns, "Time series data should have timestamp column"
        
        # Check timestamp range
        timestamps = pd.to_datetime(data['timestamp'])
        assert timestamps.min() >= start_date, "Timestamps should be >= start_date"
        assert timestamps.max() <= end_date, "Timestamps should be <= end_date"
        
        # Check temporal relationships exist
        temporal_columns = ['hour', 'day_of_week', 'month', 'quarter']
        for col in temporal_columns:
            assert col in data.columns, f"Time series should include {col}"
    
    def test_date_range_parameter(self, generator):
        """Test that date_range parameter works correctly"""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 6, 30)
        
        data = generator.generate(rows=50, date_range=(start_date, end_date))
        
        # Check that created dates are within range
        created_dates = pd.to_datetime(data['created_date'])
        assert created_dates.min().date() >= start_date, "Created dates should be >= start_date"
        assert created_dates.max().date() <= end_date, "Created dates should be <= end_date"
    
    def test_industry_impact_on_deals(self, generator):
        """Test that industry affects deal characteristics"""
        data = generator.generate(200)
        
        # Test that different industries have different average deal values
        industry_avg_values = data.groupby('industry')['deal_value'].mean()
        
        # Should have variation across industries
        assert industry_avg_values.std() > 0, "Industries should have different average deal values"
        
        # Test that sales cycle varies by industry
        for industry in data['industry'].unique():
            industry_data = data[data['industry'] == industry]
            expected_cycle = generator.industries[industry]['sales_cycle_days']
            
            # Expected close dates should reflect industry sales cycle
            days_to_close = (pd.to_datetime(industry_data['expected_close_date']) - 
                           pd.to_datetime(industry_data['created_date'])).dt.days
            
            # Should be reasonably close to expected cycle (allow variance)
            assert days_to_close.mean() >= expected_cycle * 0.5, f"{industry} sales cycle too short"
            assert days_to_close.mean() <= expected_cycle * 2.0, f"{industry} sales cycle too long"
    
    def test_company_size_impact(self, generator):
        """Test that company size affects deal characteristics"""
        data = generator.generate(200)
        
        # Test that larger companies generally have larger deals
        size_order = ['startup', 'small', 'medium', 'large', 'enterprise']
        size_avg_values = {}
        
        for size in size_order:
            size_data = data[data['company_size'] == size]
            if len(size_data) > 0:
                size_avg_values[size] = size_data['deal_value'].mean()
        
        # Generally, larger companies should have larger average deals
        if len(size_avg_values) >= 3:
            values = [size_avg_values.get(size, 0) for size in size_order if size in size_avg_values]
            # Allow some variance but should generally increase
            assert max(values) > min(values), "Company size should impact deal values"
    
    def test_sales_rep_experience_impact(self, generator):
        """Test that sales rep experience affects win rates"""
        data = generator.generate(200)
        
        # Test that more experienced reps have higher predicted win probabilities
        experience_levels = ['junior', 'mid_level', 'senior', 'expert']
        exp_win_rates = {}
        
        for level in experience_levels:
            level_data = data[data['sales_rep_experience'] == level]
            if len(level_data) > 0:
                exp_win_rates[level] = level_data['predicted_win_probability'].mean()
        
        # More experienced reps should generally have higher win rates
        if 'junior' in exp_win_rates and 'expert' in exp_win_rates:
            assert exp_win_rates['expert'] > exp_win_rates['junior'], \
                "Expert reps should have higher win rates than junior reps"
    
    def test_relationship_integrity(self, generator):
        """Test that relationships between entities are maintained"""
        data = generator.generate(100)
        
        # Test that contact and account IDs are properly formatted
        assert data['contact_id'].str.startswith('CONT_').all(), "Contact IDs should have proper format"
        assert data['account_id'].str.startswith('ACC_').all(), "Account IDs should have proper format"
        assert data['opportunity_id'].str.startswith('OPP_').all(), "Opportunity IDs should have proper format"
        
        # Test that email addresses are properly formatted
        assert data['email'].str.contains('@').all(), "All emails should contain @"
        assert data['email'].str.contains('.').all(), "All emails should contain domain"
        
        # Test that dates are logical
        created_dates = pd.to_datetime(data['created_date'])
        expected_close_dates = pd.to_datetime(data['expected_close_date'])
        
        # Expected close should be after created date
        assert (expected_close_dates >= created_dates).all(), \
            "Expected close date should be after created date"
    
    def test_realistic_patterns_application(self, generator):
        """Test that realistic patterns are properly applied"""
        data = generator.generate(100)
        
        # Test derived fields are calculated
        derived_fields = ['days_in_pipeline', 'weighted_deal_value', 'sales_velocity']
        for field in derived_fields:
            assert field in data.columns, f"Derived field {field} should be present"
            assert not data[field].isna().all(), f"Derived field {field} should have values"
        
        # Test weighted deal value calculation
        expected_weighted = data['deal_value'] * data['probability']
        pd.testing.assert_series_equal(
            data['weighted_deal_value'], 
            expected_weighted, 
            check_names=False,
            rtol=1e-10
        )
        
        # Test that data is sorted by created_date
        created_dates = pd.to_datetime(data['created_date'])
        assert created_dates.is_monotonic_increasing, "Data should be sorted by created_date"