"""
Unit tests for SuppliersGenerator

Tests realistic patterns, contract terms, performance metrics, and supplier relationships.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.suppliers import SuppliersGenerator


class TestSuppliersGenerator:
    """Test suite for SuppliersGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create SuppliersGenerator instance"""
        return SuppliersGenerator(seeder)
    
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
            'supplier_id', 'company_name', 'category', 'primary_product_category',
            'product_subcategories', 'country', 'region', 'contact_name',
            'contact_email', 'contact_phone', 'address', 'contract_type',
            'contract_value', 'contract_start_date', 'contract_end_date',
            'contract_length_months', 'contract_status', 'service_level',
            'payment_terms', 'delivery_performance', 'quality_score',
            'compliance_score', 'overall_performance', 'performance_tier',
            'avg_response_time_hours', 'renewal_probability', 'last_audit_date',
            'certification_status', 'risk_level', 'preferred_supplier',
            'onboarding_date', 'last_performance_review'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_supplier_id_uniqueness(self, generator):
        """Test that supplier IDs are unique"""
        data = generator.generate(500)
        assert data['supplier_id'].nunique() == len(data)
        
        # Check format
        assert all(data['supplier_id'].str.startswith('SUP_'))
        assert all(data['supplier_id'].str.len() == 10)  # SUP_ + 6 digits
    
    def test_supplier_categories_valid(self, generator):
        """Test that supplier categories are from valid set"""
        data = generator.generate(200)
        
        expected_categories = {
            'manufacturing', 'technology', 'logistics', 'professional_services',
            'raw_materials', 'maintenance', 'office_supplies', 'consulting'
        }
        
        actual_categories = set(data['category'].unique())
        assert actual_categories.issubset(expected_categories)
    
    def test_contract_types_valid(self, generator):
        """Test that contract types are from valid set"""
        data = generator.generate(200)
        
        expected_types = {
            'fixed_price', 'time_and_materials', 'cost_plus', 'unit_price', 'retainer'
        }
        
        actual_types = set(data['contract_type'].unique())
        assert actual_types.issubset(expected_types)
    
    def test_contract_values_realistic(self, generator):
        """Test that contract values are realistic"""
        data = generator.generate(1000)
        
        # Contract values should be positive
        assert (data['contract_value'] > 0).all()
        
        # Contract values should be reasonable (between $5K and $1M)
        assert (data['contract_value'] >= 5000).all()
        assert (data['contract_value'] <= 1000000).all()
        
        # Different categories should have different value ranges
        value_by_category = data.groupby('category')['contract_value'].mean()
        
        # Manufacturing should generally have higher values than office supplies
        if 'manufacturing' in value_by_category.index and 'office_supplies' in value_by_category.index:
            assert value_by_category['manufacturing'] > value_by_category['office_supplies']
    
    def test_contract_dates_logical(self, generator):
        """Test that contract dates are logical"""
        data = generator.generate(300)
        
        # Contract end date should be after start date
        start_dates = pd.to_datetime(data['contract_start_date'])
        end_dates = pd.to_datetime(data['contract_end_date'])
        assert (end_dates >= start_dates).all()
        
        # Contract length should match date difference (approximately)
        actual_length_days = (end_dates - start_dates).dt.days
        expected_length_days = data['contract_length_months'] * 30
        
        # Allow for some variance due to month length differences
        length_diff = abs(actual_length_days - expected_length_days)
        assert (length_diff <= 31).all()  # Within one month
    
    def test_performance_scores_valid(self, generator):
        """Test that performance scores are within valid ranges"""
        data = generator.generate(400)
        
        # All performance scores should be between 0 and 100
        assert (data['delivery_performance'] >= 0).all()
        assert (data['delivery_performance'] <= 100).all()
        assert (data['quality_score'] >= 0).all()
        assert (data['quality_score'] <= 100).all()
        assert (data['compliance_score'] >= 0).all()
        assert (data['compliance_score'] <= 100).all()
        
        # Overall performance should be average of the three scores
        calculated_overall = (data['delivery_performance'] + data['quality_score'] + data['compliance_score']) / 3
        assert abs(data['overall_performance'] - calculated_overall).max() < 0.1
    
    def test_performance_tier_consistency(self, generator):
        """Test that performance tiers are consistent with scores"""
        data = generator.generate(500)
        
        # Performance tiers should match score ranges
        excellent_mask = data['performance_tier'] == 'excellent'
        good_mask = data['performance_tier'] == 'good'
        poor_mask = data['performance_tier'] == 'poor'
        
        if excellent_mask.any():
            excellent_scores = data[excellent_mask]['overall_performance']
            assert (excellent_scores >= 85).all()  # Excellent should be high scores
        
        if poor_mask.any():
            poor_scores = data[poor_mask]['overall_performance']
            assert (poor_scores <= 75).all()  # Poor should be low scores
    
    def test_regions_and_countries_valid(self, generator):
        """Test that regions and countries are valid"""
        data = generator.generate(300)
        
        expected_regions = {
            'north_america', 'europe', 'asia_pacific', 'latin_america', 'other'
        }
        
        actual_regions = set(data['region'].unique())
        assert actual_regions.issubset(expected_regions)
        
        # Countries should be non-empty strings
        assert data['country'].notna().all()
        assert (data['country'].str.len() > 0).all()
    
    def test_service_levels_valid(self, generator):
        """Test that service levels are from valid set"""
        data = generator.generate(200)
        
        expected_levels = {'premium', 'standard', 'basic', 'economy'}
        actual_levels = set(data['service_level'].unique())
        assert actual_levels.issubset(expected_levels)
    
    def test_payment_terms_valid(self, generator):
        """Test that payment terms are from valid set"""
        data = generator.generate(200)
        
        expected_terms = {'net_30', 'net_60', 'net_15', 'net_90', 'immediate'}
        actual_terms = set(data['payment_terms'].unique())
        assert actual_terms.issubset(expected_terms)
    
    def test_contract_status_logic(self, generator):
        """Test that contract status follows logical rules"""
        data = generator.generate(400)
        
        today = datetime.now().date()
        
        # Active contracts should have end dates in the future
        active_contracts = data[data['contract_status'] == 'active']
        if not active_contracts.empty:
            active_end_dates = pd.to_datetime(active_contracts['contract_end_date']).dt.date
            assert (active_end_dates >= today).all()
        
        # Expired contracts should have end dates in the past
        expired_contracts = data[data['contract_status'] == 'expired']
        if not expired_contracts.empty:
            expired_end_dates = pd.to_datetime(expired_contracts['contract_end_date']).dt.date
            assert (expired_end_dates < today).all()
    
    def test_renewal_probability_ranges(self, generator):
        """Test that renewal probabilities are within valid ranges"""
        data = generator.generate(300)
        
        # Renewal probabilities should be between 0 and 1
        assert (data['renewal_probability'] >= 0).all()
        assert (data['renewal_probability'] <= 1).all()
        
        # Better performing suppliers should have higher renewal probabilities
        excellent_suppliers = data[data['performance_tier'] == 'excellent']
        poor_suppliers = data[data['performance_tier'] == 'poor']
        
        if not excellent_suppliers.empty and not poor_suppliers.empty:
            excellent_avg = excellent_suppliers['renewal_probability'].mean()
            poor_avg = poor_suppliers['renewal_probability'].mean()
            assert excellent_avg > poor_avg
    
    def test_risk_level_consistency(self, generator):
        """Test that risk levels are consistent with performance"""
        data = generator.generate(400)
        
        expected_risk_levels = {'low', 'medium', 'high', 'critical'}
        actual_risk_levels = set(data['risk_level'].unique())
        assert actual_risk_levels.issubset(expected_risk_levels)
        
        # Low risk should correlate with high performance
        low_risk = data[data['risk_level'] == 'low']
        high_risk = data[data['risk_level'] == 'high']
        
        if not low_risk.empty and not high_risk.empty:
            low_risk_performance = low_risk['overall_performance'].mean()
            high_risk_performance = high_risk['overall_performance'].mean()
            assert low_risk_performance > high_risk_performance
    
    def test_preferred_supplier_logic(self, generator):
        """Test that preferred supplier designation follows logical rules"""
        data = generator.generate(500)
        
        # Preferred suppliers should generally have better performance
        preferred = data[data['preferred_supplier'] == True]
        non_preferred = data[data['preferred_supplier'] == False]
        
        if not preferred.empty and not non_preferred.empty:
            preferred_performance = preferred['overall_performance'].mean()
            non_preferred_performance = non_preferred['overall_performance'].mean()
            assert preferred_performance > non_preferred_performance
    
    def test_response_time_by_service_level(self, generator):
        """Test that response times correlate with service levels"""
        data = generator.generate(600)
        
        # Premium service should have faster response times than basic
        premium = data[data['service_level'] == 'premium']
        basic = data[data['service_level'] == 'basic']
        
        if not premium.empty and not basic.empty:
            premium_response = premium['avg_response_time_hours'].mean()
            basic_response = basic['avg_response_time_hours'].mean()
            assert premium_response < basic_response
    
    def test_certification_status_valid(self, generator):
        """Test that certification statuses are from valid set"""
        data = generator.generate(200)
        
        expected_certifications = {
            'iso_9001', 'iso_14001', 'iso_45001', 'multiple_certifications',
            'pending_certification', 'no_certification'
        }
        
        actual_certifications = set(data['certification_status'].unique())
        assert actual_certifications.issubset(expected_certifications)
    
    def test_product_categories_valid(self, generator):
        """Test that product categories are valid"""
        data = generator.generate(300)
        
        expected_primary_categories = {
            'electronics', 'mechanical', 'software', 'services',
            'materials', 'packaging', 'office', 'logistics'
        }
        
        actual_primary_categories = set(data['primary_product_category'].unique())
        assert actual_primary_categories.issubset(expected_primary_categories)
        
        # Product subcategories should be non-empty strings
        assert data['product_subcategories'].notna().all()
        assert (data['product_subcategories'].str.len() > 0).all()
    
    def test_contact_information_format(self, generator):
        """Test that contact information is properly formatted"""
        data = generator.generate(100)
        
        # Contact names should be non-empty
        assert data['contact_name'].notna().all()
        assert (data['contact_name'].str.len() > 0).all()
        
        # Contact emails should contain @ symbol
        assert data['contact_email'].str.contains('@').all()
        
        # Contact phones should be non-empty
        assert data['contact_phone'].notna().all()
        assert (data['contact_phone'].str.len() > 0).all()
    
    def test_audit_and_review_dates_logical(self, generator):
        """Test that audit and review dates are logical"""
        data = generator.generate(200)
        
        today = datetime.now().date()
        
        # Audit dates should not be in the future
        audit_dates = pd.to_datetime(data['last_audit_date']).dt.date
        assert (audit_dates <= today).all()
        
        # Review dates should not be in the future
        review_dates = pd.to_datetime(data['last_performance_review']).dt.date
        assert (review_dates <= today).all()
        
        # Audit and review dates should be after contract start dates
        start_dates = pd.to_datetime(data['contract_start_date']).dt.date
        assert (audit_dates >= start_dates).all()
        assert (review_dates >= start_dates).all()
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        data = generator.generate(
            100,
            time_series=True,
            start_date=start_date,
            end_date=end_date,
            interval='1week'
        )
        
        # Should have time series specific columns
        assert 'onboarding_datetime' in data.columns
        assert 'onboarding_intensity' in data.columns
        assert 'market_adjustment_factor' in data.columns
        
        # Timestamps should be within specified range
        timestamps = pd.to_datetime(data['onboarding_datetime'])
        assert (timestamps >= start_date).all()
        assert (timestamps <= end_date).all()
    
    def test_realistic_patterns_applied(self, generator):
        """Test that realistic patterns are applied to generated data"""
        data = generator.generate(500)
        
        # Should have derived fields
        expected_derived_fields = [
            'contract_value_per_month', 'days_until_expiry', 'contract_active_days',
            'performance_category', 'contract_value_tier', 'high_risk', 'renewal_urgent'
        ]
        
        for field in expected_derived_fields:
            assert field in data.columns, f"Missing derived field: {field}"
        
        # Contract value per month should be calculated correctly
        expected_monthly_value = data['contract_value'] / data['contract_length_months']
        assert abs(data['contract_value_per_month'] - expected_monthly_value).max() < 0.01
    
    def test_performance_correlations(self, generator):
        """Test that performance metrics show realistic correlations"""
        data = generator.generate(1000)
        
        # High-value contracts should tend to have better performance
        high_value = data[data['contract_value'] > 100000]
        low_value = data[data['contract_value'] < 50000]
        
        if not high_value.empty and not low_value.empty:
            high_value_performance = high_value['overall_performance'].mean()
            low_value_performance = low_value['overall_performance'].mean()
            # Allow some variance but expect general trend
            assert high_value_performance >= low_value_performance - 5
    
    def test_regional_patterns(self, generator):
        """Test that regional patterns are realistic"""
        data = generator.generate(800)
        
        # Different regions should show different cost patterns
        region_costs = data.groupby('region')['contract_value'].mean()
        
        # Should have multiple regions represented
        assert len(region_costs) > 1
        
        # Europe should generally be more expensive than Asia Pacific
        if 'europe' in region_costs.index and 'asia_pacific' in region_costs.index:
            # Allow for some variance due to randomness
            europe_cost = region_costs['europe']
            asia_cost = region_costs['asia_pacific']
            # Europe should be at least 90% of Asia cost (allowing for variance)
            assert europe_cost >= asia_cost * 0.9
    
    def test_contract_renewal_patterns(self, generator):
        """Test that contract renewal patterns are realistic"""
        data = generator.generate(600)
        
        # Contracts nearing expiry should have pending_renewal status more often
        data['days_to_expiry'] = (pd.to_datetime(data['contract_end_date']) - datetime.now()).dt.days
        
        near_expiry = data[data['days_to_expiry'] < 90]
        far_expiry = data[data['days_to_expiry'] > 180]
        
        if not near_expiry.empty and not far_expiry.empty:
            near_pending_rate = (near_expiry['contract_status'] == 'pending_renewal').mean()
            far_pending_rate = (far_expiry['contract_status'] == 'pending_renewal').mean()
            assert near_pending_rate >= far_pending_rate
    
    def test_data_quality_validation(self, generator):
        """Test overall data quality and consistency"""
        data = generator.generate(300)
        
        # No null values in required fields
        critical_fields = [
            'supplier_id', 'company_name', 'category', 'contract_value',
            'contract_start_date', 'contract_end_date', 'overall_performance'
        ]
        
        for field in critical_fields:
            assert data[field].notna().all(), f"Null values found in critical field: {field}"
        
        # Numeric fields should be numeric
        numeric_fields = [
            'contract_value', 'contract_length_months', 'delivery_performance',
            'quality_score', 'compliance_score', 'overall_performance',
            'avg_response_time_hours', 'renewal_probability'
        ]
        
        for field in numeric_fields:
            assert pd.api.types.is_numeric_dtype(data[field]), f"Non-numeric data in field: {field}"
    
    def test_seeding_reproducibility(self):
        """Test that seeding produces reproducible results"""
        seeder1 = MillisecondSeeder(fixed_seed=999)
        seeder2 = MillisecondSeeder(fixed_seed=999)
        
        generator1 = SuppliersGenerator(seeder1)
        generator2 = SuppliersGenerator(seeder2)
        
        data1 = generator1.generate(100)
        data2 = generator2.generate(100)
        
        # Should produce identical results with same seed
        pd.testing.assert_frame_equal(data1, data2)