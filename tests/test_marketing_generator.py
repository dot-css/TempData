"""
Unit tests for MarketingGenerator

Tests realistic patterns, conversion funnels, seasonal trends, and lead scoring.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.marketing import MarketingGenerator


class TestMarketingGenerator:
    """Test suite for MarketingGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create MarketingGenerator instance"""
        return MarketingGenerator(seeder)
    
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
            'campaign_id', 'campaign_name', 'campaign_type', 'channel',
            'start_date', 'end_date', 'budget', 'total_cost', 'impressions',
            'clicks', 'conversions', 'ctr', 'conversion_rate', 'cost_per_click',
            'cost_per_conversion', 'lead_score_avg', 'campaign_objective',
            'target_audience', 'campaign_manager', 'status', 'roi'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_campaign_id_uniqueness(self, generator):
        """Test that campaign IDs are unique"""
        data = generator.generate(500)
        assert data['campaign_id'].nunique() == len(data)
        
        # Check format
        assert all(data['campaign_id'].str.startswith('CAMP_'))
        assert all(data['campaign_id'].str.len() == 11)  # CAMP_ + 6 digits
    
    def test_campaign_types_valid(self, generator):
        """Test that campaign types are from valid set"""
        data = generator.generate(200)
        
        expected_types = {
            'brand_awareness', 'lead_generation', 'product_launch',
            'retargeting', 'seasonal_promotion', 'content_marketing'
        }
        
        actual_types = set(data['campaign_type'].unique())
        assert actual_types.issubset(expected_types)
    
    def test_marketing_channels_valid(self, generator):
        """Test that marketing channels are from valid set"""
        data = generator.generate(200)
        
        expected_channels = {
            'google_ads', 'facebook_ads', 'linkedin_ads', 'email_marketing',
            'content_marketing', 'display_advertising', 'video_advertising',
            'influencer_marketing'
        }
        
        actual_channels = set(data['channel'].unique())
        assert actual_channels.issubset(expected_channels)
    
    def test_budget_ranges_realistic(self, generator):
        """Test that budget ranges are realistic"""
        data = generator.generate(1000)
        
        # Budget should be positive
        assert (data['budget'] > 0).all()
        
        # Budget should be reasonable (between $1K and $500K)
        assert (data['budget'] >= 1000).all()
        assert (data['budget'] <= 500000).all()
        
        # Total cost should not exceed budget
        assert (data['total_cost'] <= data['budget']).all()
    
    def test_conversion_funnel_logic(self, generator):
        """Test conversion funnel logic (impressions > clicks > conversions)"""
        data = generator.generate(500)
        
        # Impressions should be greater than clicks
        assert (data['impressions'] >= data['clicks']).all()
        
        # Clicks should be greater than or equal to conversions
        assert (data['clicks'] >= data['conversions']).all()
        
        # CTR should be reasonable (0.1% to 15%)
        assert (data['ctr'] >= 0.1).all()
        assert (data['ctr'] <= 15.0).all()
        
        # Conversion rate should be reasonable (0.1% to 20%)
        assert (data['conversion_rate'] >= 0.1).all()
        assert (data['conversion_rate'] <= 20.0).all()
    
    def test_cost_metrics_consistency(self, generator):
        """Test cost metrics are consistent"""
        data = generator.generate(300)
        
        # Cost per click should be positive
        assert (data['cost_per_click'] > 0).all()
        
        # Cost per conversion should be positive
        assert (data['cost_per_conversion'] > 0).all()
        
        # Cost per conversion should generally be higher than cost per click
        higher_cpc_ratio = (data['cost_per_conversion'] >= data['cost_per_click']).mean()
        assert higher_cpc_ratio > 0.8  # At least 80% should follow this pattern
    
    def test_lead_scoring_ranges(self, generator):
        """Test lead scoring is within expected ranges"""
        data = generator.generate(400)
        
        # Lead scores should be between 1 and 10
        assert (data['lead_score_avg'] >= 1.0).all()
        assert (data['lead_score_avg'] <= 10.0).all()
        
        # Different campaign types should have different average lead scores
        lead_scores_by_type = data.groupby('campaign_type')['lead_score_avg'].mean()
        
        # Lead generation should have higher scores than brand awareness
        if 'lead_generation' in lead_scores_by_type.index and 'brand_awareness' in lead_scores_by_type.index:
            assert lead_scores_by_type['lead_generation'] > lead_scores_by_type['brand_awareness']
    
    def test_channel_performance_patterns(self, generator):
        """Test channel-specific performance patterns"""
        data = generator.generate(1000)
        
        channel_metrics = data.groupby('channel').agg({
            'ctr': 'mean',
            'conversion_rate': 'mean',
            'cost_per_click': 'mean',
            'lead_score_avg': 'mean'
        })
        
        # LinkedIn should have higher cost per click than Facebook
        if 'linkedin_ads' in channel_metrics.index and 'facebook_ads' in channel_metrics.index:
            assert channel_metrics.loc['linkedin_ads', 'cost_per_click'] > channel_metrics.loc['facebook_ads', 'cost_per_click']
        
        # Email marketing should have higher conversion rates
        if 'email_marketing' in channel_metrics.index:
            avg_conversion_rate = data['conversion_rate'].mean()
            assert channel_metrics.loc['email_marketing', 'conversion_rate'] >= avg_conversion_rate * 0.8
    
    def test_seasonal_patterns(self, generator):
        """Test seasonal marketing patterns"""
        # Generate data for different months
        data_jan = generator.generate(100, date_range=(date(2024, 1, 1), date(2024, 1, 31)))
        data_nov = generator.generate(100, date_range=(date(2024, 11, 1), date(2024, 11, 30)))
        
        # November (holiday season) should have higher average performance
        jan_avg_budget = data_jan['budget'].mean()
        nov_avg_budget = data_nov['budget'].mean()
        
        # Allow some variance but November should generally be higher
        assert nov_avg_budget >= jan_avg_budget * 0.9  # At least 90% of January levels
    
    def test_roi_calculation(self, generator):
        """Test ROI calculation logic"""
        data = generator.generate(200)
        
        # ROI should be calculated correctly
        # ROI = ((conversions * 100) - total_cost) / total_cost * 100
        expected_roi = ((data['conversions'] * 100) - data['total_cost']) / data['total_cost'] * 100
        
        # Allow for small rounding differences
        roi_diff = abs(data['roi'] - expected_roi)
        assert (roi_diff <= 0.1).all()  # Within 0.1% due to rounding
    
    def test_campaign_status_distribution(self, generator):
        """Test campaign status distribution"""
        data = generator.generate(500)
        
        expected_statuses = {'active', 'paused', 'completed', 'draft'}
        actual_statuses = set(data['status'].unique())
        
        assert actual_statuses.issubset(expected_statuses)
        
        # Should have reasonable distribution (no single status > 80%)
        status_counts = data['status'].value_counts(normalize=True)
        assert (status_counts <= 0.8).all()
    
    def test_campaign_objectives_valid(self, generator):
        """Test campaign objectives are valid"""
        data = generator.generate(300)
        
        expected_objectives = {
            'increase_brand_awareness', 'generate_leads', 'drive_sales',
            'customer_acquisition', 'customer_retention', 'product_education',
            'market_expansion', 'competitive_positioning'
        }
        
        actual_objectives = set(data['campaign_objective'].unique())
        assert actual_objectives.issubset(expected_objectives)
    
    def test_target_audience_format(self, generator):
        """Test target audience format"""
        data = generator.generate(100)
        
        # Target audience should contain demographic info
        assert all(data['target_audience'].str.len() > 10)  # Should be descriptive
        
        # Should contain age ranges
        age_pattern_found = data['target_audience'].str.contains(r'\d+-\d+').any()
        assert age_pattern_found
    
    def test_campaign_name_generation(self, generator):
        """Test campaign name generation"""
        data = generator.generate(200)
        
        # Campaign names should be unique enough
        unique_ratio = data['campaign_name'].nunique() / len(data)
        assert unique_ratio > 0.7  # At least 70% unique names
        
        # Names should be reasonable length
        assert (data['campaign_name'].str.len() >= 5).all()
        assert (data['campaign_name'].str.len() <= 50).all()
    
    def test_date_consistency(self, generator):
        """Test date consistency"""
        data = generator.generate(100)
        
        # End date should be after start date
        start_dates = pd.to_datetime(data['start_date'])
        end_dates = pd.to_datetime(data['end_date'])
        
        assert (end_dates >= start_dates).all()
        
        # Campaign duration should be reasonable (7-90 days)
        duration = (end_dates - start_dates).dt.days
        assert (duration >= 7).all()
        assert (duration <= 90).all()
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = generator.generate(
            100,
            time_series=True,
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        # Should have timestamp column
        assert 'timestamp' in data.columns
        
        # Timestamps should be within range
        timestamps = pd.to_datetime(data['timestamp'])
        assert (timestamps >= start_date).all()
        assert (timestamps <= end_date).all()
        
        # Should be chronologically ordered
        timestamps_list = timestamps.tolist()
        assert timestamps_list == sorted(timestamps_list)
        
        # Should have performance_score column for time series
        assert 'performance_score' in data.columns
        assert (data['performance_score'] > 0).all()
    
    def test_time_series_temporal_patterns(self, generator):
        """Test temporal patterns in time series data"""
        # Generate weekday vs weekend data
        data = generator.generate(
            200,
            time_series=True,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            interval='1day'
        )
        
        # Add day of week
        data['day_of_week_num'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        
        # Weekday performance should generally be higher than weekend
        weekday_data = data[data['day_of_week_num'] < 5]
        weekend_data = data[data['day_of_week_num'] >= 5]
        
        if len(weekday_data) > 0 and len(weekend_data) > 0:
            weekday_avg = weekday_data['performance_score'].mean()
            weekend_avg = weekend_data['performance_score'].mean()
            
            # Allow some variance but weekdays should generally be higher
            assert weekday_avg >= weekend_avg * 0.8
    
    def test_derived_fields(self, generator):
        """Test derived fields are calculated correctly"""
        data = generator.generate(100)
        
        # Should have derived fields
        derived_fields = [
            'cost_per_impression', 'conversion_value', 'profit',
            'profit_margin', 'performance_tier', 'lead_quality_tier',
            'budget_tier'
        ]
        
        for field in derived_fields:
            assert field in data.columns
        
        # Cost per impression should be positive
        assert (data['cost_per_impression'] > 0).all()
        
        # Conversion value should equal conversions * 100
        expected_conv_value = data['conversions'] * 100
        assert (data['conversion_value'] == expected_conv_value).all()
        
        # Profit should equal conversion_value - total_cost
        expected_profit = data['conversion_value'] - data['total_cost']
        assert (data['profit'] == expected_profit).all()
    
    def test_performance_tiers(self, generator):
        """Test performance tier categorization"""
        data = generator.generate(300)
        
        expected_tiers = {'poor', 'average', 'good', 'excellent'}
        actual_tiers = set(data['performance_tier'].dropna().unique())
        
        assert actual_tiers.issubset(expected_tiers)
        
        # Verify tier logic
        poor_campaigns = data[data['performance_tier'] == 'poor']
        excellent_campaigns = data[data['performance_tier'] == 'excellent']
        
        if len(poor_campaigns) > 0 and len(excellent_campaigns) > 0:
            assert poor_campaigns['roi'].max() <= excellent_campaigns['roi'].min()
    
    def test_lead_quality_tiers(self, generator):
        """Test lead quality tier categorization"""
        data = generator.generate(300)
        
        expected_tiers = {'low', 'medium', 'high', 'premium'}
        actual_tiers = set(data['lead_quality_tier'].dropna().unique())
        
        assert actual_tiers.issubset(expected_tiers)
        
        # Verify tier logic
        low_quality = data[data['lead_quality_tier'] == 'low']
        premium_quality = data[data['lead_quality_tier'] == 'premium']
        
        if len(low_quality) > 0 and len(premium_quality) > 0:
            assert low_quality['lead_score_avg'].max() <= premium_quality['lead_score_avg'].min()
    
    def test_budget_tiers(self, generator):
        """Test budget tier categorization"""
        data = generator.generate(300)
        
        expected_tiers = {'small', 'medium', 'large', 'enterprise'}
        actual_tiers = set(data['budget_tier'].dropna().unique())
        
        assert actual_tiers.issubset(expected_tiers)
        
        # Verify tier logic
        small_budget = data[data['budget_tier'] == 'small']
        enterprise_budget = data[data['budget_tier'] == 'enterprise']
        
        if len(small_budget) > 0 and len(enterprise_budget) > 0:
            assert small_budget['budget'].max() <= enterprise_budget['budget'].min()
    
    def test_reproducibility(self, generator):
        """Test that generation is reproducible with same seed"""
        data1 = generator.generate(50)
        
        # Create new generator with same seed
        seeder2 = MillisecondSeeder(fixed_seed=123456789)
        generator2 = MarketingGenerator(seeder2)
        data2 = generator2.generate(50)
        
        # Should be identical
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_validation(self, generator):
        """Test overall data quality"""
        data = generator.generate(500)
        
        # No null values in required fields
        critical_fields = ['campaign_id', 'budget', 'impressions', 'clicks', 'conversions']
        for field in critical_fields:
            assert data[field].notna().all()
        
        # Reasonable data distributions
        assert data['budget'].std() > 0  # Should have variance
        assert data['impressions'].std() > 0
        assert data['conversions'].std() > 0
        
        # No negative values where inappropriate
        non_negative_fields = ['budget', 'total_cost', 'impressions', 'clicks', 'conversions']
        for field in non_negative_fields:
            assert (data[field] >= 0).all()
    
    def test_campaign_manager_names(self, generator):
        """Test campaign manager names are realistic"""
        data = generator.generate(100)
        
        # Should have campaign manager names
        assert data['campaign_manager'].notna().all()
        
        # Names should be reasonable length
        assert (data['campaign_manager'].str.len() >= 3).all()
        assert (data['campaign_manager'].str.len() <= 50).all()
        
        # Should have some variety
        unique_ratio = data['campaign_manager'].nunique() / len(data)
        assert unique_ratio > 0.3  # At least 30% unique names