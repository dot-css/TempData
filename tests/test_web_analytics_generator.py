"""
Unit tests for WebAnalyticsGenerator

Tests web analytics data generation patterns, user sessions, device types,
bounce rates, and conversion funnels.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.technology.web_analytics import WebAnalyticsGenerator


class TestWebAnalyticsGenerator:
    """Test suite for WebAnalyticsGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create WebAnalyticsGenerator instance with fixed seed"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        return WebAnalyticsGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        assert not data.empty
    
    def test_required_columns(self, generator):
        """Test that all required columns are present"""
        data = generator.generate(50)
        
        required_columns = [
            'session_id', 'user_id', 'page_view_id', 'timestamp',
            'page_url', 'page_category', 'device_type', 'browser',
            'user_segment', 'traffic_source', 'time_on_page_seconds',
            'is_bounce', 'is_exit', 'converted'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_device_type_distributions(self, generator):
        """Test realistic device type distributions"""
        data = generator.generate(1000)
        device_counts = data['device_type'].value_counts(normalize=True)
        
        # Check that mobile and desktop are most common
        assert device_counts['mobile'] > 0.35  # Should be around 48%
        assert device_counts['desktop'] > 0.35  # Should be around 45%
        assert device_counts['tablet'] < 0.15  # Should be around 7%
        
        # Check all expected device types are present
        expected_devices = {'desktop', 'mobile', 'tablet'}
        assert set(device_counts.index) == expected_devices
    
    def test_user_session_patterns(self, generator):
        """Test user session patterns and behavior"""
        data = generator.generate(500)
        
        # Test session grouping
        session_groups = data.groupby('session_id')
        
        # Each session should have consistent user_id, device_type, etc.
        for session_id, group in session_groups:
            assert group['user_id'].nunique() == 1, "Session should have single user"
            assert group['device_type'].nunique() == 1, "Session should have single device type"
            assert group['browser'].nunique() == 1, "Session should have single browser"
            assert group['traffic_source'].nunique() == 1, "Session should have single traffic source"
    
    def test_page_view_distributions(self, generator):
        """Test page view category distributions"""
        data = generator.generate(1000)
        page_counts = data['page_category'].value_counts(normalize=True)
        
        # Product and home pages should be most common
        assert page_counts.get('product', 0) > 0.20  # Should be around 30%
        assert page_counts.get('home', 0) > 0.15     # Should be around 25%
        
        # Check expected page categories exist
        expected_categories = {
            'home', 'product', 'category', 'checkout', 
            'blog', 'support', 'account'
        }
        assert set(page_counts.index).issubset(expected_categories)
    
    def test_bounce_rates_realistic(self, generator):
        """Test that bounce rates are realistic"""
        data = generator.generate(1000)
        
        # Overall bounce rate should be reasonable (5-70% depending on site quality)
        bounce_rate = data['is_bounce'].mean()
        assert 0.05 < bounce_rate < 0.70, f"Bounce rate {bounce_rate} seems unrealistic"
        
        # Check that we have both bounced and non-bounced sessions
        assert data['is_bounce'].sum() > 0, "Should have some bounced sessions"
        assert (~data['is_bounce']).sum() > 0, "Should have some non-bounced sessions"
        
        # Mobile should have higher bounce rate than desktop (if both exist)
        mobile_data = data[data['device_type'] == 'mobile']
        desktop_data = data[data['device_type'] == 'desktop']
        
        if len(mobile_data) > 10 and len(desktop_data) > 10:
            mobile_bounce = mobile_data['is_bounce'].mean()
            desktop_bounce = desktop_data['is_bounce'].mean()
            # Allow for some variance, but mobile should generally be higher
            assert mobile_bounce >= desktop_bounce * 0.8, "Mobile bounce rate should be comparable or higher than desktop"
    
    def test_conversion_funnels(self, generator):
        """Test conversion funnel patterns"""
        data = generator.generate(1000)
        
        # Conversion rate should be reasonable (1-15%)
        conversion_rate = data['converted'].mean()
        assert 0.01 < conversion_rate < 0.20, f"Conversion rate {conversion_rate} seems unrealistic"
        
        # Checkout pages should have higher conversion rates
        checkout_data = data[data['page_category'] == 'checkout']
        if len(checkout_data) > 0:
            checkout_conversion = checkout_data['converted'].mean()
            overall_conversion = data['converted'].mean()
            assert checkout_conversion > overall_conversion, "Checkout should have higher conversion"
    
    def test_traffic_source_patterns(self, generator):
        """Test traffic source distributions and patterns"""
        data = generator.generate(1000)
        source_counts = data['traffic_source'].value_counts(normalize=True)
        
        # Organic search and direct should be most common
        assert source_counts.get('organic_search', 0) > 0.25  # Should be around 35%
        assert source_counts.get('direct', 0) > 0.15          # Should be around 25%
        
        # Check expected traffic sources
        expected_sources = {
            'organic_search', 'direct', 'social_media', 
            'paid_search', 'email', 'referral'
        }
        assert set(source_counts.index).issubset(expected_sources)
    
    def test_time_series_generation(self, generator):
        """Test time series data generation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = generator.generate(
            200, 
            time_series=True, 
            date_range=(start_date, end_date)
        )
        
        # All timestamps should be within range
        assert data['timestamp'].min() >= start_date
        assert data['timestamp'].max() <= end_date
        
        # Data should be chronologically ordered
        timestamps = data['timestamp'].tolist()
        assert timestamps == sorted(timestamps), "Data should be chronologically ordered"
    
    def test_user_agent_generation(self, generator):
        """Test user agent string generation"""
        data = generator.generate(100)
        
        # All records should have user agents
        assert data['user_agent'].notna().all(), "All records should have user agents"
        
        # User agents should be realistic strings
        for user_agent in data['user_agent'].unique():
            assert 'Mozilla' in user_agent, "User agent should contain Mozilla"
            assert len(user_agent) > 50, "User agent should be reasonably long"
    
    def test_engagement_metrics(self, generator):
        """Test engagement score calculation"""
        data = generator.generate(200)
        
        # Engagement scores should be reasonable (0-5 range typically)
        assert data['engagement_score'].min() >= 0, "Engagement score should be non-negative"
        assert data['engagement_score'].max() <= 5, "Engagement score should be reasonable"
        
        # Higher time on page should correlate with higher engagement
        high_time_data = data[data['time_on_page_seconds'] > 120]
        low_time_data = data[data['time_on_page_seconds'] < 30]
        
        if len(high_time_data) > 0 and len(low_time_data) > 0:
            high_engagement = high_time_data['engagement_score'].mean()
            low_engagement = low_time_data['engagement_score'].mean()
            assert high_engagement > low_engagement, "Higher time should mean higher engagement"
    
    def test_geographical_data(self, generator):
        """Test geographical data generation"""
        data = generator.generate(100)
        
        # Should have IP addresses and location data
        assert data['ip_address'].notna().all(), "All records should have IP addresses"
        assert data['country'].notna().all(), "All records should have countries"
        assert data['city'].notna().all(), "All records should have cities"
        
        # IP addresses should be valid format
        for ip in data['ip_address'].unique()[:10]:  # Check first 10
            parts = ip.split('.')
            assert len(parts) == 4, f"Invalid IP format: {ip}"
            for part in parts:
                assert 0 <= int(part) <= 255, f"Invalid IP part: {part}"
    
    def test_session_duration_patterns(self, generator):
        """Test session duration patterns"""
        data = generator.generate(500)
        
        # Session durations should be positive
        assert (data['time_on_page_seconds'] > 0).all(), "Time on page should be positive"
        
        # Total session duration should make sense
        session_totals = data.groupby('session_id')['time_on_page_seconds'].sum()
        assert session_totals.min() > 0, "Session duration should be positive"
        assert session_totals.max() < 7200, "Session duration should be reasonable (< 2 hours)"
    
    def test_page_load_times(self, generator):
        """Test page load time generation"""
        data = generator.generate(100)
        
        # Page load times should be realistic (200ms - 3000ms)
        assert (data['page_load_time_ms'] >= 200).all(), "Page load time too fast"
        assert (data['page_load_time_ms'] <= 3000).all(), "Page load time too slow"
    
    def test_scroll_depth_patterns(self, generator):
        """Test scroll depth patterns"""
        data = generator.generate(200)
        
        # Scroll depth should be between 10-100%
        assert (data['scroll_depth_percent'] >= 10).all(), "Scroll depth too low"
        assert (data['scroll_depth_percent'] <= 100).all(), "Scroll depth too high"
        
        # Average scroll depth should be reasonable
        avg_scroll = data['scroll_depth_percent'].mean()
        assert 40 < avg_scroll < 90, f"Average scroll depth {avg_scroll} seems unrealistic"
    
    def test_reproducibility(self, generator):
        """Test that generation is reproducible with same seed"""
        data1 = generator.generate(50)
        
        # Create new generator with same seed
        seeder2 = MillisecondSeeder(fixed_seed=12345)
        generator2 = WebAnalyticsGenerator(seeder2)
        data2 = generator2.generate(50)
        
        # Should generate identical data
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_validation(self, generator):
        """Test overall data quality"""
        data = generator.generate(300)
        
        # No null values in critical columns
        critical_columns = [
            'session_id', 'user_id', 'timestamp', 'device_type', 
            'page_category', 'traffic_source'
        ]
        
        for column in critical_columns:
            assert data[column].notna().all(), f"Null values found in {column}"
        
        # Unique identifiers should be unique
        assert data['page_view_id'].nunique() == len(data), "Page view IDs should be unique"
        
        # Boolean columns should only contain boolean values
        boolean_columns = ['is_bounce', 'is_exit', 'converted']
        for column in boolean_columns:
            unique_values = set(data[column].unique())
            assert unique_values.issubset({True, False}), f"Non-boolean values in {column}"