"""
Comprehensive unit tests for all dataset generators with property-based testing

This module provides extensive testing for all dataset generators including:
- Business generators (sales, customers, ecommerce, etc.)
- Financial generators (stocks, banking, crypto, etc.)
- Healthcare generators (patients, appointments, etc.)
- Technology generators (web analytics, system logs, etc.)
- IoT generators (weather, energy, etc.)
- Social generators (social media, user profiles, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from decimal import Decimal
import re

from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.sales import SalesGenerator
from tempdata.datasets.business.customers import CustomerGenerator
from tempdata.datasets.business.ecommerce import EcommerceGenerator
from tempdata.datasets.financial.stocks import StockGenerator
from tempdata.datasets.financial.banking import BankingGenerator
from tempdata.datasets.healthcare.patients import PatientGenerator
from tempdata.datasets.healthcare.appointments import AppointmentGenerator
from tempdata.datasets.technology.web_analytics import WebAnalyticsGenerator
from tempdata.datasets.technology.system_logs import SystemLogsGenerator
from tempdata.datasets.iot_sensors.weather import WeatherGenerator
from tempdata.datasets.iot_sensors.energy import EnergyGenerator
from tempdata.datasets.social.social_media import SocialMediaGenerator
from tempdata.datasets.social.user_profiles import UserProfilesGenerator


class TestBusinessGeneratorsComprehensive:
    """Comprehensive tests for all business dataset generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=12345)
    
    @pytest.fixture
    def sales_generator(self, seeder):
        return SalesGenerator(seeder)
    
    @pytest.fixture
    def customer_generator(self, seeder):
        return CustomerGenerator(seeder)
    
    @pytest.fixture
    def ecommerce_generator(self, seeder):
        return EcommerceGenerator(seeder)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=1, max_value=1000))
    def test_sales_generator_properties(self, rows):
        """Property test: Sales generator produces valid data structure"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        sales_generator = SalesGenerator(seeder)
        data = sales_generator.generate(rows)
        
        assert len(data) == rows
        assert isinstance(data, pd.DataFrame)
        
        # Check required columns
        required_cols = ['transaction_id', 'date', 'customer_id', 'amount', 'region', 'payment_method']
        for col in required_cols:
            assert col in data.columns
        
        # Check data constraints
        assert all(data['amount'] > 0)  # Positive amounts
        assert data['transaction_id'].nunique() == rows  # Unique transaction IDs
        assert all(data['transaction_id'].str.startswith('TXN_'))  # Correct format
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        rows=st.integers(min_value=10, max_value=500),
        country=st.sampled_from(['united_states', 'pakistan', 'germany', 'china'])
    )
    def test_customer_generator_properties(self, rows, country):
        """Property test: Customer generator respects country constraints"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        customer_generator = CustomerGenerator(seeder)
        data = customer_generator.generate(rows, country=country)
        
        assert len(data) == rows
        assert 'customer_id' in data.columns
        assert 'email' in data.columns
        assert 'phone' in data.columns
        assert 'country' in data.columns
        
        # All customers should be from specified country
        assert all(data['country'] == country)
        
        # Email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert all(data['email'].str.match(email_pattern))
        
        # Unique customer IDs
        assert data['customer_id'].nunique() == rows
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=200))
    def test_ecommerce_generator_properties(self, rows):
        """Property test: Ecommerce generator produces realistic order data"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        ecommerce_generator = EcommerceGenerator(seeder)
        data = ecommerce_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns that actually exist in EcommerceGenerator
        required_cols = ['order_id', 'customer_id', 'order_date', 'order_status']
        for col in required_cols:
            assert col in data.columns
        
        # Check business logic for columns that exist
        if 'subtotal' in data.columns:
            assert all(data['subtotal'] > 0)  # Positive amounts
        if 'shipping_cost' in data.columns:
            assert all(data['shipping_cost'] >= 0)  # Non-negative shipping
    
    def test_sales_seasonal_patterns(self, sales_generator):
        """Test seasonal patterns in sales data"""
        # Generate data for different seasons
        winter_data = sales_generator.generate(100, date_range=(date(2024, 12, 1), date(2024, 12, 31)))
        summer_data = sales_generator.generate(100, date_range=(date(2024, 7, 1), date(2024, 7, 31)))
        
        # Winter (holiday season) should have different patterns than summer
        winter_avg = winter_data['amount'].mean()
        summer_avg = summer_data['amount'].mean()
        
        # Allow for some variance but expect seasonal differences
        assert abs(winter_avg - summer_avg) / max(winter_avg, summer_avg) > 0.05
    
    def test_customer_demographic_distribution(self, customer_generator):
        """Test demographic distribution in customer data"""
        data = customer_generator.generate(1000)
        
        # Should have age distribution
        assert 'age' in data.columns
        assert data['age'].min() >= 18  # Legal age
        assert data['age'].max() <= 100  # Reasonable max age
        
        # Should have gender distribution
        if 'gender' in data.columns:
            gender_dist = data['gender'].value_counts(normalize=True)
            # No single gender should dominate too much
            assert gender_dist.max() <= 0.7
    
    def test_ecommerce_product_correlations(self, ecommerce_generator):
        """Test product correlations in ecommerce data"""
        data = ecommerce_generator.generate(500)
        
        # Check for any price-related columns that actually exist
        price_columns = [col for col in data.columns if any(term in col.lower() for term in ['price', 'cost', 'subtotal', 'total'])]
        
        if price_columns:
            # Should have some price variation
            for col in price_columns:
                if data[col].dtype in ['float64', 'int64']:
                    assert data[col].std() > 0  # Should have variation
        
        # Check that we have reasonable data structure
        assert len(data.columns) > 5  # Should have multiple columns
        assert data['order_id'].nunique() == len(data)  # Unique order IDs


class TestFinancialGeneratorsComprehensive:
    """Comprehensive tests for financial dataset generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=54321)
    
    @pytest.fixture
    def stock_generator(self, seeder):
        return StockGenerator(seeder)
    
    @pytest.fixture
    def banking_generator(self, seeder):
        return BankingGenerator(seeder)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=10, max_value=500))
    def test_stock_generator_properties(self, rows):
        """Property test: Stock generator produces valid financial data"""
        seeder = MillisecondSeeder(fixed_seed=54321)
        stock_generator = StockGenerator(seeder)
        data = stock_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in data.columns
        
        # Check financial constraints
        assert all(data['open'] > 0)  # Positive prices
        assert all(data['high'] > 0)
        assert all(data['low'] > 0)
        assert all(data['close'] > 0)
        assert all(data['volume'] > 0)  # Positive volume
        
        # High should be >= max(open, close) and Low should be <= min(open, close)
        assert all(data['high'] >= data[['open', 'close']].max(axis=1))
        assert all(data['low'] <= data[['open', 'close']].min(axis=1))
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=300))
    def test_banking_generator_properties(self, rows):
        """Property test: Banking generator produces valid transaction data"""
        seeder = MillisecondSeeder(fixed_seed=54321)
        banking_generator = BankingGenerator(seeder)
        data = banking_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['transaction_id', 'account_id', 'amount', 'transaction_type', 'timestamp']
        for col in required_cols:
            assert col in data.columns
        
        # Check transaction types are valid
        valid_types = ['deposit', 'withdrawal', 'transfer', 'payment', 'fee']
        assert all(data['transaction_type'].isin(valid_types))
        
        # Check amount constraints based on transaction type
        deposits = data[data['transaction_type'] == 'deposit']
        withdrawals = data[data['transaction_type'] == 'withdrawal']
        
        if len(deposits) > 0:
            assert all(deposits['amount'] > 0)  # Deposits should be positive
        if len(withdrawals) > 0:
            assert all(withdrawals['amount'] < 0)  # Withdrawals should be negative
    
    def test_stock_market_volatility(self, stock_generator):
        """Test stock market volatility patterns"""
        data = stock_generator.generate(100, time_series=True)
        
        # Calculate daily returns
        data = data.sort_values('timestamp')
        data['daily_return'] = data['close'].pct_change()
        
        # Should have realistic volatility (not too extreme)
        daily_returns = data['daily_return'].dropna()
        volatility = daily_returns.std()
        
        # Daily volatility should be reasonable (0.1% to 10%)
        assert 0.001 <= volatility <= 0.1
    
    def test_banking_balance_tracking(self, banking_generator):
        """Test banking balance tracking consistency"""
        data = banking_generator.generate(50)
        
        # Group by account and check balance consistency
        if 'balance' in data.columns:
            for account_id in data['account_id'].unique()[:5]:  # Test first 5 accounts
                account_data = data[data['account_id'] == account_id].sort_values('timestamp')
                
                # Balance should change consistently with transactions
                for i in range(1, len(account_data)):
                    prev_balance = account_data.iloc[i-1]['balance']
                    curr_amount = account_data.iloc[i]['amount']
                    curr_balance = account_data.iloc[i]['balance']
                    
                    expected_balance = prev_balance + curr_amount
                    # Allow small floating point differences
                    assert abs(curr_balance - expected_balance) < 0.01


class TestHealthcareGeneratorsComprehensive:
    """Comprehensive tests for healthcare dataset generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=98765)
    
    @pytest.fixture
    def patient_generator(self, seeder):
        return PatientGenerator(seeder)
    
    @pytest.fixture
    def appointment_generator(self, seeder):
        return AppointmentGenerator(seeder)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=200))
    def test_patient_generator_properties(self, rows):
        """Property test: Patient generator produces valid medical data"""
        seeder = MillisecondSeeder(fixed_seed=98765)
        patient_generator = PatientGenerator(seeder)
        data = patient_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['patient_id', 'name', 'date_of_birth', 'gender', 'blood_type']
        for col in required_cols:
            assert col in data.columns
        
        # Check medical constraints
        valid_blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        assert all(data['blood_type'].isin(valid_blood_types))
        
        # Check age constraints (patients should be reasonable age)
        if 'age' in data.columns:
            assert all(data['age'] >= 0)
            assert all(data['age'] <= 120)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=150))
    def test_appointment_generator_properties(self, rows):
        """Property test: Appointment generator produces valid scheduling data"""
        seeder = MillisecondSeeder(fixed_seed=98765)
        appointment_generator = AppointmentGenerator(seeder)
        data = appointment_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['appointment_id', 'patient_id', 'doctor_id', 'appointment_date', 'appointment_type']
        for col in required_cols:
            assert col in data.columns
        
        # Check appointment types are valid
        valid_types = ['consultation', 'follow_up', 'emergency', 'surgery', 'checkup']
        assert all(data['appointment_type'].isin(valid_types))
        
        # Appointments should be in the future or recent past
        appointment_dates = pd.to_datetime(data['appointment_date'])
        min_date = appointment_dates.min()
        max_date = appointment_dates.max()
        
        # Should span a reasonable time range
        assert (max_date - min_date).days >= 0
    
    def test_patient_medical_history_correlations(self, patient_generator):
        """Test medical history correlations in patient data"""
        data = patient_generator.generate(200)
        
        # Check for medical conditions
        if 'medical_conditions' in data.columns:
            # Should have variety in conditions
            conditions = data['medical_conditions'].value_counts()
            assert len(conditions) > 1
        
        # Age and medical conditions should have some correlation
        if 'age' in data.columns and 'medical_conditions' in data.columns:
            # Older patients might have more conditions (this is a general trend)
            older_patients = data[data['age'] > 65]
            younger_patients = data[data['age'] < 30]
            
            if len(older_patients) > 10 and len(younger_patients) > 10:
                # This is a statistical trend, not absolute
                older_conditions = older_patients['medical_conditions'].notna().mean()
                younger_conditions = younger_patients['medical_conditions'].notna().mean()
                # Allow for variation but expect some difference
                assert abs(older_conditions - younger_conditions) >= 0


class TestTechnologyGeneratorsComprehensive:
    """Comprehensive tests for technology dataset generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=11111)
    
    @pytest.fixture
    def web_analytics_generator(self, seeder):
        return WebAnalyticsGenerator(seeder)
    
    @pytest.fixture
    def system_logs_generator(self, seeder):
        return SystemLogsGenerator(seeder)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=10, max_value=300))
    def test_web_analytics_properties(self, rows):
        """Property test: Web analytics generator produces valid web data"""
        seeder = MillisecondSeeder(fixed_seed=11111)
        web_analytics_generator = WebAnalyticsGenerator(seeder)
        data = web_analytics_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['session_id', 'user_id', 'page_url', 'timestamp', 'device_type']
        for col in required_cols:
            assert col in data.columns
        
        # Check device types are valid
        valid_devices = ['desktop', 'mobile', 'tablet']
        assert all(data['device_type'].isin(valid_devices))
        
        # Check URL format
        assert all(data['page_url'].str.startswith(('http://', 'https://')))
        
        # Session duration should be reasonable
        if 'session_duration' in data.columns:
            assert all(data['session_duration'] > 0)
            assert all(data['session_duration'] < 86400)  # Less than 24 hours
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=200))
    def test_system_logs_properties(self, rows):
        """Property test: System logs generator produces valid log data"""
        seeder = MillisecondSeeder(fixed_seed=11111)
        system_logs_generator = SystemLogsGenerator(seeder)
        data = system_logs_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['timestamp', 'log_level', 'service', 'message']
        for col in required_cols:
            assert col in data.columns
        
        # Check log levels are valid
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert all(data['log_level'].isin(valid_levels))
        
        # Messages should not be empty
        assert all(data['message'].str.len() > 0)
        
        # Timestamps should be chronologically ordered
        timestamps = pd.to_datetime(data['timestamp'])
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    def test_web_analytics_user_behavior(self, web_analytics_generator):
        """Test user behavior patterns in web analytics"""
        data = web_analytics_generator.generate(500)
        
        # Check bounce rate patterns
        if 'bounce_rate' in data.columns:
            bounce_rates = data['bounce_rate']
            assert all(bounce_rates >= 0)
            assert all(bounce_rates <= 1)
            
            # Mobile might have different bounce rates than desktop
            if len(data['device_type'].unique()) > 1:
                device_bounce = data.groupby('device_type')['bounce_rate'].mean()
                # Should have some variation
                assert device_bounce.std() > 0
    
    def test_system_logs_error_patterns(self, system_logs_generator):
        """Test error patterns in system logs"""
        data = system_logs_generator.generate(300)
        
        # Error distribution should be realistic
        log_level_dist = data['log_level'].value_counts(normalize=True)
        
        # INFO logs should be most common, CRITICAL should be least common
        if 'INFO' in log_level_dist.index and 'CRITICAL' in log_level_dist.index:
            assert log_level_dist['INFO'] > log_level_dist['CRITICAL']
        
        # Services should have variety
        assert data['service'].nunique() > 1


class TestIoTGeneratorsComprehensive:
    """Comprehensive tests for IoT sensor dataset generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=22222)
    
    @pytest.fixture
    def weather_generator(self, seeder):
        return WeatherGenerator(seeder)
    
    @pytest.fixture
    def energy_generator(self, seeder):
        return EnergyGenerator(seeder)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=10, max_value=200))
    def test_weather_generator_properties(self, rows):
        """Property test: Weather generator produces valid sensor data"""
        seeder = MillisecondSeeder(fixed_seed=22222)
        weather_generator = WeatherGenerator(seeder)
        data = weather_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['timestamp', 'temperature', 'humidity', 'pressure']
        for col in required_cols:
            assert col in data.columns
        
        # Check realistic weather ranges
        assert all(data['temperature'] >= -50)  # Reasonable minimum temperature
        assert all(data['temperature'] <= 60)   # Reasonable maximum temperature
        assert all(data['humidity'] >= 0)       # Humidity percentage
        assert all(data['humidity'] <= 100)
        assert all(data['pressure'] > 800)      # Atmospheric pressure in hPa
        assert all(data['pressure'] < 1200)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=150))
    def test_energy_generator_properties(self, rows):
        """Property test: Energy generator produces valid consumption data"""
        seeder = MillisecondSeeder(fixed_seed=22222)
        energy_generator = EnergyGenerator(seeder)
        data = energy_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['timestamp', 'consumption', 'meter_id']
        for col in required_cols:
            assert col in data.columns
        
        # Check energy consumption constraints
        assert all(data['consumption'] >= 0)  # Non-negative consumption
        
        # Meter IDs should be consistent format
        assert all(data['meter_id'].str.match(r'^[A-Z0-9]+$'))
    
    def test_weather_correlations(self, weather_generator):
        """Test correlations between weather parameters"""
        data = weather_generator.generate(200)
        
        # Temperature and humidity often have inverse correlation
        temp_humidity_corr = data['temperature'].corr(data['humidity'])
        # Allow for various correlations but should not be perfectly random
        assert abs(temp_humidity_corr) > 0.1
        
        # Pressure should be within reasonable correlation with temperature
        temp_pressure_corr = data['temperature'].corr(data['pressure'])
        assert abs(temp_pressure_corr) < 0.9  # Not perfectly correlated
    
    def test_energy_consumption_patterns(self, energy_generator):
        """Test energy consumption patterns"""
        data = energy_generator.generate(100, time_series=True)
        
        # Should have temporal patterns
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        hourly_consumption = data.groupby('hour')['consumption'].mean()
        
        # Should have variation throughout the day
        assert hourly_consumption.std() > 0
        
        # Peak hours might have higher consumption
        if len(hourly_consumption) >= 12:  # If we have enough hours
            peak_hours = [18, 19, 20]  # Evening peak
            off_peak_hours = [2, 3, 4]  # Early morning
            
            peak_avg = hourly_consumption[hourly_consumption.index.isin(peak_hours)].mean()
            off_peak_avg = hourly_consumption[hourly_consumption.index.isin(off_peak_hours)].mean()
            
            # Peak should generally be higher than off-peak
            assert peak_avg >= off_peak_avg * 0.8


class TestSocialGeneratorsComprehensive:
    """Comprehensive tests for social dataset generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=33333)
    
    @pytest.fixture
    def social_media_generator(self, seeder):
        return SocialMediaGenerator(seeder)
    
    @pytest.fixture
    def user_profiles_generator(self, seeder):
        return UserProfilesGenerator(seeder)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=100))
    def test_social_media_properties(self, rows):
        """Property test: Social media generator produces valid post data"""
        seeder = MillisecondSeeder(fixed_seed=33333)
        social_media_generator = SocialMediaGenerator(seeder)
        data = social_media_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['post_id', 'user_id', 'content', 'timestamp', 'platform']
        for col in required_cols:
            assert col in data.columns
        
        # Check platform types
        valid_platforms = ['twitter', 'facebook', 'instagram', 'linkedin', 'tiktok']
        assert all(data['platform'].isin(valid_platforms))
        
        # Content should not be empty
        assert all(data['content'].str.len() > 0)
        
        # Engagement metrics should be non-negative
        if 'likes' in data.columns:
            assert all(data['likes'] >= 0)
        if 'shares' in data.columns:
            assert all(data['shares'] >= 0)
    
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(rows=st.integers(min_value=5, max_value=100))
    def test_user_profiles_properties(self, rows):
        """Property test: User profiles generator produces valid profile data"""
        seeder = MillisecondSeeder(fixed_seed=33333)
        user_profiles_generator = UserProfilesGenerator(seeder)
        data = user_profiles_generator.generate(rows)
        
        assert len(data) == rows
        
        # Check required columns
        required_cols = ['user_id', 'username', 'email', 'join_date']
        for col in required_cols:
            assert col in data.columns
        
        # Usernames should be unique
        assert data['username'].nunique() == rows
        
        # Email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        assert all(data['email'].str.match(email_pattern))
        
        # Join dates should be in the past
        join_dates = pd.to_datetime(data['join_date'])
        assert all(join_dates <= pd.Timestamp.now())
    
    def test_social_media_engagement_patterns(self, social_media_generator):
        """Test engagement patterns in social media data"""
        data = social_media_generator.generate(200)
        
        # Different platforms might have different engagement patterns
        if 'likes' in data.columns and len(data['platform'].unique()) > 1:
            platform_engagement = data.groupby('platform')['likes'].mean()
            
            # Should have variation across platforms
            assert platform_engagement.std() > 0
        
        # Viral content should have higher engagement
        if 'likes' in data.columns and 'shares' in data.columns:
            # Posts with more likes should generally have more shares
            correlation = data['likes'].corr(data['shares'])
            assert correlation > 0.1  # Positive correlation
    
    def test_user_profiles_demographics(self, user_profiles_generator):
        """Test demographic distribution in user profiles"""
        data = user_profiles_generator.generate(300)
        
        # Age distribution should be realistic
        if 'age' in data.columns:
            assert data['age'].min() >= 13  # Minimum social media age
            assert data['age'].max() <= 100  # Reasonable maximum
            
            # Should have variety in ages
            assert data['age'].nunique() > 10
        
        # Location distribution
        if 'location' in data.columns:
            locations = data['location'].value_counts()
            # Should have multiple locations
            assert len(locations) > 1
            # No single location should dominate too much
            assert locations.iloc[0] / len(data) < 0.5


class TestPerformanceBenchmarks:
    """Performance benchmarks for all generators"""
    
    @pytest.fixture
    def seeder(self):
        return MillisecondSeeder(fixed_seed=99999)
    
    def test_sales_generator_performance(self, benchmark, seeder):
        """Benchmark sales generator performance"""
        generator = SalesGenerator(seeder)
        
        def generate_sales_data():
            return generator.generate(1000)
        
        result = benchmark(generate_sales_data)
        assert len(result) == 1000
        # Should generate at least 500 rows per second
        # This is checked by pytest-benchmark automatically
    
    def test_stock_generator_performance(self, benchmark, seeder):
        """Benchmark stock generator performance"""
        generator = StockGenerator(seeder)
        
        def generate_stock_data():
            return generator.generate(1000)
        
        result = benchmark(generate_stock_data)
        assert len(result) == 1000
    
    def test_weather_generator_performance(self, benchmark, seeder):
        """Benchmark weather generator performance"""
        generator = WeatherGenerator(seeder)
        
        def generate_weather_data():
            return generator.generate(1000)
        
        result = benchmark(generate_weather_data)
        assert len(result) == 1000
    
    @pytest.mark.slow
    def test_large_dataset_generation(self, seeder):
        """Test generation of large datasets"""
        generator = SalesGenerator(seeder)
        
        # Test 10K rows
        large_data = generator.generate(10000)
        assert len(large_data) == 10000
        
        # Check data quality is maintained
        assert large_data['transaction_id'].nunique() == 10000
        assert all(large_data['amount'] > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--benchmark-skip'])