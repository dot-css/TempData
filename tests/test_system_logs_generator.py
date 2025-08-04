"""
Unit tests for SystemLogsGenerator

Tests system log data generation patterns, log level distributions, error patterns,
service correlations, and realistic timestamp patterns.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.technology.system_logs import SystemLogsGenerator


class TestSystemLogsGenerator:
    """Test suite for SystemLogsGenerator"""
    
    @pytest.fixture
    def generator(self):
        """Create SystemLogsGenerator instance with fixed seed"""
        seeder = MillisecondSeeder(fixed_seed=12345)
        return SystemLogsGenerator(seeder)
    
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
            'timestamp', 'log_level', 'service', 'message', 'thread_id',
            'process_id', 'host', 'source_file', 'line_number', 'severity_score'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_log_level_distributions(self, generator):
        """Test realistic log level distributions"""
        data = generator.generate(1000)
        level_counts = data['log_level'].value_counts(normalize=True)
        
        # INFO should be most common (~55%)
        assert level_counts.get('INFO', 0) > 0.40, "INFO logs should be most common"
        
        # WARN should be second most common (~20%)
        assert level_counts.get('WARN', 0) > 0.10, "WARN logs should be common"
        
        # DEBUG should be present (~15%)
        assert level_counts.get('DEBUG', 0) > 0.05, "DEBUG logs should be present"
        
        # ERROR should be less common (~8%)
        assert level_counts.get('ERROR', 0) < 0.15, "ERROR logs should be less common"
        
        # FATAL should be rare (~2%)
        assert level_counts.get('FATAL', 0) < 0.05, "FATAL logs should be rare"
        
        # Check all expected log levels are present
        expected_levels = {'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'}
        assert set(level_counts.index) == expected_levels
    
    def test_service_distributions(self, generator):
        """Test service distributions"""
        data = generator.generate(1000)
        service_counts = data['service'].value_counts(normalize=True)
        
        # API and web_server should be most common
        assert service_counts.get('api', 0) > 0.20, "API service should be common"
        assert service_counts.get('web_server', 0) > 0.15, "Web server should be common"
        
        # Database should be present
        assert service_counts.get('database', 0) > 0.10, "Database service should be present"
        
        # Check expected services
        expected_services = {
            'api', 'web_server', 'database', 'cache', 
            'queue', 'payment', 'external_service'
        }
        assert set(service_counts.index).issubset(expected_services)
    
    def test_severity_score_mapping(self, generator):
        """Test severity score mapping"""
        data = generator.generate(200)
        
        # Check severity score mapping
        severity_mapping = {
            'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'FATAL': 5
        }
        
        for level, expected_score in severity_mapping.items():
            level_data = data[data['log_level'] == level]
            if len(level_data) > 0:
                assert (level_data['severity_score'] == expected_score).all(), \
                    f"Incorrect severity score for {level}"
    
    def test_error_patterns(self, generator):
        """Test error pattern generation"""
        data = generator.generate(1000)
        error_data = data[data['log_level'].isin(['ERROR', 'FATAL'])]
        
        if len(error_data) > 0:
            # Error logs should have error codes
            assert error_data['error_code'].notna().all(), "Error logs should have error codes"
            
            # Error codes should follow expected patterns
            for error_code in error_data['error_code'].unique():
                assert isinstance(error_code, str), "Error code should be string"
                assert len(error_code) > 3, "Error code should be meaningful"
    
    def test_timestamp_patterns(self, generator):
        """Test timestamp generation patterns"""
        data = generator.generate(500)
        
        # Timestamps should be datetime objects
        assert pd.api.types.is_datetime64_any_dtype(data['timestamp']), \
            "Timestamps should be datetime type"
        
        # Data should be chronologically ordered
        timestamps = data['timestamp'].tolist()
        assert timestamps == sorted(timestamps), "Data should be chronologically ordered"
        
        # Timestamps should be recent (within reasonable range)
        now = datetime.now()
        oldest_allowed = now - timedelta(days=365)  # Within last year
        
        assert data['timestamp'].min() > oldest_allowed, "Timestamps too old"
        assert data['timestamp'].max() <= now, "Timestamps in future"
    
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
        assert timestamps == sorted(timestamps), "Time series data should be ordered"
    
    def test_message_generation(self, generator):
        """Test log message generation"""
        data = generator.generate(200)
        
        # All logs should have messages
        assert data['message'].notna().all(), "All logs should have messages"
        
        # Messages should be non-empty strings
        for message in data['message']:
            assert isinstance(message, str), "Message should be string"
            assert len(message) > 10, "Message should be meaningful"
        
        # Error messages should be different from info messages
        error_messages = data[data['log_level'] == 'ERROR']['message'].tolist()
        info_messages = data[data['log_level'] == 'INFO']['message'].tolist()
        
        if error_messages and info_messages:
            # Should have some variety in messages
            assert len(set(error_messages)) > 1 or len(error_messages) == 1, \
                "Should have message variety"
    
    def test_host_generation(self, generator):
        """Test hostname generation"""
        data = generator.generate(100)
        
        # All logs should have hostnames
        assert data['host'].notna().all(), "All logs should have hostnames"
        
        # Hostnames should follow realistic patterns
        for host in data['host'].unique():
            assert isinstance(host, str), "Host should be string"
            assert '.' in host, "Host should contain domain"
            assert len(host) > 10, "Host should be realistic length"
    
    def test_source_file_generation(self, generator):
        """Test source file generation"""
        data = generator.generate(100)
        
        # All logs should have source files
        assert data['source_file'].notna().all(), "All logs should have source files"
        
        # Source files should end with .py
        for source_file in data['source_file'].unique():
            assert source_file.endswith('.py'), "Source file should be Python file"
            assert len(source_file) > 5, "Source file should be meaningful"
    
    def test_response_time_patterns(self, generator):
        """Test response time generation"""
        data = generator.generate(300)
        
        # Response times should be positive where present
        response_time_data = data[data['response_time_ms'].notna()]
        if len(response_time_data) > 0:
            assert (response_time_data['response_time_ms'] > 0).all(), \
                "Response times should be positive"
            assert (response_time_data['response_time_ms'] < 60000).all(), \
                "Response times should be reasonable (< 60s)"
        
        # Error logs should tend to have higher response times
        error_data = data[(data['log_level'] == 'ERROR') & (data['response_time_ms'].notna())]
        info_data = data[(data['log_level'] == 'INFO') & (data['response_time_ms'].notna())]
        
        if len(error_data) > 5 and len(info_data) > 5:
            error_avg = error_data['response_time_ms'].mean()
            info_avg = info_data['response_time_ms'].mean()
            # Allow for variance, but errors should generally be slower
            assert error_avg >= info_avg * 0.8, "Error responses should be slower on average"
    
    def test_status_code_patterns(self, generator):
        """Test HTTP status code generation"""
        data = generator.generate(300)
        
        # Status codes should only be present for web services
        web_services = ['api', 'web_server']
        web_data = data[data['service'].isin(web_services)]
        non_web_data = data[~data['service'].isin(web_services)]
        
        # Web services should have status codes
        if len(web_data) > 0:
            web_with_status = web_data[web_data['status_code'].notna()]
            assert len(web_with_status) > 0, "Web services should have status codes"
        
        # Status codes should be valid HTTP codes
        status_codes = data[data['status_code'].notna()]['status_code'].unique()
        for code in status_codes:
            assert 200 <= code <= 599, f"Invalid HTTP status code: {code}"
        
        # Error logs should have error status codes
        error_web_data = data[
            (data['log_level'] == 'ERROR') & 
            (data['service'].isin(web_services)) &
            (data['status_code'].notna())
        ]
        
        if len(error_web_data) > 0:
            error_codes = error_web_data['status_code'].unique()
            # Should have some 4xx or 5xx codes
            has_error_codes = any(code >= 400 for code in error_codes)
            assert has_error_codes, "Error logs should have error status codes"
    
    def test_thread_and_process_ids(self, generator):
        """Test thread and process ID generation"""
        data = generator.generate(100)
        
        # All logs should have thread and process IDs
        assert data['thread_id'].notna().all(), "All logs should have thread IDs"
        assert data['process_id'].notna().all(), "All logs should have process IDs"
        
        # Thread IDs should follow pattern
        for thread_id in data['thread_id'].unique():
            assert thread_id.startswith('thread-'), "Thread ID should have correct format"
        
        # Process IDs should be integers
        assert data['process_id'].dtype in ['int64', 'int32'], "Process IDs should be integers"
        assert (data['process_id'] >= 1000).all(), "Process IDs should be realistic"
        assert (data['process_id'] <= 9999).all(), "Process IDs should be realistic"
    
    def test_line_numbers(self, generator):
        """Test line number generation"""
        data = generator.generate(100)
        
        # All logs should have line numbers
        assert data['line_number'].notna().all(), "All logs should have line numbers"
        
        # Line numbers should be positive integers
        assert (data['line_number'] > 0).all(), "Line numbers should be positive"
        assert (data['line_number'] <= 1000).all(), "Line numbers should be reasonable"
    
    def test_stack_trace_generation(self, generator):
        """Test stack trace generation for error logs"""
        data = generator.generate(500)
        error_data = data[data['log_level'].isin(['ERROR', 'FATAL'])]
        
        if len(error_data) > 0:
            # Some error logs should have stack traces
            with_stack_trace = error_data[error_data['stack_trace'].notna()]
            
            if len(with_stack_trace) > 0:
                for stack_trace in with_stack_trace['stack_trace'].dropna():
                    assert isinstance(stack_trace, str), "Stack trace should be string"
                    assert 'File' in stack_trace, "Stack trace should contain file references"
                    assert 'line' in stack_trace, "Stack trace should contain line references"
    
    def test_business_hours_patterns(self, generator):
        """Test business hours vs off-hours patterns"""
        data = generator.generate(1000)
        
        # Should have business hours indicator
        assert 'is_business_hours' in data.columns, "Should have business hours indicator"
        
        # Should have both business and non-business hours
        business_hours_data = data[data['is_business_hours']]
        off_hours_data = data[~data['is_business_hours']]
        
        # Both should exist (with reasonable sample size)
        if len(data) > 100:
            assert len(business_hours_data) > 0, "Should have business hours logs"
            assert len(off_hours_data) > 0, "Should have off-hours logs"
    
    def test_service_error_rates(self, generator):
        """Test service error rate calculation"""
        data = generator.generate(1000)
        
        # Should have service error rate column
        assert 'service_error_rate' in data.columns, "Should have service error rates"
        
        # Error rates should be between 0 and 1
        assert (data['service_error_rate'] >= 0).all(), "Error rates should be non-negative"
        assert (data['service_error_rate'] <= 1).all(), "Error rates should not exceed 1"
        
        # Payment service should have higher error rate than cache
        payment_data = data[data['service'] == 'payment']
        cache_data = data[data['service'] == 'cache']
        
        if len(payment_data) > 10 and len(cache_data) > 10:
            payment_error_rate = payment_data['service_error_rate'].iloc[0]
            cache_error_rate = cache_data['service_error_rate'].iloc[0]
            assert payment_error_rate >= cache_error_rate, \
                "Payment should have higher error rate than cache"
    
    def test_logs_per_minute_calculation(self, generator):
        """Test logs per minute calculation"""
        data = generator.generate(500)
        
        # Should have logs per minute column
        assert 'logs_per_minute' in data.columns, "Should have logs per minute"
        
        # Values should be positive
        assert (data['logs_per_minute'] > 0).all(), "Logs per minute should be positive"
        
        # Should be reasonable (not too high)
        assert (data['logs_per_minute'] <= 1000).all(), "Logs per minute should be reasonable"
    
    def test_weekend_patterns(self, generator):
        """Test weekend vs weekday patterns"""
        data = generator.generate(500)
        
        # Should have weekend indicator
        assert 'is_weekend' in data.columns, "Should have weekend indicator"
        
        # Should be boolean
        weekend_values = set(data['is_weekend'].unique())
        assert weekend_values.issubset({True, False}), "Weekend indicator should be boolean"
    
    def test_reproducibility(self, generator):
        """Test that generation is reproducible with same seed"""
        data1 = generator.generate(50)
        
        # Create new generator with same seed
        seeder2 = MillisecondSeeder(fixed_seed=12345)
        generator2 = SystemLogsGenerator(seeder2)
        data2 = generator2.generate(50)
        
        # Should generate identical data
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_data_quality_validation(self, generator):
        """Test overall data quality"""
        data = generator.generate(300)
        
        # No null values in critical columns
        critical_columns = [
            'timestamp', 'log_level', 'service', 'message', 
            'host', 'source_file', 'severity_score'
        ]
        
        for column in critical_columns:
            assert data[column].notna().all(), f"Null values found in {column}"
        
        # Severity scores should match log levels
        for _, row in data.iterrows():
            level = row['log_level']
            score = row['severity_score']
            expected_scores = {
                'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'FATAL': 5
            }
            assert score == expected_scores[level], \
                f"Incorrect severity score {score} for level {level}"
    
    def test_realistic_log_patterns(self, generator):
        """Test that logs follow realistic patterns"""
        data = generator.generate(1000)
        
        # INFO logs should be most common
        level_counts = data['log_level'].value_counts()
        assert level_counts['INFO'] == level_counts.max(), "INFO should be most common"
        
        # FATAL logs should be least common
        if 'FATAL' in level_counts:
            assert level_counts['FATAL'] == level_counts.min(), "FATAL should be least common"
        
        # API service should generate reasonable variety of log levels
        api_data = data[data['service'] == 'api']
        if len(api_data) > 50:
            api_levels = set(api_data['log_level'].unique())
            assert len(api_levels) >= 3, "API should generate variety of log levels"
        
        # Database service should have some connection-related messages
        db_data = data[data['service'] == 'database']
        if len(db_data) > 10:
            db_messages = ' '.join(db_data['message'].tolist()).lower()
            connection_terms = ['connection', 'query', 'database', 'table']
            has_db_terms = any(term in db_messages for term in connection_terms)
            assert has_db_terms, "Database logs should contain database-related terms"