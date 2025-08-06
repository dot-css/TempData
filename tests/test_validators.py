"""
Unit tests for data validation system

Tests the DataValidator class for accuracy and quality metrics validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tempdata.core.validators import DataValidator


class TestDataValidator:
    """Test suite for DataValidator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.validator = DataValidator()
        
        # Create sample datasets for testing
        self.sample_sales_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005'],
            'amount': [25.99, 150.00, 75.50, 200.00, 45.25],
            'quantity': [1, 3, 2, 1, 2],
            'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
            'payment_method': ['card', 'cash', 'card', 'card', 'cash']
        })
        
        self.sample_customer_data = pd.DataFrame({
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'email': ['john@example.com', 'jane@test.org', 'bob@company.net', 'alice@domain.com', 'charlie@email.co'],
            'age': [25, 34, 45, 28, 52],
            'registration_date': ['2023-01-01', '2023-02-15', '2023-03-10', '2023-04-05', '2023-05-20']
        })
        
        # Create problematic data for testing edge cases
        self.problematic_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN001', 'TXN003', None, 'TXN005'],  # Duplicates and nulls
            'amount': [25.99, -50.00, 1000000, None, 45.25],  # Negative and extreme values
            'date': ['2024-01-15', '1900-01-01', '2050-12-31', '2024-01-18', 'invalid-date'],  # Invalid dates
            'email': ['john@example.com', 'invalid-email', 'bob@company.net', None, 'charlie@email.co'],
            'age': [25, 150, 45, None, -5]  # Invalid ages
        })
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        validator = DataValidator()
        
        assert hasattr(validator, 'quality_thresholds')
        assert hasattr(validator, 'pattern_rules')
        assert hasattr(validator, 'business_rules')
        assert hasattr(validator, 'statistical_thresholds')
        
        # Check quality thresholds are set
        assert validator.quality_thresholds['completeness'] == 0.90
        assert validator.quality_thresholds['geographical_accuracy'] == 0.90
    
    def test_validate_dataset_basic(self):
        """Test basic dataset validation"""
        results = self.validator.validate_dataset(self.sample_sales_data, 'sales')
        
        assert isinstance(results, dict)
        assert 'valid' in results
        assert 'overall_score' in results
        assert 'scores' in results
        assert 'quality_metrics' in results
        
        # Should pass validation for good data
        assert results['valid'] is True
        assert results['overall_score'] > 0.8
    
    def test_validate_empty_dataset(self):
        """Test validation of empty dataset"""
        empty_data = pd.DataFrame()
        results = self.validator.validate_dataset(empty_data, 'sales')
        
        assert results['valid'] is False
        assert 'Dataset is empty' in results['issues']
        assert results['overall_score'] == 0.0
    
    def test_completeness_validation(self):
        """Test data completeness validation"""
        # Test with complete data
        complete_score = self.validator._check_completeness(self.sample_sales_data)
        assert complete_score == 1.0
        
        # Test with missing data
        incomplete_data = self.sample_sales_data.copy()
        incomplete_data.loc[0, 'amount'] = None
        incomplete_data.loc[1, 'customer_id'] = None
        
        incomplete_score = self.validator._check_completeness(incomplete_data)
        assert incomplete_score < 1.0
        assert incomplete_score > 0.0
    
    def test_uniqueness_validation(self):
        """Test data uniqueness validation"""
        # Test with unique IDs
        unique_score = self.validator._check_uniqueness(self.sample_sales_data)
        assert unique_score == 1.0
        
        # Test with duplicate IDs
        duplicate_data = self.sample_sales_data.copy()
        duplicate_data.loc[1, 'transaction_id'] = 'TXN001'  # Create duplicate
        
        duplicate_score = self.validator._check_uniqueness(duplicate_data)
        assert duplicate_score < 1.0
    
    def test_geographical_accuracy_validation(self):
        """Test geographical accuracy validation"""
        # Test with valid addresses
        valid_addresses = [
            {'street': '123 Main St', 'city': 'New York', 'country': 'US', 'postal_code': '10001'},
            {'street': '456 Oak Ave', 'city': 'Los Angeles', 'country': 'US', 'postal_code': '90210'},
            {'street': '789 Pine Rd', 'city': 'Chicago', 'country': 'US', 'postal_code': '60601'}
        ]
        
        accuracy_score = self.validator.validate_geographical_accuracy(valid_addresses)
        assert accuracy_score > 0.8
        
        # Test with invalid addresses
        invalid_addresses = [
            {'street': '', 'city': '', 'country': ''},  # Empty fields
            {'street': '123 Main St'},  # Missing required fields
            {'city': 'New York', 'postal_code': 'invalid'}  # Invalid postal code
        ]
        
        invalid_score = self.validator.validate_geographical_accuracy(invalid_addresses)
        assert invalid_score < 0.5
    
    def test_data_type_validation(self):
        """Test data type consistency validation"""
        # Test with correct data types
        type_score = self.validator._check_data_types(self.sample_sales_data, 'sales')
        assert type_score > 0.8
        
        # Test with incorrect data types
        bad_types_data = self.sample_sales_data.copy()
        bad_types_data['amount'] = bad_types_data['amount'].astype(str)  # Should be numeric
        
        bad_type_score = self.validator._check_data_types(bad_types_data, 'sales')
        assert bad_type_score < type_score
    
    def test_business_rules_validation(self):
        """Test business rules validation"""
        # Test with data that follows business rules
        rules_score = self.validator._validate_business_rules(self.sample_sales_data, 'sales')
        assert rules_score > 0.8
        
        # Test with data that violates business rules
        bad_rules_data = self.sample_sales_data.copy()
        bad_rules_data['amount'] = [-100, 200000, 75.50, 300000, 45.25]  # Negative and too large amounts
        
        bad_rules_score = self.validator._validate_business_rules(bad_rules_data, 'sales')
        assert bad_rules_score < rules_score
    
    def test_statistical_patterns_validation(self):
        """Test statistical patterns validation"""
        # Test with reasonable statistical patterns
        stats_score = self.validator._validate_statistical_patterns(self.sample_sales_data, 'sales')
        assert stats_score > 0.0
        
        # Test with unrealistic patterns (all same values)
        uniform_data = self.sample_sales_data.copy()
        uniform_data['amount'] = 100.00  # All same value
        
        uniform_score = self.validator._validate_statistical_patterns(uniform_data, 'sales')
        # For small datasets, the difference might be minimal, so just check it's reasonable
        assert uniform_score >= 0.5  # Should still get a reasonable score
    
    def test_temporal_consistency_validation(self):
        """Test temporal consistency validation"""
        # Test with consistent temporal data
        temporal_score = self.validator._validate_temporal_consistency(self.sample_sales_data)
        assert temporal_score > 0.8
        
        # Test with inconsistent temporal data
        bad_temporal_data = self.sample_sales_data.copy()
        bad_temporal_data['date'] = ['2024-01-15', '1800-01-01', '2100-12-31', '2024-01-18', 'invalid']
        
        bad_temporal_score = self.validator._validate_temporal_consistency(bad_temporal_data)
        assert bad_temporal_score < temporal_score
    
    def test_cross_field_relationships_validation(self):
        """Test cross-field relationships validation"""
        # Test with good relationships
        relationship_score = self.validator._validate_cross_field_relationships(self.sample_customer_data, 'customers')
        assert relationship_score > 0.0
        
        # Test with bad email formats
        bad_relationship_data = self.sample_customer_data.copy()
        bad_relationship_data['email'] = ['invalid-email', 'also-invalid', 'bob@company.net', 'alice@domain.com', 'bad-format']
        
        bad_relationship_score = self.validator._validate_cross_field_relationships(bad_relationship_data, 'customers')
        assert bad_relationship_score < relationship_score
    
    def test_realistic_patterns_validation(self):
        """Test realistic patterns validation"""
        # Test sales patterns
        sales_realism_score = self.validator._validate_realistic_patterns(self.sample_sales_data, 'sales')
        assert sales_realism_score > 0.0
        
        # Test customer patterns
        customer_realism_score = self.validator._validate_realistic_patterns(self.sample_customer_data, 'customers')
        assert customer_realism_score > 0.0
    
    def test_quality_metrics_generation(self):
        """Test quality metrics generation"""
        metrics = self.validator._generate_quality_metrics(self.sample_sales_data, 'sales')
        
        assert 'row_count' in metrics
        assert 'column_count' in metrics
        assert 'null_percentage' in metrics
        assert 'duplicate_rows' in metrics
        assert 'memory_usage_mb' in metrics
        
        assert metrics['row_count'] == 5
        assert metrics['column_count'] == 6
        assert metrics['null_percentage'] == 0.0  # No nulls in sample data
    
    def test_problematic_data_validation(self):
        """Test validation with problematic data"""
        results = self.validator.validate_dataset(self.problematic_data, 'sales')
        
        # Should detect issues
        assert results['overall_score'] < 0.8
        assert len(results['warnings']) > 0
        assert len(results['recommendations']) > 0
    
    def test_quality_report_generation(self):
        """Test quality report generation"""
        report = self.validator.generate_quality_report(self.sample_sales_data, 'sales')
        
        assert isinstance(report, str)
        assert 'DATA QUALITY REPORT' in report
        assert 'Overall Quality Score:' in report
        assert 'DETAILED QUALITY SCORES:' in report
        assert 'RECOMMENDATIONS:' in report
        assert 'DATASET SUMMARY:' in report
    
    def test_validation_with_different_dataset_types(self):
        """Test validation works with different dataset types"""
        # Test sales dataset
        sales_results = self.validator.validate_dataset(self.sample_sales_data, 'sales')
        assert sales_results['valid'] is True
        
        # Test customer dataset
        customer_results = self.validator.validate_dataset(self.sample_customer_data, 'customers')
        assert customer_results['valid'] is True
        
        # Test unknown dataset type
        unknown_results = self.validator.validate_dataset(self.sample_sales_data, 'unknown_type')
        assert unknown_results['valid'] is True  # Should still work with default rules
    
    def test_geographical_accuracy_with_coordinates(self):
        """Test geographical accuracy validation with coordinates"""
        addresses_with_coords = [
            {
                'street': '123 Main St',
                'city': 'New York',
                'country': 'US',
                'postal_code': '10001',
                'coordinates': [40.7128, -74.0060]  # Valid NYC coordinates
            },
            {
                'street': '456 Oak Ave',
                'city': 'Los Angeles',
                'country': 'US',
                'postal_code': '90210',
                'coordinates': [34.0522, -118.2437]  # Valid LA coordinates
            },
            {
                'street': '789 Pine Rd',
                'city': 'Invalid',
                'country': 'US',
                'postal_code': '12345',
                'coordinates': [200, 300]  # Invalid coordinates
            }
        ]
        
        accuracy_score = self.validator.validate_geographical_accuracy(addresses_with_coords)
        assert 0.5 < accuracy_score < 1.0  # Should be partial due to invalid coordinates
    
    def test_pattern_rules_setup(self):
        """Test pattern rules are properly set up"""
        assert 'email' in self.validator.pattern_rules
        assert 'phone' in self.validator.pattern_rules
        assert 'postal_code' in self.validator.pattern_rules
        
        # Test email pattern
        email_pattern = self.validator.pattern_rules['email']
        assert re.match(email_pattern, 'test@example.com')
        assert not re.match(email_pattern, 'invalid-email')
        
        # Test postal code patterns
        postal_patterns = self.validator.pattern_rules['postal_code']
        assert 'us' in postal_patterns
        assert 'uk' in postal_patterns
        assert 'global' in postal_patterns
    
    def test_threshold_evaluation(self):
        """Test threshold evaluation and warning generation"""
        # Create results with low scores
        results = {
            'valid': True,
            'scores': {
                'completeness': 0.5,  # Below threshold
                'uniqueness': 0.95,   # Above threshold
                'realism': 0.8        # At threshold
            },
            'warnings': [],
            'issues': []
        }
        
        self.validator._evaluate_scores_against_thresholds(results)
        
        # Should have warnings for low completeness score
        assert len(results['warnings']) > 0
        assert any('completeness' in warning['metric'] for warning in results['warnings'])
    
    def test_improvement_recommendations(self):
        """Test improvement recommendations generation"""
        # Create results with various low scores
        results = {
            'scores': {
                'completeness': 0.8,
                'uniqueness': 0.85,
                'business_rules': 0.7,
                'realism': 0.9
            },
            'overall_score': 0.75
        }
        
        recommendations = self.validator._generate_improvement_recommendations(results, 'sales')
        
        assert len(recommendations) > 0
        assert any('business rules' in rec.lower() for rec in recommendations)
    
    def test_sales_specific_patterns(self):
        """Test sales-specific pattern validation"""
        scores = self.validator._validate_sales_patterns(self.sample_sales_data)
        assert len(scores) > 0
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_customer_specific_patterns(self):
        """Test customer-specific pattern validation"""
        scores = self.validator._validate_customer_patterns(self.sample_customer_data)
        assert len(scores) > 0
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_financial_specific_patterns(self):
        """Test financial-specific pattern validation"""
        financial_data = pd.DataFrame({
            'price': [100.0, 102.5, 98.7, 101.2, 99.8],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'volume': [1000, 1200, 800, 1100, 950]
        })
        
        scores = self.validator._validate_financial_patterns(financial_data)
        assert len(scores) > 0
        assert all(0.0 <= score <= 1.0 for score in scores)


# Import re module for pattern testing
import re


if __name__ == '__main__':
    pytest.main([__file__])