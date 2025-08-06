"""
Unit tests for realistic pattern validation and cross-dataset relationship validation

Tests the enhanced pattern detection algorithms and cross-dataset validation
functionality implemented in task 16.2.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tempdata.core.validators import DataValidator


class TestPatternValidation:
    """Test realistic pattern validation algorithms"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_sales_pattern_validation(self):
        """Test sales-specific pattern validation"""
        # Create realistic sales data
        realistic_sales = pd.DataFrame({
            'transaction_id': [f'TXN_{i:06d}' for i in range(1000)],
            'date': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'amount': np.random.lognormal(3, 1, 1000),  # Realistic distribution
            'payment_method': np.random.choice(['card', 'cash', 'digital'], 1000, p=[0.6, 0.3, 0.1]),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 1000, p=[0.4, 0.3, 0.2, 0.1])
        })
        
        scores = self.validator._validate_sales_patterns(realistic_sales)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "Sales patterns should be realistic"
        
        # Create unrealistic sales data
        unrealistic_sales = pd.DataFrame({
            'transaction_id': [f'TXN_{i:06d}' for i in range(100)],
            'date': [datetime(2023, 1, 1, 14, 0, 0)] * 100,  # All same time
            'amount': [100.0] * 100,  # All same amount
            'payment_method': ['cash'] * 100,  # All cash
            'region': ['North'] * 100,  # All same region
            'category': ['Electronics'] * 100  # All same category
        })
        
        scores = self.validator._validate_sales_patterns(unrealistic_sales)
        # Should detect unrealistic patterns
        assert any(score < 0.9 for score in scores), "Should detect unrealistic sales patterns"
    
    def test_customer_pattern_validation(self):
        """Test customer-specific pattern validation"""
        # Create realistic customer data
        realistic_customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(500)],
            'age': np.random.normal(40, 15, 500).clip(18, 80),  # Realistic age distribution
            'email': [f'user{i}@{domain}.com' for i, domain in 
                     enumerate(np.random.choice(['gmail', 'yahoo', 'outlook', 'company'], 500))],
            'name': [f'Customer {i}' for i in range(500)]
        })
        
        scores = self.validator._validate_customer_patterns(realistic_customers)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "Customer patterns should be realistic"
        
        # Create unrealistic customer data
        unrealistic_customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(100)],
            'age': [25] * 100,  # All same age
            'email': ['user@gmail.com'] * 100,  # All same email domain
            'name': ['John Doe'] * 100  # All same name
        })
        
        scores = self.validator._validate_customer_patterns(unrealistic_customers)
        # Should detect unrealistic patterns
        assert any(score < 0.9 for score in scores), "Should detect unrealistic customer patterns"
    
    def test_financial_pattern_validation(self):
        """Test financial-specific pattern validation"""
        # Create realistic financial data with proper volatility
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = [100.0]  # Starting price
        
        # Generate realistic price movements
        for i in range(1, 100):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(0.01, new_price))  # Ensure positive prices
        
        realistic_financial = pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': np.random.lognormal(10, 1, 100)
        })
        
        scores = self.validator._validate_financial_patterns(realistic_financial)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "Financial patterns should be realistic"
        
        # Create unrealistic financial data (no volatility)
        unrealistic_financial = pd.DataFrame({
            'date': dates,
            'price': [100.0] * 100,  # No price movement
            'volume': [1000] * 100   # No volume variation
        })
        
        scores = self.validator._validate_financial_patterns(unrealistic_financial)
        # Should detect lack of volatility
        assert any(score < 0.9 for score in scores), "Should detect unrealistic financial patterns"
    
    def test_healthcare_pattern_validation(self):
        """Test healthcare-specific pattern validation"""
        # Create realistic healthcare data
        realistic_healthcare = pd.DataFrame({
            'patient_id': [f'PAT_{i:06d}' for i in range(300)],
            'age': np.random.gamma(2, 20, 300).clip(0, 100),  # Realistic age distribution
            'condition': np.random.choice(['Hypertension', 'Diabetes', 'Asthma', 'Arthritis'], 300, p=[0.4, 0.3, 0.2, 0.1]),
            'treatment': np.random.choice(['Medication', 'Surgery', 'Therapy', 'Monitoring'], 300)
        })
        
        scores = self.validator._validate_healthcare_patterns(realistic_healthcare)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "Healthcare patterns should be realistic"
    
    def test_ecommerce_pattern_validation(self):
        """Test ecommerce-specific pattern validation"""
        # Create realistic ecommerce data with Pareto distribution
        realistic_ecommerce = pd.DataFrame({
            'order_id': [f'ORD_{i:06d}' for i in range(1000)],
            'order_value': np.random.pareto(1, 1000) * 50 + 10,  # Pareto distribution
            'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home'], 1000, p=[0.5, 0.2, 0.2, 0.1])
        })
        
        scores = self.validator._validate_ecommerce_patterns(realistic_ecommerce)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "Ecommerce patterns should be realistic"
    
    def test_iot_pattern_validation(self):
        """Test IoT sensor pattern validation"""
        # Create realistic IoT sensor data with continuity
        base_temp = 20.0
        temperatures = [base_temp]
        
        for i in range(1, 200):
            # Small random walk for realistic sensor readings
            change = np.random.normal(0, 0.5)
            new_temp = temperatures[-1] + change
            temperatures.append(new_temp)
        
        realistic_iot = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='min'),
            'temperature': temperatures,
            'humidity': np.random.normal(50, 10, 200).clip(0, 100)
        })
        
        scores = self.validator._validate_iot_patterns(realistic_iot)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "IoT patterns should be realistic"
        
        # Create unrealistic IoT data with large jumps
        unrealistic_iot = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='min'),
            'temperature': [20 if i % 2 == 0 else 80 for i in range(100)],  # Large jumps
            'humidity': [30, 90] * 50  # Alternating extremes
        })
        
        scores = self.validator._validate_iot_patterns(unrealistic_iot)
        # Should detect unrealistic jumps
        assert any(score < 0.9 for score in scores), "Should detect unrealistic IoT patterns"
    
    def test_social_pattern_validation(self):
        """Test social media pattern validation"""
        # Create realistic social media data
        realistic_social = pd.DataFrame({
            'post_id': [f'POST_{i:06d}' for i in range(500)],
            'likes': np.random.pareto(1, 500) * 10,  # Power law distribution
            'shares': np.random.pareto(2, 500) * 5,
            'comments': np.random.pareto(1.5, 500) * 3,
            'timestamp': pd.date_range('2023-01-01', periods=500, freq='H')
        })
        
        scores = self.validator._validate_social_patterns(realistic_social)
        assert len(scores) > 0
        assert all(score >= 0.7 for score in scores), "Social patterns should be realistic"


class TestArtificialPatternDetection:
    """Test artificial pattern detection algorithms"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_detect_arithmetic_sequences(self):
        """Test detection of artificial arithmetic sequences"""
        # Create data with arithmetic sequence
        artificial_data = pd.DataFrame({
            'id': range(100),
            'value': [i * 5 for i in range(100)],  # Perfect arithmetic sequence
            'normal': np.random.normal(0, 1, 100)
        })
        
        anomalies = self.validator._detect_numeric_anomalies(artificial_data)
        
        # Should detect arithmetic sequence in either 'id' or 'value' column
        arithmetic_anomalies = [a for a in anomalies if a['type'] == 'arithmetic_sequence']
        assert len(arithmetic_anomalies) > 0, "Should detect arithmetic sequences"
        
        # Check that at least one of the expected columns is detected
        detected_columns = [a['column'] for a in arithmetic_anomalies]
        assert 'value' in detected_columns or 'id' in detected_columns, "Should detect arithmetic sequence in 'value' or 'id' column"
        assert arithmetic_anomalies[0]['severity'] == 'high'
    
    def test_detect_repeated_decimal_patterns(self):
        """Test detection of repeated decimal patterns"""
        # Create data with repeated decimal patterns
        artificial_data = pd.DataFrame({
            'price': [10.99, 20.99, 30.99, 40.99, 50.99] * 20,  # Repeated .99 pattern
            'normal': np.random.uniform(0, 100, 100)
        })
        
        anomalies = self.validator._detect_numeric_anomalies(artificial_data)
        
        # Should detect repeated decimal pattern
        decimal_anomalies = [a for a in anomalies if a['type'] == 'repeated_decimals']
        assert len(decimal_anomalies) > 0, "Should detect repeated decimal patterns"
    
    def test_detect_uniform_string_lengths(self):
        """Test detection of uniform string lengths"""
        # Create data with uniform string lengths
        artificial_data = pd.DataFrame({
            'name': [f'Name{i:04d}' for i in range(100)],  # All same length
            'description': [f'Description {i}' for i in range(100)]  # Variable length
        })
        
        anomalies = self.validator._detect_string_anomalies(artificial_data)
        
        # Should detect uniform string length in 'name' column
        length_anomalies = [a for a in anomalies if a['type'] == 'uniform_string_length']
        assert len(length_anomalies) > 0, "Should detect uniform string lengths"
        assert length_anomalies[0]['column'] == 'name'
    
    def test_detect_temporal_clustering(self):
        """Test detection of temporal clustering"""
        # Create data with excessive clustering at specific hour
        clustered_times = [datetime(2023, 1, 1, 14, 0, 0)] * 80  # 80% at 2 PM
        other_times = pd.date_range('2023-01-01', periods=20, freq='H')
        
        artificial_data = pd.DataFrame({
            'timestamp': list(clustered_times) + list(other_times),
            'value': range(100)
        })
        
        anomalies = self.validator._detect_temporal_anomalies(artificial_data)
        
        # Should detect temporal clustering
        clustering_anomalies = [a for a in anomalies if a['type'] == 'temporal_clustering']
        assert len(clustering_anomalies) > 0, "Should detect temporal clustering"
    
    def test_detect_distribution_anomalies(self):
        """Test detection of distribution anomalies"""
        # Create data with unrealistic uniformity
        artificial_data = pd.DataFrame({
            'uniform': [50.0] * 100,  # Perfect uniformity
            'normal': np.random.normal(50, 10, 100)
        })
        
        anomalies = self.validator._detect_distribution_anomalies(artificial_data)
        
        # Should detect unrealistic uniformity
        uniformity_anomalies = [a for a in anomalies if a['type'] == 'unrealistic_uniformity']
        assert len(uniformity_anomalies) > 0, "Should detect unrealistic uniformity"
        assert uniformity_anomalies[0]['column'] == 'uniform'


class TestCrossDatasetValidation:
    """Test cross-dataset relationship validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_validate_cross_dataset_relationships(self):
        """Test cross-dataset relationship validation"""
        # Create related datasets
        customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(100)],
            'name': [f'Customer {i}' for i in range(100)],
            'country': np.random.choice(['US', 'UK', 'CA'], 100)
        })
        
        orders = pd.DataFrame({
            'order_id': [f'ORD_{i:04d}' for i in range(200)],
            'customer_id': np.random.choice([f'CUST_{i:03d}' for i in range(100)], 200),
            'amount': np.random.uniform(10, 1000, 200),
            'country': np.random.choice(['US', 'UK', 'CA'], 200)
        })
        
        datasets = {'customers': customers, 'orders': orders}
        results = self.validator.validate_cross_dataset_relationships(datasets)
        
        assert results['valid'] is True
        assert results['overall_score'] > 0.7, "Should have good relationship score"
        assert 'customers_orders' in results['relationship_scores']
        assert results['relationship_scores']['customers_orders'] > 0.8, "Should have high referential integrity"
    
    def test_validate_common_id_consistency(self):
        """Test common ID consistency validation"""
        # Create datasets with consistent IDs
        dataset1 = pd.DataFrame({
            'user_id': ['U001', 'U002', 'U003'],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        
        dataset2 = pd.DataFrame({
            'user_id': ['U001', 'U002', 'U004'],  # U004 not in dataset1
            'activity': ['login', 'purchase', 'logout']
        })
        
        datasets = {'users': dataset1, 'activities': dataset2}
        score = self.validator._validate_common_id_consistency(datasets)
        
        assert 0.5 <= score <= 1.0, "Should have reasonable ID consistency score"
    
    def test_validate_temporal_consistency(self):
        """Test temporal consistency across datasets"""
        # Create datasets with overlapping date ranges
        dataset1 = pd.DataFrame({
            'id': range(10),
            'created_date': pd.date_range('2023-01-01', periods=10, freq='D')
        })
        
        dataset2 = pd.DataFrame({
            'id': range(10),
            'updated_date': pd.date_range('2023-01-05', periods=10, freq='D')  # Overlapping range
        })
        
        datasets = {'data1': dataset1, 'data2': dataset2}
        score = self.validator._validate_cross_dataset_temporal_consistency(datasets)
        
        assert score > 0.5, "Should have reasonable temporal consistency"
    
    def test_validate_geographical_consistency(self):
        """Test geographical consistency across datasets"""
        # Create datasets with consistent geographical data
        dataset1 = pd.DataFrame({
            'id': range(10),
            'country': ['US', 'UK', 'CA'] * 3 + ['US']
        })
        
        dataset2 = pd.DataFrame({
            'id': range(10),
            'country': ['US', 'UK', 'DE'] * 3 + ['US']  # Mostly overlapping
        })
        
        datasets = {'data1': dataset1, 'data2': dataset2}
        score = self.validator._validate_cross_dataset_geographical_consistency(datasets)
        
        assert score > 0.5, "Should have reasonable geographical consistency"
    
    def test_cross_dataset_recommendations(self):
        """Test generation of cross-dataset recommendations"""
        # Create datasets with poor relationships
        customers = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(50)],
            'name': [f'Customer {i}' for i in range(50)]
        })
        
        orders = pd.DataFrame({
            'order_id': [f'ORD_{i:04d}' for i in range(100)],
            'customer_id': [f'CUST_{i:03d}' for i in range(100, 200)],  # No matching customers
            'amount': np.random.uniform(10, 1000, 100)
        })
        
        datasets = {'customers': customers, 'orders': orders}
        results = self.validator.validate_cross_dataset_relationships(datasets)
        
        assert len(results['recommendations']) > 0, "Should provide recommendations for poor relationships"
        assert any('referential integrity' in rec.lower() for rec in results['recommendations'])


class TestQualityReporting:
    """Test quality reporting and improvement suggestions"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_comprehensive_quality_report(self):
        """Test comprehensive quality report generation"""
        # Create test dataset
        test_data = pd.DataFrame({
            'id': range(100),
            'name': [f'Name {i}' for i in range(100)],
            'amount': np.random.lognormal(3, 1, 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Create related dataset
        related_data = pd.DataFrame({
            'transaction_id': range(50),
            'id': np.random.choice(range(100), 50),  # References main dataset
            'value': np.random.uniform(1, 100, 50)
        })
        
        related_datasets = {'transactions': related_data}
        
        report = self.validator.generate_comprehensive_quality_report(
            test_data, 'sales', related_datasets
        )
        
        assert 'COMPREHENSIVE DATA QUALITY REPORT' in report
        assert 'OVERALL ASSESSMENT' in report
        assert 'DETAILED QUALITY SCORES' in report
        assert 'CROSS-DATASET RELATIONSHIP ANALYSIS' in report
        assert 'RECOMMENDATIONS' in report
        assert 'PATTERN VALIDATION SUMMARY' in report
    
    def test_quality_improvement_recommendations(self):
        """Test quality improvement recommendation generation"""
        # Create low-quality dataset
        poor_data = pd.DataFrame({
            'id': [1, 1, 2, 2, 3],  # Duplicate IDs
            'name': ['', 'Name', None, 'Name', ''],  # Many nulls
            'amount': ['invalid', '100', 'bad', '200', '300'],  # Type inconsistency
            'date': ['2023-01-01', 'invalid', '2023-01-03', '2023-01-04', '2023-01-05']
        })
        
        results = self.validator.validate_dataset(poor_data, 'sales')
        
        assert len(results['recommendations']) > 0, "Should provide recommendations for poor quality data"
        
        # Check for recommendations about data quality issues
        recommendations_text = ' '.join(results['recommendations']).lower()
        
        # The test should pass if we have recommendations about the main issues
        assert any(keyword in recommendations_text for keyword in ['unique', 'duplicate', 'id']), "Should mention uniqueness issues"
        assert any(keyword in recommendations_text for keyword in ['type', 'data type', 'inconsistencies']), "Should mention data type issues"


if __name__ == '__main__':
    pytest.main([__file__])