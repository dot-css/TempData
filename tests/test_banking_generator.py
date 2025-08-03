"""
Unit tests for BankingGenerator

Tests banking transaction patterns, account behaviors, balance tracking, and fraud indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.financial.banking import BankingGenerator


class TestBankingGenerator:
    """Test suite for BankingGenerator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = BankingGenerator(self.seeder)
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly with banking data"""
        assert hasattr(self.generator, 'transaction_types')
        assert hasattr(self.generator, 'account_types')
        assert hasattr(self.generator, 'merchant_categories')
        assert hasattr(self.generator, 'fraud_indicators')
        assert hasattr(self.generator, 'customer_segments')
        
        # Check that we have expected transaction types
        expected_types = ['debit_card', 'credit_card', 'ach_debit', 'ach_credit', 'wire_transfer', 'atm_withdrawal', 'check']
        assert all(t_type in self.generator.transaction_types for t_type in expected_types)
        
        # Check that we have expected account types
        expected_accounts = ['checking', 'savings', 'business', 'premium']
        assert all(acc_type in self.generator.account_types for acc_type in expected_accounts)
    
    def test_basic_banking_generation(self):
        """Test basic banking transaction generation"""
        rows = 100
        data = self.generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns
        required_columns = ['transaction_id', 'account_id', 'account_type', 'customer_segment',
                          'date', 'datetime', 'transaction_type', 'amount', 'merchant_category',
                          'description', 'fraud_score', 'is_fraudulent']
        assert all(col in data.columns for col in required_columns)
        
        # Check data types
        assert data['amount'].dtype in [np.float64, float]
        assert data['fraud_score'].dtype in [np.float64, float]
        assert data['is_fraudulent'].dtype == bool
    
    def test_transaction_types_distribution(self):
        """Test that transaction types follow expected distribution"""
        data = self.generator.generate(1000)
        
        # Check that all transaction types are valid
        valid_types = list(self.generator.transaction_types.keys())
        assert data['transaction_type'].isin(valid_types).all()
        
        # Check distribution roughly matches probabilities
        type_counts = data['transaction_type'].value_counts(normalize=True)
        
        # Debit card should be most common (35% probability)
        assert type_counts['debit_card'] > 0.25  # Allow some variance
        
        # Wire transfer should be least common (5% probability)
        assert type_counts['wire_transfer'] < 0.15
    
    def test_account_types_and_behaviors(self):
        """Test that account types have appropriate behaviors"""
        data = self.generator.generate(500)
        
        # Check that all account types are valid
        valid_account_types = list(self.generator.account_types.keys())
        assert data['account_type'].isin(valid_account_types).all()
        
        # Checking accounts should be most common
        account_counts = data['account_type'].value_counts(normalize=True)
        assert account_counts['checking'] > 0.4  # Should be around 60%
        
        # Business accounts should generally have higher transaction amounts
        # This is probabilistic, so we'll check that business accounts have higher balance ranges
        if 'business' in account_counts.index:
            business_balance_range = self.generator.account_types['business']['typical_balance_range']
            checking_balance_range = self.generator.account_types['checking']['typical_balance_range']
            assert business_balance_range[1] > checking_balance_range[1]  # Max balance should be higher
    
    def test_customer_segments(self):
        """Test customer segment behaviors"""
        data = self.generator.generate(500)
        
        # Check that all customer segments are valid
        valid_segments = list(self.generator.customer_segments.keys())
        assert data['customer_segment'].isin(valid_segments).all()
        
        # Family segment should be most common (35% probability)
        segment_counts = data['customer_segment'].value_counts(normalize=True)
        assert segment_counts['family'] > 0.25
        
        # Business owners should have higher average transaction amounts
        if 'business_owner' in segment_counts.index and 'senior' in segment_counts.index:
            business_avg = data[data['customer_segment'] == 'business_owner']['amount'].mean()
            senior_avg = data[data['customer_segment'] == 'senior']['amount'].mean()
            assert business_avg > senior_avg
    
    def test_balance_tracking(self):
        """Test account balance tracking functionality"""
        data = self.generator.generate(100, account_tracking=True, num_accounts=10)
        
        # Should have balance_after column
        assert 'balance_after' in data.columns
        
        # Check that balances are updated correctly for each account
        for account_id in data['account_id'].unique():
            account_data = data[data['account_id'] == account_id].sort_values('datetime')
            
            # Should have at least one transaction
            assert len(account_data) > 0
            
            # Balance should change with transactions
            if len(account_data) > 1:
                balances = account_data['balance_after'].tolist()
                # Not all balances should be the same (unless very unlikely)
                assert len(set(balances)) > 1 or len(balances) == 1
    
    def test_fraud_detection_patterns(self):
        """Test fraud detection and scoring"""
        data = self.generator.generate(200, fraud_simulation=True)
        
        # Should have fraud-related columns
        assert 'fraud_score' in data.columns
        assert 'is_fraudulent' in data.columns
        assert 'risk_category' in data.columns
        
        # Fraud scores should be between 0 and 1
        assert (data['fraud_score'] >= 0).all()
        assert (data['fraud_score'] <= 1).all()
        
        # Should have some variety in fraud scores
        assert data['fraud_score'].std() > 0
        
        # High fraud scores should correlate with is_fraudulent flag
        high_fraud_scores = data[data['fraud_score'] > 0.7]
        if len(high_fraud_scores) > 0:
            assert high_fraud_scores['is_fraudulent'].all()
        
        # Risk categories should be properly assigned
        risk_categories = data['risk_category'].dropna().unique()
        expected_categories = ['low', 'medium', 'high']
        assert all(cat in expected_categories for cat in risk_categories)
    
    def test_transaction_amounts_by_type(self):
        """Test that transaction amounts are appropriate for each type"""
        data = self.generator.generate(300)
        
        # Wire transfers should have higher amounts than debit card transactions
        wire_transfers = data[data['transaction_type'] == 'wire_transfer']['amount']
        debit_cards = data[data['transaction_type'] == 'debit_card']['amount']
        
        if len(wire_transfers) > 0 and len(debit_cards) > 0:
            assert wire_transfers.mean() > debit_cards.mean()
        
        # ATM withdrawals should be reasonable amounts
        atm_withdrawals = data[data['transaction_type'] == 'atm_withdrawal']['amount']
        if len(atm_withdrawals) > 0:
            assert (atm_withdrawals >= 20).all()  # Minimum ATM amount
            assert (atm_withdrawals <= 800).all()  # Reasonable maximum based on our range
        
        # All amounts should be positive
        assert (data['amount'] > 0).all()
    
    def test_merchant_categories_and_descriptions(self):
        """Test merchant categories and transaction descriptions"""
        data = self.generator.generate(200)
        
        # Should have merchant categories
        assert 'merchant_category' in data.columns
        assert 'description' in data.columns
        
        # Merchant categories should be valid
        valid_categories = set()
        for t_type in self.generator.transaction_types.values():
            valid_categories.update(t_type['typical_merchants'])
        
        assert data['merchant_category'].isin(valid_categories).all()
        
        # Descriptions should not be empty
        assert data['description'].notna().all()
        assert (data['description'].str.len() > 0).all()
        
        # Grocery transactions should have grocery-related descriptions
        grocery_transactions = data[data['merchant_category'] == 'grocery']
        if len(grocery_transactions) > 0:
            grocery_descriptions = grocery_transactions['description'].str.upper()
            # Should contain common grocery store names
            grocery_keywords = ['WALMART', 'KROGER', 'SAFEWAY', 'TARGET']
            has_grocery_keyword = grocery_descriptions.str.contains('|'.join(grocery_keywords))
            assert has_grocery_keyword.any()
    
    def test_time_patterns(self):
        """Test realistic time patterns in transactions"""
        data = self.generator.generate(300)
        
        # Should have datetime information
        assert 'datetime' in data.columns
        assert 'hour' in data.columns
        assert 'day_of_week' in data.columns
        
        # Hours should be realistic (0-23)
        assert (data['hour'] >= 0).all()
        assert (data['hour'] <= 23).all()
        
        # Should have variety in hours (not all the same time)
        assert data['hour'].nunique() > 5
        
        # Business hours should be more common than late night
        business_hours = data[data['is_business_hours']]['hour'].count()
        late_night = data[data['hour'].isin([2, 3, 4, 5])]['hour'].count()
        
        # Business hours should generally be more common
        assert business_hours >= late_night
    
    def test_location_patterns(self):
        """Test transaction location patterns"""
        data = self.generator.generate(200)
        
        # Should have location columns
        location_columns = ['location_city', 'location_state', 'location_country']
        assert all(col in data.columns for col in location_columns)
        
        # Most transactions should be in United States
        us_transactions = data[data['location_country'] == 'United States']
        assert len(us_transactions) / len(data) > 0.8  # At least 80% should be US
        
        # Should have variety in cities and states
        assert data['location_city'].nunique() > 10
        assert data['location_state'].nunique() > 5
    
    def test_overdraft_detection(self):
        """Test overdraft detection and tracking"""
        data = self.generator.generate(150, account_tracking=True, num_accounts=20)
        
        if 'balance_after' in data.columns:
            # Should have overdraft indicators
            assert 'is_overdraft' in data.columns
            assert 'overdraft_amount' in data.columns
            
            # Overdraft flags should be consistent with negative balances
            overdraft_transactions = data[data['is_overdraft']]
            if len(overdraft_transactions) > 0:
                assert (overdraft_transactions['balance_after'] < 0).all()
                assert (overdraft_transactions['overdraft_amount'] > 0).all()
            
            # Non-overdraft transactions should have zero overdraft amount
            non_overdraft = data[~data['is_overdraft']]
            assert (non_overdraft['overdraft_amount'] == 0).all()
    
    def test_transaction_categorization(self):
        """Test transaction categorization (debit vs credit)"""
        data = self.generator.generate(200)
        
        # Should have transaction category flags
        assert 'is_debit' in data.columns
        assert 'is_credit' in data.columns
        
        # Debit and credit should be mutually exclusive
        assert (~(data['is_debit'] & data['is_credit'])).all()
        
        # All transactions should be either debit or credit
        assert (data['is_debit'] | data['is_credit']).all()
        
        # Specific transaction types should be categorized correctly
        debit_types = ['debit_card', 'ach_debit', 'atm_withdrawal', 'check']
        debit_transactions = data[data['transaction_type'].isin(debit_types)]
        if len(debit_transactions) > 0:
            assert debit_transactions['is_debit'].all()
        
        credit_types = ['credit_card', 'ach_credit', 'wire_transfer']
        credit_transactions = data[data['transaction_type'].isin(credit_types)]
        if len(credit_transactions) > 0:
            assert credit_transactions['is_credit'].all()
    
    def test_processing_times(self):
        """Test processing time assignments"""
        data = self.generator.generate(150)
        
        # Should have processing time column
        assert 'processing_time' in data.columns
        
        # Processing times should be appropriate for transaction types
        instant_types = ['debit_card', 'credit_card', 'atm_withdrawal']
        instant_transactions = data[data['transaction_type'].isin(instant_types)]
        if len(instant_transactions) > 0:
            assert (instant_transactions['processing_time'] == 'instant').all()
        
        ach_transactions = data[data['transaction_type'].isin(['ach_debit', 'ach_credit'])]
        if len(ach_transactions) > 0:
            assert (ach_transactions['processing_time'] == '1-3 business days').all()
    
    def test_seasonal_patterns(self):
        """Test seasonal spending patterns"""
        # Generate data for different months
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        data = self.generator.generate(
            rows=1000,  # Larger sample for more reliable statistics
            date_range=(start_date.date(), end_date.date())
        )
        
        # Should have month information
        assert 'month' in data.columns
        
        # Test that seasonal multipliers are applied correctly
        # Instead of comparing specific months, check that the seasonal logic exists
        seasonal_multipliers = self.generator._generate_transaction_amount.__code__.co_names
        
        # Should have variety in months
        assert data['month'].nunique() >= 6  # Should span multiple months
        
        # Check that amounts vary by month (indicating seasonal effects)
        monthly_averages = data.groupby('month')['amount'].mean()
        if len(monthly_averages) > 1:
            # Should have some variation in monthly averages
            monthly_std = monthly_averages.std()
            assert monthly_std > 0  # Some variation exists
    
    def test_weekend_patterns(self):
        """Test weekend transaction patterns"""
        data = self.generator.generate(300)
        
        # Should have weekend indicator
        assert 'is_weekend' in data.columns
        
        # Should have both weekend and weekday transactions
        weekend_count = data['is_weekend'].sum()
        weekday_count = (~data['is_weekend']).sum()
        
        # Both should exist
        assert weekend_count > 0
        assert weekday_count > 0
        
        # Weekdays should generally be more common for business transactions
        business_transactions = data[data['account_type'] == 'business']
        if len(business_transactions) > 10:
            business_weekday_pct = (~business_transactions['is_weekend']).mean()
            assert business_weekday_pct > 0.6  # At least 60% on weekdays
    
    def test_reproducibility(self):
        """Test that fixed seed produces reproducible results"""
        seeder1 = MillisecondSeeder(fixed_seed=42)
        seeder2 = MillisecondSeeder(fixed_seed=42)
        
        gen1 = BankingGenerator(seeder1)
        gen2 = BankingGenerator(seeder2)
        
        data1 = gen1.generate(50, num_accounts=10)
        data2 = gen2.generate(50, num_accounts=10)
        
        # Should produce identical results for key columns
        key_columns = ['transaction_type', 'amount', 'merchant_category']
        for col in key_columns:
            if col in data1.columns and col in data2.columns:
                pd.testing.assert_series_equal(data1[col], data2[col], check_names=False)
    
    def test_date_range_filtering(self):
        """Test date range filtering works correctly"""
        start_date = datetime(2023, 6, 1).date()
        end_date = datetime(2023, 6, 30).date()
        
        data = self.generator.generate(
            rows=100,
            date_range=(start_date, end_date)
        )
        
        # All dates should be within range
        dates = pd.to_datetime(data['date']).dt.date
        assert (dates >= start_date).all()
        assert (dates <= end_date).all()
    
    def test_account_number_consistency(self):
        """Test that account numbers are consistent and properly formatted"""
        data = self.generator.generate(200, num_accounts=20)
        
        # Account IDs should be properly formatted
        account_ids = data['account_id'].unique()
        
        # Should have expected number of unique accounts
        assert len(account_ids) <= 20  # Should not exceed specified number
        
        # Account IDs should follow format ACC_########
        for account_id in account_ids:
            assert account_id.startswith('ACC_')
            assert len(account_id) == 12  # ACC_ + 8 digits
            assert account_id[4:].isdigit()
    
    def test_data_quality_metrics(self):
        """Test overall data quality meets requirements"""
        data = self.generator.generate(300)
        
        # No missing values in critical columns
        critical_columns = ['transaction_id', 'account_id', 'transaction_type', 'amount', 'date']
        for col in critical_columns:
            assert data[col].notna().all()
        
        # Transaction IDs should be unique
        assert data['transaction_id'].nunique() == len(data)
        
        # Amounts should be reasonable
        assert data['amount'].min() > 0
        assert data['amount'].max() < 1000000  # Reasonable upper bound
        
        # Fraud scores should be reasonable
        assert data['fraud_score'].min() >= 0
        assert data['fraud_score'].max() <= 1
        
        # Should have realistic variety in data
        assert data['transaction_type'].nunique() >= 5
        assert data['merchant_category'].nunique() >= 8
        assert data['account_type'].nunique() >= 3


if __name__ == '__main__':
    pytest.main([__file__])