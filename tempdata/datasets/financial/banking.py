"""
Banking transaction data generator

Generates realistic banking data with transaction patterns and fraud indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class BankingGenerator(BaseGenerator):
    """
    Generator for realistic banking transaction data
    
    Creates banking datasets with transaction patterns, account behaviors,
    balance tracking, and fraud indicators.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_transaction_types()
        self._setup_account_types()
        self._setup_merchant_categories()
        self._setup_fraud_patterns()
        self._setup_behavioral_patterns()
    
    def _setup_transaction_types(self):
        """Setup transaction types with realistic patterns"""
        self.transaction_types = {
            'debit_card': {
                'probability': 0.35,
                'amount_range': (5, 500),
                'typical_merchants': ['grocery', 'gas_station', 'restaurant', 'retail', 'pharmacy'],
                'fraud_risk': 0.02
            },
            'credit_card': {
                'probability': 0.25,
                'amount_range': (10, 2000),
                'typical_merchants': ['retail', 'restaurant', 'online', 'travel', 'entertainment'],
                'fraud_risk': 0.03
            },
            'ach_debit': {
                'probability': 0.15,
                'amount_range': (50, 5000),
                'typical_merchants': ['utility', 'insurance', 'loan_payment', 'subscription'],
                'fraud_risk': 0.001
            },
            'ach_credit': {
                'probability': 0.10,
                'amount_range': (100, 10000),
                'typical_merchants': ['salary', 'refund', 'government', 'investment'],
                'fraud_risk': 0.0005
            },
            'wire_transfer': {
                'probability': 0.05,
                'amount_range': (1000, 100000),
                'typical_merchants': ['bank_transfer', 'real_estate', 'business'],
                'fraud_risk': 0.01
            },
            'atm_withdrawal': {
                'probability': 0.08,
                'amount_range': (20, 500),
                'typical_merchants': ['atm'],
                'fraud_risk': 0.005
            },
            'check': {
                'probability': 0.02,
                'amount_range': (50, 5000),
                'typical_merchants': ['rent', 'contractor', 'personal'],
                'fraud_risk': 0.008
            }
        }
    
    def _setup_account_types(self):
        """Setup account types with different behaviors"""
        self.account_types = {
            'checking': {
                'probability': 0.6,
                'typical_balance_range': (500, 15000),
                'transaction_frequency': 'high',  # 15-30 transactions per month
                'overdraft_risk': 0.15
            },
            'savings': {
                'probability': 0.25,
                'typical_balance_range': (1000, 50000),
                'transaction_frequency': 'low',  # 2-8 transactions per month
                'overdraft_risk': 0.02
            },
            'business': {
                'probability': 0.10,
                'typical_balance_range': (5000, 500000),
                'transaction_frequency': 'very_high',  # 50-200 transactions per month
                'overdraft_risk': 0.05
            },
            'premium': {
                'probability': 0.05,
                'typical_balance_range': (25000, 1000000),
                'transaction_frequency': 'medium',  # 10-25 transactions per month
                'overdraft_risk': 0.01
            }
        }
    
    def _setup_merchant_categories(self):
        """Setup merchant categories with realistic transaction patterns"""
        self.merchant_categories = {
            'grocery': {
                'typical_amounts': (20, 200),
                'frequency_pattern': 'weekly',
                'time_patterns': [8, 9, 10, 17, 18, 19],  # Hours of day
                'seasonal_multiplier': {'12': 1.3, '11': 1.2, '1': 0.9}  # Holiday season
            },
            'gas_station': {
                'typical_amounts': (25, 80),
                'frequency_pattern': 'weekly',
                'time_patterns': [7, 8, 17, 18],
                'seasonal_multiplier': {'6': 1.2, '7': 1.2, '8': 1.1}  # Summer travel
            },
            'restaurant': {
                'typical_amounts': (15, 150),
                'frequency_pattern': 'frequent',
                'time_patterns': [12, 13, 18, 19, 20],
                'seasonal_multiplier': {'12': 1.2, '2': 1.1}  # Holidays and Valentine's
            },
            'retail': {
                'typical_amounts': (30, 500),
                'frequency_pattern': 'irregular',
                'time_patterns': [10, 11, 14, 15, 16, 19, 20],
                'seasonal_multiplier': {'11': 1.5, '12': 1.8, '1': 0.7}  # Black Friday/Christmas
            },
            'utility': {
                'typical_amounts': (50, 300),
                'frequency_pattern': 'monthly',
                'time_patterns': [9, 10, 11, 14, 15],
                'seasonal_multiplier': {'1': 1.3, '7': 1.2, '8': 1.2}  # Winter heating, summer cooling
            },
            'salary': {
                'typical_amounts': (2000, 15000),
                'frequency_pattern': 'biweekly',
                'time_patterns': [0, 1, 2],  # Early morning processing
                'seasonal_multiplier': {'12': 1.1}  # Year-end bonuses
            },
            'atm': {
                'typical_amounts': (20, 400),
                'frequency_pattern': 'weekly',
                'time_patterns': [12, 17, 18, 19, 20, 21],
                'seasonal_multiplier': {}
            },
            'online': {
                'typical_amounts': (25, 300),
                'frequency_pattern': 'frequent',
                'time_patterns': [19, 20, 21, 22],  # Evening shopping
                'seasonal_multiplier': {'11': 1.4, '12': 1.3}  # Holiday shopping
            }
        }
    
    def _setup_fraud_patterns(self):
        """Setup fraud detection patterns"""
        self.fraud_indicators = {
            'unusual_amount': {
                'threshold_multiplier': 5.0,  # 5x normal transaction amount
                'weight': 0.3
            },
            'unusual_time': {
                'suspicious_hours': [2, 3, 4, 5],  # Late night/early morning
                'weight': 0.2
            },
            'unusual_location': {
                'foreign_country_risk': 0.8,
                'different_state_risk': 0.3,
                'weight': 0.25
            },
            'velocity': {
                'max_transactions_per_hour': 10,
                'max_amount_per_hour': 5000,
                'weight': 0.25
            }
        }
    
    def _setup_behavioral_patterns(self):
        """Setup behavioral patterns for different customer segments"""
        self.customer_segments = {
            'young_professional': {
                'age_range': (22, 35),
                'transaction_preferences': ['debit_card', 'online', 'restaurant'],
                'spending_pattern': 'moderate',
                'tech_savvy': True,
                'probability': 0.3
            },
            'family': {
                'age_range': (30, 55),
                'transaction_preferences': ['debit_card', 'grocery', 'utility', 'gas_station'],
                'spending_pattern': 'high',
                'tech_savvy': False,
                'probability': 0.35
            },
            'senior': {
                'age_range': (55, 85),
                'transaction_preferences': ['check', 'atm_withdrawal', 'pharmacy'],
                'spending_pattern': 'low',
                'tech_savvy': False,
                'probability': 0.20
            },
            'business_owner': {
                'age_range': (25, 65),
                'transaction_preferences': ['wire_transfer', 'ach_debit', 'business'],
                'spending_pattern': 'very_high',
                'tech_savvy': True,
                'probability': 0.15
            }
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate banking transaction dataset
        
        Args:
            rows: Number of transactions to generate
            **kwargs: Additional parameters (account_tracking, fraud_simulation, time_series, etc.)
            
        Returns:
            pd.DataFrame: Generated banking data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_banking(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_banking(rows, **kwargs)
    
    def _generate_snapshot_banking(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot banking data (random timestamps)"""
        account_tracking = kwargs.get('account_tracking', True)
        fraud_simulation = kwargs.get('fraud_simulation', True)
        date_range = kwargs.get('date_range', None)
        num_accounts = kwargs.get('num_accounts', min(rows // 10, 1000))
        
        # Generate account profiles first
        accounts = self._generate_account_profiles(num_accounts)
        
        data = []
        account_balances = {acc['account_id']: acc['initial_balance'] for acc in accounts}
        
        for i in range(rows):
            # Select account (weighted by transaction frequency)
            account = self._select_account_for_transaction(accounts)
            
            # Generate transaction
            transaction = self._generate_transaction(
                account, i, date_range, account_balances, fraud_simulation
            )
            
            # Update balance if tracking is enabled
            if account_tracking:
                account_id = transaction['account_id']
                if transaction['transaction_type'] in ['debit_card', 'ach_debit', 'atm_withdrawal', 'check']:
                    account_balances[account_id] -= transaction['amount']
                else:  # credit transactions
                    account_balances[account_id] += transaction['amount']
                
                transaction['balance_after'] = round(account_balances[account_id], 2)
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_banking(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series banking data using integrated time series system"""
        account_tracking = kwargs.get('account_tracking', True)
        fraud_simulation = kwargs.get('fraud_simulation', True)
        num_accounts = kwargs.get('num_accounts', min(rows // 20, 500))  # Fewer accounts for time series
        
        # Generate account profiles
        accounts = self._generate_account_profiles(num_accounts)
        account_balances = {acc['account_id']: acc['initial_balance'] for acc in accounts}
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        data = []
        
        # Track transaction patterns over time for realistic correlations
        daily_transaction_counts = {}
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows:
                break
            
            # Select account with time-based weighting
            account = self._select_account_for_time_series_transaction(accounts, timestamp, daily_transaction_counts)
            
            # Generate transaction with temporal patterns
            transaction = self._generate_time_series_transaction(
                account, i, timestamp, account_balances, fraud_simulation, daily_transaction_counts
            )
            
            # Update balance if tracking is enabled
            if account_tracking:
                account_id = transaction['account_id']
                if transaction['transaction_type'] in ['debit_card', 'ach_debit', 'atm_withdrawal', 'check']:
                    account_balances[account_id] -= transaction['amount']
                else:  # credit transactions
                    account_balances[account_id] += transaction['amount']
                
                transaction['balance_after'] = round(account_balances[account_id], 2)
            
            # Track daily transaction count
            date_key = timestamp.date()
            if date_key not in daily_transaction_counts:
                daily_transaction_counts[date_key] = {}
            if account['account_id'] not in daily_transaction_counts[date_key]:
                daily_transaction_counts[date_key][account['account_id']] = 0
            daily_transaction_counts[date_key][account['account_id']] += 1
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Rename datetime to timestamp for consistency
        if 'datetime' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = df['datetime']
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply banking-specific time series correlations
        df = self._apply_banking_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _apply_banking_time_series_correlations(self, df: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply banking-specific time series correlations"""
        if 'timestamp' not in df.columns:
            return df
        
        # Sort by timestamp for proper correlation analysis
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add time-based features for banking patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5))
        df['is_weekend'] = (df['day_of_week'] >= 5)
        
        # Apply temporal correlations to transaction amounts
        # Business hours tend to have larger transaction amounts
        business_hours_mask = df['is_business_hours']
        df.loc[business_hours_mask, 'amount'] *= self.faker.random.uniform(1.1, 1.3)
        
        # Weekend transactions tend to be smaller and more personal
        weekend_mask = df['is_weekend']
        df.loc[weekend_mask, 'amount'] *= self.faker.random.uniform(0.7, 0.9)
        
        # Add transaction velocity correlation (more transactions = smaller average amounts)
        df['transactions_per_hour'] = df.groupby([df['timestamp'].dt.floor('H'), 'account_id']).cumcount() + 1
        high_velocity_mask = df['transactions_per_hour'] > 3
        df.loc[high_velocity_mask, 'amount'] *= self.faker.random.uniform(0.6, 0.8)
        
        # Add balance-based correlations
        if 'balance_after' in df.columns:
            # Low balance accounts tend to have smaller transactions
            low_balance_mask = df['balance_after'] < 100
            df.loc[low_balance_mask, 'amount'] *= self.faker.random.uniform(0.5, 0.8)
            
            # Add balance trend correlation
            df['balance_trend'] = df.groupby('account_id')['balance_after'].pct_change()
            df['balance_trend'] = df['balance_trend'].fillna(0)
        
        # Add fraud risk correlation based on temporal patterns
        # Unusual hours increase fraud risk
        unusual_hours_mask = (df['hour'] < 6) | (df['hour'] > 22)
        df.loc[unusual_hours_mask, 'fraud_risk_score'] = df.loc[unusual_hours_mask, 'fraud_risk_score'] * 1.5
        
        # Multiple transactions in short time increase fraud risk
        df['recent_transaction_count'] = df.groupby('account_id')['timestamp'].rolling('1H').count().values
        high_frequency_mask = df['recent_transaction_count'] > 5
        df.loc[high_frequency_mask, 'fraud_risk_score'] = df.loc[high_frequency_mask, 'fraud_risk_score'] * 1.3
        
        return df
    
    def _generate_account_profiles(self, num_accounts: int) -> List[Dict]:
        """Generate account profiles with different characteristics"""
        accounts = []
        
        for i in range(num_accounts):
            # Select account type
            account_type = self._select_weighted_choice(
                list(self.account_types.keys()),
                [self.account_types[t]['probability'] for t in self.account_types.keys()]
            )
            
            # Select customer segment
            segment = self._select_weighted_choice(
                list(self.customer_segments.keys()),
                [self.customer_segments[s]['probability'] for s in self.customer_segments.keys()]
            )
            
            # Generate account details
            account_info = self.account_types[account_type]
            segment_info = self.customer_segments[segment]
            
            min_balance, max_balance = account_info['typical_balance_range']
            initial_balance = self.faker.random.uniform(min_balance, max_balance)
            
            account = {
                'account_id': f'ACC_{i+1:08d}',
                'account_type': account_type,
                'customer_segment': segment,
                'initial_balance': round(initial_balance, 2),
                'transaction_frequency': account_info['transaction_frequency'],
                'overdraft_risk': account_info['overdraft_risk'],
                'age': self.faker.random_int(*segment_info['age_range']),
                'tech_savvy': segment_info['tech_savvy'],
                'spending_pattern': segment_info['spending_pattern'],
                'preferred_transactions': segment_info['transaction_preferences']
            }
            
            accounts.append(account)
        
        return accounts
    
    def _select_account_for_transaction(self, accounts: List[Dict]) -> Dict:
        """Select account for transaction based on frequency patterns"""
        # Weight accounts by their transaction frequency
        weights = []
        frequency_weights = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 5
        }
        
        for account in accounts:
            weight = frequency_weights.get(account['transaction_frequency'], 1)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select account
        rand_val = self.faker.random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return accounts[i]
        
        return accounts[-1]  # Fallback
    
    def _generate_transaction(self, account: Dict, transaction_index: int, 
                            date_range: Tuple, account_balances: Dict, 
                            fraud_simulation: bool) -> Dict:
        """Generate a single banking transaction"""
        
        # Generate transaction date
        if date_range:
            start_date, end_date = date_range
            transaction_date = self.faker.date_between(start_date=start_date, end_date=end_date)
        else:
            transaction_date = self.faker.date_this_year()
        
        # Add time component
        hour = self._select_transaction_hour(account)
        minute = self.faker.random_int(0, 59)
        transaction_datetime = datetime.combine(transaction_date, datetime.min.time().replace(hour=hour, minute=minute))
        
        # Select transaction type based on account preferences
        transaction_type = self._select_transaction_type(account)
        
        # Generate amount based on transaction type and account behavior
        amount = self._generate_transaction_amount(transaction_type, account, transaction_datetime)
        
        # Generate merchant/description
        merchant_category = self._select_merchant_category(transaction_type)
        description = self._generate_transaction_description(transaction_type, merchant_category)
        
        # Generate location
        location = self._generate_transaction_location(account)
        
        # Calculate fraud score if enabled
        fraud_score = 0.0
        is_fraudulent = False
        
        if fraud_simulation:
            fraud_score = self._calculate_fraud_score(
                account, transaction_type, amount, transaction_datetime, location
            )
            is_fraudulent = fraud_score > 0.7  # Threshold for fraud
        
        # Generate transaction ID
        transaction_id = f'TXN_{transaction_index+1:012d}'
        
        return {
            'transaction_id': transaction_id,
            'account_id': account['account_id'],
            'account_type': account['account_type'],
            'customer_segment': account['customer_segment'],
            'date': transaction_date,
            'datetime': transaction_datetime,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'description': description,
            'location_city': location['city'],
            'location_state': location['state'],
            'location_country': location['country'],
            'fraud_score': round(fraud_score, 3),
            'is_fraudulent': is_fraudulent,
            'processing_time': self._generate_processing_time(transaction_type)
        }
    
    def _select_transaction_hour(self, account: Dict) -> int:
        """Select transaction hour based on account behavior and transaction patterns"""
        segment = account['customer_segment']
        
        # Different segments have different activity patterns
        if segment == 'young_professional':
            # More evening and lunch transactions
            hour_weights = {8: 1, 9: 2, 12: 3, 13: 3, 17: 2, 18: 3, 19: 4, 20: 3, 21: 2}
        elif segment == 'family':
            # More morning and evening transactions
            hour_weights = {7: 2, 8: 3, 9: 4, 10: 3, 17: 3, 18: 4, 19: 3, 20: 2}
        elif segment == 'senior':
            # More daytime transactions
            hour_weights = {9: 3, 10: 4, 11: 4, 14: 3, 15: 3, 16: 2}
        else:  # business_owner
            # Business hours
            hour_weights = {8: 2, 9: 4, 10: 4, 11: 3, 14: 3, 15: 4, 16: 3, 17: 2}
        
        # Select hour based on weights
        hours = list(hour_weights.keys())
        weights = list(hour_weights.values())
        
        return self._select_weighted_choice(hours, weights)
    
    def _select_transaction_type(self, account: Dict) -> str:
        """Select transaction type based on account preferences"""
        preferred_types = account['preferred_transactions']
        
        # 70% chance to use preferred transaction type
        if self.faker.random.random() < 0.7 and preferred_types:
            # Select from preferred types
            available_preferred = [t for t in preferred_types if t in self.transaction_types]
            if available_preferred:
                return self.faker.random_element(available_preferred)
        
        # Otherwise select based on general probabilities
        types = list(self.transaction_types.keys())
        probabilities = [self.transaction_types[t]['probability'] for t in types]
        
        return self._select_weighted_choice(types, probabilities)
    
    def _generate_transaction_amount(self, transaction_type: str, account: Dict, 
                                   transaction_datetime: datetime) -> float:
        """Generate transaction amount based on type and account behavior"""
        type_info = self.transaction_types[transaction_type]
        min_amount, max_amount = type_info['amount_range']
        
        # Adjust based on spending pattern
        spending_multipliers = {
            'low': 0.6,
            'moderate': 1.0,
            'high': 1.4,
            'very_high': 2.0
        }
        
        multiplier = spending_multipliers.get(account['spending_pattern'], 1.0)
        
        # Apply seasonal adjustments
        month = transaction_datetime.month
        seasonal_multiplier = 1.0
        
        # Holiday season spending increase
        if month in [11, 12]:
            seasonal_multiplier = 1.2
        elif month == 1:  # Post-holiday decrease
            seasonal_multiplier = 0.8
        
        # Generate base amount
        base_amount = self.faker.random.uniform(min_amount, max_amount)
        
        # Apply multipliers
        final_amount = base_amount * multiplier * seasonal_multiplier
        
        # Add some randomness
        variance = self.faker.random.uniform(0.8, 1.3)
        final_amount *= variance
        
        # Ensure minimum amount
        final_amount = max(final_amount, 1.0)
        
        return final_amount
    
    def _select_merchant_category(self, transaction_type: str) -> str:
        """Select merchant category based on transaction type"""
        type_info = self.transaction_types[transaction_type]
        typical_merchants = type_info['typical_merchants']
        
        return self.faker.random_element(typical_merchants)
    
    def _generate_transaction_description(self, transaction_type: str, merchant_category: str) -> str:
        """Generate realistic transaction description"""
        descriptions = {
            'grocery': ['WALMART SUPERCENTER', 'KROGER', 'SAFEWAY', 'WHOLE FOODS', 'TARGET'],
            'gas_station': ['SHELL', 'EXXON', 'BP', 'CHEVRON', 'MOBIL'],
            'restaurant': ['MCDONALDS', 'STARBUCKS', 'SUBWAY', 'CHIPOTLE', 'OLIVE GARDEN'],
            'retail': ['AMAZON.COM', 'TARGET', 'BEST BUY', 'MACYS', 'HOME DEPOT'],
            'pharmacy': ['CVS PHARMACY', 'WALGREENS', 'RITE AID'],
            'utility': ['ELECTRIC COMPANY', 'GAS COMPANY', 'WATER DEPT', 'INTERNET SERVICE'],
            'salary': ['PAYROLL DEPOSIT', 'DIRECT DEPOSIT', 'SALARY'],
            'atm': ['ATM WITHDRAWAL', 'CASH WITHDRAWAL'],
            'online': ['PAYPAL', 'AMAZON.COM', 'EBAY', 'NETFLIX'],
            'insurance': ['AUTO INSURANCE', 'HEALTH INSURANCE', 'HOME INSURANCE'],
            'loan_payment': ['MORTGAGE PAYMENT', 'AUTO LOAN', 'STUDENT LOAN'],
            'bank_transfer': ['WIRE TRANSFER', 'BANK TRANSFER', 'ACH TRANSFER']
        }
        
        if merchant_category in descriptions:
            base_description = self.faker.random_element(descriptions[merchant_category])
            
            # Add location for some merchants
            if merchant_category in ['grocery', 'gas_station', 'restaurant', 'retail']:
                city = self.faker.city().upper()
                state = self.faker.state_abbr()
                return f'{base_description} {city} {state}'
            
            return base_description
        
        return f'{merchant_category.upper()} TRANSACTION'
    
    def _generate_transaction_location(self, account: Dict) -> Dict:
        """Generate transaction location"""
        # Most transactions are local (90%)
        if self.faker.random.random() < 0.9:
            return {
                'city': self.faker.city(),
                'state': self.faker.state(),
                'country': 'United States'
            }
        else:
            # Some transactions are from different locations
            return {
                'city': self.faker.city(),
                'state': self.faker.state(),
                'country': self.faker.country()
            }
    
    def _calculate_fraud_score(self, account: Dict, transaction_type: str, 
                             amount: float, transaction_datetime: datetime, 
                             location: Dict) -> float:
        """Calculate fraud score based on various indicators"""
        score = 0.0
        
        # Base fraud risk for transaction type
        base_risk = self.transaction_types[transaction_type]['fraud_risk']
        score += base_risk
        
        # Unusual amount (compared to typical range)
        type_info = self.transaction_types[transaction_type]
        min_amount, max_amount = type_info['amount_range']
        
        if amount > max_amount * 3:  # 3x typical maximum
            score += 0.4
        elif amount > max_amount * 2:  # 2x typical maximum
            score += 0.2
        
        # Unusual time
        hour = transaction_datetime.hour
        if hour in [2, 3, 4, 5]:  # Late night/early morning
            score += 0.3
        elif hour in [0, 1, 6]:
            score += 0.1
        
        # Foreign country transactions
        if location['country'] != 'United States':
            score += 0.5
        
        # Weekend transactions for business accounts
        if account['account_type'] == 'business' and transaction_datetime.weekday() >= 5:
            score += 0.2
        
        # Large amounts for certain transaction types
        if transaction_type in ['atm_withdrawal'] and amount > 1000:
            score += 0.3
        
        # Ensure score is between 0 and 1
        return min(score, 1.0)
    
    def _generate_processing_time(self, transaction_type: str) -> str:
        """Generate processing time based on transaction type"""
        processing_times = {
            'debit_card': 'instant',
            'credit_card': 'instant',
            'ach_debit': '1-3 business days',
            'ach_credit': '1-3 business days',
            'wire_transfer': 'same day',
            'atm_withdrawal': 'instant',
            'check': '1-5 business days'
        }
        
        return processing_times.get(transaction_type, 'instant')
    
    def _select_weighted_choice(self, choices: List, weights: List) -> Any:
        """Select item from choices based on weights"""
        total_weight = sum(weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for choice, weight in zip(choices, weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return choice
        
        return choices[-1]  # Fallback
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to banking data"""
        # Add day of week and month
        data['day_of_week'] = pd.to_datetime(data['datetime']).dt.day_name()
        data['month'] = pd.to_datetime(data['datetime']).dt.month
        data['hour'] = pd.to_datetime(data['datetime']).dt.hour
        
        # Add overdraft indicators
        if 'balance_after' in data.columns:
            data['is_overdraft'] = data['balance_after'] < 0
            data['overdraft_amount'] = data['balance_after'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Add transaction categories for analysis
        data['is_debit'] = data['transaction_type'].isin(['debit_card', 'ach_debit', 'atm_withdrawal', 'check'])
        data['is_credit'] = ~data['is_debit']
        
        # Add risk categories
        data['risk_category'] = pd.cut(
            data['fraud_score'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['low', 'medium', 'high']
        )
        
        # Add business hours indicator
        data['is_business_hours'] = data['hour'].between(9, 17)
        
        # Add weekend indicator
        data['is_weekend'] = pd.to_datetime(data['datetime']).dt.weekday >= 5
        
        # Sort by datetime for realistic chronological order
        data = data.sort_values('datetime')
        
        return data
    
    def _select_account_for_time_series_transaction(self, accounts: List[Dict], 
                                                  timestamp: datetime, 
                                                  daily_counts: Dict) -> Dict:
        """Select account for time series transaction with temporal weighting"""
        # Weight accounts by their transaction frequency and time patterns
        weights = []
        frequency_weights = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 5
        }
        
        date_key = timestamp.date()
        hour = timestamp.hour
        
        for account in accounts:
            base_weight = frequency_weights.get(account['transaction_frequency'], 1)
            
            # Apply time-based weighting based on customer segment
            segment = account['customer_segment']
            time_multiplier = self._get_time_multiplier_for_segment(segment, hour)
            
            # Reduce weight if account has had many transactions today
            daily_penalty = 1.0
            if date_key in daily_counts and account['account_id'] in daily_counts[date_key]:
                daily_count = daily_counts[date_key][account['account_id']]
                if daily_count > 5:  # More than 5 transactions today
                    daily_penalty = 0.3
                elif daily_count > 2:
                    daily_penalty = 0.7
            
            final_weight = base_weight * time_multiplier * daily_penalty
            weights.append(final_weight)
        
        # Normalize weights and select
        total_weight = sum(weights)
        if total_weight == 0:
            return self.faker.random_element(accounts)
        
        probabilities = [w / total_weight for w in weights]
        
        rand_val = self.faker.random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return accounts[i]
        
        return accounts[-1]  # Fallback
    
    def _get_time_multiplier_for_segment(self, segment: str, hour: int) -> float:
        """Get time-based multiplier for customer segment"""
        # Different segments have different activity patterns
        if segment == 'young_professional':
            # More active during lunch and evening
            if hour in [12, 13]:
                return 2.0
            elif hour in [18, 19, 20]:
                return 1.8
            elif hour in [8, 9]:
                return 1.2
            elif hour in [2, 3, 4, 5]:
                return 0.1
            else:
                return 1.0
        elif segment == 'family':
            # More active during morning and evening
            if hour in [7, 8, 9]:
                return 1.8
            elif hour in [17, 18, 19]:
                return 2.0
            elif hour in [10, 11, 14, 15]:
                return 1.3
            elif hour in [0, 1, 2, 3, 4, 5]:
                return 0.1
            else:
                return 1.0
        elif segment == 'senior':
            # More active during daytime
            if hour in [9, 10, 11, 14, 15]:
                return 2.0
            elif hour in [12, 13, 16]:
                return 1.5
            elif hour in [0, 1, 2, 3, 4, 5, 22, 23]:
                return 0.1
            else:
                return 1.0
        else:  # business_owner
            # Business hours
            if hour in [9, 10, 11, 14, 15, 16]:
                return 2.0
            elif hour in [8, 17]:
                return 1.5
            elif hour in [0, 1, 2, 3, 4, 5]:
                return 0.2
            else:
                return 1.0
    
    def _generate_time_series_transaction(self, account: Dict, transaction_index: int,
                                        timestamp: datetime, account_balances: Dict,
                                        fraud_simulation: bool, daily_counts: Dict) -> Dict:
        """Generate time series transaction with temporal patterns"""
        
        # Select transaction type based on time and account preferences
        transaction_type = self._select_time_aware_transaction_type(account, timestamp)
        
        # Generate amount with temporal patterns
        amount = self._generate_time_aware_transaction_amount(
            transaction_type, account, timestamp, daily_counts
        )
        
        # Generate merchant/description
        merchant_category = self._select_merchant_category(transaction_type)
        description = self._generate_transaction_description(transaction_type, merchant_category)
        
        # Generate location
        location = self._generate_transaction_location(account)
        
        # Calculate fraud score with temporal factors
        fraud_score = 0.0
        is_fraudulent = False
        
        if fraud_simulation:
            fraud_score = self._calculate_time_aware_fraud_score(
                account, transaction_type, amount, timestamp, location, daily_counts
            )
            is_fraudulent = fraud_score > 0.7
        
        # Generate transaction ID
        transaction_id = f'TXN_{transaction_index+1:012d}'
        
        return {
            'transaction_id': transaction_id,
            'account_id': account['account_id'],
            'account_type': account['account_type'],
            'customer_segment': account['customer_segment'],
            'date': timestamp.date(),
            'datetime': timestamp,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'description': description,
            'location_city': location['city'],
            'location_state': location['state'],
            'location_country': location['country'],
            'fraud_score': round(fraud_score, 3),
            'is_fraudulent': is_fraudulent,
            'processing_time': self._generate_processing_time(transaction_type)
        }
    
    def _select_time_aware_transaction_type(self, account: Dict, timestamp: datetime) -> str:
        """Select transaction type based on time patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Time-based transaction type preferences
        if hour in [0, 1, 2]:  # Early morning - mostly automated
            preferred_types = ['ach_debit', 'ach_credit']
        elif hour in [7, 8, 9]:  # Morning - commute and coffee
            preferred_types = ['debit_card', 'atm_withdrawal']
        elif hour in [12, 13]:  # Lunch time
            preferred_types = ['debit_card', 'credit_card']
        elif hour in [17, 18, 19]:  # Evening - shopping and dining
            preferred_types = ['debit_card', 'credit_card']
        elif hour in [20, 21, 22]:  # Evening - online shopping
            preferred_types = ['credit_card', 'debit_card']
        else:
            preferred_types = list(self.transaction_types.keys())
        
        # Weekend patterns
        if day_of_week >= 5:  # Weekend
            if hour in [10, 11, 12, 13, 14, 15]:  # Weekend shopping
                preferred_types = ['debit_card', 'credit_card']
        
        # Monthly patterns (salary, bills)
        if timestamp.day <= 3:  # Beginning of month - salary and bill payments
            preferred_types.extend(['ach_credit', 'ach_debit'])
        
        # Filter by account preferences
        account_preferred = account['preferred_transactions']
        available_preferred = [t for t in preferred_types if t in account_preferred]
        
        if available_preferred and self.faker.random.random() < 0.8:
            return self.faker.random_element(available_preferred)
        
        # Fallback to general selection
        return self._select_transaction_type(account)
    
    def _generate_time_aware_transaction_amount(self, transaction_type: str, account: Dict,
                                              timestamp: datetime, daily_counts: Dict) -> float:
        """Generate transaction amount with temporal patterns"""
        base_amount = self._generate_transaction_amount(transaction_type, account, timestamp)
        
        # Apply time-based adjustments
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        date_key = timestamp.date()
        
        # Lunch time transactions tend to be smaller
        if hour in [12, 13] and transaction_type in ['debit_card', 'credit_card']:
            base_amount *= self.faker.random.uniform(0.3, 0.8)
        
        # Evening transactions tend to be larger (dinner, entertainment)
        elif hour in [18, 19, 20] and transaction_type in ['debit_card', 'credit_card']:
            base_amount *= self.faker.random.uniform(1.2, 2.0)
        
        # Weekend transactions tend to be larger
        if day_of_week >= 5:
            base_amount *= self.faker.random.uniform(1.1, 1.5)
        
        # Reduce amount if many transactions today (spending fatigue)
        if date_key in daily_counts and account['account_id'] in daily_counts[date_key]:
            daily_count = daily_counts[date_key][account['account_id']]
            if daily_count > 5:
                base_amount *= self.faker.random.uniform(0.5, 0.8)
            elif daily_count > 2:
                base_amount *= self.faker.random.uniform(0.8, 1.0)
        
        return max(base_amount, 1.0)
    
    def _calculate_time_aware_fraud_score(self, account: Dict, transaction_type: str,
                                        amount: float, timestamp: datetime,
                                        location: Dict, daily_counts: Dict) -> float:
        """Calculate fraud score with temporal factors"""
        base_score = self._calculate_fraud_score(account, transaction_type, amount, timestamp, location)
        
        # Add temporal fraud indicators
        date_key = timestamp.date()
        hour = timestamp.hour
        
        # Multiple transactions in short time period
        if date_key in daily_counts and account['account_id'] in daily_counts[date_key]:
            daily_count = daily_counts[date_key][account['account_id']]
            if daily_count > 10:  # More than 10 transactions today
                base_score += 0.4
            elif daily_count > 5:
                base_score += 0.2
        
        # Unusual time patterns for account type
        if account['account_type'] == 'business' and hour in [0, 1, 2, 3, 4, 5]:
            base_score += 0.3
        
        # Large amounts during unusual hours
        if hour in [2, 3, 4] and amount > 1000:
            base_score += 0.5
        
        return min(base_score, 1.0)
    
    def _apply_banking_time_series_correlations(self, data: pd.DataFrame, 
                                              ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations for banking data"""
        if len(data) < 2:
            return data
        
        # Sort by datetime to ensure proper time series order
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Apply temporal correlations for transaction amounts
        for account_id in data['account_id'].unique():
            account_data = data[data['account_id'] == account_id].copy()
            
            if len(account_data) > 1:
                # Apply spending persistence (people tend to spend similarly over time)
                for i in range(1, len(account_data)):
                    current_idx = account_data.index[i]
                    prev_idx = account_data.index[i-1]
                    
                    current_amount = data.loc[current_idx, 'amount']
                    prev_amount = data.loc[prev_idx, 'amount']
                    
                    # Apply correlation (spending patterns persist)
                    correlation_strength = 0.3
                    time_diff = (data.loc[current_idx, 'datetime'] - 
                               data.loc[prev_idx, 'datetime']).total_seconds() / 3600
                    
                    # Stronger correlation for transactions closer in time
                    if time_diff < 24:  # Same day
                        correlation_strength = 0.5
                    elif time_diff < 168:  # Same week
                        correlation_strength = 0.4
                    
                    # Adjust current amount based on previous amount
                    if prev_amount > 0:
                        adjustment_factor = 1 + (correlation_strength * 
                                               self.faker.random.uniform(-0.3, 0.3))
                        new_amount = current_amount * adjustment_factor
                        
                        # Ensure reasonable bounds
                        transaction_type = data.loc[current_idx, 'transaction_type']
                        type_info = self.transaction_types[transaction_type]
                        min_amount, max_amount = type_info['amount_range']
                        
                        new_amount = max(min_amount, min(max_amount * 2, new_amount))
                        data.loc[current_idx, 'amount'] = round(new_amount, 2)
        
        return data.reset_index(drop=True)
        
        return data
    
    def _select_account_for_time_series_transaction(self, accounts: List[Dict], 
                                                  timestamp: datetime, 
                                                  daily_counts: Dict) -> Dict:
        """Select account for time series transaction with temporal weighting"""
        # Weight accounts by their transaction frequency and recent activity
        weights = []
        frequency_weights = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'very_high': 5
        }
        
        date_key = timestamp.date()
        
        for account in accounts:
            base_weight = frequency_weights.get(account['transaction_frequency'], 1)
            
            # Reduce weight if account has had many transactions today
            daily_count = daily_counts.get(date_key, {}).get(account['account_id'], 0)
            if daily_count > 10:  # Limit daily transactions per account
                base_weight *= 0.3
            elif daily_count > 5:
                base_weight *= 0.7
            
            # Adjust weight based on time of day and customer segment
            hour = timestamp.hour
            segment = account['customer_segment']
            
            # Apply time-based preferences
            if segment == 'young_professional' and hour in [12, 13, 18, 19, 20]:
                base_weight *= 1.5
            elif segment == 'family' and hour in [8, 9, 17, 18, 19]:
                base_weight *= 1.5
            elif segment == 'senior' and hour in [9, 10, 11, 14, 15]:
                base_weight *= 1.5
            elif segment == 'business_owner' and hour in [9, 10, 11, 14, 15, 16]:
                base_weight *= 1.5
            
            weights.append(base_weight)
        
        return self._select_weighted_choice(accounts, weights)
    
    def _generate_time_series_transaction(self, account: Dict, transaction_index: int,
                                        timestamp: datetime, account_balances: Dict,
                                        fraud_simulation: bool, daily_counts: Dict) -> Dict:
        """Generate time series transaction with temporal patterns"""
        
        # Select transaction type with time-based preferences
        transaction_type = self._select_time_series_transaction_type(account, timestamp)
        
        # Generate amount with temporal patterns
        amount = self._generate_time_series_amount(transaction_type, account, timestamp)
        
        # Generate merchant/description
        merchant_category = self._select_merchant_category(transaction_type)
        description = self._generate_transaction_description(transaction_type, merchant_category)
        
        # Generate location
        location = self._generate_transaction_location(account)
        
        # Calculate fraud score with velocity checks
        fraud_score = 0.0
        is_fraudulent = False
        
        if fraud_simulation:
            fraud_score = self._calculate_time_series_fraud_score(
                account, transaction_type, amount, timestamp, location, daily_counts
            )
            is_fraudulent = fraud_score > 0.7
        
        # Generate transaction ID
        transaction_id = f'TXN_{transaction_index+1:012d}'
        
        return {
            'transaction_id': transaction_id,
            'account_id': account['account_id'],
            'account_type': account['account_type'],
            'customer_segment': account['customer_segment'],
            'date': timestamp.date(),
            'datetime': timestamp,
            'transaction_type': transaction_type,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'description': description,
            'location_city': location['city'],
            'location_state': location['state'],
            'location_country': location['country'],
            'fraud_score': round(fraud_score, 3),
            'is_fraudulent': is_fraudulent,
            'processing_time': self._generate_processing_time(transaction_type)
        }
    
    def _select_time_series_transaction_type(self, account: Dict, timestamp: datetime) -> str:
        """Select transaction type with time-based patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Adjust transaction type probabilities based on time
        adjusted_probs = {}
        
        for trans_type, type_info in self.transaction_types.items():
            base_prob = type_info['probability']
            
            # Time-based adjustments
            if trans_type == 'salary' and day_of_week == 4 and hour < 3:  # Friday early morning
                adjusted_probs[trans_type] = base_prob * 5  # Payroll processing
            elif trans_type == 'atm_withdrawal' and hour in [17, 18, 19, 20]:
                adjusted_probs[trans_type] = base_prob * 2  # Evening cash needs
            elif trans_type in ['debit_card', 'credit_card'] and hour in [12, 13, 18, 19, 20]:
                adjusted_probs[trans_type] = base_prob * 1.5  # Meal times and evening
            elif trans_type == 'utility' and timestamp.day <= 5:  # Early month
                adjusted_probs[trans_type] = base_prob * 3  # Bill payments
            else:
                adjusted_probs[trans_type] = base_prob
        
        # Apply account preferences
        preferred_types = account['preferred_transactions']
        if self.faker.random.random() < 0.6 and preferred_types:
            available_preferred = [t for t in preferred_types if t in adjusted_probs]
            if available_preferred:
                # Boost preferred transaction probabilities
                for pref_type in available_preferred:
                    adjusted_probs[pref_type] *= 2
        
        # Select based on adjusted probabilities
        types = list(adjusted_probs.keys())
        probabilities = list(adjusted_probs.values())
        
        return self._select_weighted_choice(types, probabilities)
    
    def _generate_time_series_amount(self, transaction_type: str, account: Dict, 
                                   timestamp: datetime) -> float:
        """Generate transaction amount with time series patterns"""
        type_info = self.transaction_types[transaction_type]
        min_amount, max_amount = type_info['amount_range']
        
        # Base amount generation
        base_amount = self.faker.random.uniform(min_amount, max_amount)
        
        # Apply spending pattern multiplier
        spending_multipliers = {
            'low': 0.6,
            'moderate': 1.0,
            'high': 1.4,
            'very_high': 2.0
        }
        multiplier = spending_multipliers.get(account['spending_pattern'], 1.0)
        
        # Apply temporal patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Time-based amount adjustments
        time_multiplier = 1.0
        
        # Higher amounts during peak hours
        if hour in [12, 13, 18, 19, 20]:  # Meal times and evening
            time_multiplier *= 1.2
        
        # Weekend spending patterns
        if day_of_week >= 5:  # Weekend
            if transaction_type in ['restaurant', 'retail', 'entertainment']:
                time_multiplier *= 1.3
        
        # Monthly patterns
        if month in [11, 12]:  # Holiday season
            time_multiplier *= 1.4
        elif month == 1:  # Post-holiday
            time_multiplier *= 0.7
        
        # Salary transactions have specific patterns
        if transaction_type == 'salary':
            # Bi-weekly pattern with some variation
            if timestamp.day in [1, 2, 15, 16]:  # Typical payroll dates
                time_multiplier *= 1.0
            else:
                time_multiplier *= 0.3  # Less likely on other days
        
        # Calculate final amount
        final_amount = base_amount * multiplier * time_multiplier
        
        # Add some randomness
        variance = self.faker.random.uniform(0.85, 1.15)
        final_amount *= variance
        
        # Ensure minimum amount
        final_amount = max(final_amount, 1.0)
        
        return final_amount
    
    def _calculate_time_series_fraud_score(self, account: Dict, transaction_type: str,
                                         amount: float, timestamp: datetime,
                                         location: Dict, daily_counts: Dict) -> float:
        """Calculate fraud score with time series velocity checks"""
        # Start with base fraud calculation
        score = self._calculate_fraud_score(account, transaction_type, amount, timestamp, location)
        
        # Add velocity-based fraud indicators
        date_key = timestamp.date()
        account_id = account['account_id']
        
        # Check daily transaction velocity
        daily_count = daily_counts.get(date_key, {}).get(account_id, 0)
        
        # High transaction frequency is suspicious
        if daily_count > 20:
            score += 0.4
        elif daily_count > 10:
            score += 0.2
        
        # Check for unusual patterns in time series
        hour = timestamp.hour
        
        # Multiple transactions in suspicious hours
        if hour in [2, 3, 4, 5] and daily_count > 2:
            score += 0.3
        
        # Business account activity on weekends
        if (account['account_type'] == 'business' and 
            timestamp.weekday() >= 5 and 
            transaction_type not in ['atm_withdrawal']):
            score += 0.2
        
        # Large amounts during off-hours
        if hour in [22, 23, 0, 1, 2, 3, 4, 5] and amount > 1000:
            score += 0.3
        
        return min(score, 1.0)
    
    def _apply_banking_time_series_correlations(self, data: pd.DataFrame, 
                                              ts_config) -> pd.DataFrame:
        """Apply banking-specific time series correlations"""
        if len(data) < 2:
            return data
        
        # Sort by datetime to ensure proper time series order
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Apply account balance persistence and realistic patterns
        if 'balance_after' in data.columns:
            # Group by account to apply account-specific patterns
            for account_id in data['account_id'].unique():
                account_mask = data['account_id'] == account_id
                account_data = data[account_mask].copy()
                
                if len(account_data) > 1:
                    # Apply balance smoothing (gradual changes are more realistic)
                    for i in range(1, len(account_data)):
                        prev_idx = account_data.index[i-1]
                        curr_idx = account_data.index[i]
                        
                        prev_balance = data.loc[prev_idx, 'balance_after']
                        curr_balance = data.loc[curr_idx, 'balance_after']
                        
                        # If balance change is too dramatic, smooth it
                        balance_change = abs(curr_balance - prev_balance)
                        if balance_change > prev_balance * 0.5:  # 50% change is suspicious
                            # Reduce the change magnitude
                            smoothed_change = balance_change * 0.7
                            if curr_balance > prev_balance:
                                data.loc[curr_idx, 'balance_after'] = prev_balance + smoothed_change
                            else:
                                data.loc[curr_idx, 'balance_after'] = prev_balance - smoothed_change
        
        # Apply fraud score temporal correlation
        # Fraud scores should have some persistence (fraudulent periods)
        for i in range(1, len(data)):
            if data.loc[i-1, 'fraud_score'] > 0.5:  # Previous transaction was suspicious
                # Increase current fraud score slightly (fraud often comes in clusters)
                current_score = data.loc[i, 'fraud_score']
                data.loc[i, 'fraud_score'] = min(1.0, current_score + 0.1)
        
        # Apply transaction amount correlation within accounts
        # Similar transaction types should have correlated amounts
        for account_id in data['account_id'].unique():
            account_mask = data['account_id'] == account_id
            account_data = data[account_mask]
            
            if len(account_data) > 1:
                # Group by transaction type within account
                for trans_type in account_data['transaction_type'].unique():
                    type_mask = (data['account_id'] == account_id) & (data['transaction_type'] == trans_type)
                    type_data = data[type_mask]
                    
                    if len(type_data) > 1:
                        # Apply amount correlation (similar amounts for same transaction type)
                        for i in range(1, len(type_data)):
                            curr_idx = type_data.index[i]
                            prev_idx = type_data.index[i-1]
                            
                            prev_amount = data.loc[prev_idx, 'amount']
                            curr_amount = data.loc[curr_idx, 'amount']
                            
                            # Apply correlation (70% correlation with previous similar transaction)
                            correlation_strength = 0.3
                            correlated_amount = (prev_amount * correlation_strength + 
                                               curr_amount * (1 - correlation_strength))
                            
                            data.loc[curr_idx, 'amount'] = round(correlated_amount, 2)
        
        return data