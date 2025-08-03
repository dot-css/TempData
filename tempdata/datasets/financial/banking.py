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
            **kwargs: Additional parameters (account_tracking, fraud_simulation, etc.)
            
        Returns:
            pd.DataFrame: Generated banking data with realistic patterns
        """
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
        data = data.sort_values('datetime').reset_index(drop=True)
        
        return data