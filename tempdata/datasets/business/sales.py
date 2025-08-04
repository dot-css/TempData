"""
Sales transaction dataset generator

Generates realistic sales data with seasonal trends and regional preferences.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from ...core.base_generator import BaseGenerator


class SalesGenerator(BaseGenerator):
    """
    Generator for realistic sales transaction data
    
    Creates sales datasets with seasonal trends, regional preferences,
    payment method distributions, and realistic amount patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_product_data()
        self._setup_regional_preferences()
        self._setup_seasonal_patterns()
        self._setup_payment_distributions()
    
    def _load_product_data(self):
        """Load product data for realistic product selection"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '../../data/business/products.json')
            with open(data_path, 'r') as f:
                product_data = json.load(f)
            
            self.products = []
            self.product_categories = product_data.get('product_categories', [])
            
            # Flatten all products from different categories
            for category, items in product_data.get('products', {}).items():
                for item in items:
                    self.products.append({
                        'name': item,
                        'category': category,
                        'base_price': self._get_base_price_for_category(category)
                    })
        except FileNotFoundError:
            # Fallback if data file not found
            self.products = [
                {'name': f'Product_{i}', 'category': 'general', 'base_price': 50.0}
                for i in range(100)
            ]
            self.product_categories = ['general']
    
    def _get_base_price_for_category(self, category: str) -> float:
        """Get base price range for product category"""
        price_ranges = {
            'technology': (200, 2000),
            'healthcare': (50, 500),
            'retail': (20, 200),
            'automotive': (500, 50000),
            'food_beverage': (5, 50),
            'home_garden': (100, 1000)
        }
        min_price, max_price = price_ranges.get(category, (20, 200))
        return self.faker.random.uniform(min_price, max_price)
    
    def _setup_regional_preferences(self):
        """Setup regional preferences for products and payment methods"""
        self.regional_data = {
            'North America': {
                'payment_methods': {'card': 0.6, 'digital': 0.3, 'cash': 0.1},
                'preferred_categories': ['technology', 'automotive', 'home_garden'],
                'price_multiplier': 1.2
            },
            'Europe': {
                'payment_methods': {'card': 0.5, 'digital': 0.4, 'cash': 0.1},
                'preferred_categories': ['retail', 'food_beverage', 'healthcare'],
                'price_multiplier': 1.1
            },
            'Asia Pacific': {
                'payment_methods': {'digital': 0.6, 'card': 0.3, 'cash': 0.1},
                'preferred_categories': ['technology', 'retail', 'food_beverage'],
                'price_multiplier': 0.9
            },
            'Latin America': {
                'payment_methods': {'cash': 0.4, 'card': 0.4, 'digital': 0.2},
                'preferred_categories': ['retail', 'food_beverage', 'home_garden'],
                'price_multiplier': 0.8
            },
            'Middle East': {
                'payment_methods': {'card': 0.5, 'cash': 0.3, 'digital': 0.2},
                'preferred_categories': ['automotive', 'technology', 'retail'],
                'price_multiplier': 1.0
            }
        }
    
    def _setup_seasonal_patterns(self):
        """Setup seasonal sales patterns"""
        # Monthly sales multipliers (1.0 = average)
        self.seasonal_multipliers = {
            1: 0.8,   # January - post-holiday lull
            2: 0.9,   # February - Valentine's Day boost
            3: 1.0,   # March - normal
            4: 1.1,   # April - spring shopping
            5: 1.2,   # May - Mother's Day
            6: 1.1,   # June - graduation season
            7: 1.0,   # July - summer
            8: 0.9,   # August - back-to-school prep
            9: 1.0,   # September - back-to-school
            10: 1.1,  # October - Halloween
            11: 1.4,  # November - Black Friday
            12: 1.6   # December - holiday season
        }
        
        # Daily patterns (hour of day effects)
        self.daily_patterns = {
            0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1,
            6: 0.2, 7: 0.4, 8: 0.6, 9: 0.8, 10: 1.0, 11: 1.2,
            12: 1.3, 13: 1.2, 14: 1.1, 15: 1.0, 16: 1.1, 17: 1.2,
            18: 1.3, 19: 1.2, 20: 1.0, 21: 0.8, 22: 0.5, 23: 0.2
        }
        
        # Day of week patterns (0=Monday, 6=Sunday)
        self.weekly_patterns = {
            0: 0.9,  # Monday
            1: 1.0,  # Tuesday
            2: 1.0,  # Wednesday
            3: 1.1,  # Thursday
            4: 1.3,  # Friday
            5: 1.4,  # Saturday - peak shopping
            6: 1.2   # Sunday
        }
    
    def _setup_payment_distributions(self):
        """Setup payment method distributions"""
        self.payment_methods = {
            'credit_card': 0.45,
            'debit_card': 0.30,
            'digital_wallet': 0.15,
            'cash': 0.08,
            'check': 0.02
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate sales transaction dataset
        
        Args:
            rows: Number of sales transactions to generate
            **kwargs: Additional parameters (time_series, date_range, region, etc.)
            
        Returns:
            pd.DataFrame: Generated sales data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_sales(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_sales(rows, **kwargs)
    
    def _generate_snapshot_sales(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot sales data (random timestamps)"""
        region = kwargs.get('region', 'North America')
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Generate random timestamp
            if date_range:
                start_date, end_date = date_range
                transaction_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                transaction_date = self.faker.date_this_year()
            
            # Add time component
            hour = self._select_sales_hour()
            minute = self.faker.random_int(0, 59)
            transaction_datetime = datetime.combine(transaction_date, datetime.min.time().replace(hour=hour, minute=minute))
            
            # Generate sales transaction
            transaction = self._generate_sales_transaction(i, transaction_datetime, region)
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_sales(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series sales data using integrated time series system"""
        region = kwargs.get('region', 'North America')
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Get regional data for base sales amount
        regional_data = self.regional_data.get(region, self.regional_data['North America'])
        base_amount = 150.0  # Base transaction amount
        
        # Create base time series for sales amounts
        amount_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_amount,
            value_range=(10.0, base_amount * 5.0)  # Allow variation
        )
        
        data = []
        
        # Generate data for each timestamp with realistic patterns
        for i, timestamp in enumerate(timestamps):
            if i >= len(amount_series) or i >= rows:
                break
                
            # Get time series amount value
            base_transaction_amount = amount_series.iloc[i]['value']
            
            # Apply sales-specific temporal patterns
            adjusted_amount = self._apply_sales_temporal_patterns(
                base_transaction_amount, timestamp, region
            )
            
            # Generate sales transaction with temporal patterns
            transaction = self._generate_time_series_sales_transaction(
                i, timestamp, adjusted_amount, region
            )
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply sales-specific time series correlations
        df = self._apply_sales_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _select_sales_hour(self) -> int:
        """Select sales hour based on daily patterns"""
        hours = list(self.daily_patterns.keys())
        weights = list(self.daily_patterns.values())
        
        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Select hour
        rand_val = self.faker.random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return hours[i]
        
        return 12  # Fallback to noon
    
    def _generate_sales_transaction(self, transaction_index: int, 
                                  transaction_datetime: datetime, region: str) -> Dict:
        """Generate a single sales transaction"""
        
        # Select product
        product = self.faker.random_element(self.products)
        
        # Generate amount based on product and seasonal patterns
        base_amount = product['base_price']
        
        # Apply seasonal multiplier
        seasonal_mult = self.seasonal_multipliers.get(transaction_datetime.month, 1.0)
        
        # Apply daily pattern
        daily_mult = self.daily_patterns.get(transaction_datetime.hour, 1.0)
        
        # Apply weekly pattern
        weekly_mult = self.weekly_patterns.get(transaction_datetime.weekday(), 1.0)
        
        # Apply regional multiplier
        regional_data = self.regional_data.get(region, self.regional_data['North America'])
        regional_mult = regional_data['price_multiplier']
        
        # Calculate final amount
        final_amount = base_amount * seasonal_mult * daily_mult * weekly_mult * regional_mult
        
        # Add random variation
        final_amount *= self.faker.random.uniform(0.8, 1.3)
        
        # Select payment method based on region
        payment_method = self._select_payment_method(regional_data)
        
        # Generate customer info
        customer_id = f'CUST_{self.faker.random_int(1, 100000):06d}'
        
        # Generate location
        location = self._generate_sales_location(region)
        
        return {
            'transaction_id': f'TXN_{transaction_index+1:010d}',
            'customer_id': customer_id,
            'date': transaction_datetime.date(),
            'datetime': transaction_datetime,
            'product_name': product['name'],
            'product_category': product['category'],
            'amount': round(final_amount, 2),
            'quantity': self.faker.random_int(1, 5),
            'payment_method': payment_method,
            'region': region,
            'store_location': location['store'],
            'city': location['city'],
            'state': location['state'],
            'sales_rep': self.faker.name(),
            'discount_applied': self.faker.random.random() < 0.3,  # 30% chance of discount
            'discount_amount': round(final_amount * 0.1, 2) if self.faker.random.random() < 0.3 else 0.0
        }
    
    def _generate_time_series_sales_transaction(self, transaction_index: int,
                                              timestamp: datetime, base_amount: float,
                                              region: str) -> Dict:
        """Generate time series sales transaction with temporal patterns"""
        
        # Select product with time-aware preferences
        product = self._select_time_aware_product(timestamp)
        
        # Use the time series base amount but adjust for product type
        product_multiplier = product['base_price'] / 150.0  # Normalize to base
        adjusted_amount = base_amount * product_multiplier
        
        # Apply additional temporal patterns
        adjusted_amount = self._apply_sales_temporal_patterns(adjusted_amount, timestamp, region)
        
        # Select payment method based on region and time
        regional_data = self.regional_data.get(region, self.regional_data['North America'])
        payment_method = self._select_time_aware_payment_method(regional_data, timestamp)
        
        # Generate customer info
        customer_id = f'CUST_{self.faker.random_int(1, 100000):06d}'
        
        # Generate location
        location = self._generate_sales_location(region)
        
        # Calculate discount based on time patterns
        discount_applied, discount_amount = self._calculate_time_aware_discount(adjusted_amount, timestamp)
        
        return {
            'transaction_id': f'TXN_{transaction_index+1:010d}',
            'customer_id': customer_id,
            'date': timestamp.date(),
            'datetime': timestamp,
            'product_name': product['name'],
            'product_category': product['category'],
            'amount': round(adjusted_amount, 2),
            'quantity': self._select_time_aware_quantity(timestamp),
            'payment_method': payment_method,
            'region': region,
            'store_location': location['store'],
            'city': location['city'],
            'state': location['state'],
            'sales_rep': self.faker.name(),
            'discount_applied': discount_applied,
            'discount_amount': discount_amount
        }
    
    def _select_payment_method(self, regional_data: Dict) -> str:
        """Select payment method based on regional preferences"""
        regional_payments = regional_data.get('payment_methods', self.payment_methods)
        
        methods = list(regional_payments.keys())
        probabilities = list(regional_payments.values())
        
        rand_val = self.faker.random.random()
        cumulative_prob = 0
        
        for method, prob in zip(methods, probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return method
        
        return 'credit_card'  # Fallback
    
    def _generate_sales_location(self, region: str) -> Dict:
        """Generate sales location based on region"""
        if region == 'North America':
            city = self.faker.city()
            state = self.faker.state()
        elif region == 'Europe':
            city = self.faker.city()
            state = self.faker.country()
        else:
            city = self.faker.city()
            state = self.faker.state()
        
        store_types = ['Mall Store', 'Standalone', 'Outlet', 'Department Store', 'Online']
        store = self.faker.random_element(store_types)
        
        return {
            'store': store,
            'city': city,
            'state': state
        }
    
    def _apply_sales_temporal_patterns(self, base_amount: float, timestamp: datetime, region: str) -> float:
        """Apply sales-specific temporal patterns to base amount"""
        
        # Apply seasonal multiplier
        seasonal_mult = self.seasonal_multipliers.get(timestamp.month, 1.0)
        
        # Apply daily pattern
        daily_mult = self.daily_patterns.get(timestamp.hour, 1.0)
        
        # Apply weekly pattern
        weekly_mult = self.weekly_patterns.get(timestamp.weekday(), 1.0)
        
        # Apply regional multiplier
        regional_data = self.regional_data.get(region, self.regional_data['North America'])
        regional_mult = regional_data['price_multiplier']
        
        # Calculate adjusted amount
        adjusted_amount = base_amount * seasonal_mult * daily_mult * weekly_mult * regional_mult
        
        # Add some random variation
        adjusted_amount *= self.faker.random.uniform(0.9, 1.1)
        
        return max(adjusted_amount, 5.0)  # Minimum transaction amount
    
    def _select_time_aware_product(self, timestamp: datetime) -> Dict:
        """Select product with time-aware preferences"""
        hour = timestamp.hour
        month = timestamp.month
        
        # Morning preferences (coffee, breakfast items)
        if hour in [6, 7, 8, 9]:
            preferred_categories = ['food_beverage']
        # Lunch preferences
        elif hour in [11, 12, 13, 14]:
            preferred_categories = ['food_beverage', 'retail']
        # Evening preferences (electronics, home items)
        elif hour in [17, 18, 19, 20]:
            preferred_categories = ['technology', 'home_garden', 'retail']
        # Holiday season preferences
        elif month in [11, 12]:
            preferred_categories = ['technology', 'retail', 'home_garden']
        else:
            preferred_categories = list(set(p['category'] for p in self.products))
        
        # Filter products by preferred categories
        preferred_products = [p for p in self.products if p['category'] in preferred_categories]
        
        if preferred_products:
            return self.faker.random_element(preferred_products)
        else:
            return self.faker.random_element(self.products)
    
    def _select_time_aware_payment_method(self, regional_data: Dict, timestamp: datetime) -> str:
        """Select payment method with time-aware preferences"""
        hour = timestamp.hour
        
        # Early morning and late night - more digital payments
        if hour in [0, 1, 2, 3, 4, 5, 22, 23]:
            digital_boost = {'digital_wallet': 0.4, 'credit_card': 0.4, 'debit_card': 0.2}
            return self._select_payment_method({'payment_methods': digital_boost})
        
        # Business hours - normal distribution
        return self._select_payment_method(regional_data)
    
    def _select_time_aware_quantity(self, timestamp: datetime) -> int:
        """Select quantity with time-aware patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Weekend shopping - larger quantities
        if day_of_week >= 5:
            return self.faker.random_int(1, 8)
        # Lunch time - smaller quantities
        elif hour in [12, 13]:
            return self.faker.random_int(1, 3)
        # Evening shopping - moderate quantities
        elif hour in [17, 18, 19]:
            return self.faker.random_int(1, 6)
        else:
            return self.faker.random_int(1, 5)
    
    def _calculate_time_aware_discount(self, amount: float, timestamp: datetime) -> tuple:
        """Calculate discount based on time patterns"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        discount_probability = 0.3  # Base probability
        
        # End of day discounts
        if hour >= 20:
            discount_probability = 0.5
        # Weekend discounts
        elif day_of_week >= 5:
            discount_probability = 0.4
        # Holiday season discounts
        elif month in [11, 12]:
            discount_probability = 0.6
        
        if self.faker.random.random() < discount_probability:
            discount_rate = self.faker.random.uniform(0.05, 0.25)  # 5-25% discount
            discount_amount = round(amount * discount_rate, 2)
            return True, discount_amount
        
        return False, 0.0
    
    def _apply_sales_time_series_correlations(self, data: pd.DataFrame, 
                                            ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations for sales data"""
        if len(data) < 2:
            return data
        
        # Sort by datetime to ensure proper time series order
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Apply temporal correlations for sales amounts
        for i in range(1, len(data)):
            prev_amount = data.iloc[i-1]['amount']
            current_amount = data.iloc[i]['amount']
            
            # Sales persistence (similar amounts tend to cluster)
            correlation_strength = 0.2
            time_diff = (data.iloc[i]['datetime'] - data.iloc[i-1]['datetime']).total_seconds() / 3600
            
            # Stronger correlation for transactions closer in time
            if time_diff < 1:  # Same hour
                correlation_strength = 0.4
            elif time_diff < 24:  # Same day
                correlation_strength = 0.3
            
            # Adjust current amount based on previous amount
            if prev_amount > 0:
                adjustment_factor = 1 + (correlation_strength * 
                                       self.faker.random.uniform(-0.2, 0.2))
                new_amount = current_amount * adjustment_factor
                
                # Ensure reasonable bounds
                new_amount = max(5.0, min(new_amount, current_amount * 2.0))
                data.loc[i, 'amount'] = round(new_amount, 2)
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to sales data"""
        # Add derived fields
        data['day_of_week'] = pd.to_datetime(data['datetime']).dt.day_name()
        data['month'] = pd.to_datetime(data['datetime']).dt.month
        data['hour'] = pd.to_datetime(data['datetime']).dt.hour
        data['is_weekend'] = pd.to_datetime(data['datetime']).dt.weekday >= 5
        
        # Calculate final amount after discount
        data['final_amount'] = data['amount'] - data['discount_amount']
        
        # Add sales performance metrics
        data['revenue'] = data['final_amount'] * data['quantity']
        
        # Add customer segment based on amount
        data['customer_segment'] = pd.cut(
            data['final_amount'],
            bins=[0, 50, 150, 500, float('inf')],
            labels=['budget', 'standard', 'premium', 'luxury']
        )
        
        # Add seasonal indicator
        data['season'] = data['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Sort by datetime for realistic chronological order
        data = data.sort_values('datetime').reset_index(drop=True)
        
        return data
        self.seasonal_multipliers = {
            1: 0.8,   # January - post-holiday low
            2: 0.9,   # February - Valentine's boost
            3: 1.0,   # March - normal
            4: 1.1,   # April - spring shopping
            5: 1.2,   # May - Mother's Day
            6: 1.1,   # June - summer start
            7: 1.0,   # July - mid-summer
            8: 0.9,   # August - vacation time
            9: 1.1,   # September - back to school
            10: 1.2,  # October - pre-holiday
            11: 1.5,  # November - Black Friday
            12: 1.8   # December - holiday season
        }
    
    def _setup_payment_distributions(self):
        """Setup payment method distributions by amount"""
        self.payment_by_amount = {
            'small': {'cash': 0.4, 'card': 0.5, 'digital': 0.1},      # < $50
            'medium': {'cash': 0.2, 'card': 0.6, 'digital': 0.2},     # $50-$500
            'large': {'cash': 0.05, 'card': 0.8, 'digital': 0.15},    # > $500
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate sales transaction dataset
        
        Args:
            rows: Number of sales transactions to generate
            **kwargs: Additional parameters (country, date_range, time_series, etc.)
            
        Returns:
            pd.DataFrame: Generated sales data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_sales(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_sales(rows, **kwargs)
    
    def _generate_snapshot_sales(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot sales data (random timestamps)"""
        country = kwargs.get('country', 'global')
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Generate transaction date with seasonal consideration
            if date_range:
                start_date, end_date = date_range
                transaction_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                transaction_date = self.faker.date_this_year()
            
            # Select region based on country or random
            region = self._select_region(country)
            
            # Select product with regional preferences
            product = self._select_product_for_region(region)
            
            # Calculate amount with seasonal and regional adjustments
            base_amount = product['base_price']
            seasonal_mult = self.seasonal_multipliers.get(transaction_date.month, 1.0)
            regional_mult = self.regional_data[region]['price_multiplier']
            
            # Add some randomness to the amount
            amount_variance = self.faker.random.uniform(0.8, 1.3)
            final_amount = round(base_amount * seasonal_mult * regional_mult * amount_variance, 2)
            
            # Select payment method based on amount and region
            payment_method = self._select_payment_method(final_amount, region)
            
            # Generate transaction record
            transaction = {
                'transaction_id': f'TXN_{i+1:08d}',
                'date': transaction_date,
                'customer_id': f'CUST_{self.faker.random_int(1, 50000):06d}',
                'product_id': f'PROD_{hash(product["name"]) % 100000:05d}',
                'product_name': product['name'],
                'product_category': product['category'],
                'amount': final_amount,
                'region': region,
                'payment_method': payment_method,
                'sales_rep_id': f'REP_{self.faker.random_int(1, 500):03d}',
                'store_id': f'STORE_{self.faker.random_int(1, 1000):04d}'
            }
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_sales(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series sales data using integrated time series system"""
        country = kwargs.get('country', 'global')
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Select region based on country
        region = self._select_region(country)
        
        # Generate base sales volume time series
        base_sales_volume = 100  # Base daily sales volume
        
        # Create base time series for sales volume
        volume_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_sales_volume,
            value_range=(base_sales_volume * 0.2, base_sales_volume * 3.0)  # Allow significant variation
        )
        
        data = []
        
        # Track product sales patterns for realistic correlations
        product_sales_history = {}
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(volume_series):
                break
            
            # Get time series volume value (represents sales intensity)
            sales_intensity = volume_series.iloc[i]['value'] / base_sales_volume
            
            # Select product with time-based preferences
            product = self._select_time_series_product(region, timestamp, product_sales_history)
            
            # Calculate amount with time series patterns
            amount = self._calculate_time_series_amount(
                product, timestamp, region, sales_intensity
            )
            
            # Select payment method based on amount, region, and time
            payment_method = self._select_time_series_payment_method(amount, region, timestamp)
            
            # Generate transaction record
            transaction = {
                'transaction_id': f'TXN_{i+1:08d}',
                'date': timestamp.date(),
                'datetime': timestamp,
                'customer_id': f'CUST_{self.faker.random_int(1, 50000):06d}',
                'product_id': f'PROD_{hash(product["name"]) % 100000:05d}',
                'product_name': product['name'],
                'product_category': product['category'],
                'amount': amount,
                'region': region,
                'payment_method': payment_method,
                'sales_rep_id': f'REP_{self.faker.random_int(1, 500):03d}',
                'store_id': f'STORE_{self.faker.random_int(1, 1000):04d}',
                'sales_intensity': round(sales_intensity, 3)
            }
            
            # Track product sales for correlation
            if product['name'] not in product_sales_history:
                product_sales_history[product['name']] = []
            product_sales_history[product['name']].append({
                'timestamp': timestamp,
                'amount': amount,
                'intensity': sales_intensity
            })
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply sales-specific time series correlations
        df = self._apply_sales_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _select_region(self, country: str) -> str:
        """Select region based on country or random selection"""
        # Map countries to regions
        country_to_region = {
            'united_states': 'North America',
            'canada': 'North America',
            'mexico': 'Latin America',
            'united_kingdom': 'Europe',
            'germany': 'Europe',
            'france': 'Europe',
            'japan': 'Asia Pacific',
            'china': 'Asia Pacific',
            'india': 'Asia Pacific',
            'brazil': 'Latin America',
            'argentina': 'Latin America',
            'saudi_arabia': 'Middle East',
            'uae': 'Middle East'
        }
        
        if country == 'global':
            return self.faker.random_element(list(self.regional_data.keys()))
        
        return country_to_region.get(country, self.faker.random_element(list(self.regional_data.keys())))
    
    def _select_product_for_region(self, region: str) -> Dict[str, Any]:
        """Select product based on regional preferences"""
        regional_prefs = self.regional_data[region]['preferred_categories']
        
        # 70% chance to select from preferred categories
        if self.faker.random.random() < 0.7 and regional_prefs:
            preferred_products = [p for p in self.products if p['category'] in regional_prefs]
            if preferred_products:
                return self.faker.random_element(preferred_products)
        
        # Fallback to any product
        return self.faker.random_element(self.products)
    
    def _select_payment_method(self, amount: float, region: str) -> str:
        """Select payment method based on amount and regional preferences"""
        # Determine amount category
        if amount < 50:
            amount_category = 'small'
        elif amount < 500:
            amount_category = 'medium'
        else:
            amount_category = 'large'
        
        # Get base distribution for amount
        amount_dist = self.payment_by_amount[amount_category]
        
        # Adjust with regional preferences
        regional_dist = self.regional_data[region]['payment_methods']
        
        # Combine distributions (60% amount-based, 40% region-based)
        combined_dist = {}
        for method in ['cash', 'card', 'digital']:
            combined_dist[method] = (0.6 * amount_dist.get(method, 0) + 
                                   0.4 * regional_dist.get(method, 0))
        
        # Select based on weighted distribution using seeded random
        choices = list(combined_dist.keys())
        weights = list(combined_dist.values())
        
        # Use faker's random for reproducibility
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        
        return choices[-1]  # Fallback
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Add discount patterns for large amounts
        data['discount_applied'] = data['amount'].apply(
            lambda x: self.faker.random.random() < (0.1 if x > 1000 else 0.05)
        )
        
        # Add return probability (higher for expensive items)
        data['return_probability'] = data['amount'].apply(
            lambda x: min(0.15, 0.02 + (x / 10000))
        )
        
        # Add customer satisfaction score (inversely related to amount for budget-conscious)
        data['satisfaction_score'] = data.apply(
            lambda row: max(1, min(5, 
                self.faker.random.gauss(4.2, 0.8) - 
                (0.0001 * row['amount'] if row['region'] in ['Latin America', 'Asia Pacific'] else 0)
            )), axis=1
        ).round(1)
        
        # Sort by date for realistic chronological order
        data = data.sort_values('date').reset_index(drop=True)
        
        return data
    
    def _select_time_series_product(self, region: str, timestamp: datetime, 
                                  product_sales_history: Dict) -> Dict[str, Any]:
        """Select product with time-based preferences and sales history correlation"""
        # Apply time-based product preferences
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Time-based category preferences
        time_category_prefs = {}
        
        # Business hours favor different products
        if 9 <= hour <= 17:  # Business hours
            time_category_prefs = {'technology': 1.5, 'automotive': 1.3}
        elif 18 <= hour <= 22:  # Evening
            time_category_prefs = {'retail': 1.4, 'food_beverage': 1.6}
        elif hour >= 23 or hour <= 6:  # Late night/early morning
            time_category_prefs = {'food_beverage': 2.0}
        
        # Weekend preferences
        if day_of_week >= 5:  # Weekend
            time_category_prefs.update({'home_garden': 1.5, 'retail': 1.3})
        
        # Seasonal preferences
        if month in [11, 12]:  # Holiday season
            time_category_prefs.update({'technology': 1.8, 'retail': 1.6})
        elif month in [3, 4, 5]:  # Spring
            time_category_prefs.update({'home_garden': 1.4})
        
        # Start with regional preferences
        regional_prefs = self.regional_data[region]['preferred_categories']
        
        # Create weighted product list
        weighted_products = []
        for product in self.products:
            weight = 1.0
            
            # Apply regional preference
            if product['category'] in regional_prefs:
                weight *= 1.5
            
            # Apply time-based preference
            if product['category'] in time_category_prefs:
                weight *= time_category_prefs[product['category']]
            
            # Apply sales history correlation (trending products)
            if product['name'] in product_sales_history:
                recent_sales = [s for s in product_sales_history[product['name']] 
                              if (timestamp - s['timestamp']).total_seconds() < 86400]  # Last 24 hours
                if len(recent_sales) > 2:  # Trending product
                    weight *= 1.3
            
            weighted_products.extend([product] * int(weight * 10))  # Multiply for selection
        
        return self.faker.random_element(weighted_products if weighted_products else self.products)
    
    def _calculate_time_series_amount(self, product: Dict, timestamp: datetime, 
                                    region: str, sales_intensity: float) -> float:
        """Calculate amount with time series patterns"""
        base_amount = product['base_price']
        
        # Apply seasonal multiplier
        seasonal_mult = self.seasonal_multipliers.get(timestamp.month, 1.0)
        
        # Apply regional multiplier
        regional_mult = self.regional_data[region]['price_multiplier']
        
        # Apply sales intensity (from time series)
        intensity_mult = 0.7 + (sales_intensity * 0.6)  # Scale intensity to 0.7-1.3 range
        
        # Apply time-of-day patterns
        hour = timestamp.hour
        time_mult = 1.0
        
        if 12 <= hour <= 14:  # Lunch time - higher amounts
            time_mult = 1.2
        elif 18 <= hour <= 20:  # Evening - higher amounts
            time_mult = 1.3
        elif hour >= 22 or hour <= 6:  # Late night/early morning - lower amounts
            time_mult = 0.8
        
        # Apply day-of-week patterns
        day_mult = 1.0
        if timestamp.weekday() >= 5:  # Weekend
            day_mult = 1.2  # Higher weekend spending
        elif timestamp.weekday() == 4:  # Friday
            day_mult = 1.1  # TGIF spending
        
        # Calculate final amount
        final_amount = (base_amount * seasonal_mult * regional_mult * 
                       intensity_mult * time_mult * day_mult)
        
        # Add randomness
        variance = self.faker.random.uniform(0.8, 1.3)
        final_amount *= variance
        
        return round(max(final_amount, 1.0), 2)
    
    def _select_time_series_payment_method(self, amount: float, region: str, 
                                         timestamp: datetime) -> str:
        """Select payment method with time-based patterns"""
        # Start with base selection logic
        base_method = self._select_payment_method(amount, region)
        
        # Apply time-based adjustments
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Time-based payment preferences
        if hour >= 22 or hour <= 6:  # Late night/early morning
            # More digital payments during off-hours
            if base_method == 'cash' and self.faker.random.random() < 0.4:
                return 'digital'
        elif 9 <= hour <= 17:  # Business hours
            # More card payments during business hours
            if base_method == 'cash' and amount > 20 and self.faker.random.random() < 0.3:
                return 'card'
        
        # Weekend patterns
        if day_of_week >= 5:  # Weekend
            # Slightly more cash usage on weekends
            if base_method == 'card' and amount < 100 and self.faker.random.random() < 0.2:
                return 'cash'
        
        return base_method
    
    def _apply_sales_time_series_correlations(self, data: pd.DataFrame, 
                                            ts_config) -> pd.DataFrame:
        """Apply sales-specific time series correlations"""
        if len(data) < 2:
            return data
        
        # Sort by datetime to ensure proper time series order
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Apply sales amount correlation within product categories
        for category in data['product_category'].unique():
            category_mask = data['product_category'] == category
            category_data = data[category_mask]
            
            if len(category_data) > 1:
                # Apply amount correlation within category (similar products have correlated prices)
                for i in range(1, len(category_data)):
                    curr_idx = category_data.index[i]
                    prev_idx = category_data.index[i-1]
                    
                    prev_amount = data.loc[prev_idx, 'amount']
                    curr_amount = data.loc[curr_idx, 'amount']
                    
                    # Apply correlation (40% correlation with previous similar product)
                    correlation_strength = 0.4
                    correlated_amount = (prev_amount * correlation_strength + 
                                       curr_amount * (1 - correlation_strength))
                    
                    data.loc[curr_idx, 'amount'] = round(correlated_amount, 2)
        
        # Apply sales intensity persistence (sales tend to cluster)
        if 'sales_intensity' in data.columns:
            for i in range(1, len(data)):
                prev_intensity = data.loc[i-1, 'sales_intensity']
                curr_intensity = data.loc[i, 'sales_intensity']
                
                # Apply intensity correlation (sales periods tend to persist)
                intensity_persistence = 0.3
                correlated_intensity = (prev_intensity * intensity_persistence + 
                                      curr_intensity * (1 - intensity_persistence))
                
                data.loc[i, 'sales_intensity'] = round(correlated_intensity, 3)
        
        # Apply regional sales correlation (same region sales are correlated)
        for region in data['region'].unique():
            region_mask = data['region'] == region
            region_data = data[region_mask]
            
            if len(region_data) > 1:
                # Calculate rolling average for regional sales amounts
                region_indices = region_data.index.tolist()
                for i in range(1, len(region_indices)):
                    curr_idx = region_indices[i]
                    prev_idx = region_indices[i-1]
                    
                    # Check if transactions are close in time (within 1 hour)
                    time_diff = (data.loc[curr_idx, 'datetime'] - 
                               data.loc[prev_idx, 'datetime']).total_seconds()
                    
                    if time_diff <= 3600:  # Within 1 hour
                        prev_amount = data.loc[prev_idx, 'amount']
                        curr_amount = data.loc[curr_idx, 'amount']
                        
                        # Apply regional correlation
                        regional_correlation = 0.2
                        correlated_amount = (prev_amount * regional_correlation + 
                                           curr_amount * (1 - regional_correlation))
                        
                        data.loc[curr_idx, 'amount'] = round(correlated_amount, 2)
        
        return data