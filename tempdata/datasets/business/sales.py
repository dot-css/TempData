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
            **kwargs: Additional parameters (country, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated sales data with realistic patterns
        """
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