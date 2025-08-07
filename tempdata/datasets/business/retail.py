"""
Retail operations dataset generator

Generates realistic retail transaction data with POS transactions, store locations,
product mix, and payment methods for retail analytics and inventory management systems.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from ...core.base_generator import BaseGenerator


class RetailGenerator(BaseGenerator):
    """
    Generator for realistic retail operations data
    
    Creates retail datasets with POS transactions, store locations, product mix,
    payment methods, and realistic transaction patterns by time of day and season.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_store_data()
        self._setup_product_data()
        self._setup_payment_methods()
        self._setup_transaction_patterns()
        self._setup_geographic_data()
    
    def _setup_store_data(self):
        """Setup store types and characteristics"""
        self.store_types = {
            'supermarket': {
                'store_size_sqft': (15000, 50000),
                'daily_transactions': (800, 2500),
                'avg_items_per_transaction': (8, 25),
                'avg_transaction_value': (25, 150),
                'weight': 0.25
            },
            'convenience': {
                'store_size_sqft': (1500, 5000),
                'daily_transactions': (300, 800),
                'avg_items_per_transaction': (1, 8),
                'avg_transaction_value': (5, 35),
                'weight': 0.20
            },
            'department_store': {
                'store_size_sqft': (25000, 100000),
                'daily_transactions': (500, 1500),
                'avg_items_per_transaction': (1, 5),
                'avg_transaction_value': (40, 300),
                'weight': 0.15
            },
            'electronics': {
                'store_size_sqft': (8000, 25000),
                'daily_transactions': (150, 500),
                'avg_items_per_transaction': (1, 3),
                'avg_transaction_value': (100, 2000),
                'weight': 0.10
            },
            'clothing': {
                'store_size_sqft': (3000, 15000),
                'daily_transactions': (200, 600),
                'avg_items_per_transaction': (1, 5),
                'avg_transaction_value': (30, 200),
                'weight': 0.12
            },
            'pharmacy': {
                'store_size_sqft': (2000, 8000),
                'daily_transactions': (200, 700),
                'avg_items_per_transaction': (2, 8),
                'avg_transaction_value': (15, 80),
                'weight': 0.10
            },
            'home_improvement': {
                'store_size_sqft': (30000, 80000),
                'daily_transactions': (250, 700),
                'avg_items_per_transaction': (3, 15),
                'avg_transaction_value': (50, 500),
                'weight': 0.08
            }
        }    

    def _setup_product_data(self):
        """Setup product categories and characteristics"""
        self.product_categories = {
            'groceries': {
                'produce': ['Organic Apples', 'Bananas', 'Fresh Spinach', 'Roma Tomatoes', 'Sharp Cheddar'],
                'dairy': ['Whole Milk', 'Greek Yogurt', 'Butter', 'Cheese Slices', 'Ice Cream'],
                'meat': ['Ground Beef', 'Chicken Breast', 'Salmon Fillet', 'Turkey Slices', 'Bacon'],
                'bakery': ['Whole Wheat Bread', 'Croissants', 'Chocolate Muffins', 'Bagels', 'Dinner Rolls'],
                'pantry': ['Pasta', 'Rice', 'Olive Oil', 'Cereal', 'Canned Tomatoes'],
                'frozen': ['Frozen Pizza', 'Ice Cream', 'Frozen Vegetables', 'Fish Sticks', 'Frozen Meals']
            },
            'electronics': {
                'phones': ['Smartphone', 'Wireless Charger', 'Phone Case', 'Bluetooth Headset', 'Screen Protector'],
                'computers': ['Laptop PC', 'Desktop PC', 'Monitor', 'Keyboard', 'Wireless Mouse'],
                'accessories': ['USB Cable', 'Power Bank', 'Bluetooth Speaker', 'Earbuds', 'Gaming Controller'],
                'audio': ['Wireless Earbuds', 'Bluetooth Speaker', 'Sound Bar', 'Microphone', 'Headphones'],
                'gaming': ['Gaming Console', 'Video Game', 'Controller', 'Gaming Headset', 'Gaming Chair']
            },
            'clothing': {
                'mens': ['T-Shirt', 'Jeans', 'Dress Shirt', 'Polo Shirt', 'Jacket'],
                'womens': ['Blouse', 'Dress', 'Leggings', 'Sweater', 'Skirt'],
                'kids': ['Kids T-Shirt', 'School Uniform', 'Pajamas', 'Jacket', 'Sneakers'],
                'shoes': ['Running Shoes', 'Dress Shoes', 'Sandals', 'Boots', 'Sneakers'],
                'accessories': ['Belt', 'Hat', 'Scarf', 'Sunglasses', 'Watch']
            },
            'home_garden': {
                'home': ['HomePro', 'GardenMax', 'Plus', 'ValueChoice', 'QualityBrand'],
                'garden': ['GardenPro', 'PlantMax', 'GrowPlus', 'NatureBest', 'GreenChoice']
            },
            'health_beauty': {
                'health': ['HealthFirst', 'WellnessPro', 'VitalityPlus', 'CareMax', 'HealthyChoice'],
                'beauty': ['BeautyPro', 'GlowMax', 'StylePlus', 'BeautyChoice', 'PerfectLook']
            },
            'automotive': {
                'parts': ['Motor Oil', 'Air Filter', 'Brake Pads', 'Spark Plugs', 'Battery'],
                'accessories': ['Car Charger', 'Phone Mount', 'Floor Mats', 'Seat Covers', 'Air Freshener']
            }
        }
        
        # Brand names by category
        self.category_brands = {
            'groceries': ['FreshChoice', 'NaturalBest', 'FarmFresh', 'OrganicPlus', 'ValueMax'],
            'electronics': ['TechPro', 'Digital', 'ElectroCore', 'MegaTech', 'SmartTech'],
            'clothing': ['FashionMax', 'TrendPlus', 'ClassicWear', 'StyleChoice', 'LivingStyle'],
            'health_beauty': ['BeautyPro', 'HealthFirst', 'WellnessPlus', 'CareMax', 'VitalityChoice'],
            'home_garden': ['HomePro', 'GardenMax', 'Plus', 'ValueChoice', 'QualityBrand']
        }
    
    def _setup_payment_methods(self):
        """Setup payment methods and processing characteristics"""
        self.payment_methods = {
            'credit_card': {
                'processing_time': 30,  # seconds
                'avg_processing_fee': 0.029,  # 2.9%
                'weight': 0.45
            },
            'debit_card': {
                'processing_time': 25,
                'avg_processing_fee': 0.015,  # 1.5%
                'weight': 0.30
            },
            'cash': {
                'processing_time': 45,
                'avg_processing_fee': 0.0,
                'weight': 0.15
            },
            'mobile_payment': {
                'processing_time': 20,
                'avg_processing_fee': 0.025,  # 2.5%
                'weight': 0.08
            },
            'gift_card': {
                'processing_time': 15,
                'avg_processing_fee': 0.0,
                'weight': 0.02
            }
        }   
 
    def _setup_transaction_patterns(self):
        """Setup realistic transaction patterns by time and season"""
        # Hourly patterns (multipliers for base activity)
        self.hourly_patterns = {
            0: 0.1, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1,
            6: 0.3, 7: 0.6, 8: 0.8, 9: 1.0, 10: 1.2, 11: 1.4,
            12: 1.6, 13: 1.5, 14: 1.3, 15: 1.2, 16: 1.4, 17: 1.6,
            18: 1.8, 19: 1.5, 20: 1.2, 21: 0.8, 22: 0.5, 23: 0.2
        }
        
        # Store type specific patterns
        self.store_type_patterns = {
            'convenience': {
                'peak_hours': [7, 8, 12, 17, 18],
                'weekend_boost': 1.2
            },
            'supermarket': {
                'peak_hours': [10, 11, 17, 18, 19],
                'weekend_boost': 1.4
            },
            'department_store': {
                'peak_hours': [12, 13, 18, 19, 20],
                'weekend_boost': 1.6
            }
        }
        
        # Monthly seasonal multipliers (1.0 = average)
        self.seasonal_multipliers = {
            1: 0.85,  # January - post-holiday low
            2: 0.90,  # February - Valentine's boost
            3: 0.95,  # March - spring preparation
            4: 1.00,  # April - normal activity
            5: 1.05,  # May - Mother's Day, graduation season
            6: 1.00,  # June - summer activities
            7: 1.05,  # July - summer
            8: 1.00,  # August - back-to-school prep
            9: 1.05,  # September - back-to-school
            10: 1.10,  # October - Halloween, fall
            11: 1.30,  # November - Thanksgiving, Black Friday
            12: 1.40   # December - holiday shopping
        }
        
        # Category-specific seasonal patterns
        self.category_seasonal_patterns = {
            'clothing': {
                3: 1.3, 4: 1.4, 5: 1.2, 8: 1.4, 9: 1.5, 10: 1.2, 11: 1.8, 12: 1.6
            },
            'electronics': {
                11: 1.6, 12: 1.8, 1: 0.7, 8: 1.3, 9: 1.2
            },
            'home_garden': {
                3: 1.4, 4: 1.6, 5: 1.5, 6: 1.3, 7: 1.2, 8: 1.0, 9: 1.1, 10: 1.2
            },
            'groceries': {
                11: 1.2, 12: 1.3  # Holiday cooking
            }
        }
    
    def _setup_geographic_data(self):
        """Setup geographic regions and constraints"""
        self.regions = {
            'northeast': {
                'states': ['NY', 'MA', 'CT', 'RI', 'VT', 'NH', 'ME', 'NJ', 'PA'],
                'population_density': 'high',
                'cost_of_living_multiplier': 1.15,
                'weight': 0.18
            },
            'southeast': {
                'states': ['FL', 'GA', 'SC', 'NC', 'VA', 'WV', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA'],
                'population_density': 'medium',
                'cost_of_living_multiplier': 0.90,
                'weight': 0.25
            },
            'midwest': {
                'states': ['OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS'],
                'population_density': 'medium',
                'cost_of_living_multiplier': 0.85,
                'weight': 0.22
            },
            'southwest': {
                'states': ['TX', 'OK', 'NM', 'AZ', 'NV', 'UT', 'CO'],
                'population_density': 'medium',
                'cost_of_living_multiplier': 0.95,
                'weight': 0.18
            },
            'west': {
                'states': ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'AK', 'HI'],
                'population_density': 'varied',
                'cost_of_living_multiplier': 1.25,
                'weight': 0.17
            }
        }   
 
    def _weighted_choice(self, choices: List[str], weights: List[float]) -> str:
        """Make a weighted random choice"""
        cumulative_weights = []
        total = 0
        for weight in weights:
            total += weight
            cumulative_weights.append(total)
        
        rand_val = self.faker.random.uniform(0, total)
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                return choices[i]
        return choices[-1]
    
    def _select_store_type(self) -> str:
        """Select store type based on distribution"""
        choices = list(self.store_types.keys())
        weights = [self.store_types[st]['weight'] for st in choices]
        return self._weighted_choice(choices, weights)
    
    def _select_region(self, geographic_constraints: Optional[Dict] = None) -> str:
        """Select region based on constraints and distribution"""
        if geographic_constraints and 'regions' in geographic_constraints:
            allowed_regions = geographic_constraints['regions']
            choices = [r for r in self.regions.keys() if r in allowed_regions]
        else:
            choices = list(self.regions.keys())
        
        if not choices:
            choices = list(self.regions.keys())
        
        weights = [self.regions[region]['weight'] for region in choices]
        return self._weighted_choice(choices, weights)
    
    def _generate_store_name(self, store_type: str) -> str:
        """Generate realistic store names based on type"""
        prefixes = {
            'supermarket': ['Fresh', 'Super', 'Grand', 'Central', 'Main', 'Premier'],
            'convenience': ['Quick', 'Express', 'Corner', 'Go', 'Stop', 'Fast'],
            'department_store': ['Grand', 'Central', 'Plaza', 'Main', 'Premier', 'Elite'],
            'electronics': ['Tech', 'Digital', 'Electronic', 'Gadget', 'Circuit', 'Smart'],
            'clothing': ['Fashion', 'Style', 'Trend', 'Boutique', 'Wardrobe', 'Closet'],
            'pharmacy': ['Health', 'Care', 'Wellness', 'Medical', 'Rx', 'Plus'],
            'home_improvement': ['Home', 'Build', 'Fix', 'Tool', 'Hardware', 'Craft']
        }
        
        suffixes = {
            'supermarket': ['Market', 'Foods', 'Grocery', 'Fresh', 'Mart', 'Store'],
            'convenience': ['Store', 'Shop', 'Mart', 'Center', 'Plaza', 'Outlet'],
            'department_store': ['Store', 'Department', 'Center', 'Plaza', 'Mart', 'Outlet'],
            'electronics': ['Electronics', 'Tech', 'Digital', 'Gadgets', 'Circuit', 'Store'],
            'clothing': ['Fashion', 'Style', 'Boutique', 'Trends', 'Wardrobe', 'Closet'],
            'pharmacy': ['Pharmacy', 'Health', 'Care', 'Wellness', 'Medical', 'Plus'],
            'home_improvement': ['Home', 'Hardware', 'Tools', 'Build', 'Depot', 'Center']
        }
        
        prefix = self.faker.random_element(prefixes.get(store_type, ['General']))
        suffix = self.faker.random_element(suffixes.get(store_type, ['Store']))
        
        return f"{prefix} {suffix}"
    
    def _generate_store_size(self, store_type: str) -> int:
        """Generate store size based on type"""
        size_min, size_max = self.store_types[store_type]['store_size_sqft']
        return self.faker.random_int(size_min, size_max)
    
    def _select_transaction_hour(self, store_type: str, transaction_date: date) -> int:
        """Select transaction hour based on store type and patterns"""
        # Get base hourly weights
        hour_weights = list(self.hourly_patterns.values())
        hours = list(range(24))
        
        # Apply store-specific patterns
        if store_type in self.store_type_patterns:
            peak_hours = self.store_type_patterns[store_type]['peak_hours']
            for hour in peak_hours:
                if hour < len(hour_weights):
                    hour_weights[hour] *= 1.5
        
        # Apply weekend boost
        if transaction_date.weekday() >= 5:  # Weekend
            weekend_boost = self.store_type_patterns.get(store_type, {}).get('weekend_boost', 1.0)
            hour_weights = [w * weekend_boost for w in hour_weights]
        
        return self._weighted_choice(hours, hour_weights)
    
    def _generate_num_items(self, store_type: str) -> int:
        """Generate number of items based on store type"""
        min_items, max_items = self.store_types[store_type]['avg_items_per_transaction']
        return self.faker.random_int(min_items, max_items)    

    def _select_product_category(self, store_type: str) -> str:
        """Select product category based on store type"""
        # Store type influences category distribution
        if store_type == 'supermarket':
            category_weights = {'groceries': 0.9, 'health_beauty': 0.1}
        elif store_type == 'convenience':
            category_weights = {'groceries': 0.6, 'automotive': 0.2, 'health_beauty': 0.1, 'electronics': 0.1}
        elif store_type == 'electronics':
            category_weights = {'electronics': 0.8, 'home_garden': 0.1, 'automotive': 0.1}
        elif store_type == 'clothing':
            category_weights = {'clothing': 0.8, 'health_beauty': 0.2}
        elif store_type == 'pharmacy':
            category_weights = {'health_beauty': 0.7, 'groceries': 0.2, 'home_garden': 0.1}
        else:
            # Default distribution
            category_weights = {}
        
        if not category_weights:
            choices = list(self.product_categories.keys())
            weights = [1.0 for _ in choices]
        else:
            choices = list(category_weights.keys())
            weights = list(category_weights.values())
        
        return self._weighted_choice(choices, weights)
    
    def _generate_product_name(self, category: str, subcategory: str) -> str:
        """Generate realistic product names"""
        if category in self.product_categories and subcategory in self.product_categories[category]:
            return self.faker.random_element(self.product_categories[category][subcategory])
        else:
            return f"{subcategory.replace('_', ' ').title()} Product"
    
    def _generate_brand_name(self, category: str) -> str:
        """Generate brand names based on category"""
        if category in self.category_brands:
            return self.faker.random_element(self.category_brands[category])
        else:
            return f"{category.replace('_', ' ').title()} Brand"
    
    def _generate_product(self, category: str, transaction_date: date) -> Dict:
        """Generate product information"""
        # Generate subcategory
        if category in self.product_categories:
            subcategory = self.faker.random_element(list(self.product_categories[category].keys()))
        else:
            subcategory = 'general'
        
        product_name = self._generate_product_name(category, subcategory)
        brand = self._generate_brand_name(category)
        
        # Generate price with seasonal adjustment
        category_config = self.store_types.get('supermarket', {})  # Default config
        price_min, price_max = category_config.get('avg_transaction_value', (5, 100))
        base_price = self.faker.random.uniform(price_min/10, price_max/10)
        
        # Apply seasonal multiplier
        seasonal_mult = self._get_seasonal_multiplier(category, transaction_date.month)
        price = round(base_price * seasonal_mult, 2)
        
        return {
            'product_id': f'P_{self.faker.random_int(100000, 999999)}',
            'product_name': product_name,
            'category': category,
            'subcategory': subcategory,
            'brand': brand,
            'price': price,
            'sku': f'SKU_{self.faker.random_int(100000, 999999)}',
            'quantity': self.faker.random_int(1, 3)
        }
    
    def _get_seasonal_multiplier(self, category: str, month: int) -> float:
        """Get seasonal price multiplier for category and month"""
        base_multiplier = self.seasonal_multipliers.get(month, 1.0)
        
        if category in self.category_seasonal_patterns:
            category_multiplier = self.category_seasonal_patterns[category].get(month, 1.0)
        else:
            category_multiplier = 1.0
        
        return base_multiplier * category_multiplier
    
    def _select_payment_method(self) -> str:
        """Select payment method based on distribution"""
        choices = list(self.payment_methods.keys())
        weights = [self.payment_methods[pm]['weight'] for pm in choices]
        return self._weighted_choice(choices, weights)
    
    def _select_customer_type(self) -> str:
        """Select customer type"""
        types = {
            'regular': 0.60,
            'loyalty_member': 0.25,
            'employee': 0.05,
            'senior': 0.10
        }
        
        choices = list(types.keys())
        weights = list(types.values())
        return self._weighted_choice(choices, weights)  
  
    def _generate_discount_amount(self, subtotal: float, customer_type: str) -> float:
        """Generate discount amount based on customer type"""
        discount_rates = {
            'regular': 0.0,
            'loyalty_member': 0.05,
            'employee': 0.10,
            'senior': 0.05
        }
        
        # Random chance of discount
        if self.faker.random.random() < 0.3:  # 30% chance of discount
            rate = discount_rates.get(customer_type, 0.0)
            return round(subtotal * rate, 2)
        
        return 0.0
    
    def _calculate_loyalty_points(self, total_amount: float, customer_type: str) -> int:
        """Calculate loyalty points earned"""
        if customer_type == 'loyalty_member':
            return int(total_amount * 1.5)  # 1.5 points per dollar
        elif customer_type == 'employee':
            return int(total_amount * 1.5)  # 1.5 points per dollar
        else:
            return 0
    
    def _generate_promotion_code(self) -> Optional[str]:
        """Generate promotion codes (sometimes)"""
        if self.faker.random.random() < 0.15:  # 15% chance
            codes = ['SAVE10', 'WELCOME25', 'FALL15', 'SUMMER20', 'WINTER30']
            return self.faker.random_element(codes)
        else:
            return None
    
    def _calculate_transaction_duration(self, num_items: int, payment_method: str) -> int:
        """Calculate transaction duration in seconds"""
        # Base time: 30 seconds
        base_time = 30
        
        # Item time: 10 seconds per item
        item_time = num_items * 10
        
        # Payment time
        payment_time = self.payment_methods[payment_method]['processing_time']
        
        total_time = base_time + item_time + payment_time
        
        # Add some random variation
        variation = self.faker.random.uniform(0.8, 1.2)
        return int(total_time * variation)
    
    def _apply_retail_temporal_patterns(self, base_activity: float, timestamp: datetime, store_type: str) -> float:
        """Apply retail-specific temporal patterns"""
        # Daily pattern
        daily_mult = self.hourly_patterns.get(timestamp.hour, 1.0)
        
        # Seasonal pattern
        seasonal_mult = self.seasonal_multipliers.get(timestamp.month, 1.0)
        
        # Store type pattern
        store_patterns = self.store_type_patterns.get(store_type, {})
        store_mult = 1.0
        if timestamp.hour in store_patterns.get('peak_hours', []):
            store_mult *= 1.2
        
        # Weekend boost
        if timestamp.weekday() >= 5:  # Weekend
            weekend_boost = store_patterns.get('weekend_boost', 1.0)
            store_mult *= weekend_boost
        
        adjusted_activity = base_activity * daily_mult * seasonal_mult * store_mult
        
        # Add some random variation
        variation = self.faker.random.uniform(0.9, 1.1)
        return adjusted_activity * variation
    
    def _generate_retail_transaction(self, transaction_index: int, transaction_date: date, store: Dict, timestamp: datetime, activity_multiplier: float) -> Dict:
        """Generate a single retail transaction record"""
        # Generate base transaction
        store_type = store['store_type']
        
        # Generate transaction time
        transaction_hour = self._select_transaction_hour(store_type, transaction_date)
        transaction_datetime = datetime.combine(
            transaction_date,
            datetime.min.time().replace(
                hour=transaction_hour,
                minute=self.faker.random_int(0, 59),
                second=self.faker.random_int(0, 59)
            )
        )
        
        # Generate products and calculate totals
        num_items = self._generate_num_items(store_type)
        items = []
        subtotal = 0
        
        for _ in range(num_items):
            category = self._select_product_category(store_type)
            product = self._generate_product(category, transaction_date)
            items.append(product)
            subtotal += product['price'] * product['quantity']
        
        # Customer information
        customer_type = self._select_customer_type()
        
        # Calculate taxes and fees
        tax_rate = self.faker.random.uniform(0.06, 0.10)  # 6-10% sales tax
        tax_amount = round(subtotal * tax_rate, 2)
        
        # Payment method
        payment_method = self._select_payment_method()
        processing_fee = round(subtotal * self.payment_methods[payment_method]['avg_processing_fee'], 2)
        
        # Apply temporal adjustment before final calculations
        temporal_adjustment = self._apply_retail_temporal_patterns(
            activity_multiplier, timestamp, store_type
        )
        
        # Apply temporal adjustment to base amounts
        adjusted_subtotal = round(subtotal * temporal_adjustment, 2)
        adjusted_tax_amount = round(tax_amount * temporal_adjustment, 2)
        
        # Calculate discounts based on adjusted subtotal
        discount_amount = self._generate_discount_amount(adjusted_subtotal, customer_type)
        adjusted_discount_amount = discount_amount
        
        total_amount = adjusted_subtotal + adjusted_tax_amount - adjusted_discount_amount
        
        # Additional metrics
        loyalty_points = self._calculate_loyalty_points(total_amount, customer_type)
        promotion_code = self._generate_promotion_code()
        transaction_duration = self._calculate_transaction_duration(num_items, payment_method)
        
        transaction = {
            'transaction_id': f'TXN_{transaction_index+1:08d}',
            'store_id': store['store_id'],
            'store_name': store['store_name'],
            'store_type': store_type,
            'region': store['region'],
            'state': store['state'],
            'city': store['city'],
            'transaction_datetime': transaction_datetime,
            'transaction_date': transaction_date,
            'day_of_week': transaction_date.strftime('%A'),
            'transaction_hour': transaction_hour,
            'customer_id': f'CUST_{self.faker.random_int(1, 50000):06d}',
            'customer_type': customer_type,
            'cashier_id': f'CASH_{self.faker.random_int(1, 1000):03d}',
            'num_items': num_items,
            'product_categories': ', '.join(set([item['category'] for item in items])),
            'primary_category': items[0]['category'] if items else 'unknown',
            'payment_method': payment_method,
            'subtotal': adjusted_subtotal,
            'tax_rate': round(tax_rate, 4),
            'tax_amount': adjusted_tax_amount,
            'discount_amount': adjusted_discount_amount,
            'discount_applied': adjusted_discount_amount > 0,
            'total_amount': round(total_amount, 2),
            'processing_fee': processing_fee,
            'net_profit': round((adjusted_subtotal * 0.3), 2),  # Assume 30% average profit margin
            'profit_margin': round(((adjusted_subtotal * 0.3) / adjusted_subtotal), 3) if adjusted_subtotal > 0 else 0,
            'avg_item_price': round(adjusted_subtotal / num_items, 2) if num_items > 0 else 0,
            'loyalty_points_earned': loyalty_points,
            'promotion_code': promotion_code,
            'transaction_duration_seconds': transaction_duration,
            'pos_terminal': f'POS_{self.faker.random_int(1, 20):02d}',
            'receipt_number': f'RCP_{transaction_index+1:08d}',
            'items_detail': json.dumps(items),
            'temporal_adjustment': round(temporal_adjustment, 3)
        }
        
        return transaction  
  
    def _generate_stores(self, num_stores: int, geographic_constraints: Optional[Dict] = None) -> List[Dict]:
        """Generate store information"""
        stores = []
        
        for i in range(num_stores):
            # Select region
            region = self._select_region(geographic_constraints)
            
            # Select store type
            store_type = self._select_store_type()
            
            store = {
                'store_id': f'ST_{i+1:04d}',
                'store_name': self._generate_store_name(store_type),
                'store_type': store_type,
                'region': region,
                'state': self.faker.random_element(self.regions[region]['states']),
                'city': self.faker.city(),
                'address': self.faker.address().replace('\n', ', '),
                'zip_code': self.faker.zipcode(),
                'phone': self.faker.phone_number(),
                'manager_name': self.faker.name(),
                'store_size_sqft': self._generate_store_size(store_type),
                'opening_date': self.faker.date_between(start_date='-5y', end_date='-1y')
            }
            stores.append(store)
        
        return stores
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to retail data"""
        # Sort by timestamp to ensure proper time ordering
        data = data.sort_values('transaction_datetime').reset_index(drop=True)
        
        # Apply temporal correlations for transaction values
        if len(data) < 2:
            return data
        
        # Transaction value clustering (similar values tend to cluster)
        correlation_strength = 0.15
        for i in range(1, len(data)):
            prev_total = data.iloc[i-1]['total_amount']
            if prev_total > 0:
                value_adjustment = 1 + (self.faker.random.uniform(-0.1, 0.1) * correlation_strength)
                new_total = data.iloc[i]['total_amount'] * value_adjustment
                data.loc[i, 'total_amount'] = max(5.0, min(new_total, 1000))  # Keep within reasonable bounds
                
                # Adjust related fields while maintaining mathematical relationships
                # Calculate new subtotal and tax amount based on new total
                discount_amount = data.iloc[i]['discount_amount']
                tax_rate = data.iloc[i]['tax_rate']
                
                # total_amount = subtotal + tax_amount - discount_amount
                # total_amount = subtotal + (subtotal * tax_rate) - discount_amount
                # total_amount = subtotal * (1 + tax_rate) - discount_amount
                # subtotal = (total_amount + discount_amount) / (1 + tax_rate)
                
                new_subtotal = (data.loc[i, 'total_amount'] + discount_amount) / (1 + tax_rate)
                new_tax_amount = new_subtotal * tax_rate
                
                data.loc[i, 'subtotal'] = round(new_subtotal, 2)
                data.loc[i, 'tax_amount'] = round(new_tax_amount, 2)
                data.loc[i, 'avg_item_price'] = round(new_subtotal / data.iloc[i]['num_items'], 2) if data.iloc[i]['num_items'] > 0 else 0
                data.loc[i, 'profit_margin'] = round(((new_subtotal * 0.3) / new_subtotal), 3) if new_subtotal > 0 else 0
                data.loc[i, 'net_profit'] = round((new_subtotal * 0.3), 2)
        
        # Add derived fields if 'transaction_datetime' in data columns:
        if 'transaction_datetime' in data.columns:
            data['transaction_date'] = pd.to_datetime(data['transaction_datetime']).dt.date
            data['transaction_month'] = pd.to_datetime(data['transaction_datetime']).dt.month
            data['transaction_quarter'] = pd.to_datetime(data['transaction_datetime']).dt.quarter
            data['is_weekend'] = pd.to_datetime(data['transaction_datetime']).dt.weekday >= 5
            data['is_peak_hour'] = data['transaction_hour'].isin([11, 12, 17, 18, 19])
        
        # Calculate additional performance indicators
        data['high_value_transaction'] = data['total_amount'] > data['total_amount'].quantile(0.8)
        data['bulk_purchase'] = data['num_items'] > 15
        data['discount_applied'] = data['discount_amount'] > 0
        data['loyalty_transaction'] = data['customer_type'] == 'loyalty_member'
        
        # Add transaction size categories
        data['transaction_size'] = pd.cut(
            data['total_amount'],
            bins=[0, 10, 25, 100, 300, float('inf')],
            labels=['small', 'medium', 'large', 'bulk', 'extra_large']
        )
        
        data['basket_size'] = pd.cut(
            data['num_items'],
            bins=[0, 3, 10, 20, float('inf')],
            labels=['small', 'medium', 'large', 'extra_large']
        )
        
        return data    

    def _generate_snapshot_retail(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot retail data (not time series)"""
        geographic_constraints = kwargs.get('geographic_constraints', None)
        date_range = kwargs.get('date_range', None)
        
        # Generate store information first
        num_stores = max(1, rows // 100)  # Roughly 100 transactions per store
        stores = self._generate_stores(num_stores, geographic_constraints)
        
        data = []
        
        for i in range(rows):
            # Select random store
            store = self.faker.random_element(stores)
            
            # Generate transaction date
            if date_range:
                start_date, end_date = date_range
                transaction_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                transaction_date = self.faker.date_this_year()
            
            # Generate transaction
            transaction = self._generate_retail_transaction(i, transaction_date, store, datetime.now(), 1.0)
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Apply realistic patterns
        df = self._apply_realistic_patterns(df)
        
        return df
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate retail operations dataset
        
        Args:
            rows: Number of transaction records to generate
            **kwargs: Additional parameters including:
                - time_series: Enable time series generation
                - date_range: Tuple of (start_date, end_date) for transactions
                - geographic_constraints: Dict with regional constraints
                
        Returns:
            pd.DataFrame: Generated retail dataset
        """
        # For now, only implement snapshot generation
        # Time series generation would require additional time series integration
        return self._generate_snapshot_retail(rows, **kwargs)