"""
Ecommerce order dataset generator

Generates realistic ecommerce data with order patterns and product correlations.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class EcommerceGenerator(BaseGenerator):
    """
    Generator for realistic ecommerce order data
    
    Creates ecommerce datasets with order patterns, shipping preferences,
    product correlations, cart abandonment patterns, and return behaviors.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_product_data()
        self._setup_order_patterns()
        self._setup_shipping_preferences()
        self._setup_product_correlations()
        self._setup_seasonal_patterns()
        self._setup_customer_behaviors()
    
    def _load_product_data(self):
        """Load product data for realistic product selection"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '../../data/business/products.json')
            with open(data_path, 'r') as f:
                product_data = json.load(f)
            
            self.products = []
            self.product_categories = product_data.get('product_categories', [])
            
            # Flatten all products from different categories with pricing
            for category, items in product_data.get('products', {}).items():
                for item in items:
                    self.products.append({
                        'name': item,
                        'category': category,
                        'base_price': self._get_base_price_for_category(category),
                        'weight_kg': self._get_weight_for_category(category)
                    })
        except FileNotFoundError:
            # Fallback if data file not found
            self.products = [
                {'name': f'Product_{i}', 'category': 'general', 'base_price': 50.0, 'weight_kg': 0.5}
                for i in range(100)
            ]
            self.product_categories = ['general']
    
    def _get_base_price_for_category(self, category: str) -> float:
        """Get base price range for product category"""
        price_ranges = {
            'technology': (50, 1500),
            'healthcare': (15, 300),
            'retail': (10, 150),
            'automotive': (25, 500),
            'food_beverage': (3, 30),
            'home_garden': (20, 400)
        }
        min_price, max_price = price_ranges.get(category, (10, 100))
        return self.faker.random.uniform(min_price, max_price)
    
    def _get_weight_for_category(self, category: str) -> float:
        """Get typical weight for product category"""
        weight_ranges = {
            'technology': (0.1, 5.0),
            'healthcare': (0.05, 2.0),
            'retail': (0.1, 3.0),
            'automotive': (0.5, 20.0),
            'food_beverage': (0.1, 2.0),
            'home_garden': (0.2, 15.0)
        }
        min_weight, max_weight = weight_ranges.get(category, (0.1, 2.0))
        return round(self.faker.random.uniform(min_weight, max_weight), 2)
    
    def _setup_order_patterns(self):
        """Setup realistic order patterns"""
        # Order status distribution
        self.order_status_distribution = {
            'delivered': 0.75,
            'shipped': 0.10,
            'processing': 0.08,
            'cancelled': 0.05,
            'returned': 0.02
        }
        
        # Order size distribution (number of items)
        self.order_size_distribution = {
            1: 0.45,  # Single item orders
            2: 0.25,  # Two items
            3: 0.15,  # Three items
            4: 0.08,  # Four items
            5: 0.04,  # Five items
            6: 0.02,  # Six items
            7: 0.01   # Seven or more items
        }
        
        # Time of day patterns (hour of day)
        self.hourly_order_patterns = {
            0: 0.01, 1: 0.005, 2: 0.005, 3: 0.005, 4: 0.005, 5: 0.01,
            6: 0.02, 7: 0.03, 8: 0.04, 9: 0.05, 10: 0.06, 11: 0.07,
            12: 0.08, 13: 0.07, 14: 0.06, 15: 0.05, 16: 0.04, 17: 0.03,
            18: 0.04, 19: 0.06, 20: 0.08, 21: 0.09, 22: 0.07, 23: 0.03
        }
    
    def _setup_shipping_preferences(self):
        """Setup shipping method preferences"""
        self.shipping_methods = {
            'standard': {
                'cost_multiplier': 1.0,
                'delivery_days': (3, 7),
                'weight_limit': 30.0,
                'popularity': 0.60
            },
            'express': {
                'cost_multiplier': 1.5,
                'delivery_days': (1, 3),
                'weight_limit': 20.0,
                'popularity': 0.25
            },
            'overnight': {
                'cost_multiplier': 2.5,
                'delivery_days': (1, 1),
                'weight_limit': 10.0,
                'popularity': 0.10
            },
            'free_shipping': {
                'cost_multiplier': 0.0,
                'delivery_days': (5, 10),
                'weight_limit': 50.0,
                'popularity': 0.05
            }
        }
        
        # Shipping preferences by order value
        self.shipping_by_order_value = {
            'low': {'standard': 0.7, 'express': 0.2, 'overnight': 0.05, 'free_shipping': 0.05},      # < $50
            'medium': {'standard': 0.5, 'express': 0.35, 'overnight': 0.1, 'free_shipping': 0.05},   # $50-$200
            'high': {'standard': 0.3, 'express': 0.4, 'overnight': 0.25, 'free_shipping': 0.05}      # > $200
        }
    
    def _setup_product_correlations(self):
        """Setup product correlation patterns"""
        # Products that are often bought together
        self.product_correlations = {
            'technology': ['technology', 'retail'],  # Tech accessories
            'healthcare': ['healthcare', 'retail'],  # Health and wellness
            'retail': ['retail', 'home_garden'],     # Fashion and home
            'automotive': ['automotive', 'technology'], # Car tech
            'food_beverage': ['food_beverage', 'home_garden'], # Kitchen items
            'home_garden': ['home_garden', 'retail']  # Home and lifestyle
        }
        
        # Seasonal product preferences
        self.seasonal_product_preferences = {
            'spring': ['home_garden', 'retail', 'healthcare'],
            'summer': ['retail', 'technology', 'automotive'],
            'fall': ['retail', 'home_garden', 'technology'],
            'winter': ['technology', 'healthcare', 'home_garden']
        }
    
    def _setup_seasonal_patterns(self):
        """Setup seasonal ordering patterns"""
        self.monthly_order_multipliers = {
            1: 0.8,   # January - post-holiday low
            2: 0.9,   # February - Valentine's boost
            3: 1.0,   # March - normal
            4: 1.1,   # April - spring shopping
            5: 1.2,   # May - Mother's Day
            6: 1.0,   # June - normal
            7: 0.9,   # July - summer vacation
            8: 0.8,   # August - back to school prep
            9: 1.1,   # September - back to school
            10: 1.2,  # October - pre-holiday
            11: 1.6,  # November - Black Friday
            12: 1.8   # December - holiday season
        }
    
    def _setup_customer_behaviors(self):
        """Setup customer behavior patterns"""
        # Cart abandonment patterns
        self.cart_abandonment_rate = 0.15  # 15% of potential orders are abandoned
        
        # Return patterns by category
        self.return_rates_by_category = {
            'technology': 0.08,
            'healthcare': 0.03,
            'retail': 0.15,  # Clothing has higher return rates
            'automotive': 0.05,
            'food_beverage': 0.02,
            'home_garden': 0.07
        }
        
        # Customer loyalty patterns
        self.repeat_customer_rate = 0.35  # 35% are repeat customers
        
        # Payment method preferences
        self.payment_methods = {
            'credit_card': 0.45,
            'debit_card': 0.25,
            'paypal': 0.15,
            'apple_pay': 0.08,
            'google_pay': 0.05,
            'bank_transfer': 0.02
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate ecommerce order dataset
        
        Args:
            rows: Number of orders to generate
            **kwargs: Additional parameters (country, date_range, time_series, etc.)
            
        Returns:
            pd.DataFrame: Generated ecommerce data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_ecommerce(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_ecommerce(rows, **kwargs)
    
    def _generate_snapshot_ecommerce(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot ecommerce data (random order dates)"""
        country = kwargs.get('country', 'global')
        date_range = kwargs.get('date_range', None)
        include_abandoned = kwargs.get('include_abandoned', False)
        
        data = []
        customer_order_history = {}  # Track repeat customers
        
        for i in range(rows):
            # Generate order date with seasonal consideration
            if date_range:
                start_date, end_date = date_range
                order_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                order_date = self.faker.date_this_year()
            
            # Add time component for realistic ordering patterns
            order_hour = self._select_order_hour()
            order_datetime = datetime.combine(order_date, datetime.min.time().replace(hour=order_hour))
            
            # Determine if this is a repeat customer
            is_repeat_customer = self.faker.random.random() < self.repeat_customer_rate
            
            if is_repeat_customer and customer_order_history:
                # Select existing customer
                customer_id = self.faker.random_element(list(customer_order_history.keys()))
                customer_order_history[customer_id] += 1
            else:
                # New customer
                customer_id = f'CUST_{self.faker.random_int(1, 50000):06d}'
                customer_order_history[customer_id] = 1
            
            # Generate order items
            order_items = self._generate_order_items(order_date)
            
            # Calculate totals
            subtotal = sum(item['price'] * item['quantity'] for item in order_items)
            total_weight = sum(item['weight'] * item['quantity'] for item in order_items)
            
            # Select shipping method based on order value and weight
            shipping_method = self._select_shipping_method(subtotal, total_weight)
            shipping_cost = self._calculate_shipping_cost(shipping_method, total_weight, subtotal)
            
            # Calculate taxes (approximate)
            tax_rate = 0.08 if country == 'united_states' else 0.10
            tax_amount = round(subtotal * tax_rate, 2)
            
            total_amount = subtotal + shipping_cost + tax_amount
            
            # Determine order status
            order_status = self._select_order_status(order_datetime)
            
            # Generate payment information
            payment_method = self._select_payment_method()
            
            # Generate shipping address
            shipping_address = self._generate_shipping_address(country)
            
            # Check for cart abandonment (only if including abandoned carts)
            is_abandoned = include_abandoned and self.faker.random.random() < self.cart_abandonment_rate
            
            if is_abandoned:
                order_status = 'abandoned'
                total_amount = 0  # No payment for abandoned carts
                shipping_cost = 0
                tax_amount = 0
            
            # Generate order record
            order = {
                'order_id': f'ORD_{i+1:08d}',
                'customer_id': customer_id,
                'order_date': order_date,
                'order_datetime': order_datetime,
                'order_status': order_status,
                'subtotal': subtotal,
                'shipping_cost': shipping_cost,
                'tax_amount': tax_amount,
                'total_amount': total_amount,
                'payment_method': payment_method,
                'shipping_method': shipping_method,
                'estimated_delivery_days': self._get_delivery_estimate(shipping_method),
                'total_items': len(order_items),
                'total_quantity': sum(item['quantity'] for item in order_items),
                'total_weight_kg': total_weight,
                'primary_category': order_items[0]['category'] if order_items else 'unknown',
                'shipping_country': shipping_address['country'],
                'shipping_city': shipping_address['city'],
                'shipping_postal_code': shipping_address['postal_code'],
                'is_repeat_customer': customer_order_history[customer_id] > 1,
                'customer_order_number': customer_order_history[customer_id],
                'is_gift': self.faker.random.random() < 0.12,  # 12% are gifts
            }
            
            # Handle discount logic consistently
            has_discount = self.faker.random.random() < 0.25  # 25% have discounts
            order['has_discount'] = has_discount
            order['discount_amount'] = round(subtotal * 0.1, 2) if has_discount else 0.0
            
            # Add item details as separate columns for the first few items
            for idx, item in enumerate(order_items[:3]):  # Include up to 3 items
                order[f'item_{idx+1}_name'] = item['name']
                order[f'item_{idx+1}_category'] = item['category']
                order[f'item_{idx+1}_price'] = item['price']
                order[f'item_{idx+1}_quantity'] = item['quantity']
            
            # Fill empty item slots
            for idx in range(len(order_items), 3):
                order[f'item_{idx+1}_name'] = None
                order[f'item_{idx+1}_category'] = None
                order[f'item_{idx+1}_price'] = None
                order[f'item_{idx+1}_quantity'] = None
            
            data.append(order)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_ecommerce(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series ecommerce data using integrated time series system"""
        country = kwargs.get('country', 'global')
        include_abandoned = kwargs.get('include_abandoned', False)
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Generate base order volume time series
        base_order_volume = 50  # Base daily order volume
        
        # Create base time series for order volume
        volume_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_order_volume,
            value_range=(base_order_volume * 0.1, base_order_volume * 4.0)
        )
        
        data = []
        customer_order_history = {}  # Track repeat customers
        product_sales_history = {}  # Track product trends
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(volume_series):
                break
            
            # Get time series volume value (represents order intensity)
            order_intensity = volume_series.iloc[i]['value'] / base_order_volume
            
            # Apply time-based order patterns
            order_date = timestamp.date()
            order_hour = timestamp.hour
            
            # Determine if this is a repeat customer with time-based patterns
            repeat_probability = self._calculate_time_series_repeat_probability(
                timestamp, customer_order_history, order_intensity
            )
            is_repeat_customer = self.faker.random.random() < repeat_probability
            
            if is_repeat_customer and customer_order_history:
                # Select existing customer with recency bias
                customer_id = self._select_time_series_customer(customer_order_history, timestamp)
                customer_order_history[customer_id]['order_count'] += 1
                customer_order_history[customer_id]['last_order'] = timestamp
            else:
                # New customer
                customer_id = f'CUST_{self.faker.random_int(1, 50000):06d}'
                customer_order_history[customer_id] = {
                    'order_count': 1,
                    'first_order': timestamp,
                    'last_order': timestamp
                }
            
            # Generate order items with time-based trends
            order_items = self._generate_time_series_order_items(
                order_date, timestamp, product_sales_history, order_intensity
            )
            
            # Calculate totals with time-based adjustments
            subtotal = sum(item['price'] * item['quantity'] for item in order_items)
            total_weight = sum(item['weight'] * item['quantity'] for item in order_items)
            
            # Apply time-based pricing adjustments
            subtotal = self._apply_time_series_pricing(subtotal, timestamp, order_intensity)
            
            # Select shipping method with time-based preferences
            shipping_method = self._select_time_series_shipping_method(
                subtotal, total_weight, timestamp, order_intensity
            )
            shipping_cost = self._calculate_shipping_cost(shipping_method, total_weight, subtotal)
            
            # Calculate taxes
            tax_rate = 0.08 if country == 'united_states' else 0.10
            tax_amount = round(subtotal * tax_rate, 2)
            
            total_amount = subtotal + shipping_cost + tax_amount
            
            # Determine order status with time-based logic
            order_status = self._select_time_series_order_status(timestamp)
            
            # Generate payment information with time trends
            payment_method = self._select_time_series_payment_method(timestamp, total_amount)
            
            # Generate shipping address
            shipping_address = self._generate_shipping_address(country)
            
            # Check for cart abandonment with time-based patterns
            abandonment_probability = self._calculate_time_series_abandonment_probability(
                timestamp, order_intensity, total_amount
            )
            is_abandoned = (include_abandoned and 
                          self.faker.random.random() < abandonment_probability)
            
            if is_abandoned:
                order_status = 'abandoned'
                total_amount = 0
                shipping_cost = 0
                tax_amount = 0
            
            # Generate order record
            order = {
                'order_id': f'ORD_{i+1:08d}',
                'customer_id': customer_id,
                'order_date': order_date,
                'order_datetime': timestamp,
                'order_status': order_status,
                'subtotal': subtotal,
                'shipping_cost': shipping_cost,
                'tax_amount': tax_amount,
                'total_amount': total_amount,
                'payment_method': payment_method,
                'shipping_method': shipping_method,
                'estimated_delivery_days': self._get_delivery_estimate(shipping_method),
                'total_items': len(order_items),
                'total_quantity': sum(item['quantity'] for item in order_items),
                'total_weight_kg': total_weight,
                'primary_category': order_items[0]['category'] if order_items else 'unknown',
                'shipping_country': shipping_address['country'],
                'shipping_city': shipping_address['city'],
                'shipping_postal_code': shipping_address['postal_code'],
                'is_repeat_customer': customer_order_history[customer_id]['order_count'] > 1,
                'customer_order_number': customer_order_history[customer_id]['order_count'],
                'is_gift': self._calculate_time_series_gift_probability(timestamp),
                'order_intensity': round(order_intensity, 3)
            }
            
            # Handle discount logic with time-based patterns
            discount_probability = self._calculate_time_series_discount_probability(
                timestamp, order_intensity, is_repeat_customer
            )
            has_discount = self.faker.random.random() < discount_probability
            order['has_discount'] = has_discount
            order['discount_amount'] = round(subtotal * 0.1, 2) if has_discount else 0.0
            
            # Add item details
            for idx, item in enumerate(order_items[:3]):
                order[f'item_{idx+1}_name'] = item['name']
                order[f'item_{idx+1}_category'] = item['category']
                order[f'item_{idx+1}_price'] = item['price']
                order[f'item_{idx+1}_quantity'] = item['quantity']
            
            # Fill empty item slots
            for idx in range(len(order_items), 3):
                order[f'item_{idx+1}_name'] = None
                order[f'item_{idx+1}_category'] = None
                order[f'item_{idx+1}_price'] = None
                order[f'item_{idx+1}_quantity'] = None
            
            data.append(order)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply ecommerce-specific time series correlations
        df = self._apply_ecommerce_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _calculate_time_series_repeat_probability(self, timestamp: datetime, 
                                                customer_history: Dict, order_intensity: float) -> float:
        """Calculate repeat customer probability with time-based patterns"""
        base_probability = self.repeat_customer_rate
        
        # Time-based adjustments
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Evening and weekend orders more likely to be repeat customers
        if 18 <= hour <= 22:
            base_probability *= 1.3
        if day_of_week >= 5:  # Weekend
            base_probability *= 1.2
        
        # Holiday season increases repeat purchases
        if month in [11, 12]:
            base_probability *= 1.4
        
        # Order intensity correlation (busy periods = more repeat customers)
        intensity_multiplier = 0.8 + (order_intensity * 0.4)
        base_probability *= intensity_multiplier
        
        # Customer base growth effect (more customers = higher repeat probability)
        if len(customer_history) > 100:
            base_probability *= 1.1
        
        return min(0.8, base_probability)
    
    def _select_time_series_customer(self, customer_history: Dict, timestamp: datetime) -> str:
        """Select existing customer with recency bias"""
        # Weight customers by recency and order frequency
        weighted_customers = []
        
        for customer_id, data in customer_history.items():
            # Recency weight (more recent customers more likely to order again)
            days_since_last = (timestamp - data['last_order']).days
            recency_weight = max(0.1, 1.0 / (1 + days_since_last * 0.1))
            
            # Frequency weight (frequent customers more likely to order again)
            frequency_weight = min(2.0, 1.0 + data['order_count'] * 0.1)
            
            total_weight = recency_weight * frequency_weight
            weighted_customers.extend([customer_id] * int(total_weight * 10))
        
        return self.faker.random_element(weighted_customers if weighted_customers else list(customer_history.keys()))
    
    def _generate_time_series_order_items(self, order_date: date, timestamp: datetime,
                                        product_history: Dict, order_intensity: float) -> List[Dict[str, Any]]:
        """Generate order items with time-based trends and product correlations"""
        # Determine order size with time-based patterns
        order_size = self._select_time_series_order_size(timestamp, order_intensity)
        
        # Get seasonal preferences
        season = self._get_season(order_date)
        preferred_categories = self.seasonal_product_preferences.get(season, self.product_categories)
        
        items = []
        selected_categories = []
        
        for i in range(order_size):
            if i == 0:
                # First item - consider trending products and seasonal preferences
                category = self._select_time_series_category(
                    preferred_categories, product_history, timestamp
                )
            else:
                # Subsequent items - use correlation patterns
                if selected_categories:
                    last_category = selected_categories[-1]
                    correlated_categories = self.product_correlations.get(last_category, [last_category])
                    category = self.faker.random_element(correlated_categories)
                else:
                    category = self.faker.random_element(self.product_categories)
            
            # Select product from category with trending bias
            product = self._select_time_series_product(category, product_history, timestamp)
            
            # Generate quantity with time-based patterns
            quantity = self._select_time_series_quantity(timestamp, order_intensity)
            
            # Apply time-based pricing
            price_variation = self._calculate_time_series_price_variation(timestamp, order_intensity)
            final_price = round(product['base_price'] * price_variation, 2)
            
            item = {
                'name': product['name'],
                'category': product['category'],
                'price': final_price,
                'quantity': quantity,
                'weight': product['weight_kg']
            }
            
            items.append(item)
            selected_categories.append(category)
            
            # Track product sales for trending
            if product['name'] not in product_history:
                product_history[product['name']] = []
            product_history[product['name']].append({
                'timestamp': timestamp,
                'price': final_price,
                'quantity': quantity
            })
        
        return items
    
    def _select_time_series_order_size(self, timestamp: datetime, order_intensity: float) -> int:
        """Select order size with time-based patterns"""
        base_size = self._select_order_size()
        
        # Time-based adjustments
        hour = timestamp.hour
        month = timestamp.month
        
        # Evening orders tend to be larger (leisure shopping)
        if 18 <= hour <= 22:
            if self.faker.random.random() < 0.3:
                base_size += 1
        
        # Holiday season orders tend to be larger
        if month in [11, 12]:
            if self.faker.random.random() < 0.4:
                base_size += self.faker.random_int(1, 2)
        
        # High intensity periods may have larger orders
        if order_intensity > 1.5:
            if self.faker.random.random() < 0.2:
                base_size += 1
        
        return min(7, max(1, base_size))
    
    def _select_time_series_category(self, preferred_categories: List[str], 
                                   product_history: Dict, timestamp: datetime) -> str:
        """Select category considering trends and time patterns"""
        # Calculate trending categories based on recent sales
        trending_categories = self._calculate_trending_categories(product_history, timestamp)
        
        # Combine seasonal preferences with trending data
        weighted_categories = []
        
        for category in preferred_categories:
            weight = 2  # Base weight for seasonal preference
            
            # Add trending weight
            if category in trending_categories:
                weight += trending_categories[category]
            
            weighted_categories.extend([category] * weight)
        
        return self.faker.random_element(weighted_categories if weighted_categories else preferred_categories)
    
    def _calculate_trending_categories(self, product_history: Dict, timestamp: datetime) -> Dict[str, int]:
        """Calculate trending categories based on recent sales"""
        trending = {}
        cutoff_time = timestamp - timedelta(days=7)  # Look at last 7 days
        
        for product_name, sales_history in product_history.items():
            # Find product category
            product = next((p for p in self.products if p['name'] == product_name), None)
            if not product:
                continue
            
            category = product['category']
            
            # Count recent sales
            recent_sales = [s for s in sales_history if s['timestamp'] >= cutoff_time]
            
            if recent_sales:
                if category not in trending:
                    trending[category] = 0
                trending[category] += len(recent_sales)
        
        return trending
    
    def _select_time_series_product(self, category: str, product_history: Dict, 
                                  timestamp: datetime) -> Dict[str, Any]:
        """Select product with trending bias"""
        category_products = [p for p in self.products if p['category'] == category]
        if not category_products:
            return self.faker.random_element(self.products)
        
        # Weight products by recent sales (trending products)
        weighted_products = []
        
        for product in category_products:
            weight = 1  # Base weight
            
            # Add trending weight
            if product['name'] in product_history:
                cutoff_time = timestamp - timedelta(days=3)  # Recent trend window
                recent_sales = [s for s in product_history[product['name']] 
                              if s['timestamp'] >= cutoff_time]
                if recent_sales:
                    weight += len(recent_sales)
            
            weighted_products.extend([product] * weight)
        
        return self.faker.random_element(weighted_products)
    
    def _select_time_series_quantity(self, timestamp: datetime, order_intensity: float) -> int:
        """Select quantity with time-based patterns"""
        base_quantity = 1
        
        # Higher intensity periods may have higher quantities
        if order_intensity > 1.5 and self.faker.random.random() < 0.3:
            base_quantity += 1
        
        # Holiday season bulk buying
        if timestamp.month in [11, 12] and self.faker.random.random() < 0.25:
            base_quantity += self.faker.random_int(1, 2)
        
        # Weekend bulk shopping
        if timestamp.weekday() >= 5 and self.faker.random.random() < 0.2:
            base_quantity += 1
        
        return min(5, base_quantity)
    
    def _calculate_time_series_price_variation(self, timestamp: datetime, order_intensity: float) -> float:
        """Calculate price variation with time-based patterns"""
        base_variation = self.faker.random.uniform(0.9, 1.1)
        
        # High demand periods (high intensity) = higher prices
        if order_intensity > 1.5:
            base_variation *= self.faker.random.uniform(1.0, 1.15)
        elif order_intensity < 0.7:
            base_variation *= self.faker.random.uniform(0.85, 1.0)
        
        # Holiday season pricing
        if timestamp.month in [11, 12]:
            # Mix of sales and premium pricing
            if self.faker.random.random() < 0.3:  # 30% chance of sale
                base_variation *= self.faker.random.uniform(0.7, 0.9)
            else:
                base_variation *= self.faker.random.uniform(1.0, 1.1)
        
        return base_variation
    
    def _apply_time_series_pricing(self, subtotal: float, timestamp: datetime, 
                                 order_intensity: float) -> float:
        """Apply time-based pricing adjustments to subtotal"""
        # Peak demand pricing
        if order_intensity > 2.0:
            subtotal *= self.faker.random.uniform(1.02, 1.08)
        
        # Holiday season adjustments
        if timestamp.month in [11, 12]:
            # Black Friday / Cyber Monday effects
            if timestamp.month == 11 and timestamp.day >= 20:
                if self.faker.random.random() < 0.4:  # 40% chance of discount
                    subtotal *= self.faker.random.uniform(0.8, 0.95)
        
        return round(subtotal, 2)
    
    def _select_time_series_shipping_method(self, order_value: float, total_weight: float,
                                          timestamp: datetime, order_intensity: float) -> str:
        """Select shipping method with time-based preferences"""
        base_method = self._select_shipping_method(order_value, total_weight)
        
        # Time-based shipping preferences
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Late orders more likely to use express shipping
        if hour >= 20:
            if base_method == 'standard' and self.faker.random.random() < 0.3:
                base_method = 'express'
        
        # Friday orders more likely to use express (weekend delivery)
        if day_of_week == 4:  # Friday
            if base_method == 'standard' and self.faker.random.random() < 0.25:
                base_method = 'express'
        
        # High intensity periods = more express shipping
        if order_intensity > 1.5:
            if base_method == 'standard' and self.faker.random.random() < 0.2:
                base_method = 'express'
        
        return base_method
    
    def _select_time_series_order_status(self, timestamp: datetime) -> str:
        """Select order status with time-based logic"""
        days_ago = (datetime.now() - timestamp).days
        
        # Same logic as base method but with time-aware adjustments
        if days_ago < 1:
            status_dist = {'processing': 0.6, 'shipped': 0.3, 'delivered': 0.05, 'cancelled': 0.05}
        elif days_ago < 3:
            status_dist = {'processing': 0.2, 'shipped': 0.5, 'delivered': 0.25, 'cancelled': 0.05}
        elif days_ago < 7:
            status_dist = {'processing': 0.05, 'shipped': 0.2, 'delivered': 0.7, 'cancelled': 0.03, 'returned': 0.02}
        else:
            status_dist = {'delivered': 0.85, 'cancelled': 0.05, 'returned': 0.1}
        
        return self._weighted_choice(status_dist)
    
    def _select_time_series_payment_method(self, timestamp: datetime, total_amount: float) -> str:
        """Select payment method with time-based trends"""
        base_method = self._select_payment_method()
        
        # Time-based payment trends
        year = timestamp.year
        
        # Digital payment adoption over time
        if year >= 2020:
            if base_method == 'credit_card' and self.faker.random.random() < 0.3:
                base_method = self.faker.random_element(['paypal', 'apple_pay', 'google_pay'])
        
        # High-value orders more likely to use secure methods
        if total_amount > 500:
            if base_method in ['apple_pay', 'google_pay'] and self.faker.random.random() < 0.4:
                base_method = 'credit_card'
        
        return base_method
    
    def _calculate_time_series_abandonment_probability(self, timestamp: datetime,
                                                     order_intensity: float, total_amount: float) -> float:
        """Calculate cart abandonment probability with time-based patterns"""
        base_probability = self.cart_abandonment_rate
        
        # Time-based adjustments
        hour = timestamp.hour
        
        # Late night orders more likely to be abandoned
        if hour >= 23 or hour <= 5:
            base_probability *= 1.5
        
        # High-value orders more likely to be abandoned
        if total_amount > 200:
            base_probability *= 1.3
        elif total_amount > 500:
            base_probability *= 1.6
        
        # High intensity periods = lower abandonment (urgency effect)
        if order_intensity > 1.5:
            base_probability *= 0.8
        
        return min(0.5, base_probability)
    
    def _calculate_time_series_gift_probability(self, timestamp: datetime) -> bool:
        """Calculate gift probability with seasonal patterns"""
        base_probability = 0.12
        
        # Seasonal adjustments
        month = timestamp.month
        day = timestamp.day
        
        # Holiday season
        if month == 12:
            base_probability = 0.4
        elif month == 11:
            base_probability = 0.25
        elif month == 2 and day <= 14:  # Valentine's Day
            base_probability = 0.3
        elif month == 5 and day >= 8 and day <= 14:  # Mother's Day
            base_probability = 0.35
        
        return self.faker.random.random() < base_probability
    
    def _calculate_time_series_discount_probability(self, timestamp: datetime,
                                                  order_intensity: float, is_repeat: bool) -> float:
        """Calculate discount probability with time-based patterns"""
        base_probability = 0.25
        
        # Time-based discount patterns
        month = timestamp.month
        day_of_week = timestamp.weekday()
        
        # Holiday season discounts
        if month in [11, 12]:
            base_probability = 0.45
        
        # Weekend promotions
        if day_of_week >= 5:
            base_probability *= 1.2
        
        # Low intensity periods = more discounts to drive sales
        if order_intensity < 0.8:
            base_probability *= 1.3
        
        # Repeat customer discounts
        if is_repeat:
            base_probability *= 1.1
        
        return min(0.6, base_probability)
    
    def _weighted_choice(self, choices_dict: Dict[str, float]) -> str:
        """Make weighted choice from dictionary"""
        choices = list(choices_dict.keys())
        weights = list(choices_dict.values())
        
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
    
    def _apply_ecommerce_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply ecommerce-specific time series correlations"""
        if len(data) < 2:
            return data
        
        # Sort by order datetime to ensure proper time series order
        data = data.sort_values('order_datetime').reset_index(drop=True)
        
        # Apply order intensity persistence (viral/promotional effects)
        if 'order_intensity' in data.columns:
            for i in range(1, len(data)):
                prev_intensity = data.loc[i-1, 'order_intensity']
                curr_intensity = data.loc[i, 'order_intensity']
                
                # Apply intensity correlation (promotional campaigns persist)
                intensity_persistence = 0.4
                correlated_intensity = (prev_intensity * intensity_persistence + 
                                      curr_intensity * (1 - intensity_persistence))
                
                data.loc[i, 'order_intensity'] = round(correlated_intensity, 3)
        
        # Apply customer behavior correlation (repeat customers cluster)
        for i in range(1, len(data)):
            prev_customer = data.loc[i-1, 'customer_id']
            curr_customer = data.loc[i, 'customer_id']
            
            # Check if orders are close in time (within 1 hour)
            time_diff = (data.loc[i, 'order_datetime'] - 
                        data.loc[i-1, 'order_datetime']).total_seconds()
            
            if time_diff <= 3600:  # Within 1 hour
                # Apply customer correlation (word-of-mouth effects)
                if prev_customer != curr_customer and self.faker.random.random() < 0.15:
                    # Similar order patterns for customers ordering at similar times
                    prev_category = data.loc[i-1, 'primary_category']
                    if self.faker.random.random() < 0.3:
                        data.loc[i, 'primary_category'] = prev_category
        
        # Apply product category clustering (trending products)
        category_momentum = {}
        for i in range(len(data)):
            category = data.loc[i, 'primary_category']
            timestamp = data.loc[i, 'order_datetime']
            
            # Track category momentum
            if category not in category_momentum:
                category_momentum[category] = []
            category_momentum[category].append(timestamp)
            
            # Apply momentum effect (trending categories continue to trend)
            if len(category_momentum[category]) > 3:
                recent_orders = [t for t in category_momentum[category] 
                               if (timestamp - t).total_seconds() <= 7200]  # Last 2 hours
                
                if len(recent_orders) >= 3:  # Category is trending
                    # Boost order amount for trending categories
                    current_amount = data.loc[i, 'total_amount']
                    trending_boost = 1.05  # 5% boost
                    data.loc[i, 'total_amount'] = round(current_amount * trending_boost, 2)
        
        return data
    
    def _select_order_hour(self) -> int:
        """Select order hour based on realistic patterns"""
        choices = list(self.hourly_order_patterns.keys())
        weights = list(self.hourly_order_patterns.values())
        
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
    
    def _generate_order_items(self, order_date: date) -> List[Dict[str, Any]]:
        """Generate items for an order with realistic correlations"""
        # Determine order size
        order_size = self._select_order_size()
        
        # Get seasonal preferences
        season = self._get_season(order_date)
        preferred_categories = self.seasonal_product_preferences.get(season, self.product_categories)
        
        items = []
        selected_categories = []
        
        for i in range(order_size):
            if i == 0:
                # First item - select from seasonal preferences
                category = self.faker.random_element(preferred_categories)
            else:
                # Subsequent items - use correlation patterns
                if selected_categories:
                    last_category = selected_categories[-1]
                    correlated_categories = self.product_correlations.get(last_category, [last_category])
                    category = self.faker.random_element(correlated_categories)
                else:
                    category = self.faker.random_element(self.product_categories)
            
            # Select product from category
            category_products = [p for p in self.products if p['category'] == category]
            if not category_products:
                category_products = self.products[:10]  # Fallback
            
            product = self.faker.random_element(category_products)
            
            # Generate quantity (most orders have quantity 1, some have more)
            quantity = 1
            if self.faker.random.random() < 0.2:  # 20% chance of multiple quantity
                quantity = self.faker.random_int(2, 5)
            
            # Add some price variation
            price_variation = self.faker.random.uniform(0.9, 1.1)
            final_price = round(product['base_price'] * price_variation, 2)
            
            item = {
                'name': product['name'],
                'category': product['category'],
                'price': final_price,
                'quantity': quantity,
                'weight': product['weight_kg']
            }
            
            items.append(item)
            selected_categories.append(category)
        
        return items
    
    def _select_order_size(self) -> int:
        """Select number of items in order"""
        choices = list(self.order_size_distribution.keys())
        weights = list(self.order_size_distribution.values())
        
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
    
    def _get_season(self, order_date: date) -> str:
        """Determine season based on order date"""
        month = order_date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _select_shipping_method(self, order_value: float, total_weight: float) -> str:
        """Select shipping method based on order value and weight"""
        # Determine order value category
        if order_value < 50:
            value_category = 'low'
        elif order_value < 200:
            value_category = 'medium'
        else:
            value_category = 'high'
        
        # Get distribution for this value category
        distribution = self.shipping_by_order_value[value_category]
        
        # Filter by weight limits
        available_methods = {}
        for method, probability in distribution.items():
            weight_limit = self.shipping_methods[method]['weight_limit']
            if total_weight <= weight_limit:
                available_methods[method] = probability
        
        # If no methods available due to weight, use standard
        if not available_methods:
            return 'standard'
        
        # Normalize probabilities
        total_prob = sum(available_methods.values())
        for method in available_methods:
            available_methods[method] /= total_prob
        
        # Select method
        choices = list(available_methods.keys())
        weights = list(available_methods.values())
        
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
    
    def _calculate_shipping_cost(self, shipping_method: str, total_weight: float, order_value: float) -> float:
        """Calculate shipping cost based on method, weight, and order value"""
        if shipping_method == 'free_shipping' or order_value > 100:  # Free shipping over $100
            return 0.0
        
        base_cost = 5.99  # Base shipping cost
        method_multiplier = self.shipping_methods[shipping_method]['cost_multiplier']
        weight_cost = max(0, (total_weight - 1.0) * 2.0)  # $2 per kg over 1kg
        
        total_cost = (base_cost + weight_cost) * method_multiplier
        return round(total_cost, 2)
    
    def _get_delivery_estimate(self, shipping_method: str) -> int:
        """Get delivery estimate for shipping method"""
        min_days, max_days = self.shipping_methods[shipping_method]['delivery_days']
        return self.faker.random_int(min_days, max_days)
    
    def _select_order_status(self, order_datetime: datetime) -> str:
        """Select order status based on how recent the order is"""
        days_ago = (datetime.now() - order_datetime).days
        
        # Recent orders are more likely to be in processing/shipped
        if days_ago < 1:
            status_dist = {'processing': 0.6, 'shipped': 0.3, 'delivered': 0.05, 'cancelled': 0.05}
        elif days_ago < 3:
            status_dist = {'processing': 0.2, 'shipped': 0.5, 'delivered': 0.25, 'cancelled': 0.05}
        elif days_ago < 7:
            status_dist = {'processing': 0.05, 'shipped': 0.2, 'delivered': 0.7, 'cancelled': 0.03, 'returned': 0.02}
        else:
            status_dist = {'delivered': 0.85, 'cancelled': 0.05, 'returned': 0.1}
        
        choices = list(status_dist.keys())
        weights = list(status_dist.values())
        
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
    
    def _select_payment_method(self) -> str:
        """Select payment method based on distribution"""
        choices = list(self.payment_methods.keys())
        weights = list(self.payment_methods.values())
        
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
    
    def _generate_shipping_address(self, country: str) -> Dict[str, str]:
        """Generate shipping address"""
        return {
            'country': country if country != 'global' else self.faker.country(),
            'city': self.faker.city(),
            'postal_code': self.faker.postcode()
        }
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Add customer satisfaction score based on order experience
        data['customer_satisfaction'] = data.apply(
            lambda row: self._calculate_satisfaction_score(row), axis=1
        ).round(1)
        
        # Add delivery performance score
        data['delivery_performance'] = data.apply(
            lambda row: self._calculate_delivery_performance(row), axis=1
        ).round(1)
        
        # Add return likelihood
        data['return_likelihood'] = data.apply(
            lambda row: self._calculate_return_likelihood(row), axis=1
        ).round(2)
        
        # Add customer lifetime value estimate
        data['estimated_clv'] = data.apply(
            lambda row: self._estimate_customer_lifetime_value(row), axis=1
        ).round(2)
        
        # Sort by order date for realistic chronological order
        data = data.sort_values('order_datetime').reset_index(drop=True)
        
        return data
    
    def _calculate_satisfaction_score(self, row) -> float:
        """Calculate customer satisfaction score"""
        base_score = 4.0
        
        # Shipping method impact
        if row['shipping_method'] == 'overnight':
            base_score += 0.5
        elif row['shipping_method'] == 'express':
            base_score += 0.2
        elif row['shipping_method'] == 'free_shipping':
            base_score += 0.3
        
        # Order status impact
        if row['order_status'] == 'delivered':
            base_score += 0.3
        elif row['order_status'] == 'cancelled':
            base_score -= 1.5
        elif row['order_status'] == 'returned':
            base_score -= 1.0
        
        # Discount impact
        if row['has_discount']:
            base_score += 0.2
        
        # Add randomness
        base_score += self.faker.random.gauss(0, 0.3)
        
        return max(1.0, min(5.0, base_score))
    
    def _calculate_delivery_performance(self, row) -> float:
        """Calculate delivery performance score"""
        base_score = 4.0
        
        # Shipping method reliability
        method_scores = {
            'overnight': 4.8,
            'express': 4.5,
            'standard': 4.0,
            'free_shipping': 3.5
        }
        
        base_score = method_scores.get(row['shipping_method'], 4.0)
        
        # Order status impact
        if row['order_status'] == 'delivered':
            base_score += 0.2
        elif row['order_status'] in ['cancelled', 'returned']:
            base_score -= 1.0
        
        # Add randomness
        base_score += self.faker.random.gauss(0, 0.2)
        
        return max(1.0, min(5.0, base_score))
    
    def _calculate_return_likelihood(self, row) -> float:
        """Calculate return likelihood based on order characteristics"""
        base_likelihood = 0.05  # 5% base return rate
        
        # Category impact
        if row['primary_category'] == 'retail':
            base_likelihood += 0.10  # Clothing has higher return rates
        elif row['primary_category'] == 'technology':
            base_likelihood += 0.03
        
        # Order value impact (expensive items more likely to be returned)
        if row['total_amount'] > 200:
            base_likelihood += 0.05
        elif row['total_amount'] > 500:
            base_likelihood += 0.10
        
        # Multiple items increase return likelihood
        if row['total_items'] > 3:
            base_likelihood += 0.03
        
        # Add randomness
        base_likelihood += self.faker.random.gauss(0, 0.02)
        
        return max(0.0, min(1.0, base_likelihood))
    
    def _estimate_customer_lifetime_value(self, row) -> float:
        """Estimate customer lifetime value"""
        base_clv = row['total_amount'] * 5  # Assume 5x current order value
        
        # Repeat customer multiplier
        if row['is_repeat_customer']:
            base_clv *= 1.5
        
        # Order frequency impact
        if row['customer_order_number'] > 5:
            base_clv *= 1.3
        
        # Satisfaction impact
        satisfaction_multiplier = row['customer_satisfaction'] / 4.0
        base_clv *= satisfaction_multiplier
        
        # Add some randomness
        base_clv *= self.faker.random.uniform(0.8, 1.2)
        
        return max(50.0, base_clv)