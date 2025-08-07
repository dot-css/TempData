"""
Inventory dataset generator

Generates realistic inventory and stock management data with SKUs, quantities,
reorder points, warehouse locations, movement history, and realistic demand patterns.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
from ...core.base_generator import BaseGenerator


class InventoryGenerator(BaseGenerator):
    """
    Generator for realistic inventory and stock management data
    
    Creates inventory datasets with product SKUs, stock levels, warehouse locations,
    movement history, and realistic demand patterns with seasonal variations.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_product_categories()
        self._setup_warehouse_data()
        self._setup_demand_patterns()
        self._setup_supplier_data()
        self._setup_abc_analysis()
        self._setup_movement_types()
    
    def _setup_product_categories(self):
        """Setup product categories and characteristics"""
        self.product_categories = {
            'electronics': {
                'weight': 0.20,
                'avg_unit_cost': (50, 2000),
                'shelf_life_days': None,  # No expiration
                'storage_requirements': 'climate_controlled',
                'demand_volatility': 0.25,
                'seasonal_factor': 1.2  # Higher during holidays
            },
            'clothing': {
                'weight': 0.18,
                'avg_unit_cost': (15, 300),
                'shelf_life_days': None,
                'storage_requirements': 'standard',
                'demand_volatility': 0.35,
                'seasonal_factor': 1.4  # Very seasonal
            },
            'food_beverage': {
                'weight': 0.15,
                'avg_unit_cost': (2, 50),
                'shelf_life_days': (30, 365),
                'storage_requirements': 'temperature_controlled',
                'demand_volatility': 0.15,
                'seasonal_factor': 1.1
            },
            'home_garden': {
                'weight': 0.12,
                'avg_unit_cost': (10, 500),
                'shelf_life_days': None,
                'storage_requirements': 'standard',
                'demand_volatility': 0.30,
                'seasonal_factor': 1.3  # Spring/summer peak
            },
            'automotive': {
                'weight': 0.10,
                'avg_unit_cost': (25, 1500),
                'shelf_life_days': None,
                'storage_requirements': 'standard',
                'demand_volatility': 0.20,
                'seasonal_factor': 1.0
            },
            'health_beauty': {
                'weight': 0.08,
                'avg_unit_cost': (5, 200),
                'shelf_life_days': (180, 1095),  # 6 months to 3 years
                'storage_requirements': 'climate_controlled',
                'demand_volatility': 0.18,
                'seasonal_factor': 1.1
            },
            'sports_outdoors': {
                'weight': 0.07,
                'avg_unit_cost': (20, 800),
                'shelf_life_days': None,
                'storage_requirements': 'standard',
                'demand_volatility': 0.40,
                'seasonal_factor': 1.5  # Very seasonal
            },
            'books_media': {
                'weight': 0.05,
                'avg_unit_cost': (8, 100),
                'shelf_life_days': None,
                'storage_requirements': 'standard',
                'demand_volatility': 0.25,
                'seasonal_factor': 1.2
            },
            'office_supplies': {
                'weight': 0.05,
                'avg_unit_cost': (3, 150),
                'shelf_life_days': None,
                'storage_requirements': 'standard',
                'demand_volatility': 0.12,
                'seasonal_factor': 1.1
            }
        }
    
    def _setup_warehouse_data(self):
        """Setup warehouse locations and characteristics"""
        self.warehouses = {
            'main_distribution': {
                'weight': 0.40,
                'capacity_units': (50000, 100000),
                'location_type': 'distribution_center',
                'automation_level': 'high',
                'operating_cost_per_unit': 2.50
            },
            'regional_hub': {
                'weight': 0.25,
                'capacity_units': (20000, 50000),
                'location_type': 'regional_warehouse',
                'automation_level': 'medium',
                'operating_cost_per_unit': 3.00
            },
            'local_store': {
                'weight': 0.20,
                'capacity_units': (5000, 20000),
                'location_type': 'retail_store',
                'automation_level': 'low',
                'operating_cost_per_unit': 4.00
            },
            'specialty_storage': {
                'weight': 0.10,
                'capacity_units': (10000, 30000),
                'location_type': 'specialty_facility',
                'automation_level': 'medium',
                'operating_cost_per_unit': 5.00
            },
            'overflow_storage': {
                'weight': 0.05,
                'capacity_units': (30000, 80000),
                'location_type': 'overflow_facility',
                'automation_level': 'low',
                'operating_cost_per_unit': 1.80
            }
        }
        
        # Geographic regions for warehouses
        self.warehouse_regions = {
            'northeast': ['New York', 'Boston', 'Philadelphia', 'Newark'],
            'southeast': ['Atlanta', 'Miami', 'Charlotte', 'Jacksonville'],
            'midwest': ['Chicago', 'Detroit', 'Cleveland', 'Indianapolis'],
            'southwest': ['Dallas', 'Houston', 'Phoenix', 'San Antonio'],
            'west': ['Los Angeles', 'San Francisco', 'Seattle', 'Portland']
        }
    
    def _setup_demand_patterns(self):
        """Setup realistic demand patterns and seasonality"""
        # Monthly demand multipliers (1.0 = average)
        self.seasonal_demand = {
            1: 0.85,  # January - post-holiday low
            2: 0.90,  # February
            3: 1.05,  # March - spring preparation
            4: 1.10,  # April
            5: 1.15,  # May - spring peak
            6: 1.05,  # June
            7: 1.00,  # July
            8: 0.95,  # August
            9: 1.10,  # September - back to school/work
            10: 1.15,  # October - preparation for holidays
            11: 1.35,  # November - Black Friday, Thanksgiving
            12: 1.40   # December - holiday peak
        }
        
        # Category-specific seasonal patterns
        self.category_seasonality = {
            'clothing': {
                3: 1.4, 4: 1.3, 8: 1.5, 9: 1.4, 11: 1.8, 12: 1.6
            },
            'sports_outdoors': {
                3: 1.6, 4: 1.8, 5: 1.7, 6: 1.5, 7: 1.4, 8: 1.3
            },
            'home_garden': {
                3: 1.8, 4: 2.0, 5: 1.9, 6: 1.4, 7: 1.2, 8: 1.1
            },
            'electronics': {
                11: 1.6, 12: 1.9, 1: 0.7
            }
        }
        
        # Weekly patterns (Monday = 0, Sunday = 6)
        self.weekly_demand = {
            0: 1.2,  # Monday - restocking after weekend
            1: 1.1,  # Tuesday
            2: 1.0,  # Wednesday
            3: 1.0,  # Thursday
            4: 1.1,  # Friday
            5: 0.9,  # Saturday
            6: 0.7   # Sunday
        }
    
    def _setup_supplier_data(self):
        """Setup supplier relationships and lead times"""
        self.supplier_profiles = {
            'domestic_premium': {
                'weight': 0.25,
                'lead_time_days': (3, 7),
                'reliability_score': (90, 98),
                'cost_multiplier': 1.15
            },
            'domestic_standard': {
                'weight': 0.35,
                'lead_time_days': (5, 14),
                'reliability_score': (80, 92),
                'cost_multiplier': 1.0
            },
            'international_premium': {
                'weight': 0.15,
                'lead_time_days': (14, 28),
                'reliability_score': (85, 95),
                'cost_multiplier': 0.85
            },
            'international_standard': {
                'weight': 0.20,
                'lead_time_days': (21, 45),
                'reliability_score': (70, 85),
                'cost_multiplier': 0.70
            },
            'local_supplier': {
                'weight': 0.05,
                'lead_time_days': (1, 3),
                'reliability_score': (75, 90),
                'cost_multiplier': 1.25
            }
        }
    
    def _setup_abc_analysis(self):
        """Setup ABC analysis patterns for inventory classification"""
        self.abc_classification = {
            'A': {
                'weight': 0.20,  # 20% of items
                'value_contribution': 0.80,  # 80% of value
                'turnover_rate': (8, 20),  # High turnover
                'service_level': (95, 99),
                'safety_stock_days': (3, 7)
            },
            'B': {
                'weight': 0.30,  # 30% of items
                'value_contribution': 0.15,  # 15% of value
                'turnover_rate': (4, 8),  # Medium turnover
                'service_level': (90, 95),
                'safety_stock_days': (7, 14)
            },
            'C': {
                'weight': 0.50,  # 50% of items
                'value_contribution': 0.05,  # 5% of value
                'turnover_rate': (1, 4),  # Low turnover
                'service_level': (80, 90),
                'safety_stock_days': (14, 30)
            }
        }
    
    def _setup_movement_types(self):
        """Setup inventory movement types and patterns"""
        self.movement_types = {
            'inbound': {
                'receipt': 0.40,
                'return_from_customer': 0.05,
                'transfer_in': 0.15,
                'adjustment_positive': 0.03
            },
            'outbound': {
                'sale': 0.60,
                'return_to_supplier': 0.02,
                'transfer_out': 0.15,
                'shrinkage': 0.03,
                'damage': 0.02,
                'adjustment_negative': 0.02
            }
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate inventory dataset
        
        Args:
            rows: Number of inventory records to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated inventory data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_inventory(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_inventory(rows, **kwargs)
    
    def _generate_snapshot_inventory(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot inventory data"""
        data = []
        
        for i in range(rows):
            inventory_record = self._generate_inventory_record(i, **kwargs)
            data.append(inventory_record)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_inventory(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series inventory data"""
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Create base time series for inventory activity
        base_activity_rate = 1.0  # Base inventory activity level
        
        activity_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_activity_rate,
            value_range=(base_activity_rate * 0.5, base_activity_rate * 2.0)
        )
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(activity_series):
                break
            
            # Get time series activity intensity
            activity_intensity = activity_series.iloc[i]['value'] / base_activity_rate
            
            # Generate inventory record with temporal patterns
            inventory_record = self._generate_time_series_inventory_record(
                i, timestamp, activity_intensity, **kwargs
            )
            
            data.append(inventory_record)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply inventory-specific time series correlations
        df = self._apply_inventory_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_inventory_record(self, record_index: int, **kwargs) -> Dict:
        """Generate a single inventory record"""
        
        # Select product category and ABC classification
        category = self._select_product_category()
        abc_class = self._select_abc_classification()
        
        # Get category and ABC configurations
        category_config = self.product_categories[category]
        abc_config = self.abc_classification[abc_class]
        
        # Generate product information
        product_name = self._generate_product_name(category)
        sku = self._generate_sku(category, record_index)
        
        # Generate cost information
        cost_min, cost_max = category_config['avg_unit_cost']
        unit_cost = round(self.faker.random.uniform(cost_min, cost_max), 2)
        
        # Generate warehouse and location
        warehouse_type = self._select_warehouse_type()
        warehouse_config = self.warehouses[warehouse_type]
        warehouse_location = self._generate_warehouse_location(warehouse_type)
        
        # Generate demand and turnover patterns
        turnover_min, turnover_max = abc_config['turnover_rate']
        annual_turnover = self.faker.random.uniform(turnover_min, turnover_max)
        
        # Calculate average daily demand
        avg_daily_demand = round(annual_turnover * unit_cost / 365, 2)
        
        # Generate current stock levels
        safety_stock_min, safety_stock_max = abc_config['safety_stock_days']
        safety_stock_days = self.faker.random.uniform(safety_stock_min, safety_stock_max)
        safety_stock_quantity = max(1, int(avg_daily_demand * safety_stock_days))
        
        # Calculate reorder point and quantity
        supplier_type = self._select_supplier_type()
        supplier_config = self.supplier_profiles[supplier_type]
        lead_time_min, lead_time_max = supplier_config['lead_time_days']
        lead_time_days = self.faker.random.uniform(lead_time_min, lead_time_max)
        
        reorder_point = max(1, int((avg_daily_demand * lead_time_days) + safety_stock_quantity))
        
        # Economic Order Quantity (EOQ) calculation
        ordering_cost = self.faker.random.uniform(50, 200)  # Cost per order
        holding_cost_rate = self.faker.random.uniform(0.15, 0.25)  # Annual holding cost rate
        
        if avg_daily_demand > 0:
            annual_demand = avg_daily_demand * 365
            eoq = int((2 * annual_demand * ordering_cost / (unit_cost * holding_cost_rate)) ** 0.5)
            reorder_quantity = max(10, eoq)
        else:
            reorder_quantity = safety_stock_quantity * 2
        
        # Generate current stock level
        current_stock = self._generate_current_stock_level(
            reorder_point, reorder_quantity, abc_class
        )
        
        # Generate supplier information
        supplier_id = f'SUP_{self.faker.random_int(1, 500):03d}'
        reliability_min, reliability_max = supplier_config['reliability_score']
        supplier_reliability = round(self.faker.random.uniform(reliability_min, reliability_max), 1)
        
        # Generate dates
        last_received_date = self.faker.date_between(start_date='-90d', end_date='today')
        last_sold_date = self.faker.date_between(start_date='-30d', end_date='today')
        
        # Generate expiration date if applicable
        expiration_date = None
        if category_config['shelf_life_days']:
            shelf_life_min, shelf_life_max = category_config['shelf_life_days']
            shelf_life = self.faker.random_int(shelf_life_min, shelf_life_max)
            expiration_date = last_received_date + timedelta(days=shelf_life)
        
        # Calculate inventory metrics
        stock_value = round(current_stock * unit_cost, 2)
        days_of_supply = round(current_stock / max(avg_daily_demand, 0.1), 1)
        
        # Generate location details
        bin_location = self._generate_bin_location(warehouse_type)
        
        return {
            'sku': sku,
            'product_name': product_name,
            'category': category,
            'abc_classification': abc_class,
            'current_stock': current_stock,
            'unit_cost': unit_cost,
            'stock_value': stock_value,
            'reorder_point': reorder_point,
            'reorder_quantity': reorder_quantity,
            'safety_stock_quantity': safety_stock_quantity,
            'warehouse_id': f'WH_{warehouse_type.upper()[:3]}_{self.faker.random_int(1, 10):02d}',
            'warehouse_type': warehouse_type,
            'warehouse_location': warehouse_location,
            'bin_location': bin_location,
            'storage_requirements': category_config['storage_requirements'],
            'supplier_id': supplier_id,
            'supplier_type': supplier_type,
            'supplier_reliability': supplier_reliability,
            'lead_time_days': round(lead_time_days, 1),
            'avg_daily_demand': avg_daily_demand,
            'annual_turnover': round(annual_turnover, 2),
            'days_of_supply': days_of_supply,
            'last_received_date': last_received_date,
            'last_sold_date': last_sold_date,
            'expiration_date': expiration_date,
            'expired': expiration_date and expiration_date < datetime.now().date(),
            'stock_status': self._determine_stock_status(current_stock, reorder_point, safety_stock_quantity),
            'demand_volatility': category_config['demand_volatility'],
            'seasonal_factor': category_config['seasonal_factor'],
            'holding_cost_annual': round(stock_value * holding_cost_rate, 2),
            'ordering_cost': ordering_cost,
            'service_level_target': round(self.faker.random.uniform(*abc_config['service_level']), 1),
            'created_date': self.faker.date_between(start_date='-2y', end_date='-30d'),
            'last_updated': self.faker.date_between(start_date='-7d', end_date='today')
        }
    
    def _generate_time_series_inventory_record(self, record_index: int, 
                                             timestamp: datetime, 
                                             activity_intensity: float, 
                                             **kwargs) -> Dict:
        """Generate time series inventory record with temporal patterns"""
        
        # Generate base inventory record
        inventory_record = self._generate_inventory_record(record_index, **kwargs)
        
        # Apply temporal patterns
        seasonal_mult = self._get_seasonal_multiplier(
            inventory_record['category'], timestamp.month
        )
        
        weekly_mult = self.weekly_demand.get(timestamp.weekday(), 1.0)
        
        # Adjust demand based on temporal patterns
        base_demand = inventory_record['avg_daily_demand']
        adjusted_demand = base_demand * seasonal_mult * weekly_mult * activity_intensity
        inventory_record['avg_daily_demand'] = round(adjusted_demand, 2)
        
        # Recalculate dependent fields
        inventory_record['days_of_supply'] = round(
            inventory_record['current_stock'] / max(adjusted_demand, 0.1), 1
        )
        
        # Add time series specific fields
        inventory_record.update({
            'timestamp': timestamp,
            'seasonal_multiplier': round(seasonal_mult, 3),
            'weekly_multiplier': round(weekly_mult, 3),
            'activity_intensity': round(activity_intensity, 3),
            'demand_forecast': round(adjusted_demand * 30, 1)  # 30-day forecast
        })
        
        return inventory_record
    
    def _select_product_category(self) -> str:
        """Select product category based on distribution"""
        choices = list(self.product_categories.keys())
        weights = [self.product_categories[cat]['weight'] for cat in choices]
        return self._weighted_choice(choices, weights)
    
    def _select_abc_classification(self) -> str:
        """Select ABC classification based on distribution"""
        choices = list(self.abc_classification.keys())
        weights = [self.abc_classification[abc]['weight'] for abc in choices]
        return self._weighted_choice(choices, weights)
    
    def _select_warehouse_type(self) -> str:
        """Select warehouse type based on distribution"""
        choices = list(self.warehouses.keys())
        weights = [self.warehouses[wh]['weight'] for wh in choices]
        return self._weighted_choice(choices, weights)
    
    def _select_supplier_type(self) -> str:
        """Select supplier type based on distribution"""
        choices = list(self.supplier_profiles.keys())
        weights = [self.supplier_profiles[sp]['weight'] for sp in choices]
        return self._weighted_choice(choices, weights)
    
    def _generate_product_name(self, category: str) -> str:
        """Generate realistic product names based on category"""
        category_products = {
            'electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Smart Watch', 'Camera', 'Speaker'],
            'clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sneakers', 'Sweater', 'Pants'],
            'food_beverage': ['Organic Pasta', 'Premium Coffee', 'Energy Drink', 'Protein Bar', 'Olive Oil'],
            'home_garden': ['Garden Hose', 'Power Drill', 'Lawn Mower', 'Paint Brush', 'Fertilizer'],
            'automotive': ['Motor Oil', 'Brake Pads', 'Air Filter', 'Spark Plugs', 'Car Battery'],
            'health_beauty': ['Shampoo', 'Face Cream', 'Vitamin C', 'Toothpaste', 'Perfume'],
            'sports_outdoors': ['Running Shoes', 'Yoga Mat', 'Camping Tent', 'Bicycle', 'Fishing Rod'],
            'books_media': ['Novel', 'Cookbook', 'DVD Movie', 'Video Game', 'Magazine'],
            'office_supplies': ['Printer Paper', 'Pen Set', 'Stapler', 'Notebook', 'Desk Lamp']
        }
        
        base_products = category_products.get(category, ['Generic Product'])
        base_name = self.faker.random_element(base_products)
        
        # Add brand/model variation
        brands = ['Pro', 'Max', 'Elite', 'Premium', 'Standard', 'Classic', 'Deluxe']
        if self.faker.random.random() < 0.6:
            brand = self.faker.random_element(brands)
            return f"{brand} {base_name}"
        else:
            return base_name
    
    def _generate_sku(self, category: str, index: int) -> str:
        """Generate SKU based on category and index"""
        category_codes = {
            'electronics': 'ELC',
            'clothing': 'CLT',
            'food_beverage': 'FDB',
            'home_garden': 'HGD',
            'automotive': 'AUT',
            'health_beauty': 'HBT',
            'sports_outdoors': 'SPT',
            'books_media': 'BKM',
            'office_supplies': 'OFC'
        }
        
        code = category_codes.get(category, 'GEN')
        return f"{code}-{index+1:06d}"
    
    def _generate_warehouse_location(self, warehouse_type: str) -> str:
        """Generate warehouse location based on type"""
        region = self.faker.random_element(list(self.warehouse_regions.keys()))
        city = self.faker.random_element(self.warehouse_regions[region])
        return f"{city}, {region.title()}"
    
    def _generate_bin_location(self, warehouse_type: str) -> str:
        """Generate bin location within warehouse"""
        if warehouse_type in ['main_distribution', 'regional_hub']:
            # Large warehouses have structured bin locations
            aisle = self.faker.random_element(['A', 'B', 'C', 'D', 'E'])
            section = self.faker.random_int(1, 20)
            level = self.faker.random_int(1, 5)
            position = self.faker.random_int(1, 10)
            return f"{aisle}{section:02d}-{level}-{position:02d}"
        else:
            # Smaller locations have simpler systems
            zone = self.faker.random_element(['F', 'B', 'R'])  # Front, Back, Receiving
            shelf = self.faker.random_int(1, 50)
            return f"{zone}{shelf:02d}"
    
    def _generate_current_stock_level(self, reorder_point: int, reorder_quantity: int, abc_class: str) -> int:
        """Generate realistic current stock level"""
        # Stock levels vary based on ABC classification and reorder patterns
        if abc_class == 'A':
            # A items are managed more tightly
            if self.faker.random.random() < 0.7:
                # Usually well-stocked
                return self.faker.random_int(reorder_point, reorder_point + reorder_quantity)
            else:
                # Sometimes running low
                return self.faker.random_int(0, reorder_point)
        elif abc_class == 'B':
            # B items have moderate management
            if self.faker.random.random() < 0.6:
                return self.faker.random_int(reorder_point, reorder_point + reorder_quantity)
            else:
                return self.faker.random_int(0, reorder_point)
        else:
            # C items may have excess stock or stockouts
            if self.faker.random.random() < 0.4:
                # Sometimes overstocked
                return self.faker.random_int(reorder_point + reorder_quantity, 
                                           reorder_point + reorder_quantity * 2)
            elif self.faker.random.random() < 0.3:
                # Sometimes understocked
                return self.faker.random_int(0, reorder_point // 2)
            else:
                return self.faker.random_int(reorder_point, reorder_point + reorder_quantity)
    
    def _determine_stock_status(self, current_stock: int, reorder_point: int, safety_stock: int) -> str:
        """Determine stock status based on levels"""
        if current_stock == 0:
            return 'out_of_stock'
        elif current_stock <= safety_stock:
            return 'critical'
        elif current_stock <= reorder_point:
            return 'reorder_needed'
        elif current_stock > reorder_point * 2:
            return 'overstock'
        else:
            return 'normal'
    
    def _get_seasonal_multiplier(self, category: str, month: int) -> float:
        """Get seasonal demand multiplier for category and month"""
        base_multiplier = self.seasonal_demand.get(month, 1.0)
        
        if category in self.category_seasonality:
            category_multiplier = self.category_seasonality[category].get(month, 1.0)
        else:
            category_multiplier = 1.0
        
        return base_multiplier * category_multiplier
    
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
    
    def _apply_inventory_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply inventory-specific time series correlations"""
        if 'timestamp' not in data.columns or len(data) < 2:
            return data
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply demand correlation (demand tends to be correlated over time)
        data = self._apply_time_series_correlation(data, ts_config, 'avg_daily_demand')
        
        # Apply stock level correlation (stock levels change gradually)
        data = self._apply_time_series_correlation(data, ts_config, 'current_stock')
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to inventory data"""
        if data.empty:
            return data
        
        # Calculate derived metrics
        data['stock_turn_ratio'] = data['annual_turnover'] / (data['current_stock'] + 1)
        data['reorder_urgency'] = (data['reorder_point'] - data['current_stock']) / data['reorder_point']
        data['reorder_urgency'] = data['reorder_urgency'].clip(0, 2)  # Cap at 2x urgency
        
        # Calculate inventory age (days since last received)
        today = datetime.now().date()
        received_dates = pd.to_datetime(data['last_received_date']).dt.date
        data['days_since_received'] = (pd.to_datetime(today) - pd.to_datetime(received_dates)).dt.days
        
        # Calculate days until expiration
        data['days_until_expiration'] = None
        expired_mask = data['expiration_date'].notna()
        if expired_mask.any():
            exp_dates = pd.to_datetime(data.loc[expired_mask, 'expiration_date']).dt.date
            data.loc[expired_mask, 'days_until_expiration'] = (
                pd.to_datetime(exp_dates) - pd.to_datetime(today)
            ).dt.days
        
        # Add inventory health score (0-100)
        data['inventory_health_score'] = self._calculate_inventory_health_score(data)
        
        # Add movement frequency (based on turnover and stock status)
        data['movement_frequency'] = self._calculate_movement_frequency(data)
        
        # Add cost per day of holding
        data['daily_holding_cost'] = data['holding_cost_annual'] / 365
        
        # Add stockout risk score
        data['stockout_risk'] = self._calculate_stockout_risk(data)
        
        return data
    
    def _calculate_inventory_health_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate overall inventory health score (0-100)"""
        scores = pd.Series(index=data.index, dtype=float)
        
        for idx, row in data.iterrows():
            score = 100  # Start with perfect score
            
            # Penalize for stock status issues
            if row['stock_status'] == 'out_of_stock':
                score -= 50
            elif row['stock_status'] == 'critical':
                score -= 30
            elif row['stock_status'] == 'reorder_needed':
                score -= 15
            elif row['stock_status'] == 'overstock':
                score -= 20
            
            # Penalize for expired items
            if row['expired']:
                score -= 40
            
            # Penalize for slow-moving items (low turnover)
            if row['annual_turnover'] < 2:
                score -= 15
            
            # Penalize for old inventory
            if row['days_since_received'] > 180:
                score -= 10
            
            # Bonus for good ABC management
            if row['abc_classification'] == 'A' and row['stock_status'] == 'normal':
                score += 5
            
            scores.iloc[idx] = max(0, min(100, score))
        
        return scores
    
    def _calculate_movement_frequency(self, data: pd.DataFrame) -> pd.Series:
        """Calculate expected movement frequency (movements per month)"""
        # Base frequency on turnover rate and current stock status
        base_frequency = data['annual_turnover'] / 12  # Monthly turnover
        
        # Adjust based on stock status
        status_multipliers = {
            'out_of_stock': 0.0,
            'critical': 0.5,
            'reorder_needed': 0.8,
            'normal': 1.0,
            'overstock': 1.2
        }
        
        multipliers = data['stock_status'].map(status_multipliers).fillna(1.0)
        return (base_frequency * multipliers).round(2)
    
    def _calculate_stockout_risk(self, data: pd.DataFrame) -> pd.Series:
        """Calculate stockout risk score (0-100, higher = more risk)"""
        risk_scores = pd.Series(index=data.index, dtype=float)
        
        for idx, row in data.iterrows():
            risk = 0
            
            # Base risk on current stock vs reorder point
            if row['current_stock'] <= 0:
                risk = 100
            elif row['current_stock'] <= row['safety_stock_quantity']:
                risk = 80
            elif row['current_stock'] <= row['reorder_point']:
                risk = 50
            else:
                # Risk decreases as stock increases above reorder point
                excess_stock = row['current_stock'] - row['reorder_point']
                risk = max(0, 30 - (excess_stock / row['reorder_point'] * 20))
            
            # Adjust for demand volatility
            volatility_adjustment = row['demand_volatility'] * 20
            risk += volatility_adjustment
            
            # Adjust for supplier reliability
            reliability_adjustment = (100 - row['supplier_reliability']) * 0.3
            risk += reliability_adjustment
            
            # Adjust for lead time
            if row['lead_time_days'] > 14:
                risk += 10
            elif row['lead_time_days'] > 30:
                risk += 20
            
            risk_scores.iloc[idx] = max(0, min(100, risk))
        
        return risk_scores.round(1)