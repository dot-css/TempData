"""
Energy consumption data generator

Generates realistic energy consumption data with consumption patterns, peak usage times,
seasonal variations, and realistic meter readings with power factor calculations.
"""

import pandas as pd
import math
import json
import os
from collections import OrderedDict
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from ...core.base_generator import BaseGenerator


class EnergyGenerator(BaseGenerator):
    """
    Generator for realistic energy consumption data
    
    Creates energy datasets with consumption patterns, peak usage times,
    seasonal variations, and realistic meter readings with power factor calculations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_building_profiles()
        self._setup_seasonal_patterns()
        self._setup_daily_patterns()
        self._setup_regional_patterns()
    
    def _setup_building_profiles(self):
        """Setup energy consumption profiles for different building types"""
        self.building_profiles = {
            'residential': {
                'base_consumption': 25,  # kWh per day average
                'peak_multiplier': 2.5,
                'off_peak_multiplier': 0.4,
                'voltage_range': (220, 240),
                'power_factor_range': (0.85, 0.95),
                'seasonal_variation': 0.3,
                'appliance_types': ['hvac', 'water_heater', 'lighting', 'electronics', 'kitchen'],
                'peak_hours': [(7, 9), (18, 22)],  # Morning and evening peaks
                'weekend_multiplier': 1.2
            },
            'commercial': {
                'base_consumption': 150,  # kWh per day average
                'peak_multiplier': 1.8,
                'off_peak_multiplier': 0.6,
                'voltage_range': (380, 415),  # Three-phase
                'power_factor_range': (0.80, 0.90),
                'seasonal_variation': 0.4,
                'appliance_types': ['hvac', 'lighting', 'computers', 'elevators', 'security'],
                'peak_hours': [(8, 12), (13, 18)],  # Business hours
                'weekend_multiplier': 0.3
            },
            'industrial': {
                'base_consumption': 800,  # kWh per day average
                'peak_multiplier': 1.4,
                'off_peak_multiplier': 0.8,
                'voltage_range': (380, 415),  # Three-phase
                'power_factor_range': (0.75, 0.85),
                'seasonal_variation': 0.2,
                'appliance_types': ['machinery', 'motors', 'furnaces', 'compressors', 'lighting'],
                'peak_hours': [(6, 14), (14, 22)],  # Shift patterns
                'weekend_multiplier': 0.7
            },
            'retail': {
                'base_consumption': 80,  # kWh per day average
                'peak_multiplier': 2.0,
                'off_peak_multiplier': 0.3,
                'voltage_range': (220, 240),
                'power_factor_range': (0.82, 0.92),
                'seasonal_variation': 0.5,
                'appliance_types': ['lighting', 'hvac', 'refrigeration', 'pos_systems', 'security'],
                'peak_hours': [(10, 14), (16, 20)],  # Shopping hours
                'weekend_multiplier': 1.4
            },
            'hospital': {
                'base_consumption': 400,  # kWh per day average
                'peak_multiplier': 1.2,
                'off_peak_multiplier': 0.9,
                'voltage_range': (380, 415),
                'power_factor_range': (0.88, 0.95),
                'seasonal_variation': 0.15,
                'appliance_types': ['medical_equipment', 'hvac', 'lighting', 'elevators', 'kitchen'],
                'peak_hours': [(6, 22)],  # Extended hours
                'weekend_multiplier': 0.95
            }
        }
    
    def _setup_seasonal_patterns(self):
        """Setup seasonal energy consumption patterns"""
        # Northern hemisphere seasonal multipliers (month 1-12)
        self.northern_seasonal = {
            1: 1.3,   # January - winter heating
            2: 1.2,   # February
            3: 1.0,   # March
            4: 0.9,   # April
            5: 0.8,   # May
            6: 1.0,   # June - AC starts
            7: 1.2,   # July - peak AC
            8: 1.2,   # August - peak AC
            9: 1.0,   # September
            10: 0.9,  # October
            11: 1.1,  # November - heating starts
            12: 1.3   # December - winter heating
        }
        
        # Southern hemisphere (opposite pattern)
        self.southern_seasonal = {
            1: 1.2,   # January - summer AC
            2: 1.2,   # February - summer AC
            3: 1.0,   # March
            4: 0.9,   # April
            5: 0.8,   # May
            6: 1.0,   # June - winter heating
            7: 1.3,   # July - peak heating
            8: 1.2,   # August - heating
            9: 1.0,   # September
            10: 0.9,  # October
            11: 1.0,  # November
            12: 1.1   # December - summer starts
        }
    
    def _setup_daily_patterns(self):
        """Setup daily consumption patterns by hour"""
        # Typical daily patterns (0-23 hours)
        self.daily_patterns = {
            'residential': {
                0: 0.3, 1: 0.25, 2: 0.2, 3: 0.2, 4: 0.25, 5: 0.4,
                6: 0.7, 7: 1.0, 8: 0.8, 9: 0.5, 10: 0.4, 11: 0.4,
                12: 0.6, 13: 0.5, 14: 0.4, 15: 0.4, 16: 0.5, 17: 0.7,
                18: 1.2, 19: 1.3, 20: 1.2, 21: 1.0, 22: 0.8, 23: 0.5
            },
            'commercial': {
                0: 0.2, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.2,
                6: 0.3, 7: 0.6, 8: 1.0, 9: 1.2, 10: 1.3, 11: 1.2,
                12: 1.0, 13: 1.2, 14: 1.3, 15: 1.2, 16: 1.1, 17: 1.0,
                18: 0.8, 19: 0.5, 20: 0.3, 21: 0.25, 22: 0.2, 23: 0.2
            },
            'industrial': {
                0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.9,
                6: 1.2, 7: 1.3, 8: 1.3, 9: 1.2, 10: 1.2, 11: 1.2,
                12: 1.0, 13: 1.2, 14: 1.3, 15: 1.2, 16: 1.2, 17: 1.2,
                18: 1.1, 19: 1.0, 20: 1.0, 21: 1.0, 22: 0.9, 23: 0.8
            }
        }
    
    def _setup_regional_patterns(self):
        """Setup regional energy consumption patterns"""
        self.regional_patterns = {
            'north_america': {
                'voltage_standard': 120,  # Residential
                'frequency': 60,
                'seasonal_ac_factor': 1.4,
                'heating_fuel_mix': {'electric': 0.4, 'gas': 0.5, 'other': 0.1}
            },
            'europe': {
                'voltage_standard': 230,
                'frequency': 50,
                'seasonal_ac_factor': 1.1,
                'heating_fuel_mix': {'electric': 0.3, 'gas': 0.6, 'other': 0.1}
            },
            'asia_pacific': {
                'voltage_standard': 220,
                'frequency': 50,
                'seasonal_ac_factor': 1.5,
                'heating_fuel_mix': {'electric': 0.6, 'gas': 0.3, 'other': 0.1}
            },
            'middle_east': {
                'voltage_standard': 220,
                'frequency': 50,
                'seasonal_ac_factor': 1.8,
                'heating_fuel_mix': {'electric': 0.7, 'gas': 0.2, 'other': 0.1}
            }
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate energy consumption dataset with realistic patterns
        
        Args:
            rows: Number of energy readings to generate
            **kwargs: Additional parameters (country, time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated energy data with realistic consumption patterns
        """
        country = kwargs.get('country', 'global')
        time_series = kwargs.get('time_series', False)
        date_range = kwargs.get('date_range', None)
        interval = kwargs.get('interval', '1hour')
        building_type = kwargs.get('building_type', None)
        
        data = []
        
        # Generate timestamps
        timestamps = self._generate_timestamps(rows, time_series, date_range, interval)
        
        for i in range(rows):
            timestamp = timestamps[i]
            
            # Select building type
            if building_type:
                selected_building_type = building_type
            else:
                selected_building_type = self._select_building_type()
            
            # Determine regional pattern for each row (for variety in global data)
            region = self._get_region(country)
            
            # Generate meter reading
            meter_data = self._generate_meter_reading(
                timestamp, selected_building_type, region, i
            )
            
            # Create energy record
            energy_record = {
                'meter_id': f'MTR_{i+1:08d}',
                'timestamp': timestamp,
                'building_type': selected_building_type,
                'region': region,
                **meter_data
            }
            
            data.append(energy_record)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_timestamps(self, rows: int, time_series: bool, 
                           date_range: Optional[Tuple], interval: str) -> List[datetime]:
        """Generate timestamps for energy readings"""
        if time_series and date_range:
            start_date, end_date = date_range
            
            # Convert date objects to datetime objects
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            elif isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
                
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.max.time())
            elif isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            
            # Calculate interval in minutes
            interval_minutes = self._parse_interval(interval)
            
            timestamps = []
            current_time = start_date
            
            for _ in range(rows):
                timestamps.append(current_time)
                current_time += timedelta(minutes=interval_minutes)
                if current_time > end_date:
                    break
            
            # Fill remaining with random timestamps if needed
            while len(timestamps) < rows:
                timestamps.append(self.faker.date_time_between(start_date, end_date))
                
        else:
            # Generate random timestamps
            timestamps = [self.faker.date_time_this_year() for _ in range(rows)]
        
        return sorted(timestamps)
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to minutes"""
        interval_map = {
            '1min': 1,
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1hour': 60,
            '2hour': 120,
            '6hour': 360,
            '12hour': 720,
            '1day': 1440
        }
        return interval_map.get(interval, 60)  # Default to 1 hour
    
    def _get_region(self, country: str) -> str:
        """Determine region based on country"""
        country_to_region = {
            'united_states': 'north_america',
            'canada': 'north_america',
            'mexico': 'north_america',
            'germany': 'europe',
            'france': 'europe',
            'united_kingdom': 'europe',
            'italy': 'europe',
            'spain': 'europe',
            'china': 'asia_pacific',
            'japan': 'asia_pacific',
            'india': 'asia_pacific',
            'australia': 'asia_pacific',
            'saudi_arabia': 'middle_east',
            'uae': 'middle_east',
            'egypt': 'middle_east'
        }
        
        if country == 'global':
            # Use weighted distribution for global data to ensure variety
            region_weights = OrderedDict([
                ('north_america', 0.3),
                ('europe', 0.25),
                ('asia_pacific', 0.35),
                ('middle_east', 0.1)
            ])
            return self.faker.random_element(elements=region_weights)
        
        return country_to_region.get(country.lower(), 'north_america')
    
    def _select_building_type(self) -> str:
        """Select building type with realistic distribution"""
        # Weighted distribution of building types using OrderedDict
        building_weights = OrderedDict([
            ('residential', 0.6),
            ('commercial', 0.2),
            ('industrial', 0.1),
            ('retail', 0.08),
            ('hospital', 0.02)
        ])
        
        return self.faker.random_element(elements=building_weights)
    
    def _generate_meter_reading(self, timestamp: datetime, building_type: str, 
                              region: str, meter_index: int) -> Dict[str, float]:
        """Generate realistic meter reading for specific conditions"""
        profile = self.building_profiles[building_type]
        regional_data = self.regional_patterns[region]
        
        # Determine hemisphere for seasonal patterns
        is_northern = region in ['north_america', 'europe']
        seasonal_pattern = self.northern_seasonal if is_northern else self.southern_seasonal
        
        # Base consumption
        base_consumption = profile['base_consumption']
        
        # Apply seasonal variation
        seasonal_mult = seasonal_pattern[timestamp.month]
        seasonal_variation = profile['seasonal_variation']
        consumption_base = base_consumption * (1 + (seasonal_mult - 1) * seasonal_variation)
        
        # Apply daily pattern
        hour_pattern = self.daily_patterns.get(building_type, self.daily_patterns['residential'])
        daily_mult = hour_pattern.get(timestamp.hour, 1.0)
        
        # Apply weekend pattern
        is_weekend = timestamp.weekday() >= 5
        weekend_mult = profile['weekend_multiplier'] if is_weekend else 1.0
        
        # Calculate final consumption (per hour)
        hourly_consumption = (consumption_base / 24) * daily_mult * weekend_mult
        
        # Add random variation
        hourly_consumption *= self.faker.random.uniform(0.8, 1.2)
        
        # Generate voltage based on building type and region
        voltage_range = profile['voltage_range']
        if building_type == 'residential' and region == 'north_america':
            voltage = self.faker.random.uniform(110, 125)  # US residential
        else:
            voltage = self.faker.random.uniform(voltage_range[0], voltage_range[1])
        
        # Add voltage fluctuation
        voltage += self.faker.random.gauss(0, 2)
        
        # Generate current based on consumption and voltage
        # P = V * I * PF (for single phase) or P = √3 * V * I * PF (for three phase)
        power_factor_range = profile['power_factor_range']
        power_factor = self.faker.random.uniform(power_factor_range[0], power_factor_range[1])
        
        # Calculate current (assuming single phase for simplicity)
        current = (hourly_consumption * 1000) / (voltage * power_factor)  # Convert kWh to Wh
        current = max(0.1, current)  # Minimum current
        
        # Generate additional parameters
        frequency = regional_data['frequency']
        
        # Calculate apparent power and reactive power
        real_power = hourly_consumption * 1000  # Convert to watts
        apparent_power = real_power / power_factor
        reactive_power = math.sqrt(apparent_power**2 - real_power**2)
        
        # Generate meter readings (cumulative)
        # Use meter index to simulate cumulative readings
        base_reading = meter_index * 50 + self.faker.random.uniform(0, 100)
        cumulative_reading = base_reading + hourly_consumption
        
        return {
            'consumption_kwh': round(hourly_consumption, 3),
            'cumulative_kwh': round(cumulative_reading, 2),
            'voltage_v': round(voltage, 1),
            'current_a': round(current, 2),
            'power_factor': round(power_factor, 3),
            'frequency_hz': frequency,
            'real_power_w': round(real_power, 1),
            'apparent_power_va': round(apparent_power, 1),
            'reactive_power_var': round(reactive_power, 1),
            'demand_kw': round(real_power / 1000, 2)
        }
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to energy data"""
        # Add energy cost calculation
        data['cost_per_kwh'] = data.apply(self._calculate_energy_cost, axis=1)
        data['total_cost'] = (data['consumption_kwh'] * data['cost_per_kwh']).round(2)
        
        # Add efficiency metrics
        data['efficiency_rating'] = data.apply(self._calculate_efficiency_rating, axis=1)
        
        # Add peak/off-peak classification
        data['rate_period'] = data.apply(self._classify_rate_period, axis=1)
        
        # Add meter status
        data['meter_status'] = data.apply(self._determine_meter_status, axis=1)
        
        # Add carbon footprint estimation
        data['carbon_footprint_kg'] = data.apply(self._calculate_carbon_footprint, axis=1)
        
        # Sort by timestamp for realistic chronological order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
    
    def _calculate_energy_cost(self, row) -> float:
        """Calculate energy cost per kWh based on region and building type"""
        # Base rates by region (USD per kWh)
        base_rates = {
            'north_america': 0.12,
            'europe': 0.25,
            'asia_pacific': 0.15,
            'middle_east': 0.08
        }
        
        base_rate = base_rates.get(row['region'], 0.12)
        
        # Building type multipliers
        building_multipliers = {
            'residential': 1.0,
            'commercial': 0.9,
            'industrial': 0.7,
            'retail': 0.95,
            'hospital': 0.85
        }
        
        multiplier = building_multipliers.get(row['building_type'], 1.0)
        
        # Time-of-use pricing
        hour = pd.to_datetime(row['timestamp']).hour
        if 18 <= hour <= 22:  # Peak hours
            time_multiplier = 1.5
        elif 22 <= hour <= 6:  # Off-peak hours
            time_multiplier = 0.7
        else:  # Standard hours
            time_multiplier = 1.0
        
        return round(base_rate * multiplier * time_multiplier, 4)
    
    def _calculate_efficiency_rating(self, row) -> str:
        """Calculate efficiency rating based on power factor and consumption patterns"""
        pf = row['power_factor']
        
        if pf >= 0.95:
            return 'excellent'
        elif pf >= 0.90:
            return 'good'
        elif pf >= 0.85:
            return 'fair'
        elif pf >= 0.80:
            return 'poor'
        else:
            return 'very_poor'
    
    def _classify_rate_period(self, row) -> str:
        """Classify time period for rate calculation"""
        hour = pd.to_datetime(row['timestamp']).hour
        weekday = pd.to_datetime(row['timestamp']).weekday()
        
        # Weekend rates
        if weekday >= 5:
            return 'weekend'
        
        # Weekday rates
        if 18 <= hour <= 22:
            return 'peak'
        elif 22 <= hour <= 6:
            return 'off_peak'
        else:
            return 'standard'
    
    def _determine_meter_status(self, row) -> str:
        """Determine meter status based on readings"""
        # Most meters work fine
        if self.faker.random.random() < 0.97:
            return 'normal'
        elif self.faker.random.random() < 0.02:
            return 'maintenance_required'
        else:
            return 'calibration_needed'
    
    def _calculate_carbon_footprint(self, row) -> float:
        """Calculate carbon footprint based on energy consumption and region"""
        # Carbon intensity by region (kg CO2 per kWh)
        carbon_intensity = {
            'north_america': 0.45,
            'europe': 0.35,
            'asia_pacific': 0.65,
            'middle_east': 0.55
        }
        
        intensity = carbon_intensity.get(row['region'], 0.45)
        return round(row['consumption_kwh'] * intensity, 3)