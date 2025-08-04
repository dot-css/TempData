"""
Weather sensor data generator

Generates realistic weather sensor data with seasonal patterns, geographical variations,
and realistic correlations between temperature, humidity, and pressure.
"""

import pandas as pd
import math
import json
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Any
from ...core.base_generator import BaseGenerator


class WeatherGenerator(BaseGenerator):
    """
    Generator for realistic weather sensor data
    
    Creates weather datasets with realistic temperature, humidity, pressure
    correlations, seasonal patterns, and geographical weather variations.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_climate_zones()
        self._setup_seasonal_patterns()
        self._setup_weather_correlations()
        self._load_geographical_data()
    
    def _setup_climate_zones(self):
        """Setup climate zone characteristics for different regions"""
        self.climate_zones = {
            'tropical': {
                'temp_range': (20, 35),
                'humidity_range': (60, 95),
                'pressure_range': (1005, 1020),
                'seasonal_variation': 5,  # Low seasonal variation
                'countries': ['brazil', 'india', 'indonesia', 'thailand', 'philippines']
            },
            'subtropical': {
                'temp_range': (10, 30),
                'humidity_range': (40, 85),
                'pressure_range': (1000, 1025),
                'seasonal_variation': 15,
                'countries': ['united_states', 'china', 'japan', 'australia']
            },
            'temperate': {
                'temp_range': (-5, 25),
                'humidity_range': (30, 80),
                'pressure_range': (995, 1030),
                'seasonal_variation': 20,
                'countries': ['germany', 'france', 'united_kingdom', 'canada']
            },
            'continental': {
                'temp_range': (-20, 30),
                'humidity_range': (25, 75),
                'pressure_range': (990, 1035),
                'seasonal_variation': 35,
                'countries': ['russia', 'kazakhstan', 'mongolia']
            },
            'arid': {
                'temp_range': (5, 45),
                'humidity_range': (10, 50),
                'pressure_range': (1000, 1025),
                'seasonal_variation': 25,
                'countries': ['saudi_arabia', 'egypt', 'australia', 'mexico']
            },
            'polar': {
                'temp_range': (-40, 10),
                'humidity_range': (40, 90),
                'pressure_range': (980, 1020),
                'seasonal_variation': 30,
                'countries': ['greenland', 'antarctica', 'alaska']
            }
        }
    
    def _setup_seasonal_patterns(self):
        """Setup seasonal temperature and weather patterns"""
        # Northern hemisphere seasonal multipliers (month 1-12)
        self.northern_seasonal_temp = {
            1: -0.8,   # January - coldest
            2: -0.6,   # February
            3: -0.3,   # March
            4: 0.1,    # April
            5: 0.4,    # May
            6: 0.7,    # June
            7: 0.8,    # July - warmest
            8: 0.7,    # August
            9: 0.3,    # September
            10: 0.0,   # October
            11: -0.4,  # November
            12: -0.7   # December
        }
        
        # Southern hemisphere (opposite pattern)
        self.southern_seasonal_temp = {
            month: -multiplier for month, multiplier in self.northern_seasonal_temp.items()
        }
        
        # Humidity patterns (inverse relationship with temperature in many climates)
        self.humidity_seasonal = {
            month: -0.3 * temp_mult for month, temp_mult in self.northern_seasonal_temp.items()
        }
    
    def _setup_weather_correlations(self):
        """Setup realistic correlations between weather parameters"""
        self.correlations = {
            'temp_humidity': -0.4,      # Generally inverse relationship
            'temp_pressure': 0.2,       # Slight positive correlation
            'humidity_pressure': -0.3,  # Inverse relationship
            'altitude_pressure': -0.12, # Pressure decreases with altitude
            'altitude_temp': -0.0065    # Temperature lapse rate (°C per meter)
        }
    
    def _load_geographical_data(self):
        """Load geographical data for location-based weather patterns"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '../../data/countries')
            
            # Try to load city data with coordinates
            cities_file = os.path.join(data_path, 'cities.json')
            if os.path.exists(cities_file):
                with open(cities_file, 'r', encoding='utf-8') as f:
                    self.cities_data = json.load(f)
            else:
                self.cities_data = {}
                
        except (FileNotFoundError, json.JSONDecodeError):
            self.cities_data = {}
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate weather sensor dataset with realistic patterns
        
        Args:
            rows: Number of weather readings to generate
            **kwargs: Additional parameters (country, time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated weather data with realistic correlations
        """
        country = kwargs.get('country', 'global')
        time_series = kwargs.get('time_series', False)
        date_range = kwargs.get('date_range', None)
        interval = kwargs.get('interval', '1hour')
        
        data = []
        
        # Determine climate zone based on country
        climate_zone = self._get_climate_zone(country)
        
        # Generate base timestamps
        timestamps = self._generate_timestamps(rows, time_series, date_range, interval)
        
        for i in range(rows):
            timestamp = timestamps[i]
            
            # Generate location
            location_data = self._generate_location(country)
            
            # Generate correlated weather data
            weather_data = self._generate_weather_reading(
                timestamp, climate_zone, location_data
            )
            
            # Create sensor record
            sensor_record = {
                'sensor_id': f'WS_{i+1:06d}',
                'timestamp': timestamp,
                'location': location_data['name'],
                'latitude': location_data['latitude'],
                'longitude': location_data['longitude'],
                'altitude_m': location_data['altitude'],
                **weather_data
            }
            
            data.append(sensor_record)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _get_climate_zone(self, country: str) -> str:
        """Determine climate zone based on country"""
        if country == 'global':
            # Random climate zone for global data
            return self.faker.random_element(list(self.climate_zones.keys()))
        
        # Find climate zone for specific country
        for zone, zone_data in self.climate_zones.items():
            if country.lower() in zone_data['countries']:
                return zone
        
        # Default to temperate if country not found
        return 'temperate'
    
    def _generate_timestamps(self, rows: int, time_series: bool, 
                           date_range: Optional[Tuple], interval: str) -> List[datetime]:
        """Generate timestamps for weather readings"""
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
    
    def _generate_location(self, country: str) -> Dict[str, Any]:
        """Generate location data with coordinates and altitude"""
        if country == 'global':
            # Random global location
            city = self.faker.city()
            latitude = float(self.faker.latitude())
            longitude = float(self.faker.longitude())
        else:
            # Get realistic coordinates for specific countries
            city = self.faker.city()
            latitude, longitude = self._get_country_coordinates(country)
            
            # Adjust coordinates based on country if we have data
            if country.lower() in self.cities_data:
                country_cities = self.cities_data[country.lower()]
                if country_cities:
                    city_data = self.faker.random_element(country_cities)
                    city = city_data.get('name', city)
                    latitude = city_data.get('latitude', latitude)
                    longitude = city_data.get('longitude', longitude)
        
        # Generate realistic altitude based on location
        # Most weather stations are at reasonable altitudes
        altitude = max(0, self.faker.random.gauss(200, 500))  # Mean 200m, std 500m
        altitude = min(altitude, 3000)  # Cap at 3000m for typical weather stations
        
        return {
            'name': city,
            'latitude': round(latitude, 6),
            'longitude': round(longitude, 6),
            'altitude': round(altitude, 1)
        }
    
    def _get_country_coordinates(self, country: str) -> Tuple[float, float]:
        """Get realistic coordinates for a specific country"""
        # Country coordinate ranges (ensuring correct hemisphere)
        country_bounds = {
            'united_states': {'lat': (24.0, 49.0), 'lon': (-125.0, -66.0)},
            'canada': {'lat': (41.0, 84.0), 'lon': (-141.0, -52.0)},
            'germany': {'lat': (47.0, 55.0), 'lon': (5.0, 15.0)},
            'france': {'lat': (41.0, 51.0), 'lon': (-5.0, 10.0)},
            'united_kingdom': {'lat': (49.0, 61.0), 'lon': (-8.0, 2.0)},
            'brazil': {'lat': (-34.0, 5.0), 'lon': (-74.0, -32.0)},
            'australia': {'lat': (-44.0, -10.0), 'lon': (113.0, 154.0)},
            'china': {'lat': (18.0, 54.0), 'lon': (73.0, 135.0)},
            'japan': {'lat': (24.0, 46.0), 'lon': (123.0, 146.0)},
            'india': {'lat': (6.0, 37.0), 'lon': (68.0, 97.0)},
            'russia': {'lat': (41.0, 82.0), 'lon': (19.0, 170.0)},
            'pakistan': {'lat': (23.0, 37.0), 'lon': (60.0, 77.0)},
            'saudi_arabia': {'lat': (16.0, 33.0), 'lon': (34.0, 56.0)},
            'egypt': {'lat': (22.0, 32.0), 'lon': (25.0, 37.0)},
            'mexico': {'lat': (14.0, 33.0), 'lon': (-118.0, -86.0)},
            'argentina': {'lat': (-55.0, -22.0), 'lon': (-74.0, -53.0)}
        }
        
        country_key = country.lower()
        if country_key in country_bounds:
            bounds = country_bounds[country_key]
            lat = self.faker.random.uniform(bounds['lat'][0], bounds['lat'][1])
            lon = self.faker.random.uniform(bounds['lon'][0], bounds['lon'][1])
            return (lat, lon)
        
        # Fallback to faker for unknown countries
        return (float(self.faker.latitude()), float(self.faker.longitude()))
    
    def _generate_weather_reading(self, timestamp: datetime, climate_zone: str, 
                                location_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate correlated weather reading for specific conditions"""
        zone_data = self.climate_zones[climate_zone]
        
        # Determine hemisphere for seasonal patterns
        is_northern = location_data['latitude'] >= 0
        seasonal_temp = (self.northern_seasonal_temp if is_northern 
                        else self.southern_seasonal_temp)
        
        # Base temperature from climate zone
        temp_min, temp_max = zone_data['temp_range']
        base_temp = (temp_min + temp_max) / 2
        
        # Apply seasonal variation
        seasonal_mult = seasonal_temp[timestamp.month]
        seasonal_variation = zone_data['seasonal_variation']
        temperature = base_temp + (seasonal_mult * seasonal_variation)
        
        # Debug: ensure seasonal patterns are working
        # July (month 7) should be warmer in northern hemisphere
        # January (month 1) should be colder in northern hemisphere
        
        # Add daily variation (cooler at night, warmer during day)
        hour_variation = 3 * math.sin((timestamp.hour - 6) * math.pi / 12)
        temperature += hour_variation
        
        # Add altitude effect
        altitude_effect = location_data['altitude'] * self.correlations['altitude_temp']
        temperature += altitude_effect
        
        # Add random variation
        temperature += self.faker.random.gauss(0, 2)
        
        # Generate humidity with inverse correlation to temperature
        humidity_min, humidity_max = zone_data['humidity_range']
        base_humidity = (humidity_min + humidity_max) / 2
        
        # Temperature correlation
        temp_normalized = (temperature - base_temp) / seasonal_variation if seasonal_variation > 0 else 0
        humidity_temp_effect = self.correlations['temp_humidity'] * temp_normalized * 20
        
        # Seasonal humidity pattern
        humidity_seasonal_effect = self.humidity_seasonal[timestamp.month] * 15
        
        humidity = base_humidity + humidity_temp_effect + humidity_seasonal_effect
        humidity += self.faker.random.gauss(0, 5)  # Random variation
        humidity = max(humidity_min, min(humidity_max, humidity))  # Clamp to range
        
        # Generate pressure with correlations
        pressure_min, pressure_max = zone_data['pressure_range']
        base_pressure = (pressure_min + pressure_max) / 2
        
        # Altitude effect (major factor)
        pressure_altitude_effect = location_data['altitude'] * self.correlations['altitude_pressure']
        
        # Temperature correlation
        pressure_temp_effect = self.correlations['temp_pressure'] * temp_normalized * 5
        
        # Humidity correlation
        humidity_normalized = (humidity - base_humidity) / 20 if base_humidity > 0 else 0
        pressure_humidity_effect = self.correlations['humidity_pressure'] * humidity_normalized * 3
        
        pressure = (base_pressure + pressure_altitude_effect + 
                   pressure_temp_effect + pressure_humidity_effect)
        pressure += self.faker.random.gauss(0, 2)  # Random variation
        pressure = max(pressure_min, min(pressure_max, pressure))  # Clamp to range
        
        # Generate additional weather parameters
        wind_speed = max(0, self.faker.random.gauss(8, 5))  # km/h, mean 8, std 5
        wind_direction = self.faker.random.uniform(0, 360)  # degrees
        
        # UV index (related to time of day and season)
        uv_base = 5 if 6 <= timestamp.hour <= 18 else 0  # Daylight hours
        uv_seasonal = abs(seasonal_mult) * 3  # Higher in summer
        uv_index = max(0, min(11, uv_base + uv_seasonal + self.faker.random.gauss(0, 1)))
        
        # Visibility (inversely related to humidity)
        visibility_base = 20  # km
        visibility_humidity_effect = -0.1 * (humidity - 50)  # Reduced by high humidity
        visibility = max(0.1, visibility_base + visibility_humidity_effect + 
                        self.faker.random.gauss(0, 3))
        
        return {
            'temperature_c': round(temperature, 1),
            'humidity_percent': round(humidity, 1),
            'pressure_hpa': round(pressure, 1),
            'wind_speed_kmh': round(wind_speed, 1),
            'wind_direction_deg': round(wind_direction, 0),
            'uv_index': round(uv_index, 1),
            'visibility_km': round(visibility, 1)
        }
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to weather data"""
        # Add weather condition categories based on parameters
        data['weather_condition'] = data.apply(self._determine_weather_condition, axis=1)
        
        # Add heat index calculation
        data['heat_index_c'] = data.apply(
            lambda row: self._calculate_heat_index(row['temperature_c'], row['humidity_percent']), 
            axis=1
        )
        
        # Add dew point calculation
        data['dew_point_c'] = data.apply(
            lambda row: self._calculate_dew_point(row['temperature_c'], row['humidity_percent']), 
            axis=1
        )
        
        # Add data quality indicators
        data['sensor_status'] = data.apply(self._determine_sensor_status, axis=1)
        
        # Sort by timestamp for realistic chronological order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
    
    def _determine_weather_condition(self, row) -> str:
        """Determine weather condition based on parameters"""
        temp = row['temperature_c']
        humidity = row['humidity_percent']
        pressure = row['pressure_hpa']
        visibility = row['visibility_km']
        
        # More varied weather condition logic
        if visibility < 1:
            return 'fog'
        elif temp < 0:
            return 'freezing'
        elif humidity > 85 and pressure < 1010:
            return 'rainy'
        elif pressure < 1000:
            return 'stormy'
        elif humidity < 35 and temp > 25:
            return 'sunny'
        elif humidity > 75 and temp > 18:
            return 'humid'
        elif temp > 30:
            return 'hot'
        elif temp < 5:
            return 'cold'
        elif humidity > 60:
            return 'overcast'
        else:
            return 'clear'
    
    def _calculate_heat_index(self, temp_c: float, humidity: float) -> float:
        """Calculate heat index (feels like temperature)"""
        if temp_c < 27:  # Heat index only relevant for high temperatures
            return temp_c
        
        # Convert to Fahrenheit for calculation
        temp_f = temp_c * 9/5 + 32
        
        # Simplified heat index formula - should be higher than temperature
        # when both temperature and humidity are high
        if humidity > 40:
            heat_factor = 1 + (humidity - 40) * 0.01  # Increase based on humidity
            hi_f = temp_f * heat_factor
        else:
            hi_f = temp_f
        
        # Convert back to Celsius
        return round((hi_f - 32) * 5/9, 1)
    
    def _calculate_dew_point(self, temp_c: float, humidity: float) -> float:
        """Calculate dew point temperature"""
        # Magnus formula approximation
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp_c) / (b + temp_c)) + math.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return round(dew_point, 1)
    
    def _determine_sensor_status(self, row) -> str:
        """Determine sensor status based on reading patterns"""
        # Most sensors work fine
        if self.faker.random.random() < 0.95:
            return 'normal'
        elif self.faker.random.random() < 0.03:
            return 'maintenance_required'
        else:
            return 'calibration_needed'