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
        
        # Create time series configuration if requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_weather(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_weather(rows, country, **kwargs)
    
    def _generate_snapshot_weather(self, rows: int, country: str, **kwargs) -> pd.DataFrame:
        """Generate snapshot weather data (random timestamps)"""
        data = []
        
        # Determine climate zone based on country
        climate_zone = self._get_climate_zone(country)
        
        for i in range(rows):
            # Generate random timestamp
            timestamp = self.faker.date_time_this_year()
            
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
    
    def _generate_time_series_weather(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series weather data using integrated time series system"""
        country = kwargs.get('country', 'global')
        data = []
        
        # Determine climate zone based on country
        climate_zone = self._get_climate_zone(country)
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Generate location (consistent for time series)
        location_data = self._generate_location(country)
        
        # Generate base temperature time series
        zone_data = self.climate_zones[climate_zone]
        temp_min, temp_max = zone_data['temp_range']
        base_temp = (temp_min + temp_max) / 2
        
        # Create base time series for temperature with seasonal patterns
        temp_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_temp,
            value_range=(temp_min - 10, temp_max + 10)  # Allow some variation beyond normal range
        )
        
        # Generate data for each timestamp with realistic correlations
        for i, timestamp in enumerate(timestamps):
            if i >= len(temp_series):
                break
                
            # Get time series temperature value
            temperature = temp_series.iloc[i]['value']
            
            # Apply additional weather-specific temporal patterns
            temperature = self._apply_weather_temporal_patterns(
                temperature, timestamp, climate_zone, location_data
            )
            
            # Generate correlated weather parameters using time series values as base
            weather_data = self._generate_correlated_weather_from_temp(
                timestamp, temperature, climate_zone, location_data
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
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply realistic time-based correlations between weather parameters
        df = self._apply_weather_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _apply_weather_time_series_correlations(self, df: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply weather-specific time series correlations"""
        if 'timestamp' not in df.columns:
            return df
        
        # Sort by timestamp for proper correlation analysis
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add time-based features for weather patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_year'] = pd.to_datetime(df['timestamp']).dt.dayofyear
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Apply diurnal temperature patterns (daily temperature cycle)
        # Temperature typically peaks around 2-4 PM and is lowest around 6 AM
        diurnal_adjustment = np.sin((df['hour'] - 6) * np.pi / 12) * 3  # ±3°C variation
        df['temperature_celsius'] += diurnal_adjustment
        
        # Apply seasonal correlations
        # Northern hemisphere: warmest around day 200 (mid-July), coldest around day 20 (mid-January)
        seasonal_adjustment = np.sin((df['day_of_year'] - 20) * 2 * np.pi / 365) * 10  # ±10°C seasonal variation
        df['temperature_celsius'] += seasonal_adjustment
        
        # Apply humidity-temperature inverse correlation
        # Higher temperatures generally mean lower relative humidity
        temp_normalized = (df['temperature_celsius'] - df['temperature_celsius'].mean()) / df['temperature_celsius'].std()
        humidity_adjustment = -temp_normalized * 15  # Inverse correlation
        df['humidity_percent'] = np.clip(df['humidity_percent'] + humidity_adjustment, 0, 100)
        
        # Apply pressure-weather correlations
        # Lower pressure often correlates with higher humidity and precipitation
        pressure_humidity_corr = -0.3 * (df['humidity_percent'] - 50) / 50  # Normalize around 50%
        df['pressure_hpa'] += pressure_humidity_corr * 20  # ±20 hPa variation
        
        # Add wind speed correlations with pressure differences
        # Higher pressure differences typically mean higher wind speeds
        df['pressure_gradient'] = df['pressure_hpa'].diff().fillna(0)
        wind_adjustment = np.abs(df['pressure_gradient']) * 2  # Pressure gradient affects wind
        df['wind_speed_kmh'] += wind_adjustment
        df['wind_speed_kmh'] = np.clip(df['wind_speed_kmh'], 0, 150)  # Reasonable wind speed limits
        
        # Add precipitation correlations
        # High humidity + low pressure often leads to precipitation
        precip_probability = ((df['humidity_percent'] - 60) / 40) * ((1020 - df['pressure_hpa']) / 20)
        precip_probability = np.clip(precip_probability, 0, 1)
        
        # Generate precipitation based on probability
        df['precipitation_mm'] = np.where(
            np.random.random(len(df)) < precip_probability,
            np.random.exponential(2.0, len(df)),  # Exponential distribution for rain amounts
            0
        )
        
        # Add temporal persistence (weather tends to be similar to recent weather)
        for col in ['temperature_celsius', 'humidity_percent', 'pressure_hpa']:
            if col in df.columns:
                # Add moving average component for persistence
                df[f'{col}_smooth'] = df[col].rolling(window=3, center=True).mean().fillna(df[col])
                # Blend original with smoothed (70% original, 30% smoothed for persistence)
                df[col] = 0.7 * df[col] + 0.3 * df[f'{col}_smooth']
                df.drop(f'{col}_smooth', axis=1, inplace=True)
        
        return df
    
    def _generate_correlated_weather_from_temp(self, timestamp: datetime, temperature: float,
                                             climate_zone: str, location_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate correlated weather parameters from base temperature"""
        zone_data = self.climate_zones[climate_zone]
        
        # Generate humidity with inverse correlation to temperature
        humidity_min, humidity_max = zone_data['humidity_range']
        base_humidity = (humidity_min + humidity_max) / 2
        
        # Apply temperature correlation
        temp_range = zone_data['temp_range']
        temp_normalized = (temperature - sum(temp_range)/2) / (temp_range[1] - temp_range[0])
        humidity_temp_effect = self.correlations['temp_humidity'] * temp_normalized * 20
        
        # Apply seasonal humidity pattern
        is_northern = location_data['latitude'] >= 0
        seasonal_humidity = (self.humidity_seasonal if is_northern 
                           else {k: -v for k, v in self.humidity_seasonal.items()})
        humidity_seasonal_effect = seasonal_humidity[timestamp.month] * 15
        
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
        is_northern = location_data['latitude'] >= 0
        seasonal_temp = (self.northern_seasonal_temp if is_northern 
                        else self.southern_seasonal_temp)
        uv_seasonal = abs(seasonal_temp[timestamp.month]) * 3  # Higher in summer
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
    
    def _apply_weather_temporal_patterns(self, temperature: float, timestamp: datetime, 
                                       climate_zone: str, location_data: Dict[str, Any]) -> float:
        """Apply weather-specific temporal patterns to temperature"""
        zone_data = self.climate_zones[climate_zone]
        
        # Apply daily temperature variation (cooler at night, warmer during day)
        hour_variation = 3 * math.sin((timestamp.hour - 6) * math.pi / 12)
        temperature += hour_variation
        
        # Apply seasonal patterns based on hemisphere
        is_northern = location_data['latitude'] >= 0
        seasonal_temp = (self.northern_seasonal_temp if is_northern 
                        else self.southern_seasonal_temp)
        
        seasonal_mult = seasonal_temp[timestamp.month]
        seasonal_variation = zone_data['seasonal_variation']
        seasonal_adjustment = seasonal_mult * seasonal_variation * 0.3  # Moderate seasonal effect
        temperature += seasonal_adjustment
        
        # Apply altitude effect
        altitude_effect = location_data['altitude'] * self.correlations['altitude_temp']
        temperature += altitude_effect
        
        return temperature
    
    def _apply_weather_time_series_correlations(self, data: pd.DataFrame, 
                                              ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations between weather parameters"""
        if len(data) < 2:
            return data
        
        # Sort by timestamp to ensure proper time series order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply temporal correlations between temperature, humidity, and pressure
        for i in range(1, len(data)):
            prev_row = data.iloc[i-1]
            current_row = data.iloc[i]
            
            # Temperature persistence (weather tends to change gradually)
            temp_persistence = 0.8
            temp_change = (current_row['temperature_c'] - prev_row['temperature_c'])
            if abs(temp_change) > 5:  # Limit sudden temperature changes
                adjusted_temp = prev_row['temperature_c'] + temp_change * temp_persistence
                data.loc[i, 'temperature_c'] = round(adjusted_temp, 1)
            
            # Pressure persistence (pressure changes gradually)
            pressure_persistence = 0.9
            pressure_change = (current_row['pressure_hpa'] - prev_row['pressure_hpa'])
            if abs(pressure_change) > 3:  # Limit sudden pressure changes
                adjusted_pressure = prev_row['pressure_hpa'] + pressure_change * pressure_persistence
                data.loc[i, 'pressure_hpa'] = round(adjusted_pressure, 1)
            
            # Humidity correlation with temperature changes
            temp_change_effect = -0.5 * temp_change  # Inverse relationship
            current_humidity = data.loc[i, 'humidity_percent']
            adjusted_humidity = current_humidity + temp_change_effect
            
            # Clamp humidity to realistic range
            humidity_min = 10
            humidity_max = 100
            adjusted_humidity = max(humidity_min, min(humidity_max, adjusted_humidity))
            data.loc[i, 'humidity_percent'] = round(adjusted_humidity, 1)
        
        return data