"""
Data loading utilities for efficient reference data access

Provides enhanced data loading with multi-level caching and geographical utilities.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import threading
from functools import lru_cache
import logging

from ..core.caching import LazyDataLoader, CacheConfig

logger = logging.getLogger(__name__)


class GeographicalDataLoader(LazyDataLoader):
    """
    Specialized data loader for geographical and reference data
    
    Extends the enhanced LazyDataLoader with geographical-specific
    methods and fallback logic for country data.
    """
    
    def __init__(self, config: Optional[Union[CacheConfig, Path]] = None, data_root: Optional[Path] = None):
        """
        Initialize geographical data loader
        
        Args:
            config: Cache configuration or data root path (for backward compatibility)
            data_root: Root path for data files (defaults to package data directory)
        """
        # Handle backward compatibility - if config is a Path, treat it as data_root
        if isinstance(config, Path):
            data_root = config
            config = None
        
        if data_root is None:
            data_root = Path(__file__).parent
        
        super().__init__(config, data_root)
        
        # Validate data directory exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
    
    def get_country_data(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get country data with fallback to comprehensive data
        
        Args:
            country: Specific country code (if None, returns all countries)
            
        Returns:
            Dict[str, Any]: Country data
        """
        # Try comprehensive data first, fallback to original
        try:
            data = self.load_data("countries/comprehensive_country_data.json")
        except (FileNotFoundError, RuntimeError):
            try:
                data = self.load_data("countries/country_data.json")
            except (FileNotFoundError, RuntimeError):
                logger.warning("No country data files found, using minimal fallback")
                data = {
                    'global': {
                        'name': 'Global',
                        'locale': 'en_US',
                        'currency': 'USD',
                        'currency_symbol': '$'
                    }
                }
        
        if country:
            country_key = country.lower()
            if country_key in data:
                return data[country_key]
            else:
                logger.warning(f"Country '{country}' not found, using global fallback")
                return data.get('global', {})
        
        return data
    
    def get_city_boundaries(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get city boundary data for geographical coordinate generation
        
        Args:
            country: Specific country code (if None, returns all countries)
            
        Returns:
            Dict[str, Any]: City boundary data
        """
        try:
            # Try comprehensive boundaries first
            data = self.load_data("countries/comprehensive_city_boundaries.json")
        except (FileNotFoundError, RuntimeError):
            try:
                data = self.load_data("countries/city_boundaries.json")
            except (FileNotFoundError, RuntimeError):
                logger.warning("No city boundary data found")
                return {}
        
        if country:
            country_key = country.lower()
            return data.get(country_key, {})
        
        return data
    
    def get_states_provinces(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get states/provinces data with cities
        
        Args:
            country: Specific country code (if None, returns all countries)
            
        Returns:
            Dict[str, Any]: States/provinces data
        """
        try:
            data = self.load_data("countries/states_provinces.json")
        except (FileNotFoundError, RuntimeError):
            logger.warning("No states/provinces data found")
            return {}
        
        if country:
            country_key = country.lower()
            return data.get(country_key, {})
        
        return data
    
    def get_postal_codes(self, country: str) -> List[str]:
        """
        Get postal codes for a specific country
        
        Args:
            country: Country code
            
        Returns:
            List[str]: List of postal codes
        """
        try:
            data = self.load_data(f"countries/postal_codes/{country.lower()}.json")
            return data.get('postal_codes', [])
        except (FileNotFoundError, RuntimeError):
            logger.warning(f"No postal codes found for country: {country}")
            return []


class CountryDataManager:
    """
    High-level manager for country-specific data access
    
    Provides convenient methods for accessing geographical and
    localization data with built-in validation and fallbacks.
    """
    
    def __init__(self, data_loader: Optional[GeographicalDataLoader] = None):
        """
        Initialize country data manager
        
        Args:
            data_loader: Optional data loader instance (creates new if None)
        """
        self.data_loader = data_loader or GeographicalDataLoader()
        self._supported_countries_cache = None
    
    @lru_cache(maxsize=128)
    def get_supported_countries(self) -> List[str]:
        """
        Get list of all supported countries with caching
        
        Returns:
            List[str]: List of supported country codes
        """
        if self._supported_countries_cache is None:
            country_data = self.data_loader.get_country_data()
            self._supported_countries_cache = list(country_data.keys())
        
        return self._supported_countries_cache
    
    def is_country_supported(self, country: str) -> bool:
        """
        Check if a country is supported
        
        Args:
            country: Country code
            
        Returns:
            bool: True if country is supported
        """
        return country.lower() in self.get_supported_countries()
    
    def get_country_info(self, country: str) -> Dict[str, Any]:
        """
        Get comprehensive country information
        
        Args:
            country: Country code
            
        Returns:
            Dict[str, Any]: Country information
        """
        return self.data_loader.get_country_data(country)
    
    def get_cities_for_country(self, country: str) -> List[str]:
        """
        Get list of cities for a country
        
        Args:
            country: Country code
            
        Returns:
            List[str]: List of city names
        """
        # Try states/provinces data first
        states_data = self.data_loader.get_states_provinces(country)
        if states_data:
            cities = []
            for state_cities in states_data.values():
                cities.extend(state_cities)
            return cities
        
        # Fallback to city boundaries data
        boundaries_data = self.data_loader.get_city_boundaries(country)
        return list(boundaries_data.keys()) if boundaries_data else []
    
    def get_city_coordinates(self, country: str, city: str) -> Optional[Dict[str, Any]]:
        """
        Get coordinates and boundaries for a specific city
        
        Args:
            country: Country code
            city: City name
            
        Returns:
            Optional[Dict[str, Any]]: City coordinate data or None if not found
        """
        boundaries_data = self.data_loader.get_city_boundaries(country)
        city_key = city.lower().replace(' ', '_')
        return boundaries_data.get(city_key)
    
    def validate_geographical_data(self, country: str, city: Optional[str] = None) -> bool:
        """
        Validate that geographical data exists for country/city combination
        
        Args:
            country: Country code
            city: Optional city name
            
        Returns:
            bool: True if data exists and is valid
        """
        if not self.is_country_supported(country):
            return False
        
        if city:
            cities = self.get_cities_for_country(country)
            return city.lower() in [c.lower() for c in cities]
        
        return True


# Global instances for easy access
_default_loader = None
_default_manager = None


def get_data_loader() -> GeographicalDataLoader:
    """Get default data loader instance (singleton)"""
    global _default_loader
    if _default_loader is None:
        _default_loader = GeographicalDataLoader()
    return _default_loader


def get_country_manager() -> CountryDataManager:
    """Get default country data manager instance (singleton)"""
    global _default_manager
    if _default_manager is None:
        _default_manager = CountryDataManager(get_data_loader())
    return _default_manager