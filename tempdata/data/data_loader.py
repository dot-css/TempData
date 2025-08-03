"""
Data loading utilities for efficient reference data access

Provides lazy loading system for memory efficiency and caching
for frequently accessed geographical and reference data.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import threading
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class LazyDataLoader:
    """
    Lazy loading system for reference data with memory efficiency
    
    Implements caching and lazy loading to minimize memory usage
    while providing fast access to frequently used data.
    """
    
    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize lazy data loader
        
        Args:
            data_root: Root path for data files (defaults to package data directory)
        """
        if data_root is None:
            data_root = Path(__file__).parent
        
        self.data_root = Path(data_root)
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self._loaded_files = set()
        
        # Validate data directory exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
    
    def _get_cache_key(self, file_path: str, section: Optional[str] = None) -> str:
        """Generate cache key for file and optional section"""
        if section:
            return f"{file_path}:{section}"
        return file_path
    
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON file with error handling
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dict[str, Any]: Loaded JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise
    
    def load_data(self, file_path: str, section: Optional[str] = None, 
                  force_reload: bool = False) -> Union[Dict[str, Any], Any]:
        """
        Load data from file with lazy loading and caching
        
        Args:
            file_path: Relative path to data file from data root
            section: Optional section key to extract from loaded data
            force_reload: Force reload even if cached
            
        Returns:
            Union[Dict[str, Any], Any]: Loaded data or section
        """
        cache_key = self._get_cache_key(file_path, section)
        
        with self._cache_lock:
            # Return cached data if available and not forcing reload
            if not force_reload and cache_key in self._cache:
                return self._cache[cache_key]
            
            # Load file if not already loaded or forcing reload
            full_path = self.data_root / file_path
            
            if force_reload or file_path not in self._loaded_files:
                try:
                    data = self._load_json_file(full_path)
                    
                    # Cache the full file data
                    file_cache_key = self._get_cache_key(file_path)
                    self._cache[file_cache_key] = data
                    self._loaded_files.add(file_path)
                    
                    logger.debug(f"Loaded data file: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load data file {file_path}: {e}")
                    raise
            
            # Get the full data from cache
            full_data = self._cache[self._get_cache_key(file_path)]
            
            # Extract section if specified
            if section:
                if section in full_data:
                    section_data = full_data[section]
                    self._cache[cache_key] = section_data
                    return section_data
                else:
                    raise KeyError(f"Section '{section}' not found in {file_path}")
            
            return full_data
    
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
        except FileNotFoundError:
            try:
                data = self.load_data("countries/country_data.json")
            except FileNotFoundError:
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
        except FileNotFoundError:
            try:
                data = self.load_data("countries/city_boundaries.json")
            except FileNotFoundError:
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
        except FileNotFoundError:
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
        except FileNotFoundError:
            logger.warning(f"No postal codes found for country: {country}")
            return []
    
    def clear_cache(self, file_path: Optional[str] = None) -> None:
        """
        Clear cache for specific file or all cached data
        
        Args:
            file_path: Specific file to clear from cache (if None, clears all)
        """
        with self._cache_lock:
            if file_path:
                # Clear specific file and its sections
                keys_to_remove = [k for k in self._cache.keys() 
                                if k.startswith(file_path)]
                for key in keys_to_remove:
                    del self._cache[key]
                
                if file_path in self._loaded_files:
                    self._loaded_files.remove(file_path)
            else:
                # Clear all cache
                self._cache.clear()
                self._loaded_files.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._cache_lock:
            return {
                'cached_items': len(self._cache),
                'loaded_files': len(self._loaded_files),
                'cache_keys': list(self._cache.keys()),
                'loaded_files_list': list(self._loaded_files)
            }


class CountryDataManager:
    """
    High-level manager for country-specific data access
    
    Provides convenient methods for accessing geographical and
    localization data with built-in validation and fallbacks.
    """
    
    def __init__(self, data_loader: Optional[LazyDataLoader] = None):
        """
        Initialize country data manager
        
        Args:
            data_loader: Optional data loader instance (creates new if None)
        """
        self.data_loader = data_loader or LazyDataLoader()
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


def get_data_loader() -> LazyDataLoader:
    """Get default data loader instance (singleton)"""
    global _default_loader
    if _default_loader is None:
        _default_loader = LazyDataLoader()
    return _default_loader


def get_country_manager() -> CountryDataManager:
    """Get default country data manager instance (singleton)"""
    global _default_manager
    if _default_manager is None:
        _default_manager = CountryDataManager(get_data_loader())
    return _default_manager