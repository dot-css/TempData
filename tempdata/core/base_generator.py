"""
Base generator class for all dataset generators

Provides common functionality including seeding, localization, and validation.
"""

import pandas as pd
from faker import Faker
from typing import Any
from .seeding import MillisecondSeeder


class BaseGenerator:
    """
    Abstract base class for all dataset generators
    
    Provides common functionality including seeder integration, localization,
    and data validation that all specific generators inherit.
    """
    
    def __init__(self, seeder: MillisecondSeeder, locale: str = 'en_US'):
        """
        Initialize base generator with seeder and locale
        
        Args:
            seeder: MillisecondSeeder instance for reproducible randomness
            locale: Locale string for localization (default: 'en_US')
        """
        self.seeder = seeder
        self.locale = locale
        self.faker = Faker(locale)
        self.faker.seed_instance(seeder.get_contextual_seed(self.__class__.__name__))
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Abstract method for generating dataset
        
        Args:
            rows: Number of rows to generate
            **kwargs: Additional generation parameters
            
        Returns:
            pd.DataFrame: Generated dataset
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement generate() method")
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply domain-specific realistic patterns to generated data
        
        Args:
            data: Generated data to apply patterns to
            
        Returns:
            pd.DataFrame: Data with realistic patterns applied
        """
        # Base implementation - subclasses should override for specific patterns
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate generated data meets quality standards
        
        Args:
            data: Generated data to validate
            
        Returns:
            bool: True if data passes validation
        """
        # Basic validation - subclasses can extend
        return not data.empty and len(data) > 0