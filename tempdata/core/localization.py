"""
Localization engine for country-specific data generation

Handles country-specific data loading, formatting, and locale management.
"""

from typing import Dict, Any, Optional, List
import json
import os
import re
import random
from pathlib import Path


class LocalizationEngine:
    """
    Engine for handling country-specific localization and data formatting
    
    Manages locale-specific data loading and provides country-specific
    formatting for addresses, phone numbers, and other localized data.
    Supports 20+ countries with proper locale handling.
    """
    
    def __init__(self):
        """Initialize localization engine"""
        self._country_data: Dict[str, Any] = {}
        self._loaded_countries = set()
        self._data_path = Path(__file__).parent.parent / "data" / "countries"
        self._load_all_country_data()
    
    def _load_all_country_data(self) -> None:
        """Load all country data from JSON file"""
        try:
            country_data_file = self._data_path / "country_data.json"
            if country_data_file.exists():
                with open(country_data_file, 'r', encoding='utf-8') as f:
                    self._country_data = json.load(f)
                    self._loaded_countries = set(self._country_data.keys())
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Fallback to default data if file loading fails
            self._country_data = {
                'global': {
                    'name': 'Global',
                    'locale': 'en_US',
                    'currency': 'USD',
                    'currency_symbol': '$',
                    'phone_format': '+#-###-###-####',
                    'postal_format': '#####',
                    'postal_regex': '^[A-Z0-9]{3,10}$',
                    'address_format': '{street}\n{city}, {state} {postal_code}',
                    'date_format': 'YYYY-MM-DD',
                    'decimal_separator': '.',
                    'thousands_separator': ',',
                    'timezone': 'UTC'
                }
            }
            self._loaded_countries = {'global'}
    
    def get_supported_countries(self) -> List[str]:
        """
        Get list of all supported countries
        
        Returns:
            List[str]: List of supported country codes
        """
        return list(self._loaded_countries)
    
    def is_country_supported(self, country: str) -> bool:
        """
        Check if a country is supported
        
        Args:
            country: Country code or name
            
        Returns:
            bool: True if country is supported
        """
        return country.lower() in self._loaded_countries
    
    def load_country_data(self, country: str) -> Dict[str, Any]:
        """
        Load country-specific data for localization
        
        Args:
            country: Country code or name
            
        Returns:
            Dict[str, Any]: Country-specific data
        """
        if not country:
            # Handle None or empty string
            country = 'global'
            
        country_key = country.lower()
        
        if country_key in self._country_data:
            return self._country_data[country_key]
        
        # Fallback to global if country not found
        return self._country_data.get('global', {
            'locale': 'en_US',
            'currency': 'USD',
            'phone_format': '+1-###-###-####',
            'postal_format': '#####'
        })
    
    def get_locale(self, country: str) -> str:
        """
        Get appropriate locale for country
        
        Args:
            country: Country code or name
            
        Returns:
            str: Locale string (e.g., 'en_US', 'ur_PK')
        """
        country_data = self.load_country_data(country)
        return country_data.get('locale', 'en_US')
    
    def get_currency_info(self, country: str) -> Dict[str, str]:
        """
        Get currency information for country
        
        Args:
            country: Country code or name
            
        Returns:
            Dict[str, str]: Currency code and symbol
        """
        country_data = self.load_country_data(country)
        return {
            'code': country_data.get('currency', 'USD'),
            'symbol': country_data.get('currency_symbol', '$')
        }
    
    def get_number_formatting(self, country: str) -> Dict[str, str]:
        """
        Get number formatting rules for country
        
        Args:
            country: Country code or name
            
        Returns:
            Dict[str, str]: Decimal and thousands separators
        """
        country_data = self.load_country_data(country)
        return {
            'decimal_separator': country_data.get('decimal_separator', '.'),
            'thousands_separator': country_data.get('thousands_separator', ',')
        }
    
    def format_phone(self, country: str, number: str = None) -> str:
        """
        Format phone number according to country standards
        
        Args:
            country: Country code or name
            number: Raw phone number (if None, generates random)
            
        Returns:
            str: Formatted phone number
        """
        country_data = self.load_country_data(country)
        format_pattern = country_data.get('phone_format', '+1-###-###-####')
        
        if number is None:
            # Generate random phone number
            formatted = format_pattern
            for char in formatted:
                if char == '#':
                    formatted = formatted.replace('#', str(random.randint(0, 9)), 1)
            return formatted
        
        # Format existing number
        digits = re.sub(r'\D', '', number)
        formatted = format_pattern
        digit_index = 0
        
        result = ""
        for char in formatted:
            if char == '#' and digit_index < len(digits):
                result += digits[digit_index]
                digit_index += 1
            elif char != '#':
                result += char
        
        return result
    
    def format_postal_code(self, country: str, code: str = None) -> str:
        """
        Format postal code according to country standards
        
        Args:
            country: Country code or name
            code: Raw postal code (if None, generates random)
            
        Returns:
            str: Formatted postal code
        """
        country_data = self.load_country_data(country)
        format_pattern = country_data.get('postal_format', '#####')
        
        if code is None:
            # Generate random postal code
            formatted = format_pattern
            for char in formatted:
                if char == '#':
                    formatted = formatted.replace('#', str(random.randint(0, 9)), 1)
            return formatted
        
        # Format existing code
        if format_pattern == '### ###':  # UK/Canada style
            if len(code) >= 6:
                return f"{code[:3]} {code[3:6]}"
        elif format_pattern == '####-###':  # Brazil style
            if len(code) >= 8:
                return f"{code[:5]}-{code[5:8]}"
        elif format_pattern == '###-####':  # Japan style
            if len(code) >= 7:
                return f"{code[:3]}-{code[3:7]}"
        
        return code
    
    def validate_postal_code(self, country: str, code: str) -> bool:
        """
        Validate postal code format for country
        
        Args:
            country: Country code or name
            code: Postal code to validate
            
        Returns:
            bool: True if valid format
        """
        country_data = self.load_country_data(country)
        regex_pattern = country_data.get('postal_regex', r'^\d{5}$')
        
        try:
            return bool(re.match(regex_pattern, code))
        except re.error:
            return False
    
    def get_address_format(self, country: str) -> str:
        """
        Get address formatting template for country
        
        Args:
            country: Country code or name
            
        Returns:
            str: Address format template
        """
        country_data = self.load_country_data(country)
        return country_data.get('address_format', '{street}\n{city}, {state} {postal_code}')
    
    def format_address(self, country: str, address_parts: Dict[str, str]) -> str:
        """
        Format address according to country standards
        
        Args:
            country: Country code or name
            address_parts: Dictionary with address components
            
        Returns:
            str: Formatted address
        """
        format_template = self.get_address_format(country)
        
        try:
            return format_template.format(**address_parts)
        except KeyError:
            # Fallback if some keys are missing
            safe_parts = {k: v for k, v in address_parts.items() if v}
            return format_template.format(**safe_parts)
    
    def get_date_format(self, country: str) -> str:
        """
        Get date format for country
        
        Args:
            country: Country code or name
            
        Returns:
            str: Date format string
        """
        country_data = self.load_country_data(country)
        return country_data.get('date_format', 'YYYY-MM-DD')
    
    def get_timezone(self, country: str) -> str:
        """
        Get primary timezone for country
        
        Args:
            country: Country code or name
            
        Returns:
            str: Timezone string
        """
        country_data = self.load_country_data(country)
        return country_data.get('timezone', 'UTC')
    
    def format_currency(self, country: str, amount: float) -> str:
        """
        Format currency amount according to country standards
        
        Args:
            country: Country code or name
            amount: Amount to format
            
        Returns:
            str: Formatted currency string
        """
        currency_info = self.get_currency_info(country)
        number_format = self.get_number_formatting(country)
        
        # Format the number with appropriate separators
        decimal_sep = number_format['decimal_separator']
        thousands_sep = number_format['thousands_separator']
        
        # Convert to string with 2 decimal places
        amount_str = f"{amount:.2f}"
        integer_part, decimal_part = amount_str.split('.')
        
        # Add thousands separators
        if len(integer_part) > 3:
            formatted_integer = ""
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_integer = thousands_sep + formatted_integer
                formatted_integer = digit + formatted_integer
            integer_part = formatted_integer
        
        # Combine with decimal separator
        formatted_amount = integer_part + decimal_sep + decimal_part
        
        return f"{currency_info['symbol']}{formatted_amount}"