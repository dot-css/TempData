"""
Unit tests for LocalizationEngine

Tests locale-specific data generation and formatting functionality.
"""

import pytest
import json
from tempdata.core.localization import LocalizationEngine


class TestLocalizationEngine:
    """Test cases for LocalizationEngine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = LocalizationEngine()
    
    def test_initialization(self):
        """Test LocalizationEngine initializes correctly"""
        assert self.engine is not None
        assert isinstance(self.engine._country_data, dict)
        assert len(self.engine._loaded_countries) > 0
    
    def test_supported_countries_count(self):
        """Test that 20+ countries are supported"""
        supported = self.engine.get_supported_countries()
        assert len(supported) >= 20, f"Expected 20+ countries, got {len(supported)}"
    
    def test_country_support_check(self):
        """Test country support validation"""
        # Test supported countries
        assert self.engine.is_country_supported('united_states')
        assert self.engine.is_country_supported('pakistan')
        assert self.engine.is_country_supported('germany')
        assert self.engine.is_country_supported('global')
        
        # Test unsupported country
        assert not self.engine.is_country_supported('nonexistent_country')
    
    def test_load_country_data(self):
        """Test loading country-specific data"""
        # Test valid country
        us_data = self.engine.load_country_data('united_states')
        assert us_data['locale'] == 'en_US'
        assert us_data['currency'] == 'USD'
        assert us_data['phone_format'] == '+1-###-###-####'
        
        # Test Pakistan data
        pk_data = self.engine.load_country_data('pakistan')
        assert pk_data['locale'] == 'ur_PK'
        assert pk_data['currency'] == 'PKR'
        assert pk_data['currency_symbol'] == '₨'
        
        # Test fallback for unsupported country
        fallback_data = self.engine.load_country_data('nonexistent')
        assert fallback_data['locale'] == 'en_US'  # Should fallback to global
    
    def test_get_locale(self):
        """Test locale retrieval for different countries"""
        assert self.engine.get_locale('united_states') == 'en_US'
        assert self.engine.get_locale('pakistan') == 'ur_PK'
        assert self.engine.get_locale('germany') == 'de_DE'
        assert self.engine.get_locale('japan') == 'ja_JP'
        assert self.engine.get_locale('china') == 'zh_CN'
        
        # Test fallback
        assert self.engine.get_locale('nonexistent') == 'en_US'
    
    def test_currency_info(self):
        """Test currency information retrieval"""
        # Test US currency
        us_currency = self.engine.get_currency_info('united_states')
        assert us_currency['code'] == 'USD'
        assert us_currency['symbol'] == '$'
        
        # Test Pakistan currency
        pk_currency = self.engine.get_currency_info('pakistan')
        assert pk_currency['code'] == 'PKR'
        assert pk_currency['symbol'] == '₨'
        
        # Test Euro countries
        de_currency = self.engine.get_currency_info('germany')
        assert de_currency['code'] == 'EUR'
        assert de_currency['symbol'] == '€'
    
    def test_number_formatting(self):
        """Test number formatting rules"""
        # Test US formatting
        us_format = self.engine.get_number_formatting('united_states')
        assert us_format['decimal_separator'] == '.'
        assert us_format['thousands_separator'] == ','
        
        # Test German formatting
        de_format = self.engine.get_number_formatting('germany')
        assert de_format['decimal_separator'] == ','
        assert de_format['thousands_separator'] == '.'
        
        # Test French formatting
        fr_format = self.engine.get_number_formatting('france')
        assert fr_format['decimal_separator'] == ','
        assert fr_format['thousands_separator'] == ' '
    
    def test_phone_formatting(self):
        """Test phone number formatting"""
        # Test US phone formatting
        us_phone = self.engine.format_phone('united_states', '1234567890')
        assert us_phone.startswith('+1-')
        assert len(us_phone.replace('-', '').replace('+', '')) == 11
        
        # Test random phone generation
        random_us_phone = self.engine.format_phone('united_states')
        assert random_us_phone.startswith('+1-')
        assert '-' in random_us_phone
        
        # Test Pakistan phone formatting
        pk_phone = self.engine.format_phone('pakistan')
        assert pk_phone.startswith('+92-')
    
    def test_postal_code_formatting(self):
        """Test postal code formatting"""
        # Test US postal code
        us_postal = self.engine.format_postal_code('united_states', '12345')
        assert len(us_postal) == 5
        assert us_postal.isdigit()
        
        # Test UK postal code formatting
        uk_postal = self.engine.format_postal_code('united_kingdom', 'SW1A1AA')
        assert ' ' in uk_postal or len(uk_postal) >= 6
        
        # Test random postal code generation
        random_postal = self.engine.format_postal_code('united_states')
        assert len(random_postal) == 5
        assert random_postal.isdigit()
    
    def test_postal_code_validation(self):
        """Test postal code validation"""
        # Test US postal codes
        assert self.engine.validate_postal_code('united_states', '12345')
        assert self.engine.validate_postal_code('united_states', '12345-6789')
        assert not self.engine.validate_postal_code('united_states', 'ABCDE')
        
        # Test UK postal codes
        assert self.engine.validate_postal_code('united_kingdom', 'SW1A 1AA')
        assert not self.engine.validate_postal_code('united_kingdom', '12345')
        
        # Test German postal codes
        assert self.engine.validate_postal_code('germany', '12345')
        assert not self.engine.validate_postal_code('germany', '123')
    
    def test_address_formatting(self):
        """Test address formatting"""
        address_parts = {
            'street': '123 Main St',
            'city': 'New York',
            'state': 'NY',
            'postal_code': '10001'
        }
        
        # Test US address format
        us_address = self.engine.format_address('united_states', address_parts)
        assert '123 Main St' in us_address
        assert 'New York, NY 10001' in us_address
        
        # Test German address format (postal code before city)
        de_address = self.engine.format_address('germany', address_parts)
        assert '10001 New York' in de_address
    
    def test_date_format(self):
        """Test date format retrieval"""
        assert self.engine.get_date_format('united_states') == 'MM/DD/YYYY'
        assert self.engine.get_date_format('united_kingdom') == 'DD/MM/YYYY'
        assert self.engine.get_date_format('germany') == 'DD.MM.YYYY'
        assert self.engine.get_date_format('japan') == 'YYYY/MM/DD'
    
    def test_timezone(self):
        """Test timezone retrieval"""
        assert self.engine.get_timezone('united_states') == 'America/New_York'
        assert self.engine.get_timezone('pakistan') == 'Asia/Karachi'
        assert self.engine.get_timezone('germany') == 'Europe/Berlin'
        assert self.engine.get_timezone('japan') == 'Asia/Tokyo'
    
    def test_currency_formatting(self):
        """Test currency amount formatting"""
        # Test US currency formatting
        us_formatted = self.engine.format_currency('united_states', 1234.56)
        assert us_formatted == '$1,234.56'
        
        # Test German currency formatting (different separators)
        de_formatted = self.engine.format_currency('germany', 1234.56)
        assert de_formatted == '€1.234,56'
        
        # Test large amounts
        large_amount = self.engine.format_currency('united_states', 1234567.89)
        assert large_amount == '$1,234,567.89'
    
    def test_multiple_countries_support(self):
        """Test that multiple specific countries are supported"""
        required_countries = [
            'united_states', 'pakistan', 'united_kingdom', 'germany',
            'france', 'japan', 'china', 'india', 'brazil', 'canada',
            'australia', 'russia', 'south_korea', 'mexico', 'italy',
            'spain', 'netherlands', 'sweden', 'norway', 'switzerland',
            'turkey', 'south_africa', 'global'
        ]
        
        supported = self.engine.get_supported_countries()
        
        for country in required_countries:
            assert country in supported, f"Country {country} should be supported"
    
    def test_locale_specific_data_generation(self):
        """Test that locale-specific data is properly loaded"""
        # Test that different countries have different locales
        locales = set()
        test_countries = ['united_states', 'pakistan', 'germany', 'japan', 'france']
        
        for country in test_countries:
            locale = self.engine.get_locale(country)
            locales.add(locale)
        
        # Should have multiple different locales
        assert len(locales) >= 4, "Should have multiple different locales"
        
        # Test specific locale mappings
        assert self.engine.get_locale('pakistan') == 'ur_PK'
        assert self.engine.get_locale('japan') == 'ja_JP'
        assert self.engine.get_locale('china') == 'zh_CN'
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with None input
        assert self.engine.get_locale(None) == 'en_US'
        
        # Test with empty string
        assert self.engine.get_locale('') == 'en_US'
        
        # Test postal code validation with invalid regex
        # Should not crash and return False
        result = self.engine.validate_postal_code('nonexistent', '12345')
        assert isinstance(result, bool)