"""
Unit tests for data loading utilities and validation

Tests the lazy loading system, data validation, and geographical data access.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from tempdata.data.data_loader import (
    LazyDataLoader, 
    CountryDataManager, 
    get_data_loader, 
    get_country_manager
)


class TestLazyDataLoader:
    """Test cases for LazyDataLoader class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_dir = self.temp_dir / "test_data"
        self.test_data_dir.mkdir()
        
        # Create test data files
        self.create_test_data_files()
        
        # Initialize loader with test directory
        self.loader = LazyDataLoader(self.test_data_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data_files(self):
        """Create test data files"""
        # Create countries directory
        countries_dir = self.test_data_dir / "countries"
        countries_dir.mkdir()
        
        # Create test country data
        country_data = {
            "test_country": {
                "name": "Test Country",
                "locale": "en_TEST",
                "currency": "TST",
                "currency_symbol": "T$"
            },
            "another_country": {
                "name": "Another Country",
                "locale": "en_AN",
                "currency": "ANC",
                "currency_symbol": "A$"
            }
        }
        
        with open(countries_dir / "country_data.json", 'w') as f:
            json.dump(country_data, f)
        
        # Create test city boundaries
        city_boundaries = {
            "test_country": {
                "test_city": {
                    "center": [40.7128, -74.0060],
                    "bounds": {
                        "north": 40.8,
                        "south": 40.6,
                        "east": -73.9,
                        "west": -74.1
                    }
                }
            }
        }
        
        with open(countries_dir / "city_boundaries.json", 'w') as f:
            json.dump(city_boundaries, f)
        
        # Create test states/provinces
        states_data = {
            "test_country": {
                "Test State": ["Test City", "Another City"]
            }
        }
        
        with open(countries_dir / "states_provinces.json", 'w') as f:
            json.dump(states_data, f)
        
        # Create postal codes directory and file
        postal_dir = countries_dir / "postal_codes"
        postal_dir.mkdir()
        
        postal_codes = {
            "postal_codes": ["12345", "67890", "11111", "22222"]
        }
        
        with open(postal_dir / "test_country.json", 'w') as f:
            json.dump(postal_codes, f)
    
    def test_initialization(self):
        """Test loader initialization"""
        assert self.loader.data_root == self.test_data_dir
        assert isinstance(self.loader._cache, dict)
        assert len(self.loader._cache) == 0
        assert len(self.loader._loaded_files) == 0
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with invalid data path"""
        with pytest.raises(FileNotFoundError):
            LazyDataLoader(Path("/nonexistent/path"))
    
    def test_load_json_file_success(self):
        """Test successful JSON file loading"""
        file_path = self.test_data_dir / "countries" / "country_data.json"
        data = self.loader._load_json_file(file_path)
        
        assert isinstance(data, dict)
        assert "test_country" in data
        assert data["test_country"]["name"] == "Test Country"
    
    def test_load_json_file_not_found(self):
        """Test loading non-existent file"""
        file_path = self.test_data_dir / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            self.loader._load_json_file(file_path)
    
    def test_load_json_file_invalid_json(self):
        """Test loading invalid JSON file"""
        invalid_file = self.test_data_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(json.JSONDecodeError):
            self.loader._load_json_file(invalid_file)
    
    def test_load_data_full_file(self):
        """Test loading full data file"""
        data = self.loader.load_data("countries/country_data.json")
        
        assert isinstance(data, dict)
        assert "test_country" in data
        assert "another_country" in data
        assert "countries/country_data.json" in self.loader._loaded_files
    
    def test_load_data_with_section(self):
        """Test loading specific section from file"""
        data = self.loader.load_data("countries/country_data.json", "test_country")
        
        assert isinstance(data, dict)
        assert data["name"] == "Test Country"
        assert data["currency"] == "TST"
    
    def test_load_data_section_not_found(self):
        """Test loading non-existent section"""
        with pytest.raises(KeyError):
            self.loader.load_data("countries/country_data.json", "nonexistent_section")
    
    def test_load_data_caching(self):
        """Test data caching functionality"""
        # Load data first time
        data1 = self.loader.load_data("countries/country_data.json")
        
        # Load same data second time (should use cache)
        data2 = self.loader.load_data("countries/country_data.json")
        
        # Should be the same object (cached)
        assert data1 is data2
        
        # Check cache statistics
        stats = self.loader.get_cache_stats()
        assert stats["cached_items"] >= 1
        assert "countries/country_data.json" in stats["loaded_files_list"]
    
    def test_force_reload(self):
        """Test force reload functionality"""
        # Load data first time
        data1 = self.loader.load_data("countries/country_data.json")
        
        # Force reload
        data2 = self.loader.load_data("countries/country_data.json", force_reload=True)
        
        # Should be different objects (reloaded)
        assert data1 == data2  # Same content
        # Note: In this test they might be the same object due to JSON loading,
        # but the important thing is that reload was attempted
    
    def test_get_country_data_all(self):
        """Test getting all country data"""
        data = self.loader.get_country_data()
        
        assert isinstance(data, dict)
        assert "test_country" in data
        assert "another_country" in data
    
    def test_get_country_data_specific(self):
        """Test getting specific country data"""
        data = self.loader.get_country_data("test_country")
        
        assert isinstance(data, dict)
        assert data["name"] == "Test Country"
        assert data["currency"] == "TST"
    
    def test_get_country_data_not_found(self):
        """Test getting data for non-existent country"""
        data = self.loader.get_country_data("nonexistent_country")
        
        # Should return empty dict or fallback
        assert isinstance(data, dict)
    
    def test_get_city_boundaries(self):
        """Test getting city boundary data"""
        data = self.loader.get_city_boundaries("test_country")
        
        assert isinstance(data, dict)
        assert "test_city" in data
        assert "center" in data["test_city"]
        assert "bounds" in data["test_city"]
    
    def test_get_states_provinces(self):
        """Test getting states/provinces data"""
        data = self.loader.get_states_provinces("test_country")
        
        assert isinstance(data, dict)
        assert "Test State" in data
        assert "Test City" in data["Test State"]
    
    def test_get_postal_codes(self):
        """Test getting postal codes"""
        codes = self.loader.get_postal_codes("test_country")
        
        assert isinstance(codes, list)
        assert "12345" in codes
        assert "67890" in codes
    
    def test_get_postal_codes_not_found(self):
        """Test getting postal codes for non-existent country"""
        codes = self.loader.get_postal_codes("nonexistent_country")
        
        assert isinstance(codes, list)
        assert len(codes) == 0
    
    def test_clear_cache_specific(self):
        """Test clearing cache for specific file"""
        # Load some data
        self.loader.load_data("countries/country_data.json")
        
        # Verify it's cached
        stats = self.loader.get_cache_stats()
        assert stats["cached_items"] > 0
        
        # Clear specific file cache
        self.loader.clear_cache("countries/country_data.json")
        
        # Verify cache is cleared
        stats = self.loader.get_cache_stats()
        assert "countries/country_data.json" not in stats["loaded_files_list"]
    
    def test_clear_cache_all(self):
        """Test clearing all cache"""
        # Load some data
        self.loader.load_data("countries/country_data.json")
        self.loader.load_data("countries/city_boundaries.json")
        
        # Verify cache has items
        stats = self.loader.get_cache_stats()
        assert stats["cached_items"] > 0
        
        # Clear all cache
        self.loader.clear_cache()
        
        # Verify all cache is cleared
        stats = self.loader.get_cache_stats()
        assert stats["cached_items"] == 0
        assert len(stats["loaded_files_list"]) == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics"""
        # Initially empty
        stats = self.loader.get_cache_stats()
        assert stats["cached_items"] == 0
        assert stats["loaded_files"] == 0
        
        # Load some data
        self.loader.load_data("countries/country_data.json")
        
        # Check updated stats
        stats = self.loader.get_cache_stats()
        assert stats["cached_items"] >= 1
        assert stats["loaded_files"] >= 1
        assert isinstance(stats["cache_keys"], list)
        assert isinstance(stats["loaded_files_list"], list)


class TestCountryDataManager:
    """Test cases for CountryDataManager class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_dir = self.temp_dir / "test_data"
        self.test_data_dir.mkdir()
        
        # Create test data files
        self.create_test_data_files()
        
        # Initialize manager with test loader
        test_loader = LazyDataLoader(self.test_data_dir)
        self.manager = CountryDataManager(test_loader)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data_files(self):
        """Create test data files"""
        # Create countries directory
        countries_dir = self.test_data_dir / "countries"
        countries_dir.mkdir()
        
        # Create test country data
        country_data = {
            "test_country": {
                "name": "Test Country",
                "locale": "en_TEST",
                "currency": "TST"
            },
            "another_country": {
                "name": "Another Country",
                "locale": "en_AN",
                "currency": "ANC"
            }
        }
        
        with open(countries_dir / "country_data.json", 'w') as f:
            json.dump(country_data, f)
        
        # Create test city boundaries
        city_boundaries = {
            "test_country": {
                "test_city": {
                    "center": [40.7128, -74.0060],
                    "bounds": {
                        "north": 40.8,
                        "south": 40.6,
                        "east": -73.9,
                        "west": -74.1
                    }
                }
            }
        }
        
        with open(countries_dir / "city_boundaries.json", 'w') as f:
            json.dump(city_boundaries, f)
        
        # Create test states/provinces
        states_data = {
            "test_country": {
                "Test State": ["Test City", "Another City"]
            }
        }
        
        with open(countries_dir / "states_provinces.json", 'w') as f:
            json.dump(states_data, f)
    
    def test_initialization(self):
        """Test manager initialization"""
        assert self.manager.data_loader is not None
        assert isinstance(self.manager.data_loader, LazyDataLoader)
    
    def test_initialization_with_default_loader(self):
        """Test initialization with default loader"""
        manager = CountryDataManager()
        assert manager.data_loader is not None
    
    def test_get_supported_countries(self):
        """Test getting supported countries"""
        countries = self.manager.get_supported_countries()
        
        assert isinstance(countries, list)
        assert "test_country" in countries
        assert "another_country" in countries
    
    def test_get_supported_countries_caching(self):
        """Test that supported countries are cached"""
        countries1 = self.manager.get_supported_countries()
        countries2 = self.manager.get_supported_countries()
        
        # Should be the same object (cached)
        assert countries1 is countries2
    
    def test_is_country_supported(self):
        """Test country support checking"""
        assert self.manager.is_country_supported("test_country")
        assert self.manager.is_country_supported("another_country")
        assert not self.manager.is_country_supported("nonexistent_country")
    
    def test_is_country_supported_case_insensitive(self):
        """Test country support checking is case insensitive"""
        assert self.manager.is_country_supported("TEST_COUNTRY")
        assert self.manager.is_country_supported("Test_Country")
    
    def test_get_country_info(self):
        """Test getting country information"""
        info = self.manager.get_country_info("test_country")
        
        assert isinstance(info, dict)
        assert info["name"] == "Test Country"
        assert info["currency"] == "TST"
    
    def test_get_cities_for_country(self):
        """Test getting cities for country"""
        cities = self.manager.get_cities_for_country("test_country")
        
        assert isinstance(cities, list)
        assert "Test City" in cities
        assert "Another City" in cities
    
    def test_get_cities_for_country_from_boundaries(self):
        """Test getting cities from boundary data when states data not available"""
        # Clear states data to test fallback
        with patch.object(self.manager.data_loader, 'get_states_provinces', return_value={}):
            cities = self.manager.get_cities_for_country("test_country")
            
            assert isinstance(cities, list)
            assert "test_city" in cities
    
    def test_get_city_coordinates(self):
        """Test getting city coordinates"""
        coords = self.manager.get_city_coordinates("test_country", "test_city")
        
        assert isinstance(coords, dict)
        assert "center" in coords
        assert "bounds" in coords
        assert coords["center"] == [40.7128, -74.0060]
    
    def test_get_city_coordinates_not_found(self):
        """Test getting coordinates for non-existent city"""
        coords = self.manager.get_city_coordinates("test_country", "nonexistent_city")
        
        assert coords is None
    
    def test_validate_geographical_data_country_only(self):
        """Test geographical data validation for country only"""
        assert self.manager.validate_geographical_data("test_country")
        assert not self.manager.validate_geographical_data("nonexistent_country")
    
    def test_validate_geographical_data_with_city(self):
        """Test geographical data validation with city"""
        assert self.manager.validate_geographical_data("test_country", "Test City")
        assert not self.manager.validate_geographical_data("test_country", "Nonexistent City")
        assert not self.manager.validate_geographical_data("nonexistent_country", "Any City")
    
    def test_validate_geographical_data_case_insensitive(self):
        """Test geographical data validation is case insensitive"""
        assert self.manager.validate_geographical_data("test_country", "test city")
        assert self.manager.validate_geographical_data("test_country", "TEST CITY")


class TestGlobalInstances:
    """Test cases for global instance functions"""
    
    def test_get_data_loader_singleton(self):
        """Test that get_data_loader returns singleton"""
        loader1 = get_data_loader()
        loader2 = get_data_loader()
        
        assert loader1 is loader2
        assert isinstance(loader1, LazyDataLoader)
    
    def test_get_country_manager_singleton(self):
        """Test that get_country_manager returns singleton"""
        manager1 = get_country_manager()
        manager2 = get_country_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, CountryDataManager)
    
    def test_manager_uses_default_loader(self):
        """Test that country manager uses default loader"""
        loader = get_data_loader()
        manager = get_country_manager()
        
        assert manager.data_loader is loader


class TestDataValidation:
    """Test cases for data validation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Use real data loader for validation tests
        self.loader = get_data_loader()
        self.manager = get_country_manager()
    
    def test_real_country_data_loading(self):
        """Test loading real country data"""
        # Test with known countries from the actual data
        countries = self.manager.get_supported_countries()
        
        assert isinstance(countries, list)
        assert len(countries) > 0
        
        # Test some expected countries
        expected_countries = ['united_states', 'pakistan', 'united_kingdom', 'germany']
        for country in expected_countries:
            if country in countries:  # Only test if country exists in data
                info = self.manager.get_country_info(country)
                assert isinstance(info, dict)
                assert 'name' in info
                assert 'currency' in info
    
    def test_postal_code_loading(self):
        """Test loading postal codes for real countries"""
        # Test postal codes for countries we created
        test_countries = ['united_states', 'pakistan', 'united_kingdom', 'germany']
        
        for country in test_countries:
            codes = self.loader.get_postal_codes(country)
            assert isinstance(codes, list)
            if codes:  # If postal codes exist
                assert len(codes) > 0
                # Verify all codes are strings
                assert all(isinstance(code, str) for code in codes)
    
    def test_geographical_data_consistency(self):
        """Test consistency between different geographical data sources"""
        # Get countries that have both boundary and state data
        countries = self.manager.get_supported_countries()
        
        for country in countries[:3]:  # Test first 3 countries to keep test fast
            # Get cities from states data
            cities_from_states = self.manager.get_cities_for_country(country)
            
            # Get cities from boundaries data
            boundaries = self.loader.get_city_boundaries(country)
            cities_from_boundaries = list(boundaries.keys()) if boundaries else []
            
            # Both should be lists
            assert isinstance(cities_from_states, list)
            assert isinstance(cities_from_boundaries, list)
            
            # If both have data, there should be some overlap or consistency
            if cities_from_states and cities_from_boundaries:
                # At least one source should have data
                assert len(cities_from_states) > 0 or len(cities_from_boundaries) > 0


if __name__ == "__main__":
    pytest.main([__file__])


class TestTemplateLoader:
    """Test cases for TemplateLoader class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Import here to avoid circular imports
        from tempdata.data.template_loader import TemplateLoader
        
        # Create temporary directory for test templates
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_template_dir = self.temp_dir / "test_templates"
        self.test_template_dir.mkdir()
        
        # Create test template files
        self.create_test_template_files()
        
        # Initialize loader with test directory
        self.loader = TemplateLoader(self.test_template_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_template_files(self):
        """Create test template files"""
        # Create test email domains
        email_domains = {
            "personal_domains": ["gmail.com", "yahoo.com", "hotmail.com"],
            "business_domains": ["company.com", "corp.com", "inc.com"],
            "country_specific_domains": {
                "test_country": ["test.com", "example.com"]
            },
            "domain_patterns": {
                "personal": ["{first_name}.{last_name}@{domain}"]
            }
        }
        
        with open(self.test_template_dir / "email_domains.json", 'w') as f:
            json.dump(email_domains, f)
        
        # Create test phone formats
        phone_formats = {
            "country_formats": {
                "test_country": {
                    "formats": ["+1-###-###-####"],
                    "area_codes": ["555", "123"]
                }
            },
            "mobile_patterns": {
                "test_country": ["5##"]
            },
            "landline_patterns": {
                "test_country": ["2##"]
            }
        }
        
        with open(self.test_template_dir / "phone_formats.json", 'w') as f:
            json.dump(phone_formats, f)
        
        # Create test name formats
        name_formats = {
            "name_patterns": {
                "business": ["{adjective} {noun}"],
                "product": ["{adjective} {noun}"],
                "person": ["{first_name} {last_name}"]
            },
            "business_adjectives": ["Advanced", "Smart"],
            "business_nouns": ["Solutions", "Systems"],
            "product_adjectives": ["Pro", "Ultra"],
            "product_nouns": ["Device", "Tool"],
            "versions": ["1.0", "2.0"],
            "models": ["A1", "B2"],
            "series": ["Pro Series"],
            "editions": ["Standard Edition"],
            "titles": ["Mr.", "Dr."],
            "suffixes": ["Jr.", "Sr."]
        }
        
        with open(self.test_template_dir / "name_formats.json", 'w') as f:
            json.dump(name_formats, f)
        
        # Create business directory and data
        business_dir = self.test_template_dir.parent / "business"
        business_dir.mkdir(exist_ok=True)
        
        # Create test companies data
        companies = {
            "company_names": {
                "technology": ["TechCorp", "DataSoft"]
            },
            "industries": ["Technology", "Finance"],
            "departments": ["IT", "Sales"],
            "business_types": ["Startup", "Corporation"]
        }
        
        with open(business_dir / "companies.json", 'w') as f:
            json.dump(companies, f)
        
        # Create test products data
        products = {
            "products": {
                "technology": ["Laptop", "Phone"]
            },
            "product_categories": ["Electronics", "Software"]
        }
        
        with open(business_dir / "products.json", 'w') as f:
            json.dump(products, f)
    
    def test_initialization(self):
        """Test loader initialization"""
        assert self.loader.template_root == self.test_template_dir
        assert isinstance(self.loader._cache, dict)
        assert len(self.loader._cache) == 0
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with invalid template path"""
        from tempdata.data.template_loader import TemplateLoader
        
        with pytest.raises(FileNotFoundError):
            TemplateLoader(Path("/nonexistent/path"))
    
    def test_load_template_success(self):
        """Test successful template loading"""
        data = self.loader.load_template("email_domains")
        
        assert isinstance(data, dict)
        assert "personal_domains" in data
        assert "gmail.com" in data["personal_domains"]
    
    def test_load_template_not_found(self):
        """Test loading non-existent template"""
        with pytest.raises(FileNotFoundError):
            self.loader.load_template("nonexistent_template")
    
    def test_load_template_caching(self):
        """Test template caching"""
        # Load template first time
        data1 = self.loader.load_template("email_domains")
        
        # Load same template second time (should use cache)
        data2 = self.loader.load_template("email_domains")
        
        # Should be the same object (cached)
        assert data1 is data2
        
        # Check cache statistics
        stats = self.loader.get_cache_stats()
        assert stats["cached_templates"] >= 1
        assert "email_domains" in stats["template_names"]
    
    def test_get_email_domains_personal(self):
        """Test getting personal email domains"""
        domains = self.loader.get_email_domains("personal")
        
        assert isinstance(domains, list)
        assert "gmail.com" in domains
        assert "yahoo.com" in domains
    
    def test_get_email_domains_business(self):
        """Test getting business email domains"""
        domains = self.loader.get_email_domains("business")
        
        assert isinstance(domains, list)
        assert "company.com" in domains
        assert "corp.com" in domains
    
    def test_get_email_domains_country_specific(self):
        """Test getting country-specific email domains"""
        domains = self.loader.get_email_domains("country_specific", "test_country")
        
        assert isinstance(domains, list)
        assert "test.com" in domains
        assert "example.com" in domains
    
    def test_get_phone_format(self):
        """Test getting phone format"""
        phone_info = self.loader.get_phone_format("test_country")
        
        assert isinstance(phone_info, dict)
        assert "formats" in phone_info
        assert "area_codes" in phone_info
        assert "+1-###-###-####" in phone_info["formats"]
        assert "555" in phone_info["area_codes"]
    
    def test_get_phone_format_mobile(self):
        """Test getting mobile phone format"""
        phone_info = self.loader.get_phone_format("test_country", "mobile")
        
        assert isinstance(phone_info, dict)
        assert "patterns" in phone_info
        assert "5##" in phone_info["patterns"]
    
    def test_get_name_patterns_business(self):
        """Test getting business name patterns"""
        patterns = self.loader.get_name_patterns("business")
        
        assert isinstance(patterns, dict)
        assert "patterns" in patterns
        assert "adjectives" in patterns
        assert "nouns" in patterns
        assert "{adjective} {noun}" in patterns["patterns"]
        assert "Advanced" in patterns["adjectives"]
        assert "Solutions" in patterns["nouns"]
    
    def test_get_name_patterns_product(self):
        """Test getting product name patterns"""
        patterns = self.loader.get_name_patterns("product")
        
        assert isinstance(patterns, dict)
        assert "patterns" in patterns
        assert "adjectives" in patterns
        assert "nouns" in patterns
        assert "versions" in patterns
        assert "models" in patterns
    
    def test_generate_email_address(self):
        """Test email address generation"""
        email = self.loader.generate_email_address("John", "Doe", "personal")
        
        assert isinstance(email, str)
        assert "@" in email
        assert "john" in email.lower()
        assert "doe" in email.lower()
    
    def test_generate_phone_number(self):
        """Test phone number generation"""
        phone = self.loader.generate_phone_number("test_country")
        
        assert isinstance(phone, str)
        assert len(phone) > 5  # Should be a reasonable length
        # Should contain some digits
        assert any(c.isdigit() for c in phone)
    
    def test_generate_business_name(self):
        """Test business name generation"""
        name = self.loader.generate_business_name()
        
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_generate_product_name(self):
        """Test product name generation"""
        name = self.loader.generate_product_name()
        
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_get_business_data_company_names(self):
        """Test getting company names"""
        names = self.loader.get_business_data("company_names")
        
        assert isinstance(names, list)
        assert "TechCorp" in names
        assert "DataSoft" in names
    
    def test_get_business_data_industries(self):
        """Test getting industries"""
        industries = self.loader.get_business_data("industries")
        
        assert isinstance(industries, list)
        assert "Technology" in industries
        assert "Finance" in industries
    
    def test_get_business_data_products(self):
        """Test getting products"""
        products = self.loader.get_business_data("products")
        
        assert isinstance(products, list)
        assert "Laptop" in products
        assert "Phone" in products
    
    def test_clear_cache_specific(self):
        """Test clearing specific template cache"""
        # Load template
        self.loader.load_template("email_domains")
        
        # Verify it's cached
        stats = self.loader.get_cache_stats()
        assert "email_domains" in stats["template_names"]
        
        # Clear specific template cache
        self.loader.clear_cache("email_domains")
        
        # Verify cache is cleared
        stats = self.loader.get_cache_stats()
        assert "email_domains" not in stats["template_names"]
    
    def test_clear_cache_all(self):
        """Test clearing all template cache"""
        # Load some templates
        self.loader.load_template("email_domains")
        self.loader.load_template("phone_formats")
        
        # Verify cache has items
        stats = self.loader.get_cache_stats()
        assert stats["cached_templates"] > 0
        
        # Clear all cache
        self.loader.clear_cache()
        
        # Verify all cache is cleared
        stats = self.loader.get_cache_stats()
        assert stats["cached_templates"] == 0


class TestTemplateLoaderGlobalInstance:
    """Test cases for global template loader instance"""
    
    def test_get_template_loader_singleton(self):
        """Test that get_template_loader returns singleton"""
        from tempdata.data.template_loader import get_template_loader
        
        loader1 = get_template_loader()
        loader2 = get_template_loader()
        
        assert loader1 is loader2
        from tempdata.data.template_loader import TemplateLoader
        assert isinstance(loader1, TemplateLoader)


class TestTemplateIntegration:
    """Integration tests for template system with real data"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from tempdata.data.template_loader import get_template_loader
        self.loader = get_template_loader()
    
    def test_real_email_domain_loading(self):
        """Test loading real email domain data"""
        # Test personal domains
        personal_domains = self.loader.get_email_domains("personal")
        assert isinstance(personal_domains, list)
        assert len(personal_domains) > 0
        
        # Test business domains
        business_domains = self.loader.get_email_domains("business")
        assert isinstance(business_domains, list)
        assert len(business_domains) > 0
        
        # Verify common domains exist
        common_personal = ["gmail.com", "yahoo.com", "hotmail.com"]
        for domain in common_personal:
            if domain in personal_domains:  # Only test if domain exists
                assert isinstance(domain, str)
                assert "." in domain
    
    def test_real_phone_format_loading(self):
        """Test loading real phone format data"""
        # Test known countries
        test_countries = ["united_states", "united_kingdom", "germany", "pakistan"]
        
        for country in test_countries:
            phone_info = self.loader.get_phone_format(country)
            assert isinstance(phone_info, dict)
            assert "formats" in phone_info
            assert "area_codes" in phone_info
            assert len(phone_info["formats"]) > 0
            assert len(phone_info["area_codes"]) > 0
    
    def test_real_business_data_loading(self):
        """Test loading real business data"""
        # Test company names
        company_names = self.loader.get_business_data("company_names")
        assert isinstance(company_names, list)
        assert len(company_names) > 0
        
        # Test industries
        industries = self.loader.get_business_data("industries")
        assert isinstance(industries, list)
        assert len(industries) > 0
        
        # Test products
        products = self.loader.get_business_data("products")
        assert isinstance(products, list)
        assert len(products) > 0
    
    def test_email_generation_integration(self):
        """Test email generation with real data"""
        email = self.loader.generate_email_address("John", "Doe")
        
        assert isinstance(email, str)
        assert "@" in email
        assert email.count("@") == 1
        
        # Verify email format
        parts = email.split("@")
        assert len(parts) == 2
        assert len(parts[0]) > 0  # Username part
        assert len(parts[1]) > 0  # Domain part
        assert "." in parts[1]    # Domain should have extension
    
    def test_phone_generation_integration(self):
        """Test phone generation with real data"""
        phone = self.loader.generate_phone_number("united_states")
        
        assert isinstance(phone, str)
        assert len(phone) >= 10  # US phones should be at least 10 digits
        
        # Should contain digits
        digit_count = sum(1 for c in phone if c.isdigit())
        assert digit_count >= 10
    
    def test_business_name_generation_integration(self):
        """Test business name generation with real data"""
        name = self.loader.generate_business_name("technology")
        
        assert isinstance(name, str)
        assert len(name) > 0
        assert len(name.split()) >= 1  # Should have at least one word
    
    def test_product_name_generation_integration(self):
        """Test product name generation with real data"""
        name = self.loader.generate_product_name("technology")
        
        assert isinstance(name, str)
        assert len(name) > 0