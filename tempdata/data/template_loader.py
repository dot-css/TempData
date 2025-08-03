"""
Template loading and caching system

Provides efficient loading and caching of template data for
email domains, phone formats, name patterns, and other templates.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from functools import lru_cache
import threading
import logging

logger = logging.getLogger(__name__)


class TemplateLoader:
    """
    Template loading and caching system for various data templates
    
    Manages loading and caching of template data including email domains,
    phone formats, name patterns, and other template-based data generation.
    """
    
    def __init__(self, template_root: Optional[Path] = None):
        """
        Initialize template loader
        
        Args:
            template_root: Root path for template files (defaults to package templates directory)
        """
        if template_root is None:
            template_root = Path(__file__).parent / "templates"
        
        self.template_root = Path(template_root)
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        
        # Validate template directory exists
        if not self.template_root.exists():
            raise FileNotFoundError(f"Template directory not found: {self.template_root}")
    
    def _load_template_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load template file with error handling
        
        Args:
            file_path: Path to template file
            
        Returns:
            Dict[str, Any]: Loaded template data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in template file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading template file {file_path}: {e}")
            raise
    
    def load_template(self, template_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load template data with caching
        
        Args:
            template_name: Name of template file (without .json extension)
            force_reload: Force reload even if cached
            
        Returns:
            Dict[str, Any]: Template data
        """
        with self._cache_lock:
            # Return cached data if available and not forcing reload
            if not force_reload and template_name in self._cache:
                return self._cache[template_name]
            
            # Load template file
            template_file = self.template_root / f"{template_name}.json"
            
            try:
                data = self._load_template_file(template_file)
                self._cache[template_name] = data
                logger.debug(f"Loaded template: {template_name}")
                return data
                
            except Exception as e:
                logger.error(f"Failed to load template {template_name}: {e}")
                raise
    
    def get_email_domains(self, domain_type: str = "personal", 
                         country: Optional[str] = None) -> List[str]:
        """
        Get email domains for specified type and country
        
        Args:
            domain_type: Type of domains ('personal', 'business', 'country_specific')
            country: Country code for country-specific domains
            
        Returns:
            List[str]: List of email domains
        """
        email_data = self.load_template("email_domains")
        
        if domain_type == "country_specific" and country:
            country_domains = email_data.get("country_specific_domains", {})
            return country_domains.get(country.lower(), email_data.get("personal_domains", []))
        elif domain_type == "business":
            return email_data.get("business_domains", [])
        elif domain_type == "personal":
            return email_data.get("personal_domains", [])
        else:
            return email_data.get("personal_domains", [])
    
    def get_phone_format(self, country: str, phone_type: str = "any") -> Dict[str, Any]:
        """
        Get phone format information for country
        
        Args:
            country: Country code
            phone_type: Type of phone ('mobile', 'landline', 'any')
            
        Returns:
            Dict[str, Any]: Phone format information
        """
        phone_data = self.load_template("phone_formats")
        country_formats = phone_data.get("country_formats", {})
        
        country_key = country.lower()
        if country_key not in country_formats:
            # Fallback to US format
            country_key = "united_states"
        
        country_info = country_formats.get(country_key, {})
        
        result = {
            "formats": country_info.get("formats", ["+1-###-###-####"]),
            "area_codes": country_info.get("area_codes", ["555"])
        }
        
        # Add specific patterns if requested
        if phone_type == "mobile":
            mobile_patterns = phone_data.get("mobile_patterns", {})
            result["patterns"] = mobile_patterns.get(country_key, ["###"])
        elif phone_type == "landline":
            landline_patterns = phone_data.get("landline_patterns", {})
            result["patterns"] = landline_patterns.get(country_key, ["###"])
        
        return result
    
    def get_name_patterns(self, pattern_type: str = "person") -> Dict[str, Any]:
        """
        Get name patterns for specified type
        
        Args:
            pattern_type: Type of name patterns ('person', 'business', 'product')
            
        Returns:
            Dict[str, Any]: Name pattern data
        """
        name_data = self.load_template("name_formats")
        
        if pattern_type == "business":
            return {
                "patterns": name_data.get("name_patterns", {}).get("business", []),
                "adjectives": name_data.get("business_adjectives", []),
                "nouns": name_data.get("business_nouns", [])
            }
        elif pattern_type == "product":
            return {
                "patterns": name_data.get("name_patterns", {}).get("product", []),
                "adjectives": name_data.get("product_adjectives", []),
                "nouns": name_data.get("product_nouns", []),
                "versions": name_data.get("versions", []),
                "models": name_data.get("models", []),
                "series": name_data.get("series", []),
                "editions": name_data.get("editions", [])
            }
        else:  # person
            return {
                "patterns": name_data.get("name_patterns", {}).get("person", []),
                "titles": name_data.get("titles", []),
                "suffixes": name_data.get("suffixes", [])
            }
    
    def generate_email_address(self, first_name: str, last_name: str, 
                              domain_type: str = "personal", 
                              country: Optional[str] = None) -> str:
        """
        Generate email address using templates
        
        Args:
            first_name: First name
            last_name: Last name
            domain_type: Type of email domain
            country: Country for country-specific domains
            
        Returns:
            str: Generated email address
        """
        email_data = self.load_template("email_domains")
        
        # Get domain
        domains = self.get_email_domains(domain_type, country)
        domain = random.choice(domains) if domains else "example.com"
        
        # Get pattern
        patterns = email_data.get("domain_patterns", {}).get("personal", [
            "{first_name}.{last_name}@{domain}",
            "{first_name}{last_name}@{domain}",
            "{first_name}_{last_name}@{domain}"
        ])
        
        pattern = random.choice(patterns)
        
        # Generate email
        email = pattern.format(
            first_name=first_name.lower(),
            last_name=last_name.lower(),
            first_initial=first_name[0].lower() if first_name else "a",
            last_initial=last_name[0].lower() if last_name else "a",
            domain=domain,
            birth_year=random.randint(1970, 2005),
            random_number=random.randint(1, 999),
            username=f"{first_name.lower()}{random.randint(1, 99)}"
        )
        
        return email
    
    def generate_phone_number(self, country: str, phone_type: str = "any") -> str:
        """
        Generate phone number using country-specific formats
        
        Args:
            country: Country code
            phone_type: Type of phone ('mobile', 'landline', 'any')
            
        Returns:
            str: Generated phone number
        """
        phone_info = self.get_phone_format(country, phone_type)
        
        # Choose format
        formats = phone_info.get("formats", ["+1-###-###-####"])
        format_pattern = random.choice(formats)
        
        # Choose area code
        area_codes = phone_info.get("area_codes", ["555"])
        area_code = random.choice(area_codes)
        
        # Generate phone number
        phone = format_pattern
        
        # Replace area code placeholder if present
        if "###" in phone and area_code:
            phone = phone.replace("###", area_code, 1)
        
        # Replace remaining # with random digits
        while "#" in phone:
            phone = phone.replace("#", str(random.randint(0, 9)), 1)
        
        return phone
    
    def generate_business_name(self, industry: Optional[str] = None) -> str:
        """
        Generate business name using templates
        
        Args:
            industry: Industry type for industry-specific names
            
        Returns:
            str: Generated business name
        """
        name_patterns = self.get_name_patterns("business")
        
        # Choose pattern
        patterns = name_patterns.get("patterns", ["{adjective} {noun}"])
        pattern = random.choice(patterns)
        
        # Get components
        adjectives = name_patterns.get("adjectives", ["Advanced"])
        nouns = name_patterns.get("nouns", ["Solutions"])
        
        # Generate name
        name = pattern.format(
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            suffix=random.choice(["Inc", "Corp", "LLC", "Ltd"]),
            founder_name=f"{random.choice(['Smith', 'Johnson', 'Williams'])}",
            location=random.choice(["Metro", "Central", "Global"]),
            industry=industry or random.choice(["Tech", "Business", "Professional"]),
            initials=f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
        )
        
        return name
    
    def generate_product_name(self, category: Optional[str] = None) -> str:
        """
        Generate product name using templates
        
        Args:
            category: Product category
            
        Returns:
            str: Generated product name
        """
        name_patterns = self.get_name_patterns("product")
        
        # Choose pattern
        patterns = name_patterns.get("patterns", ["{adjective} {noun}"])
        pattern = random.choice(patterns)
        
        # Get components
        adjectives = name_patterns.get("adjectives", ["Smart"])
        nouns = name_patterns.get("nouns", ["Device"])
        versions = name_patterns.get("versions", ["1.0"])
        models = name_patterns.get("models", ["A1"])
        series = name_patterns.get("series", ["Pro Series"])
        editions = name_patterns.get("editions", ["Standard Edition"])
        
        # Generate name
        name = pattern.format(
            adjective=random.choice(adjectives),
            noun=random.choice(nouns),
            brand=random.choice(["TechPro", "SmartChoice", "Premium"]),
            version=random.choice(versions),
            model=random.choice(models),
            series=random.choice(series),
            edition=random.choice(editions)
        )
        
        return name
    
    def get_business_data(self, data_type: str) -> List[str]:
        """
        Get business-related data from templates
        
        Args:
            data_type: Type of business data to retrieve
            
        Returns:
            List[str]: List of business data items
        """
        # Load business data from business directory
        try:
            business_file = self.template_root.parent / "business" / "companies.json"
            products_file = self.template_root.parent / "business" / "products.json"
            
            business_data = {}
            products_data = {}
            
            if business_file.exists():
                business_data = self._load_template_file(business_file)
            
            if products_file.exists():
                products_data = self._load_template_file(products_file)
            
            if data_type == "company_names":
                # Flatten all company names from different industries
                all_names = []
                company_names = business_data.get("company_names", {})
                for industry_names in company_names.values():
                    all_names.extend(industry_names)
                return all_names
            
            elif data_type == "industries":
                return business_data.get("industries", [])
            
            elif data_type == "departments":
                return business_data.get("departments", [])
            
            elif data_type == "business_types":
                return business_data.get("business_types", [])
            
            elif data_type == "products":
                # Flatten all products from different categories
                all_products = []
                products = products_data.get("products", {})
                for category_products in products.values():
                    all_products.extend(category_products)
                return all_products
            
            elif data_type == "product_categories":
                return products_data.get("product_categories", [])
            
            elif data_type in business_data:
                return business_data[data_type]
            
            elif data_type in products_data:
                return products_data[data_type]
            
            else:
                logger.warning(f"Unknown business data type: {data_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading business data for type {data_type}: {e}")
            return []
    
    def clear_cache(self, template_name: Optional[str] = None) -> None:
        """
        Clear template cache
        
        Args:
            template_name: Specific template to clear (if None, clears all)
        """
        with self._cache_lock:
            if template_name:
                if template_name in self._cache:
                    del self._cache[template_name]
            else:
                self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._cache_lock:
            return {
                "cached_templates": len(self._cache),
                "template_names": list(self._cache.keys())
            }


# Global instance for easy access
_default_template_loader = None


def get_template_loader() -> TemplateLoader:
    """Get default template loader instance (singleton)"""
    global _default_template_loader
    if _default_template_loader is None:
        _default_template_loader = TemplateLoader()
    return _default_template_loader