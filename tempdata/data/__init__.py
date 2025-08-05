"""
Static reference data for realistic data generation

Contains geographical, business, and template data used by generators
to create realistic patterns and localized content.
"""

from .data_loader import (
    CountryDataManager,
    get_data_loader,
    get_country_manager
)
from ..core.caching import LazyDataLoader
from .template_loader import (
    TemplateLoader,
    get_template_loader
)

__all__ = [
    'LazyDataLoader',
    'CountryDataManager', 
    'get_data_loader',
    'get_country_manager',
    'TemplateLoader',
    'get_template_loader'
]