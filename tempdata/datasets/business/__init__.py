"""
Business dataset generators

Provides generators for business-related datasets including sales, customers,
ecommerce, inventory, marketing, employees, suppliers, retail, reviews, and CRM.
"""

from .sales import SalesGenerator
from .customers import CustomerGenerator
from .ecommerce import EcommerceGenerator
from .marketing import MarketingGenerator

__all__ = [
    "SalesGenerator",
    "CustomerGenerator",
    "EcommerceGenerator",
    "MarketingGenerator"
]