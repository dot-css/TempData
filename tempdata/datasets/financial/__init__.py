"""
Financial dataset generators

Provides generators for financial datasets including stocks, banking, crypto,
insurance, loans, investments, accounting, and payments.
"""

from .stocks import StockGenerator
from .banking import BankingGenerator

__all__ = [
    "StockGenerator", 
    "BankingGenerator"
]