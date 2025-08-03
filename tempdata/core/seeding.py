"""
Time-based seeding system for TempData

Provides millisecond-precision seeding for unique yet reproducible data generation.
"""

import time
import random
import hashlib
from typing import Dict


class MillisecondSeeder:
    """
    Millisecond-precision seeder for unique data generation
    
    This class provides the foundation for TempData's time-based seeding system,
    ensuring unique data generation while maintaining reproducibility when needed.
    """
    
    def __init__(self, fixed_seed: int = None):
        """
        Initialize the seeder with optional fixed seed
        
        Args:
            fixed_seed: Optional fixed seed for reproducible results
        """
        if fixed_seed:
            self.seed = fixed_seed
        else:
            # Millisecond precision ensures uniqueness
            self.seed = int(time.time() * 1000) % (2**32)
        
        random.seed(self.seed)
        self.base_time = time.time()
        self._context_seeds: Dict[str, int] = {}
    
    def get_contextual_seed(self, context: str) -> int:
        """
        Generate consistent seed for specific contexts
        
        Args:
            context: Context identifier for seed generation
            
        Returns:
            int: Context-specific seed value
        """
        if context not in self._context_seeds:
            context_hash = hashlib.md5(f"{self.seed}_{context}".encode()).hexdigest()
            self._context_seeds[context] = int(context_hash[:8], 16) % (2**31)
        return self._context_seeds[context]
    
    def get_temporal_seed(self, offset_seconds: int = 0) -> int:
        """
        Generate seed with time offset for time series
        
        Args:
            offset_seconds: Time offset in seconds
            
        Returns:
            int: Time-offset seed value
        """
        time_offset = int((self.base_time + offset_seconds) * 1000)
        return time_offset % (2**32)