"""
Time-based seeding system for TempData

Provides millisecond-precision seeding for unique yet reproducible data generation.
"""

import time
import os
import hashlib
import secrets
from typing import Dict, Optional


class MillisecondSeeder:
    """
    Advanced seeder for unique data generation
    
    This class provides a sophisticated seeding system that combines multiple entropy sources
    for unpredictable yet reproducible data generation when needed.
    """
    
    def __init__(self, fixed_seed: Optional[int] = None):
        """
        Initialize the seeder with optional fixed seed
        
        Args:
            fixed_seed: Optional fixed seed for reproducible results
        """
        if fixed_seed is not None:
            self.seed = fixed_seed
            self.is_fixed = True
        else:
            # Create complex seed from multiple entropy sources
            self.seed = self._generate_complex_seed()
            self.is_fixed = False
        
        self.base_time = time.time()
        self._context_seeds: Dict[str, int] = {}
        self._entropy_pool = self._initialize_entropy_pool()
    
    def _generate_complex_seed(self) -> int:
        """
        Generate a complex seed using multiple entropy sources
        
        Returns:
            int: Complex seed value
        """
        # Combine multiple entropy sources
        entropy_sources = []
        
        # High-precision timestamp with microseconds
        timestamp = int(time.time() * 1_000_000)
        entropy_sources.append(timestamp.to_bytes(8, 'big'))
        
        # Process ID for uniqueness across processes
        pid = os.getpid()
        entropy_sources.append(pid.to_bytes(4, 'big'))
        
        # Cryptographically secure random bytes
        crypto_random = secrets.token_bytes(16)
        entropy_sources.append(crypto_random)
        
        # Memory address of a new object for additional entropy
        memory_addr = id(object())
        entropy_sources.append(memory_addr.to_bytes(8, 'big'))
        
        # Thread ID if available
        try:
            import threading
            thread_id = threading.get_ident()
            entropy_sources.append(thread_id.to_bytes(8, 'big'))
        except:
            pass
        
        # Combine all entropy sources
        combined_entropy = b''.join(entropy_sources)
        
        # Use SHA-256 to create final seed
        seed_hash = hashlib.sha256(combined_entropy).digest()
        
        # Convert to integer and ensure it fits in 32-bit range
        return int.from_bytes(seed_hash[:4], 'big')
    
    def _initialize_entropy_pool(self) -> bytes:
        """
        Initialize entropy pool for additional randomness
        
        Returns:
            bytes: Initial entropy pool
        """
        if self.is_fixed:
            # For fixed seeds, create deterministic entropy pool
            return hashlib.sha256(str(self.seed).encode()).digest()
        else:
            # For dynamic seeds, use cryptographic randomness
            return secrets.token_bytes(32)
    
    def get_contextual_seed(self, context: str) -> int:
        """
        Generate consistent seed for specific contexts using advanced hashing
        
        Args:
            context: Context identifier for seed generation
            
        Returns:
            int: Context-specific seed value
        """
        if context not in self._context_seeds:
            # Create complex context-specific seed
            context_data = f"{self.seed}_{context}_{len(self._context_seeds)}"
            
            # Use PBKDF2 for key stretching to make seed derivation more complex
            import hashlib
            context_bytes = context_data.encode('utf-8')
            salt = self._entropy_pool[:16]  # Use part of entropy pool as salt
            
            # Perform key derivation with multiple iterations
            derived_key = hashlib.pbkdf2_hmac('sha256', context_bytes, salt, 10000, 32)
            
            # Convert to integer seed
            self._context_seeds[context] = int.from_bytes(derived_key[:4], 'big') % (2**31)
        
        return self._context_seeds[context]
    
    def get_temporal_seed(self, offset_seconds: int = 0) -> int:
        """
        Generate seed with time offset for time series using enhanced entropy
        
        Args:
            offset_seconds: Time offset in seconds
            
        Returns:
            int: Time-offset seed value
        """
        # Create time-based seed with additional complexity
        time_offset = int((self.base_time + offset_seconds) * 1_000_000)  # Microsecond precision
        
        # Combine with base seed and entropy
        time_data = f"{self.seed}_{time_offset}_{offset_seconds}"
        time_hash = hashlib.sha256(time_data.encode() + self._entropy_pool[:8]).digest()
        
        return int.from_bytes(time_hash[:4], 'big') % (2**32)
    
    def get_derived_seed(self, derivation_key: str, iteration: int = 0) -> int:
        """
        Generate derived seed for specific use cases
        
        Args:
            derivation_key: Key for seed derivation
            iteration: Iteration number for multiple derived seeds
            
        Returns:
            int: Derived seed value
        """
        derivation_data = f"{self.seed}_{derivation_key}_{iteration}"
        derivation_hash = hashlib.sha256(derivation_data.encode() + self._entropy_pool).digest()
        
        return int.from_bytes(derivation_hash[:4], 'big') % (2**31)
    
    def refresh_entropy(self) -> None:
        """
        Refresh the entropy pool with new random data
        """
        if not self.is_fixed:
            # Only refresh for non-fixed seeds to maintain reproducibility
            new_entropy = secrets.token_bytes(16)
            current_hash = hashlib.sha256(self._entropy_pool).digest()
            self._entropy_pool = hashlib.sha256(current_hash + new_entropy).digest()
    
    def get_seed_info(self) -> Dict[str, any]:
        """
        Get information about the current seed state
        
        Returns:
            Dict containing seed information
        """
        return {
            'seed': self.seed,
            'is_fixed': self.is_fixed,
            'base_time': self.base_time,
            'context_count': len(self._context_seeds),
            'entropy_pool_hash': hashlib.sha256(self._entropy_pool).hexdigest()[:16]
        }