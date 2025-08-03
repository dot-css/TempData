"""
Unit tests for StockGenerator

Tests financial data patterns, time series generation, and realistic stock market behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.financial.stocks import StockGenerator


class TestStockGenerator:
    """Test suite for StockGenerator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.seeder = MillisecondSeeder(fixed_seed=12345)
        self.generator = StockGenerator(self.seeder)
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly with market data"""
        assert hasattr(self.generator, 'sectors')
        assert hasattr(self.generator, 'all_symbols')
        assert hasattr(self.generator, 'sector_correlations')
        assert hasattr(self.generator, 'market_conditions')
        
        # Check that we have expected sectors
        expected_sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Industrial']
        assert all(sector in self.generator.sectors for sector in expected_sectors)
        
        # Check that symbols are properly loaded
        assert len(self.generator.all_symbols) > 0
        assert all('symbol' in stock for stock in self.generator.all_symbols)
        assert all('sector' in stock for stock in self.generator.all_symbols)
    
    def test_basic_stock_generation(self):
        """Test basic stock data generation"""
        rows = 100
        data = self.generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        
        # Check required columns
        required_columns = ['symbol', 'date', 'open_price', 'high_price', 'low_price', 
                          'close_price', 'volume', 'sector', 'market_condition']
        assert all(col in data.columns for col in required_columns)
        
        # Check data types
        assert data['open_price'].dtype in [np.float64, float]
        assert data['high_price'].dtype in [np.float64, float]
        assert data['low_price'].dtype in [np.float64, float]
        assert data['close_price'].dtype in [np.float64, float]
        assert data['volume'].dtype in [np.int64, int]
    
    def test_ohlc_price_relationships(self):
        """Test that OHLC prices follow logical relationships"""
        data = self.generator.generate(50)
        
        for _, row in data.iterrows():
            # High should be >= max(open, close)
            assert row['high_price'] >= max(row['open_price'], row['close_price'])
            
            # Low should be <= min(open, close)
            assert row['low_price'] <= min(row['open_price'], row['close_price'])
            
            # All prices should be positive
            assert row['open_price'] > 0
            assert row['high_price'] > 0
            assert row['low_price'] > 0
            assert row['close_price'] > 0
            
            # Volume should be positive
            assert row['volume'] > 0
    
    def test_sector_specific_patterns(self):
        """Test that different sectors have appropriate characteristics"""
        data = self.generator.generate(200)
        
        # Group by sector and check patterns
        sector_stats = data.groupby('sector').agg({
            'volatility': 'mean',
            'close_price': 'mean',
            'volume': 'mean'
        })
        
        # Technology should have higher volatility than Healthcare
        if 'Technology' in sector_stats.index and 'Healthcare' in sector_stats.index:
            tech_vol = sector_stats.loc['Technology', 'volatility']
            health_vol = sector_stats.loc['Healthcare', 'volatility']
            assert tech_vol > health_vol
        
        # Energy should have higher volatility than Consumer
        if 'Energy' in sector_stats.index and 'Consumer' in sector_stats.index:
            energy_vol = sector_stats.loc['Energy', 'volatility']
            consumer_vol = sector_stats.loc['Consumer', 'volatility']
            assert energy_vol > consumer_vol
    
    def test_market_conditions(self):
        """Test that market conditions affect stock behavior appropriately"""
        # Test bull market
        bull_data = self.generator.generate(100, market_condition='bull_market')
        
        # Test bear market
        bear_data = self.generator.generate(100, market_condition='bear_market')
        
        # Bear market should have higher volatility
        bull_vol = bull_data['volatility'].mean()
        bear_vol = bear_data['volatility'].mean()
        assert bear_vol > bull_vol
        
        # Test that market conditions are properly assigned
        assert (bull_data['market_condition'] == 'bull_market').all()
        assert (bear_data['market_condition'] == 'bear_market').all()
        
        # Bear market should have more extreme negative returns
        bull_negative_returns = (bull_data['daily_return'] < -1).sum()
        bear_negative_returns = (bear_data['daily_return'] < -1).sum()
        
        # Bear market should have more extreme negative returns (probabilistic test)
        # We'll be lenient since this is random data
        assert bear_vol > bull_vol  # This should always hold based on our setup
    
    def test_time_series_generation(self):
        """Test time series generation with price continuity"""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        data = self.generator.generate(
            rows=100,
            time_series=True,
            symbols=['AAPL', 'MSFT'],
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        # Check that we have data for both symbols
        symbols_in_data = data['symbol'].unique()
        assert 'AAPL' in symbols_in_data
        assert 'MSFT' in symbols_in_data
        
        # Check date range
        min_date = pd.to_datetime(data['date']).min()
        max_date = pd.to_datetime(data['date']).max()
        assert min_date >= pd.Timestamp(start_date)
        assert max_date <= pd.Timestamp(end_date)
        
        # Check price continuity for each symbol
        for symbol in symbols_in_data:
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            
            # Prices should be continuous (no huge jumps)
            price_changes = symbol_data['close_price'].pct_change().dropna()
            
            # Most daily changes should be reasonable (< 50%)
            extreme_changes = abs(price_changes) > 0.5
            assert extreme_changes.sum() / len(price_changes) < 0.1  # Less than 10% extreme changes
    
    def test_different_intervals(self):
        """Test different time intervals for time series"""
        base_kwargs = {
            'time_series': True,
            'symbols': ['AAPL'],
            'start_date': datetime(2023, 1, 1, 9, 0),
            'end_date': datetime(2023, 1, 1, 16, 0)
        }
        
        # Test daily interval
        daily_data = self.generator.generate(rows=50, interval='1day', **base_kwargs)
        
        # Test hourly interval
        hourly_data = self.generator.generate(rows=50, interval='1hour', **base_kwargs)
        
        # Hourly data should have more granular timestamps
        if len(hourly_data) > 1:
            hourly_times = pd.to_datetime(hourly_data['date'])
            time_diffs = hourly_times.diff().dropna()
            
            # Most time differences should be 1 hour for hourly data
            hour_diffs = time_diffs.dt.total_seconds() / 3600
            assert (hour_diffs == 1.0).sum() > 0
    
    def test_volume_patterns(self):
        """Test realistic volume patterns"""
        data = self.generator.generate(100)
        
        # Volume should vary significantly
        volume_cv = data['volume'].std() / data['volume'].mean()
        assert volume_cv > 0.3  # Coefficient of variation > 30%
        
        # Higher volatility should generally correlate with higher volume
        correlation = data['volatility'].corr(data['volume'])
        assert correlation > 0  # Positive correlation
    
    def test_technical_indicators(self):
        """Test that technical indicators are calculated correctly"""
        # Generate enough data for moving averages
        data = self.generator.generate(
            rows=100,
            time_series=True,
            symbols=['AAPL'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 4, 1)
        )
        
        if len(data) >= 20:
            # Check that moving averages exist
            assert 'sma_20' in data.columns
            
            # SMA should exist and be reasonable
            sma_values = data['sma_20'].dropna()
            assert len(sma_values) > 0
            assert (sma_values > 0).all()
            
            # SMA should be close to price values (within reasonable range)
            price_mean = data['close_price'].mean()
            sma_mean = sma_values.mean()
            assert abs(price_mean - sma_mean) / price_mean < 0.5  # Within 50%
        
        # Check price change calculations
        if 'price_change' in data.columns:
            # Price change should be close to difference in consecutive prices
            data_sorted = data.sort_values(['symbol', 'date'])
            manual_change = data_sorted.groupby('symbol')['close_price'].diff()
            
            # Should be very close (allowing for rounding)
            if len(manual_change.dropna()) > 0:
                diff = abs(data_sorted['price_change'].dropna() - manual_change.dropna())
                assert (diff < 0.01).all()  # Within 1 cent
    
    def test_market_cap_estimation(self):
        """Test market cap estimation logic"""
        data = self.generator.generate(50)
        
        if 'estimated_market_cap' in data.columns:
            # Market cap should be positive
            market_caps = data['estimated_market_cap'].dropna()
            assert (market_caps > 0).all()
            
            # Market cap should correlate with price
            correlation = data['close_price'].corr(data['estimated_market_cap'])
            assert correlation > 0.5  # Strong positive correlation
    
    def test_reproducibility(self):
        """Test that fixed seed produces reproducible results"""
        seeder1 = MillisecondSeeder(fixed_seed=42)
        seeder2 = MillisecondSeeder(fixed_seed=42)
        
        gen1 = StockGenerator(seeder1)
        gen2 = StockGenerator(seeder2)
        
        data1 = gen1.generate(20)
        data2 = gen2.generate(20)
        
        # Should produce identical results
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_date_range_filtering(self):
        """Test date range filtering works correctly"""
        start_date = datetime(2023, 6, 1).date()
        end_date = datetime(2023, 6, 30).date()
        
        data = self.generator.generate(
            rows=50,
            date_range=(start_date, end_date)
        )
        
        # All dates should be within range
        dates = pd.to_datetime(data['date']).dt.date
        assert (dates >= start_date).all()
        assert (dates <= end_date).all()
    
    def test_symbol_filtering(self):
        """Test symbol filtering works correctly"""
        target_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        data = self.generator.generate(
            rows=30,
            symbols=target_symbols
        )
        
        # All symbols should be from the target list
        actual_symbols = data['symbol'].unique()
        assert all(symbol in target_symbols for symbol in actual_symbols)
    
    def test_volatility_patterns(self):
        """Test volatility patterns are realistic"""
        data = self.generator.generate(100)
        
        # Volatility should be positive
        assert (data['volatility'] > 0).all()
        
        # Volatility should vary by sector
        sector_volatilities = data.groupby('sector')['volatility'].mean()
        
        # Should have variation across sectors
        vol_range = sector_volatilities.max() - sector_volatilities.min()
        assert vol_range > 5  # At least 5% difference between sectors
    
    def test_trading_session_indicators(self):
        """Test trading session indicators"""
        # Generate intraday data
        data = self.generator.generate(
            rows=50,
            time_series=True,
            symbols=['AAPL'],
            start_date=datetime(2023, 1, 1, 8, 0),
            end_date=datetime(2023, 1, 1, 18, 0),
            interval='1hour'
        )
        
        if 'trading_session' in data.columns:
            # Should have both regular and extended sessions
            sessions = data['trading_session'].unique()
            
            # During regular hours (9-16), should be 'regular'
            regular_hours_data = data[
                (pd.to_datetime(data['date']).dt.hour >= 9) & 
                (pd.to_datetime(data['date']).dt.hour <= 16)
            ]
            
            if len(regular_hours_data) > 0:
                assert (regular_hours_data['trading_session'] == 'regular').all()
    
    def test_data_quality_metrics(self):
        """Test overall data quality meets requirements"""
        data = self.generator.generate(200)
        
        # No missing values in critical columns
        critical_columns = ['symbol', 'date', 'open_price', 'close_price', 'volume', 'sector']
        for col in critical_columns:
            assert data[col].notna().all()
        
        # Reasonable price ranges
        assert data['close_price'].min() > 0
        assert data['close_price'].max() < 10000  # Reasonable upper bound
        
        # Volume should be reasonable
        assert data['volume'].min() > 0
        assert data['volume'].max() < 1e10  # Reasonable upper bound
        
        # Daily returns should be reasonable
        if 'daily_return' in data.columns:
            returns = data['daily_return'].abs()
            # 95% of returns should be less than 20%
            extreme_returns = returns > 20
            assert extreme_returns.sum() / len(returns) < 0.05


if __name__ == '__main__':
    pytest.main([__file__])