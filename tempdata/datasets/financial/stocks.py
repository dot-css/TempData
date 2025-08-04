"""
Stock market data generator

Generates realistic stock market data with volatility and trading patterns.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class StockGenerator(BaseGenerator):
    """
    Generator for realistic stock market data
    
    Creates stock datasets with market volatility, trading volumes,
    sector correlations, and realistic price movements.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_market_data()
        self._setup_sector_correlations()
        self._setup_volatility_patterns()
        self._setup_trading_patterns()
        
        # Initialize time series generator for integrated time series support
        from ...core.time_series import TimeSeriesGenerator
        self.time_series_generator = TimeSeriesGenerator(self.seeder, self.locale)
    
    def _setup_market_data(self):
        """Setup market sectors and stock symbols"""
        self.sectors = {
            'Technology': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM'],
                'base_volatility': 0.25,
                'growth_bias': 0.15,
                'price_range': (50, 3000),
                'volume_multiplier': 1.2
            },
            'Healthcare': {
                'symbols': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN'],
                'base_volatility': 0.18,
                'growth_bias': 0.08,
                'price_range': (40, 500),
                'volume_multiplier': 0.8
            },
            'Financial': {
                'symbols': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB'],
                'base_volatility': 0.22,
                'growth_bias': 0.05,
                'price_range': (30, 400),
                'volume_multiplier': 1.0
            },
            'Energy': {
                'symbols': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OKE', 'WMB', 'MPC'],
                'base_volatility': 0.35,
                'growth_bias': 0.02,
                'price_range': (20, 200),
                'volume_multiplier': 0.9
            },
            'Consumer': {
                'symbols': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT', 'HD', 'LOW'],
                'base_volatility': 0.16,
                'growth_bias': 0.06,
                'price_range': (60, 400),
                'volume_multiplier': 0.7
            },
            'Industrial': {
                'symbols': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR'],
                'base_volatility': 0.20,
                'growth_bias': 0.04,
                'price_range': (40, 300),
                'volume_multiplier': 0.8
            }
        }
        
        # Create comprehensive symbol list
        self.all_symbols = []
        for sector, data in self.sectors.items():
            for symbol in data['symbols']:
                self.all_symbols.append({
                    'symbol': symbol,
                    'sector': sector,
                    'base_volatility': data['base_volatility'],
                    'growth_bias': data['growth_bias'],
                    'price_range': data['price_range'],
                    'volume_multiplier': data['volume_multiplier']
                })
    
    def _setup_sector_correlations(self):
        """Setup correlations between sectors"""
        self.sector_correlations = {
            'Technology': {'Technology': 1.0, 'Healthcare': 0.3, 'Financial': 0.4, 'Energy': -0.1, 'Consumer': 0.2, 'Industrial': 0.3},
            'Healthcare': {'Technology': 0.3, 'Healthcare': 1.0, 'Financial': 0.2, 'Energy': 0.1, 'Consumer': 0.4, 'Industrial': 0.2},
            'Financial': {'Technology': 0.4, 'Healthcare': 0.2, 'Financial': 1.0, 'Energy': 0.3, 'Consumer': 0.3, 'Industrial': 0.5},
            'Energy': {'Technology': -0.1, 'Healthcare': 0.1, 'Financial': 0.3, 'Energy': 1.0, 'Consumer': 0.2, 'Industrial': 0.4},
            'Consumer': {'Technology': 0.2, 'Healthcare': 0.4, 'Financial': 0.3, 'Energy': 0.2, 'Consumer': 1.0, 'Industrial': 0.3},
            'Industrial': {'Technology': 0.3, 'Healthcare': 0.2, 'Financial': 0.5, 'Energy': 0.4, 'Consumer': 0.3, 'Industrial': 1.0}
        }
    
    def _setup_volatility_patterns(self):
        """Setup volatility patterns for different market conditions"""
        self.market_conditions = {
            'bull_market': {'volatility_multiplier': 0.8, 'growth_multiplier': 1.5, 'probability': 0.4},
            'bear_market': {'volatility_multiplier': 1.8, 'growth_multiplier': -0.8, 'probability': 0.2},
            'sideways_market': {'volatility_multiplier': 1.0, 'growth_multiplier': 0.1, 'probability': 0.4}
        }
        
        # Intraday volatility patterns (hour of day effects)
        self.intraday_volatility = {
            9: 1.5,   # Market open - high volatility
            10: 1.3,  # Early morning
            11: 1.0,  # Mid morning
            12: 0.8,  # Lunch time - lower volatility
            13: 0.9,  # Early afternoon
            14: 1.1,  # Mid afternoon
            15: 1.4,  # Late afternoon - increased activity
            16: 1.6   # Market close - high volatility
        }
    
    def _setup_trading_patterns(self):
        """Setup realistic trading volume patterns"""
        # Day of week effects on volume
        self.day_of_week_volume = {
            0: 1.1,  # Monday - higher volume
            1: 1.0,  # Tuesday - normal
            2: 1.0,  # Wednesday - normal
            3: 1.0,  # Thursday - normal
            4: 0.9,  # Friday - slightly lower
        }
        
        # Monthly patterns (earnings seasons, etc.)
        self.monthly_volume = {
            1: 1.2,  # January - new year trading
            2: 1.0,  # February - normal
            3: 1.1,  # March - quarter end
            4: 1.3,  # April - earnings season
            5: 1.0,  # May - normal
            6: 1.1,  # June - quarter end
            7: 0.9,  # July - summer lull
            8: 0.8,  # August - vacation time
            9: 1.1,  # September - back to business
            10: 1.3, # October - earnings season
            11: 1.0, # November - normal
            12: 0.9  # December - holiday season
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate stock market dataset
        
        Args:
            rows: Number of stock records to generate
            **kwargs: Additional parameters (time_series, date_range, symbols, etc.)
            
        Returns:
            pd.DataFrame: Generated stock data with realistic patterns
        """
        # Create time series configuration if requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_stocks(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot(rows, **kwargs)
    
    def _generate_snapshot(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot stock data (random dates)"""
        symbols = kwargs.get('symbols', None)
        date_range = kwargs.get('date_range', None)
        
        data = []
        
        for i in range(rows):
            # Select stock symbol
            if symbols:
                stock_info = self.faker.random_element([s for s in self.all_symbols if s['symbol'] in symbols])
            else:
                stock_info = self.faker.random_element(self.all_symbols)
            
            # Generate date
            if date_range:
                start_date, end_date = date_range
                trade_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                trade_date = self.faker.date_this_year()
            
            # Generate stock data
            stock_data = self._generate_stock_record(stock_info, trade_date, **kwargs)
            data.append(stock_data)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_integrated(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate time series stock data using integrated TimeSeriesGenerator"""
        from ...core.time_series import create_time_series_config
        
        # Extract time series parameters
        start_date = kwargs.get('start_date', datetime.now() - timedelta(days=365))
        end_date = kwargs.get('end_date', datetime.now())
        interval = kwargs.get('interval', '1day')
        symbols = kwargs.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        
        # Convert string dates if needed
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Create time series configuration
        ts_config = create_time_series_config(
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            seasonal_patterns=True,
            trend_direction=kwargs.get('trend_direction', 'random'),
            volatility_level=kwargs.get('volatility_level', 0.2),
            correlation_strength=0.6  # Financial correlations
        )
        
        data = []
        
        # Generate data for each symbol
        for symbol in symbols:
            stock_info = next((s for s in self.all_symbols if s['symbol'] == symbol), self.all_symbols[0])
            
            # Get base price for this symbol
            min_price, max_price = stock_info['price_range']
            base_price = self.faker.random.uniform(min_price, max_price)
            
            # Generate base time series using TimeSeriesGenerator
            base_series = self.time_series_generator.generate_time_series_base(
                ts_config, 
                base_value=base_price,
                value_range=(max(1.0, min_price * 0.1), max_price * 2.0)
            )
            
            # Apply financial-specific patterns to each timestamp
            for _, row in base_series.iterrows():
                timestamp = row['timestamp']
                price = row['value']
                
                # Generate realistic OHLC from the time series price
                stock_data = self._generate_financial_record_from_price(
                    stock_info, timestamp, price, **kwargs
                )
                data.append(stock_data)
        
        # Limit to requested rows if needed
        if len(data) > rows:
            data = data[:rows]
        
        df = pd.DataFrame(data)
        
        # Add cross-symbol correlations for realistic market behavior
        df = self._apply_sector_correlations(df)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_stocks(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series stock data using integrated time series system"""
        symbols = kwargs.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        data = []
        
        # Generate data for each symbol
        for symbol in symbols:
            stock_info = next((s for s in self.all_symbols if s['symbol'] == symbol), self.all_symbols[0])
            
            # Get base price for this symbol
            min_price, max_price = stock_info['price_range']
            base_price = self.faker.random.uniform(min_price, max_price)
            
            # Generate base time series using TimeSeriesGenerator
            base_series = self.time_series_generator.generate_time_series_base(
                ts_config, 
                base_value=base_price,
                value_range=(max(1.0, min_price * 0.1), max_price * 2.0)
            )
            
            # Apply financial-specific patterns to each timestamp
            for i, timestamp in enumerate(timestamps):
                if i >= len(base_series):
                    break
                    
                # Get time series price value
                price = base_series.iloc[i]['value']
                
                # Generate realistic OHLC from the time series price
                stock_data = self._generate_financial_record_from_price(
                    stock_info, timestamp, price, **kwargs
                )
                data.append(stock_data)
                
                # Break if we have enough rows
                if len(data) >= rows:
                    break
            
            if len(data) >= rows:
                break
        
        # Limit to requested rows if needed
        if len(data) > rows:
            data = data[:rows]
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships using base generator functionality
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply realistic time-based correlations for financial data
        df = self._apply_financial_time_series_correlations(df, ts_config)
        
        # Add cross-symbol correlations for realistic market behavior
        df = self._apply_sector_correlations(df)
        
        return self._apply_realistic_patterns(df)
    
    def _apply_financial_time_series_correlations(self, df: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply financial-specific time series correlations"""
        if 'date' not in df.columns:
            return df
        
        # Sort by timestamp for proper correlation analysis
        df = df.sort_values('date').reset_index(drop=True)
        
        # Rename date to timestamp for consistency
        if 'date' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = df['date']
        
        # Add time-based features for financial patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        df['is_trading_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16))
        df['is_weekend'] = (df['day_of_week'] >= 5)
        
        # Apply market hours volatility patterns
        trading_hours_mask = df['is_trading_hours']
        df.loc[trading_hours_mask, 'volatility'] *= self.faker.random.uniform(1.1, 1.3)
        
        # Apply end-of-day effects (higher volatility near market close)
        end_of_day_mask = (df['hour'] >= 15)
        df.loc[end_of_day_mask, 'volatility'] *= self.faker.random.uniform(1.2, 1.5)
        
        # Apply volume correlations with volatility
        high_volatility_mask = df['volatility'] > df['volatility'].quantile(0.75)
        volume_multiplier = self.faker.random.uniform(1.3, 1.8)
        df.loc[high_volatility_mask, 'volume'] = (df.loc[high_volatility_mask, 'volume'] * volume_multiplier).astype(int)
        
        # Apply monthly earnings season effects
        earnings_months = [1, 4, 7, 10]  # Quarterly earnings months
        earnings_mask = df['month'].isin(earnings_months)
        df.loc[earnings_mask, 'volatility'] *= self.faker.random.uniform(1.1, 1.4)
        volume_multiplier = self.faker.random.uniform(1.2, 1.6)
        df.loc[earnings_mask, 'volume'] = (df.loc[earnings_mask, 'volume'] * volume_multiplier).astype(int)
        
        # Add momentum effects (trending behavior)
        if len(df) > 5:
            df['price_momentum'] = df.groupby('symbol')['close_price'].pct_change(periods=5).fillna(0)
            
            # Positive momentum tends to continue (with some mean reversion)
            momentum_mask = df['price_momentum'] > 0.02  # 2% positive momentum
            df.loc[momentum_mask, 'daily_return'] *= self.faker.random.uniform(1.05, 1.15)
            
            # Negative momentum also tends to continue
            negative_momentum_mask = df['price_momentum'] < -0.02
            df.loc[negative_momentum_mask, 'daily_return'] *= self.faker.random.uniform(0.85, 0.95)
        
        return df
    
    def _generate_financial_record_from_price(self, stock_info: Dict, timestamp: datetime, 
                                            base_price: float, **kwargs) -> Dict:
        """Generate financial record from time series base price"""
        symbol = stock_info['symbol']
        sector = stock_info['sector']
        base_volatility = stock_info['base_volatility']
        volume_multiplier = stock_info['volume_multiplier']
        
        # Use base_price as close price and generate OHLC around it
        close_price = base_price
        
        # Generate realistic intraday spread
        interval = kwargs.get('interval', '1day')
        if interval in ['1min', '5min']:
            spread_factor = 0.002  # 0.2% spread for short intervals
        elif interval == '1hour':
            spread_factor = 0.01   # 1% spread for hourly
        else:
            spread_factor = 0.03   # 3% spread for daily
        
        # Generate OHLC with realistic relationships
        daily_volatility = base_volatility * spread_factor
        
        # Open price (slight gap from previous close)
        open_gap = self.faker.random.gauss(0, daily_volatility * 0.5)
        open_price = close_price * (1 + open_gap)
        
        # High and low prices
        high_move = abs(self.faker.random.gauss(0, daily_volatility))
        low_move = abs(self.faker.random.gauss(0, daily_volatility))
        
        high_price = max(open_price, close_price) * (1 + high_move)
        low_price = min(open_price, close_price) * (1 - low_move)
        
        # Ensure logical price relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Calculate return
        if open_price > 0:
            period_return = (close_price - open_price) / open_price
        else:
            period_return = 0
        
        # Generate volume with time-based patterns
        base_volume = self.faker.random_int(100000, 10000000)
        volume = int(base_volume * volume_multiplier)
        
        # Apply temporal volume patterns
        if hasattr(timestamp, 'weekday'):
            day_multiplier = self.day_of_week_volume.get(timestamp.weekday(), 1.0)
            volume = int(volume * day_multiplier)
        
        month_multiplier = self.monthly_volume.get(timestamp.month, 1.0)
        volume = int(volume * month_multiplier)
        
        # Volume increases with volatility
        volatility_multiplier = 1 + abs(period_return) * 5
        volume = int(volume * volatility_multiplier)
        
        # Apply intraday volume patterns for short intervals
        if interval in ['1min', '5min', '1hour'] and hasattr(timestamp, 'hour'):
            hour_volume_mult = self.intraday_volatility.get(timestamp.hour, 1.0)
            volume = int(volume * hour_volume_mult)
        
        return {
            'symbol': symbol,
            'date': timestamp,
            'open_price': round(open_price, 2),
            'high_price': round(high_price, 2),
            'low_price': round(low_price, 2),
            'close_price': round(close_price, 2),
            'volume': volume,
            'sector': sector,
            'market_condition': self._select_market_condition(),
            'daily_return': round(period_return * 100, 4),
            'volatility': round(base_volatility * 100, 2)
        }
    
    def _apply_sector_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply realistic correlations between sectors in time series data"""
        if len(data) == 0 or 'sector' not in data.columns:
            return data
        
        # Group by timestamp to apply correlations
        data = data.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # For each timestamp, adjust prices based on sector correlations
        for timestamp in data['date'].unique():
            timestamp_data = data[data['date'] == timestamp].copy()
            
            if len(timestamp_data) > 1:
                # Calculate sector average returns
                sector_returns = timestamp_data.groupby('sector')['daily_return'].mean()
                
                # Apply correlation adjustments
                for idx, row in timestamp_data.iterrows():
                    sector = row['sector']
                    base_return = row['daily_return']
                    
                    # Calculate correlated adjustment
                    correlation_adjustment = 0
                    for other_sector, other_return in sector_returns.items():
                        if other_sector != sector:
                            correlation = self.sector_correlations.get(sector, {}).get(other_sector, 0)
                            correlation_adjustment += correlation * other_return * 0.1  # Dampen effect
                    
                    # Apply correlation adjustment to price
                    adjusted_return = base_return + correlation_adjustment
                    original_close = data.loc[data.index[idx], 'close_price']
                    original_open = data.loc[data.index[idx], 'open_price']
                    
                    if original_open > 0:
                        new_close = original_open * (1 + adjusted_return / 100)
                        data.loc[data.index[idx], 'close_price'] = round(new_close, 2)
                        data.loc[data.index[idx], 'daily_return'] = round(adjusted_return, 4)
                        
                        # Adjust high/low if needed
                        current_high = data.loc[data.index[idx], 'high_price']
                        current_low = data.loc[data.index[idx], 'low_price']
                        
                        data.loc[data.index[idx], 'high_price'] = round(max(current_high, new_close, original_open), 2)
                        data.loc[data.index[idx], 'low_price'] = round(min(current_low, new_close, original_open), 2)
        
        return data
    
    def _generate_time_series(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate time series stock data"""
        symbols = kwargs.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])  # Default symbols
        start_date = kwargs.get('start_date', datetime.now() - timedelta(days=365))
        end_date = kwargs.get('end_date', datetime.now())
        interval = kwargs.get('interval', '1day')
        
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range based on interval
        dates = self._generate_date_range(start_date, end_date, interval)
        
        # Limit to requested rows if specified
        if len(dates) * len(symbols) > rows:
            dates = dates[:rows // len(symbols)]
        
        data = []
        
        # Initialize price tracking for each symbol
        symbol_prices = {}
        for symbol in symbols:
            stock_info = next((s for s in self.all_symbols if s['symbol'] == symbol), self.all_symbols[0])
            min_price, max_price = stock_info['price_range']
            symbol_prices[symbol] = {
                'current_price': self.faker.random.uniform(min_price, max_price),
                'info': stock_info
            }
        
        # Generate market condition for the period
        market_condition = self._select_market_condition()
        
        for date in dates:
            for symbol in symbols:
                stock_info = symbol_prices[symbol]['info']
                
                # Generate correlated price movement
                price_data = self._generate_time_series_record(
                    symbol_prices[symbol], date, market_condition, **kwargs
                )
                
                # Update current price for next iteration
                symbol_prices[symbol]['current_price'] = price_data['close_price']
                
                data.append(price_data)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_date_range(self, start_date: datetime, end_date: datetime, interval: str) -> List[datetime]:
        """Generate date range based on interval"""
        dates = []
        current_date = start_date
        
        if interval == '1min':
            delta = timedelta(minutes=1)
        elif interval == '5min':
            delta = timedelta(minutes=5)
        elif interval == '1hour':
            delta = timedelta(hours=1)
        elif interval == '1day':
            delta = timedelta(days=1)
        else:
            delta = timedelta(days=1)  # Default to daily
        
        while current_date <= end_date:
            # Skip weekends for daily data
            if interval == '1day' and current_date.weekday() >= 5:
                current_date += delta
                continue
            
            # Skip non-trading hours for intraday data
            if interval in ['1min', '5min', '1hour']:
                if current_date.hour < 9 or current_date.hour > 16:
                    current_date += delta
                    continue
                if current_date.weekday() >= 5:
                    current_date += delta
                    continue
            
            dates.append(current_date)
            current_date += delta
        
        return dates
    
    def _select_market_condition(self) -> str:
        """Select market condition based on probabilities"""
        conditions = list(self.market_conditions.keys())
        probabilities = [self.market_conditions[c]['probability'] for c in conditions]
        
        # Use seeded random for reproducibility
        rand_val = self.faker.random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return conditions[i]
        
        return conditions[-1]  # Fallback
    
    def _generate_stock_record(self, stock_info: Dict, trade_date: datetime, **kwargs) -> Dict:
        """Generate a single stock record"""
        symbol = stock_info['symbol']
        sector = stock_info['sector']
        base_volatility = stock_info['base_volatility']
        growth_bias = stock_info['growth_bias']
        min_price, max_price = stock_info['price_range']
        volume_multiplier = stock_info['volume_multiplier']
        
        # Generate base price
        base_price = self.faker.random.uniform(min_price, max_price)
        
        # Apply market condition effects
        market_condition = kwargs.get('market_condition', self._select_market_condition())
        condition_data = self.market_conditions[market_condition]
        
        volatility = base_volatility * condition_data['volatility_multiplier']
        growth_effect = growth_bias * condition_data['growth_multiplier']
        
        # Generate OHLC prices
        open_price = base_price
        
        # Generate price movements with realistic patterns
        daily_return = self.faker.random.gauss(growth_effect / 252, volatility / np.sqrt(252))  # Daily return
        close_price = open_price * (1 + daily_return)
        
        # Generate high and low with realistic spreads
        high_spread = abs(self.faker.random.gauss(0, volatility * 0.5))
        low_spread = abs(self.faker.random.gauss(0, volatility * 0.5))
        
        high_price = max(open_price, close_price) * (1 + high_spread)
        low_price = min(open_price, close_price) * (1 - low_spread)
        
        # Ensure logical price relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume with realistic patterns
        base_volume = self.faker.random_int(100000, 10000000)
        
        # Apply volume multipliers
        volume = int(base_volume * volume_multiplier)
        
        # Apply day of week effects
        if hasattr(trade_date, 'weekday'):
            day_multiplier = self.day_of_week_volume.get(trade_date.weekday(), 1.0)
            volume = int(volume * day_multiplier)
        
        # Apply monthly effects
        month_multiplier = self.monthly_volume.get(trade_date.month, 1.0)
        volume = int(volume * month_multiplier)
        
        # Higher volume on high volatility days
        volatility_volume_multiplier = 1 + abs(daily_return) * 5
        volume = int(volume * volatility_volume_multiplier)
        
        return {
            'symbol': symbol,
            'date': trade_date,
            'open_price': round(open_price, 2),
            'high_price': round(high_price, 2),
            'low_price': round(low_price, 2),
            'close_price': round(close_price, 2),
            'volume': volume,
            'sector': sector,
            'market_condition': market_condition,
            'daily_return': round(daily_return * 100, 4),  # Percentage
            'volatility': round(volatility * 100, 2)  # Percentage
        }
    
    def _generate_time_series_record(self, symbol_data: Dict, date: datetime, 
                                   market_condition: str, **kwargs) -> Dict:
        """Generate time series record with price continuity"""
        stock_info = symbol_data['info']
        current_price = symbol_data['current_price']
        
        symbol = stock_info['symbol']
        sector = stock_info['sector']
        base_volatility = stock_info['base_volatility']
        growth_bias = stock_info['growth_bias']
        volume_multiplier = stock_info['volume_multiplier']
        
        # Apply market condition effects
        condition_data = self.market_conditions[market_condition]
        volatility = base_volatility * condition_data['volatility_multiplier']
        growth_effect = growth_bias * condition_data['growth_multiplier']
        
        # Generate price movement from current price
        open_price = current_price
        
        # Generate return based on interval
        interval = kwargs.get('interval', '1day')
        if interval == '1min':
            time_factor = 1 / (252 * 24 * 60)  # Minutes in trading year
        elif interval == '5min':
            time_factor = 5 / (252 * 24 * 60)
        elif interval == '1hour':
            time_factor = 1 / (252 * 24)  # Hours in trading year
        else:  # 1day
            time_factor = 1 / 252  # Trading days in year
        
        # Apply intraday volatility if applicable
        if interval in ['1min', '5min', '1hour']:
            hour_multiplier = self.intraday_volatility.get(date.hour, 1.0)
            volatility *= hour_multiplier
        
        # Generate return
        period_return = self.faker.random.gauss(growth_effect * time_factor, volatility * np.sqrt(time_factor))
        close_price = open_price * (1 + period_return)
        
        # Generate high and low
        if interval in ['1min', '5min']:
            # Smaller spreads for short intervals
            high_spread = abs(self.faker.random.gauss(0, volatility * 0.1))
            low_spread = abs(self.faker.random.gauss(0, volatility * 0.1))
        else:
            high_spread = abs(self.faker.random.gauss(0, volatility * 0.3))
            low_spread = abs(self.faker.random.gauss(0, volatility * 0.3))
        
        high_price = max(open_price, close_price) * (1 + high_spread)
        low_price = min(open_price, close_price) * (1 - low_spread)
        
        # Ensure logical relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume
        base_volume = self.faker.random_int(100000, 10000000)
        volume = int(base_volume * volume_multiplier)
        
        # Apply time-based volume patterns
        if hasattr(date, 'weekday'):
            day_multiplier = self.day_of_week_volume.get(date.weekday(), 1.0)
            volume = int(volume * day_multiplier)
        
        month_multiplier = self.monthly_volume.get(date.month, 1.0)
        volume = int(volume * month_multiplier)
        
        # Volume increases with volatility
        volatility_multiplier = 1 + abs(period_return) * 10
        volume = int(volume * volatility_multiplier)
        
        return {
            'symbol': symbol,
            'date': date,
            'open_price': round(open_price, 2),
            'high_price': round(high_price, 2),
            'low_price': round(low_price, 2),
            'close_price': round(close_price, 2),
            'volume': volume,
            'sector': sector,
            'market_condition': market_condition,
            'daily_return': round(period_return * 100, 4),
            'volatility': round(volatility * 100, 2)
        }
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to stock data"""
        # Add technical indicators
        if 'close_price' in data.columns:
            # Simple moving averages (if we have enough data)
            if len(data) >= 20:
                data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
                data['sma_20'] = data.groupby('symbol')['close_price'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True).round(2)
                data['sma_50'] = data.groupby('symbol')['close_price'].rolling(window=50, min_periods=1).mean().reset_index(level=0, drop=True).round(2)
            
            # Price change and percentage change
            data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
            data['price_change'] = data.groupby('symbol')['close_price'].diff().round(2)
            data['price_change_pct'] = (data['price_change'] / data.groupby('symbol')['close_price'].shift(1) * 100).round(2)
        
        # Add market cap estimation (simplified)
        if 'close_price' in data.columns:
            # Estimate shares outstanding based on sector
            sector_shares = {
                'Technology': (1000000000, 5000000000),  # 1B - 5B shares
                'Healthcare': (500000000, 2000000000),   # 500M - 2B shares
                'Financial': (1000000000, 8000000000),   # 1B - 8B shares
                'Energy': (500000000, 3000000000),       # 500M - 3B shares
                'Consumer': (800000000, 4000000000),     # 800M - 4B shares
                'Industrial': (300000000, 1500000000)    # 300M - 1.5B shares
            }
            
            def estimate_market_cap(row):
                if row['sector'] in sector_shares:
                    min_shares, max_shares = sector_shares[row['sector']]
                    shares = self.faker.random_int(min_shares, max_shares)
                    return int(row['close_price'] * shares)
                return None
            
            data['estimated_market_cap'] = data.apply(estimate_market_cap, axis=1)
        
        # Add trading session indicators
        if 'date' in data.columns:
            data['trading_session'] = data['date'].apply(
                lambda x: 'regular' if hasattr(x, 'hour') and 9 <= x.hour <= 16 else 'extended'
            )
        
        # Add volatility ranking within sector
        if 'volatility' in data.columns and 'sector' in data.columns:
            data['volatility_rank'] = data.groupby('sector')['volatility'].rank(pct=True).round(2)
        
        # Sort by date and symbol for better readability
        if 'date' in data.columns and 'symbol' in data.columns:
            data = data.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        return data