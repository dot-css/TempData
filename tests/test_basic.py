"""
Basic tests for TempData
"""

import pytest
import tempdata
import os
import tempfile

def test_version():
    """Test that version is defined"""
    assert hasattr(tempdata, '__version__')
    assert tempdata.__version__ == '0.1.0'

def test_create_dataset_basic():
    """Test basic dataset creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test.csv')
        result = tempdata.create_dataset(filename, rows=10)
        
        assert result == filename
        assert os.path.exists(filename)
        
        # Check file content
        import pandas as pd
        df = pd.read_csv(filename)
        assert len(df) == 10
        assert 'id' in df.columns
        assert 'name' in df.columns

def test_create_dataset_with_seed():
    """Test dataset creation with seed for reproducibility"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename1 = os.path.join(tmpdir, 'test1.csv')
        filename2 = os.path.join(tmpdir, 'test2.csv')
        
        result1 = tempdata.create_dataset(filename1, rows=5, seed=12345)
        result2 = tempdata.create_dataset(filename2, rows=5, seed=12345)
        
        # Both files should exist
        assert os.path.exists(result1)
        assert os.path.exists(result2)
        
        # Content should be identical
        import pandas as pd
        df1 = pd.read_csv(result1)
        df2 = pd.read_csv(result2)
        
        assert df1.equals(df2)

def test_create_batch():
    """Test batch dataset creation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        datasets = [
            {'filename': os.path.join(tmpdir, 'customers.csv'), 'rows': 5},
            {'filename': os.path.join(tmpdir, 'orders.csv'), 'rows': 10}
        ]
        
        results = tempdata.create_batch(datasets, seed=12345)
        
        assert len(results) == 2
        for result in results:
            assert os.path.exists(result)

def test_geo_addresses():
    """Test geographical address generation"""
    addresses = tempdata.geo.addresses('united_states', count=5)
    
    assert len(addresses) == 5
    assert all('address' in addr for addr in addresses)
    assert all(addr['country'] == 'united_states' for addr in addresses)

def test_geo_coordinates():
    """Test coordinate generation"""
    coords = tempdata.geo.coordinates('new_york', count=3)
    
    assert len(coords) == 3
    assert all('lat' in coord for coord in coords)
    assert all('lng' in coord for coord in coords)

def test_geo_routes():
    """Test route generation"""
    route = tempdata.geo.routes('new_york', 'boston', waypoints=3)
    
    assert route['start'] == 'new_york'
    assert route['end'] == 'boston'
    assert route['waypoints'] == 3