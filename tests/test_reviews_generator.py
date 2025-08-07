"""
Unit tests for ReviewsGenerator

Tests realistic patterns, sentiment correlation, rating distributions, and reviewer behavior.
"""

import pytest
import pandas as pd
from datetime import datetime, date, timedelta
from tempdata.core.seeding import MillisecondSeeder
from tempdata.datasets.business.reviews import ReviewsGenerator


class TestReviewsGenerator:
    """Test suite for ReviewsGenerator functionality"""
    
    @pytest.fixture
    def seeder(self):
        """Create a fixed seeder for reproducible tests"""
        return MillisecondSeeder(fixed_seed=123456789)
    
    @pytest.fixture
    def generator(self, seeder):
        """Create ReviewsGenerator instance"""
        return ReviewsGenerator(seeder)
    
    def test_basic_generation(self, generator):
        """Test basic data generation functionality"""
        rows = 100
        data = generator.generate(rows)
        
        # Check basic structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == rows
        assert not data.empty
    
    def test_required_columns(self, generator):
        """Test that all required columns are present"""
        data = generator.generate(50)
        
        required_columns = [
            'review_id', 'product_id', 'product_name', 'product_category', 'brand',
            'reviewer_id', 'reviewer_name', 'reviewer_type', 'rating', 'review_text',
            'review_title', 'sentiment_score', 'sentiment_label', 'word_count',
            'helpful_votes', 'total_votes', 'verified_purchase', 'review_date',
            'reviewer_total_reviews', 'reviewer_avg_rating', 'product_price_tier'
        ]
        
        for column in required_columns:
            assert column in data.columns, f"Missing required column: {column}"
    
    def test_review_id_uniqueness(self, generator):
        """Test that review IDs are unique"""
        data = generator.generate(500)
        assert data['review_id'].nunique() == len(data)
        
        # Check format
        assert all(data['review_id'].str.startswith('REV_'))
        assert all(data['review_id'].str.len() == 12)  # REV_ + 8 digits
    
    def test_rating_distribution(self, generator):
        """Test that rating distribution follows realistic patterns"""
        data = generator.generate(2000)
        
        # Check rating range
        assert data['rating'].min() >= 1
        assert data['rating'].max() <= 5
        assert set(data['rating'].unique()).issubset({1, 2, 3, 4, 5})
        
        # Check J-curve pattern (5-star and 1-star should be most common)
        rating_counts = data['rating'].value_counts(normalize=True).sort_index()
        
        # 5-star should be most common
        assert rating_counts[5] > rating_counts[4]
        assert rating_counts[5] > rating_counts[3]
        
        # 1-star should be more common than 2-star (allowing some flexibility for 3-star)
        assert rating_counts[1] > rating_counts[2]
        # 1-star should be at least close to 3-star (within reasonable range)
        assert abs(rating_counts[1] - rating_counts[3]) < 0.05 or rating_counts[1] > rating_counts[3]
    
    def test_sentiment_rating_correlation(self, generator):
        """Test correlation between sentiment scores and ratings"""
        data = generator.generate(1000)
        
        # Check sentiment score range
        assert data['sentiment_score'].min() >= 0
        assert data['sentiment_score'].max() <= 100
        
        # Test correlation patterns
        high_ratings = data[data['rating'] >= 4]
        low_ratings = data[data['rating'] <= 2]
        neutral_ratings = data[data['rating'] == 3]
        
        # High ratings should have higher sentiment scores
        assert high_ratings['sentiment_score'].mean() > neutral_ratings['sentiment_score'].mean()
        assert high_ratings['sentiment_score'].mean() > low_ratings['sentiment_score'].mean()
        
        # Low ratings should have lower sentiment scores
        assert low_ratings['sentiment_score'].mean() < neutral_ratings['sentiment_score'].mean()
        
        # Check sentiment labels align with scores
        very_positive = data[data['sentiment_label'] == 'very_positive']
        very_negative = data[data['sentiment_label'] == 'very_negative']
        
        if len(very_positive) > 0:
            assert very_positive['sentiment_score'].min() >= 80
        if len(very_negative) > 0:
            assert very_negative['sentiment_score'].max() <= 20
    
    def test_reviewer_types_valid(self, generator):
        """Test that reviewer types are from valid set"""
        data = generator.generate(200)
        
        expected_types = {'casual', 'enthusiast', 'critic', 'influencer'}
        actual_types = set(data['reviewer_type'].unique())
        assert actual_types.issubset(expected_types)
        
        # Casual reviewers should be most common
        type_counts = data['reviewer_type'].value_counts()
        assert type_counts['casual'] > type_counts.get('enthusiast', 0)
        assert type_counts['casual'] > type_counts.get('critic', 0)
        assert type_counts['casual'] > type_counts.get('influencer', 0)
    
    def test_reviewer_behavior_patterns(self, generator):
        """Test realistic reviewer behavior patterns"""
        data = generator.generate(1000)
        
        # Test reviewer total reviews by type
        enthusiasts = data[data['reviewer_type'] == 'enthusiast']
        casuals = data[data['reviewer_type'] == 'casual']
        
        if len(enthusiasts) > 0 and len(casuals) > 0:
            # Enthusiasts should have more total reviews on average
            assert enthusiasts['reviewer_total_reviews'].mean() > casuals['reviewer_total_reviews'].mean()
        
        # Test verified purchase rates
        verified_rate = data['verified_purchase'].mean()
        assert 0.7 < verified_rate < 0.95  # Should be between 70-95%
        
        # Test helpful votes correlation with review quality
        long_reviews = data[data['word_count'] > 100]
        short_reviews = data[data['word_count'] < 50]
        
        if len(long_reviews) > 0 and len(short_reviews) > 0:
            # Longer reviews should tend to get more helpful votes
            assert long_reviews['helpful_votes'].mean() >= short_reviews['helpful_votes'].mean()
    
    def test_review_text_quality(self, generator):
        """Test review text generation quality"""
        data = generator.generate(200)
        
        # Check that all reviews have text
        assert data['review_text'].notna().all()
        assert (data['review_text'].str.len() > 0).all()
        
        # Check word count accuracy
        calculated_word_counts = data['review_text'].apply(lambda x: len(x.split()))
        assert (calculated_word_counts == data['word_count']).all()
        
        # Check that review titles exist and are reasonable
        assert data['review_title'].notna().all()
        assert (data['review_title'].str.len() > 0).all()
        assert (data['review_title'].str.len() < 100).all()  # Titles should be short
        
        # Test that review text correlates with rating
        high_rating_reviews = data[data['rating'] == 5]['review_text'].str.lower()
        low_rating_reviews = data[data['rating'] == 1]['review_text'].str.lower()
        
        # High rating reviews should contain more positive words
        positive_words = ['great', 'excellent', 'amazing', 'perfect', 'love', 'recommend']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'avoid']
        
        if len(high_rating_reviews) > 0:
            high_positive_count = sum(high_rating_reviews.str.contains('|'.join(positive_words), na=False))
            assert high_positive_count > 0
        
        if len(low_rating_reviews) > 0:
            low_negative_count = sum(low_rating_reviews.str.contains('|'.join(negative_words), na=False))
            assert low_negative_count > 0
    
    def test_product_categories_valid(self, generator):
        """Test that product categories are valid"""
        data = generator.generate(200)
        
        # Should have valid categories
        assert data['product_category'].notna().all()
        
        # Categories should be from expected set
        expected_categories = set(generator.products.keys())
        actual_categories = set(data['product_category'].unique())
        assert actual_categories.issubset(expected_categories)
        
        # Product names should correspond to categories
        for category in actual_categories:
            category_data = data[data['product_category'] == category]
            category_products = set(category_data['product_name'].unique())
            expected_products = set(generator.products[category])
            assert category_products.issubset(expected_products)
    
    def test_temporal_patterns(self, generator):
        """Test temporal review patterns"""
        # Generate data for different time periods
        data_holiday = generator.generate(100, date_range=(date(2024, 12, 1), date(2024, 12, 31)))
        data_normal = generator.generate(100, date_range=(date(2024, 6, 1), date(2024, 6, 30)))
        
        # Holiday season might have slightly higher ratings on average
        holiday_avg_rating = data_holiday['rating'].mean()
        normal_avg_rating = data_normal['rating'].mean()
        
        # Both should be reasonable
        assert 1.0 <= holiday_avg_rating <= 5.0
        assert 1.0 <= normal_avg_rating <= 5.0
        
        # Test review timing patterns
        verified_reviews = data_normal[data_normal['verified_purchase'] == True]
        if len(verified_reviews) > 0:
            # Days after purchase should be reasonable
            assert verified_reviews['days_after_purchase'].min() >= 0
            assert verified_reviews['days_after_purchase'].max() <= 365
            
            # Most reviews should be within 30 days
            recent_reviews = verified_reviews[verified_reviews['days_after_purchase'] <= 30]
            assert len(recent_reviews) / len(verified_reviews) > 0.5
    
    def test_time_series_generation(self, generator):
        """Test time series review generation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = generator.generate(
            100,
            time_series=True,
            start_date=start_date,
            end_date=end_date,
            interval='1day'
        )
        
        # Should have timestamp column
        assert 'timestamp' in data.columns
        assert 'volume_factor' in data.columns
        
        # Timestamps should be within range
        timestamps = pd.to_datetime(data['timestamp'])
        assert timestamps.min() >= start_date
        assert timestamps.max() <= end_date
        
        # Should have time-based derived columns
        assert 'day_of_week' in data.columns
        assert 'month' in data.columns
        assert 'is_weekend' in data.columns
    
    def test_product_relationships(self, generator):
        """Test product relationship support"""
        # Create mock product relationships
        product_relationships = {
            'products': [
                {'name': 'Test Smartphone', 'category': 'technology'},
                {'name': 'Test Laptop', 'category': 'technology'},
                {'name': 'Test Shoes', 'category': 'retail'}
            ]
        }
        
        data = generator.generate(50, product_relationships=product_relationships)
        
        # All products should be from the provided list
        expected_products = {'Test Smartphone', 'Test Laptop', 'Test Shoes'}
        actual_products = set(data['product_name'].unique())
        assert actual_products.issubset(expected_products)
    
    def test_data_quality_metrics(self, generator):
        """Test additional data quality metrics"""
        data = generator.generate(500)
        
        # Test helpful ratio calculation
        assert 'helpful_ratio' in data.columns
        assert (data['helpful_ratio'] >= 0).all()
        assert (data['helpful_ratio'] <= 1).all()
        
        # Test review length categories
        assert 'review_length_category' in data.columns
        expected_categories = {'very_short', 'short', 'medium', 'long', 'very_long'}
        actual_categories = set(data['review_length_category'].dropna().unique())
        assert actual_categories.issubset(expected_categories)
        
        # Test quality score
        assert 'quality_score' in data.columns
        assert (data['quality_score'] >= 0).all()
        assert (data['quality_score'] <= 1).all()
        
        # Test sentiment-rating alignment
        assert 'sentiment_rating_aligned' in data.columns
        aligned_rate = data['sentiment_rating_aligned'].mean()
        assert aligned_rate > 0.6  # Most reviews should have aligned sentiment and rating
        
        # Test reviewer credibility
        assert 'reviewer_credibility' in data.columns
        assert (data['reviewer_credibility'] >= 0).all()
        assert (data['reviewer_credibility'] <= 1).all()
    
    def test_price_tier_distribution(self, generator):
        """Test price tier distribution"""
        data = generator.generate(300)
        
        expected_tiers = {'budget', 'economy', 'standard', 'premium', 'luxury'}
        actual_tiers = set(data['product_price_tier'].unique())
        assert actual_tiers.issubset(expected_tiers)
        
        # Standard tier should be most common
        tier_counts = data['product_price_tier'].value_counts()
        assert tier_counts['standard'] > tier_counts.get('luxury', 0)
    
    def test_statistical_validation(self, generator):
        """Test statistical properties of generated data"""
        data = generator.generate(2000)
        
        # Test rating distribution follows expected pattern
        rating_dist = data['rating'].value_counts(normalize=True).sort_index()
        
        # 5-star should be most common (around 45%)
        assert rating_dist[5] > 0.35
        assert rating_dist[5] < 0.55
        
        # 1-star should be second most common (around 10%)
        assert rating_dist[1] > 0.05
        assert rating_dist[1] < 0.15
        
        # 3-star should be least common (around 12%)
        assert rating_dist[3] < 0.20
        
        # Test sentiment score distribution
        sentiment_stats = data['sentiment_score'].describe()
        assert 0 <= sentiment_stats['min'] <= 100
        assert 0 <= sentiment_stats['max'] <= 100
        assert 60 <= sentiment_stats['mean'] <= 85  # Should be positive-skewed due to J-curve rating distribution
        
        # Test word count distribution
        word_count_stats = data['word_count'].describe()
        assert word_count_stats['min'] > 0
        assert word_count_stats['mean'] > 20  # Average should be reasonable
        assert word_count_stats['max'] < 1000  # No extremely long reviews
    
    def test_edge_cases(self, generator):
        """Test edge cases and error handling"""
        # Test with minimal rows
        data = generator.generate(1)
        assert len(data) == 1
        assert not data.empty
        
        # Test with large number of rows
        data = generator.generate(5000)
        assert len(data) == 5000
        
        # Test data consistency
        assert data['review_id'].nunique() == len(data)  # All IDs unique
        assert data['rating'].between(1, 5).all()  # All ratings valid
        assert data['sentiment_score'].between(0, 100).all()  # All sentiment scores valid
        assert (data['helpful_votes'] <= data['total_votes']).all()  # Logical consistency
    
    def test_reproducibility(self, seeder):
        """Test that generation is reproducible with same seed"""
        generator1 = ReviewsGenerator(seeder)
        generator2 = ReviewsGenerator(seeder)
        
        data1 = generator1.generate(100)
        data2 = generator2.generate(100)
        
        # Should generate identical data with same seed
        pd.testing.assert_frame_equal(data1, data2)