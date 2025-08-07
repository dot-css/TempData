"""
Product reviews dataset generator

Generates realistic product review data with sentiment patterns, rating distributions,
and reviewer behavior correlations.
"""

import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from ...core.base_generator import BaseGenerator



class ReviewsGenerator(BaseGenerator):
    """
    Generator for realistic product review data
    
    Creates review datasets with ratings, review text, sentiment scores,
    reviewer information, and realistic correlation patterns between ratings and sentiment.
    """
    
    def __init__(self, seeder, locale: str = 'en_US'):
        super().__init__(seeder, locale)
        self._setup_product_data()
        self._setup_review_templates()
        self._setup_sentiment_patterns()
        self._setup_reviewer_behavior()
        self._setup_rating_distributions()
    
    def _setup_product_data(self):
        """Setup product information from business data"""
        try:
            data_path = os.path.join(os.path.dirname(__file__), '../../data/business/products.json')
            with open(data_path, 'r') as f:
                business_data = json.load(f)
            
            self.products = business_data.get('products', {})
            self.product_categories = business_data.get('product_categories', [])
            self.product_attributes = business_data.get('product_attributes', {})
        except:
            # Fallback data if loading fails
            self.products = {
                'technology': ['Smartphone', 'Laptop', 'Headphones', 'Smart Watch', 'Tablet'],
                'retail': ['Running Shoes', 'T-Shirt', 'Jeans', 'Backpack', 'Sunglasses'],
                'home_garden': ['Coffee Maker', 'Vacuum Cleaner', 'Bed Sheets', 'Lamp', 'Sofa']
            }
            self.product_categories = ['Electronics', 'Clothing', 'Home & Garden']
            self.product_attributes = {
                'brands': ['TechPro', 'SmartChoice', 'QualityFirst', 'PremiumPlus'],
                'colors': ['Black', 'White', 'Blue', 'Red', 'Gray'],
                'sizes': ['S', 'M', 'L', 'XL', 'One Size']
            }
    
    def _setup_review_templates(self):
        """Setup review text templates by rating and sentiment"""
        self.review_templates = {
            5: {  # 5-star reviews
                'positive': [
                    "Absolutely love this {product}! {positive_aspect} exceeded my expectations. {recommendation}",
                    "Outstanding {product}! {quality_praise} and {feature_praise}. Highly recommend!",
                    "Perfect {product}! {experience_positive} and {value_positive}. Will buy again!",
                    "Amazing quality! {durability_praise} and {performance_praise}. Worth every penny!",
                    "Best {product} I've ever owned! {comparison_positive} and {satisfaction_high}."
                ]
            },
            4: {  # 4-star reviews
                'positive': [
                    "Really good {product}! {positive_aspect} but {minor_issue}. Still recommend it.",
                    "Great {product} overall. {quality_good} though {small_complaint}. Happy with purchase.",
                    "Solid {product}! {performance_good} and {value_good}. {minor_suggestion}.",
                    "Very pleased with this {product}. {feature_praise} but {room_improvement}.",
                    "Good quality {product}. {positive_experience} with only {minor_negative}."
                ]
            },
            3: {  # 3-star reviews
                'neutral': [
                    "Decent {product}. {mixed_feelings} - {positive_aspect} but {negative_aspect}.",
                    "Average {product}. {meets_expectations} though {could_be_better}.",
                    "It's okay. {some_positives} but {some_negatives}. {neutral_recommendation}.",
                    "Mixed feelings about this {product}. {good_points} but {bad_points}.",
                    "Fair {product}. {adequate_performance} but {room_for_improvement}."
                ]
            },
            2: {  # 2-star reviews
                'negative': [
                    "Disappointed with this {product}. {major_issue} and {quality_concern}.",
                    "Not great. {performance_poor} and {value_poor}. {regret_purchase}.",
                    "Below expectations. {problem_description} and {frustration}.",
                    "Poor quality {product}. {durability_issue} and {functionality_problem}.",
                    "Wouldn't recommend. {negative_experience} and {warning_others}."
                ]
            },
            1: {  # 1-star reviews
                'negative': [
                    "Terrible {product}! {major_failure} and {complete_disappointment}. Waste of money!",
                    "Awful quality! {broke_quickly} and {customer_service_bad}. Avoid at all costs!",
                    "Worst {product} ever! {multiple_problems} and {total_regret}. Don't buy!",
                    "Complete garbage! {defective_product} and {money_wasted}. Returning immediately!",
                    "Absolutely horrible! {failed_expectations} and {angry_customer}. Never again!"
                ]
            }
        }
        
        # Template components for building reviews
        self.template_components = {
            'positive_aspect': [
                'The quality', 'The design', 'The functionality', 'The performance', 
                'The durability', 'The value', 'The features', 'The build quality'
            ],
            'negative_aspect': [
                'the price is high', 'it feels cheap', 'the battery life is poor',
                'it\'s not very durable', 'the instructions are unclear', 'it\'s too complicated'
            ],
            'minor_issue': [
                'the packaging could be better', 'delivery took a while', 'it\'s a bit pricey',
                'the color is slightly different', 'setup was a bit tricky'
            ],
            'quality_praise': [
                'Excellent build quality', 'Superior materials', 'Top-notch craftsmanship',
                'Premium feel', 'Solid construction', 'High-end quality'
            ],
            'performance_praise': [
                'Works flawlessly', 'Performs excellently', 'Runs smoothly', 
                'Very responsive', 'Lightning fast', 'Reliable performance'
            ],
            'recommendation': [
                'Definitely recommend!', 'Would buy again!', 'Tell all your friends!',
                'Don\'t hesitate to purchase!', 'You won\'t regret it!', 'Must-have item!'
            ],
            'major_issue': [
                'It broke after a week', 'Doesn\'t work as advertised', 'Poor build quality',
                'Multiple defects', 'Stopped working suddenly', 'Completely unreliable'
            ],
            'complete_disappointment': [
                'total waste of money', 'completely useless', 'nothing like described',
                'worst purchase ever', 'complete failure', 'absolutely terrible'
            ]
        }
    
    def _setup_sentiment_patterns(self):
        """Setup sentiment scoring patterns correlated with ratings"""
        # Sentiment score ranges by rating (0-100 scale)
        self.sentiment_ranges = {
            5: (85, 100),  # Very positive
            4: (65, 84),   # Positive
            3: (40, 64),   # Neutral
            2: (15, 39),   # Negative
            1: (0, 14)     # Very negative
        }
        
        # Sentiment keywords for scoring
        self.sentiment_keywords = {
            'very_positive': [
                'amazing', 'excellent', 'outstanding', 'perfect', 'fantastic', 'incredible',
                'awesome', 'brilliant', 'superb', 'wonderful', 'exceptional', 'flawless'
            ],
            'positive': [
                'good', 'great', 'nice', 'solid', 'decent', 'satisfied', 'happy',
                'pleased', 'recommend', 'quality', 'reliable', 'useful'
            ],
            'neutral': [
                'okay', 'average', 'fair', 'adequate', 'acceptable', 'reasonable',
                'standard', 'normal', 'typical', 'expected', 'mixed', 'moderate'
            ],
            'negative': [
                'bad', 'poor', 'disappointing', 'cheap', 'flimsy', 'unreliable',
                'frustrating', 'annoying', 'problematic', 'defective', 'faulty'
            ],
            'very_negative': [
                'terrible', 'awful', 'horrible', 'worst', 'garbage', 'useless',
                'broken', 'waste', 'regret', 'avoid', 'disaster', 'nightmare'
            ]
        }
    
    def _setup_reviewer_behavior(self):
        """Setup realistic reviewer behavior patterns"""
        # Reviewer types with different behaviors
        self.reviewer_types = {
            'casual': {
                'review_frequency': (1, 5),      # Reviews per year
                'avg_review_length': (20, 80),   # Words
                'rating_bias': 0.0,              # No systematic bias
                'helpful_vote_rate': 0.3,        # 30% get helpful votes
                'verified_purchase_rate': 0.85   # 85% verified purchases
            },
            'enthusiast': {
                'review_frequency': (10, 50),
                'avg_review_length': (100, 300),
                'rating_bias': 0.2,              # Slightly more positive
                'helpful_vote_rate': 0.6,
                'verified_purchase_rate': 0.95
            },
            'critic': {
                'review_frequency': (5, 20),
                'avg_review_length': (150, 400),
                'rating_bias': -0.3,             # More critical
                'helpful_vote_rate': 0.7,
                'verified_purchase_rate': 0.90
            },
            'influencer': {
                'review_frequency': (20, 100),
                'avg_review_length': (80, 200),
                'rating_bias': 0.1,
                'helpful_vote_rate': 0.8,
                'verified_purchase_rate': 0.70   # Some promotional reviews
            }
        }
        
        # Review timing patterns
        self.review_timing = {
            'immediate': 0.15,    # Within 1 day
            'quick': 0.35,        # 2-7 days
            'normal': 0.35,       # 1-4 weeks
            'delayed': 0.15       # 1+ months
        }
    
    def _setup_rating_distributions(self):
        """Setup realistic rating distribution patterns"""
        # Overall rating distribution (J-curve pattern common in reviews)
        self.rating_distribution = {
            5: 0.45,  # 45% - Most common
            4: 0.25,  # 25%
            3: 0.12,  # 12%
            2: 0.08,  # 8%
            1: 0.10   # 10% - Second most common (polarization)
        }
        
        # Product category rating adjustments
        self.category_rating_adjustments = {
            'technology': {'avg_boost': 0.1, 'variance': 0.8},
            'healthcare': {'avg_boost': 0.2, 'variance': 0.6},
            'retail': {'avg_boost': 0.0, 'variance': 1.0},
            'automotive': {'avg_boost': 0.15, 'variance': 0.7},
            'home_garden': {'avg_boost': 0.05, 'variance': 0.9}
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate product reviews dataset
        
        Args:
            rows: Number of review records to generate
            **kwargs: Additional parameters (time_series, product_relationships, etc.)
            
        Returns:
            pd.DataFrame: Generated product reviews data with realistic patterns
        """
        # Check if time series generation is requested
        ts_config = self._create_time_series_config(**kwargs)
        
        if ts_config:
            return self._generate_time_series_reviews(rows, ts_config, **kwargs)
        else:
            return self._generate_snapshot_reviews(rows, **kwargs)
    
    def _generate_snapshot_reviews(self, rows: int, **kwargs) -> pd.DataFrame:
        """Generate snapshot review data (random timestamps)"""
        date_range = kwargs.get('date_range', None)
        product_relationships = kwargs.get('product_relationships', None)
        
        data = []
        
        for i in range(rows):
            # Generate review date
            if date_range:
                start_date, end_date = date_range
                review_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            else:
                review_date = self.faker.date_between(start_date='-2y', end_date='today')
            
            # Generate review record
            review = self._generate_product_review(i, review_date, product_relationships)
            data.append(review)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_time_series_reviews(self, rows: int, ts_config, **kwargs) -> pd.DataFrame:
        """Generate time series review data using integrated time series system"""
        
        # Generate timestamps from time series config
        timestamps = self._generate_time_series_timestamps(ts_config, rows)
        
        # Create base time series for review volume
        base_volume = 50.0  # Base reviews per period
        
        volume_series = self.time_series_generator.generate_time_series_base(
            ts_config,
            base_value=base_volume,
            value_range=(base_volume * 0.2, base_volume * 3.0)
        )
        
        data = []
        product_relationships = kwargs.get('product_relationships', None)
        
        for i, timestamp in enumerate(timestamps):
            if i >= rows or i >= len(volume_series):
                break
            
            # Get time series volume multiplier
            volume_multiplier = volume_series.iloc[i]['value'] / base_volume
            
            # Generate review with temporal patterns
            review = self._generate_time_series_review(
                i, timestamp, volume_multiplier, product_relationships
            )
            
            data.append(review)
        
        df = pd.DataFrame(data)
        
        # Add temporal relationships
        df = self._add_temporal_relationships(df, ts_config)
        
        # Apply review-specific time series correlations
        df = self._apply_review_time_series_correlations(df, ts_config)
        
        return self._apply_realistic_patterns(df)
    
    def _generate_product_review(self, review_index: int, review_date: datetime.date, 
                                product_relationships: Optional[Dict] = None) -> Dict:
        """Generate a single product review record"""
        
        # Select product and category
        if product_relationships and 'products' in product_relationships:
            # Use provided product relationships
            product_info = self.faker.random_element(product_relationships['products'])
            product_name = product_info.get('name', 'Unknown Product')
            category = product_info.get('category', 'general')
        else:
            # Generate random product
            category = self.faker.random_element(list(self.products.keys()))
            product_name = self.faker.random_element(self.products[category])
        
        # Generate reviewer information
        reviewer_type = self._select_reviewer_type()
        reviewer_info = self._generate_reviewer_info(reviewer_type)
        
        # Generate rating with realistic distribution
        rating = self._generate_realistic_rating(category, reviewer_type)
        
        # Generate review text correlated with rating
        review_text = self._generate_review_text(product_name, rating, reviewer_type)
        
        # Calculate sentiment score correlated with rating
        sentiment_score = self._calculate_sentiment_score(review_text, rating)
        
        # Generate additional review metrics
        helpful_votes = self._generate_helpful_votes(reviewer_type, rating, len(review_text.split()))
        
        # Generate purchase verification
        verified_purchase = self._is_verified_purchase(reviewer_type)
        
        # Calculate review timing (days after purchase)
        days_after_purchase = self._generate_review_timing()
        purchase_date = review_date - timedelta(days=days_after_purchase)
        
        return {
            'review_id': f'REV_{review_index+1:08d}',
            'product_id': f'PROD_{hash(product_name) % 100000:05d}',
            'product_name': product_name,
            'product_category': category,
            'brand': self.faker.random_element(self.product_attributes.get('brands', ['Generic'])),
            'reviewer_id': reviewer_info['reviewer_id'],
            'reviewer_name': reviewer_info['reviewer_name'],
            'reviewer_type': reviewer_type,
            'rating': rating,
            'review_text': review_text,
            'review_title': self._generate_review_title(rating, product_name),
            'sentiment_score': round(sentiment_score, 2),
            'sentiment_label': self._get_sentiment_label(sentiment_score),
            'word_count': len(review_text.split()),
            'helpful_votes': helpful_votes,
            'total_votes': helpful_votes + self.faker.random_int(0, max(helpful_votes // 2, 1)),
            'verified_purchase': verified_purchase,
            'review_date': review_date,
            'purchase_date': purchase_date if verified_purchase else None,
            'days_after_purchase': days_after_purchase if verified_purchase else None,
            'reviewer_total_reviews': reviewer_info['total_reviews'],
            'reviewer_avg_rating': reviewer_info['avg_rating'],
            'product_price_tier': self._get_price_tier(category),
            'review_language': 'en',
            'contains_images': self.faker.boolean(chance_of_getting_true=25),
            'contains_video': self.faker.boolean(chance_of_getting_true=5)
        }
    
    def _generate_time_series_review(self, review_index: int, timestamp: datetime, 
                                   volume_multiplier: float, 
                                   product_relationships: Optional[Dict] = None) -> Dict:
        """Generate time series review with temporal patterns"""
        
        # Select product with time-aware preferences
        if product_relationships and 'products' in product_relationships:
            product_info = self.faker.random_element(product_relationships['products'])
            product_name = product_info.get('name', 'Unknown Product')
            category = product_info.get('category', 'general')
        else:
            category = self._select_time_aware_category(timestamp)
            product_name = self.faker.random_element(self.products[category])
        
        # Generate reviewer with temporal patterns
        reviewer_type = self._select_time_aware_reviewer_type(timestamp, volume_multiplier)
        reviewer_info = self._generate_reviewer_info(reviewer_type)
        
        # Generate rating with temporal adjustments
        rating = self._generate_temporal_rating(category, reviewer_type, timestamp, volume_multiplier)
        
        # Generate review text with temporal context
        review_text = self._generate_temporal_review_text(product_name, rating, reviewer_type, timestamp)
        
        # Calculate sentiment with temporal patterns
        sentiment_score = self._calculate_temporal_sentiment(review_text, rating, timestamp)
        
        # Generate metrics with temporal adjustments
        helpful_votes = self._generate_temporal_helpful_votes(reviewer_type, rating, volume_multiplier)
        verified_purchase = self._is_verified_purchase(reviewer_type)
        days_after_purchase = self._generate_review_timing()
        purchase_date = timestamp.date() - timedelta(days=days_after_purchase)
        
        return {
            'review_id': f'REV_{review_index+1:08d}',
            'product_id': f'PROD_{hash(product_name) % 100000:05d}',
            'product_name': product_name,
            'product_category': category,
            'brand': self.faker.random_element(self.product_attributes.get('brands', ['Generic'])),
            'reviewer_id': reviewer_info['reviewer_id'],
            'reviewer_name': reviewer_info['reviewer_name'],
            'reviewer_type': reviewer_type,
            'rating': rating,
            'review_text': review_text,
            'review_title': self._generate_review_title(rating, product_name),
            'sentiment_score': round(sentiment_score, 2),
            'sentiment_label': self._get_sentiment_label(sentiment_score),
            'word_count': len(review_text.split()),
            'helpful_votes': helpful_votes,
            'total_votes': helpful_votes + self.faker.random_int(0, max(helpful_votes // 2, 1)),
            'verified_purchase': verified_purchase,
            'timestamp': timestamp,
            'review_date': timestamp.date(),
            'purchase_date': purchase_date if verified_purchase else None,
            'days_after_purchase': days_after_purchase if verified_purchase else None,
            'reviewer_total_reviews': reviewer_info['total_reviews'],
            'reviewer_avg_rating': reviewer_info['avg_rating'],
            'product_price_tier': self._get_price_tier(category),
            'review_language': 'en',
            'contains_images': self.faker.boolean(chance_of_getting_true=25),
            'contains_video': self.faker.boolean(chance_of_getting_true=5),
            'volume_factor': round(volume_multiplier, 2)
        }
    
    def _select_reviewer_type(self) -> str:
        """Select reviewer type based on realistic distribution"""
        weights = [0.60, 0.20, 0.15, 0.05]  # casual, enthusiast, critic, influencer
        return self.faker.random.choices(
            list(self.reviewer_types.keys()), 
            weights=weights
        )[0]
    
    def _generate_reviewer_info(self, reviewer_type: str) -> Dict:
        """Generate reviewer information based on type"""
        reviewer_config = self.reviewer_types[reviewer_type]
        
        # Generate total reviews based on reviewer type
        freq_min, freq_max = reviewer_config['review_frequency']
        total_reviews = self.faker.random_int(freq_min, freq_max)
        
        # Generate average rating with bias
        base_avg = 3.8  # Typical average rating
        bias = reviewer_config['rating_bias']
        avg_rating = max(1.0, min(5.0, base_avg + bias + self.faker.random.uniform(-0.3, 0.3)))
        
        return {
            'reviewer_id': f'USER_{self.faker.random_int(100000, 999999)}',
            'reviewer_name': self.faker.name(),
            'total_reviews': total_reviews,
            'avg_rating': round(avg_rating, 1)
        }
    
    def _generate_realistic_rating(self, category: str, reviewer_type: str) -> int:
        """Generate rating following realistic distribution patterns"""
        
        # Start with base distribution
        rating_choices = list(self.rating_distribution.keys())
        rating_weights = list(self.rating_distribution.values())
        
        # Apply category adjustments
        if category in self.category_rating_adjustments:
            adj = self.category_rating_adjustments[category]
            # Boost higher ratings slightly for certain categories
            if adj['avg_boost'] > 0:
                rating_weights[3] *= (1 + adj['avg_boost'])  # 4-star boost
                rating_weights[4] *= (1 + adj['avg_boost'])  # 5-star boost
        
        # Apply reviewer type bias
        reviewer_config = self.reviewer_types[reviewer_type]
        bias = reviewer_config['rating_bias']
        
        if bias > 0:  # More positive ratings
            rating_weights[3] *= (1 + bias)
            rating_weights[4] *= (1 + bias)
        elif bias < 0:  # More negative ratings
            rating_weights[0] *= (1 + abs(bias))
            rating_weights[1] *= (1 + abs(bias))
        
        # Normalize weights
        total_weight = sum(rating_weights)
        rating_weights = [w / total_weight for w in rating_weights]
        
        return self.faker.random.choices(rating_choices, weights=rating_weights)[0]
    
    def _generate_review_text(self, product_name: str, rating: int, reviewer_type: str) -> str:
        """Generate review text correlated with rating"""
        
        # Get appropriate templates for rating
        if rating >= 4:
            templates = self.review_templates[rating]['positive']
        elif rating == 3:
            templates = self.review_templates[rating]['neutral']
        else:
            templates = self.review_templates[rating]['negative']
        
        # Select template
        template = self.faker.random_element(templates)
        
        # Get target length based on reviewer type
        reviewer_config = self.reviewer_types[reviewer_type]
        min_words, max_words = reviewer_config['avg_review_length']
        target_words = self.faker.random_int(min_words, max_words)
        
        # Fill template with appropriate components
        review_text = self._fill_review_template(template, product_name, rating)
        
        # Extend or trim to target length
        review_text = self._adjust_review_length(review_text, target_words, rating)
        
        return review_text
    
    def _fill_review_template(self, template: str, product_name: str, rating: int) -> str:
        """Fill review template with appropriate content"""
        
        # Replace product placeholder
        text = template.replace('{product}', product_name.lower())
        
        # Replace other placeholders with appropriate content
        for placeholder, options in self.template_components.items():
            if f'{{{placeholder}}}' in text:
                replacement = self.faker.random_element(options)
                text = text.replace(f'{{{placeholder}}}', replacement)
        
        return text
    
    def _adjust_review_length(self, text: str, target_words: int, rating: int) -> str:
        """Adjust review length to match target word count"""
        current_words = len(text.split())
        
        if current_words < target_words:
            # Add more content
            additional_sentences = self._generate_additional_sentences(rating, target_words - current_words)
            text += ' ' + additional_sentences
        elif current_words > target_words * 1.5:
            # Trim if significantly over target
            words = text.split()
            text = ' '.join(words[:target_words])
        
        return text
    
    def _generate_additional_sentences(self, rating: int, needed_words: int) -> str:
        """Generate additional sentences to reach target word count"""
        
        additional_content = {
            5: ["The shipping was fast.", "Great customer service.", "Exactly as described.", 
                "Would definitely buy again.", "Exceeded all expectations."],
            4: ["Good value for money.", "Minor issues but overall satisfied.", "Quick delivery.", 
                "Would recommend to others.", "Pretty happy with this purchase."],
            3: ["It's okay for the price.", "Does what it's supposed to do.", "Average quality.", 
                "Not bad, not great.", "Meets basic expectations."],
            2: ["Had some problems with it.", "Not worth the money.", "Quality could be better.", 
                "Disappointed with purchase.", "Several issues encountered."],
            1: ["Complete waste of money.", "Broke almost immediately.", "Terrible quality.", 
                "Would not recommend.", "Save your money and buy something else."]
        }
        
        sentences = []
        current_words = 0
        
        while current_words < needed_words:
            sentence = self.faker.random_element(additional_content[rating])
            sentences.append(sentence)
            current_words += len(sentence.split())
        
        return ' '.join(sentences)
    
    def _calculate_sentiment_score(self, review_text: str, rating: int) -> float:
        """Calculate sentiment score correlated with rating"""
        
        # Get base sentiment range for rating
        min_score, max_score = self.sentiment_ranges[rating]
        base_score = self.faker.random.uniform(min_score, max_score)
        
        # Analyze text for sentiment keywords
        text_lower = review_text.lower()
        sentiment_adjustment = 0
        
        # Count sentiment keywords
        for sentiment_type, keywords in self.sentiment_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            
            if sentiment_type == 'very_positive':
                sentiment_adjustment += keyword_count * 5
            elif sentiment_type == 'positive':
                sentiment_adjustment += keyword_count * 2
            elif sentiment_type == 'negative':
                sentiment_adjustment -= keyword_count * 2
            elif sentiment_type == 'very_negative':
                sentiment_adjustment -= keyword_count * 5
        
        # Apply adjustment with dampening
        adjusted_score = base_score + (sentiment_adjustment * 2)
        
        # Ensure score stays within valid range
        return max(0, min(100, adjusted_score))
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score >= 80:
            return 'very_positive'
        elif sentiment_score >= 60:
            return 'positive'
        elif sentiment_score >= 40:
            return 'neutral'
        elif sentiment_score >= 20:
            return 'negative'
        else:
            return 'very_negative'
    
    def _generate_review_title(self, rating: int, product_name: str) -> str:
        """Generate review title based on rating"""
        
        title_templates = {
            5: ["Excellent {product}!", "Love this {product}!", "Perfect {product}!", 
                "Amazing quality!", "Highly recommend!"],
            4: ["Great {product}", "Really good purchase", "Happy with this {product}", 
                "Good quality", "Solid choice"],
            3: ["Decent {product}", "It's okay", "Average quality", "Fair purchase", "Mixed feelings"],
            2: ["Disappointed", "Not great", "Below expectations", "Poor quality", "Issues with {product}"],
            1: ["Terrible!", "Waste of money", "Awful quality", "Don't buy!", "Complete disaster"]
        }
        
        template = self.faker.random_element(title_templates[rating])
        return template.replace('{product}', product_name)
    
    def _generate_helpful_votes(self, reviewer_type: str, rating: int, word_count: int) -> int:
        """Generate helpful votes based on reviewer type and review quality"""
        
        reviewer_config = self.reviewer_types[reviewer_type]
        base_rate = reviewer_config['helpful_vote_rate']
        
        # Adjust based on rating (extreme ratings get more attention)
        if rating in [1, 5]:
            attention_multiplier = 1.5
        elif rating in [2, 4]:
            attention_multiplier = 1.2
        else:
            attention_multiplier = 1.0
        
        # Adjust based on review length (longer reviews tend to get more votes)
        length_multiplier = min(1.0 + (word_count / 200), 2.0)
        
        # Calculate expected votes
        expected_votes = base_rate * attention_multiplier * length_multiplier * 10
        
        # Add randomness using exponential distribution to simulate Poisson-like behavior
        if expected_votes > 0:
            # Use exponential distribution as approximation for Poisson
            random_factor = self.faker.random.expovariate(1.0 / expected_votes) if expected_votes > 0 else 0
            return max(0, int(random_factor))
        else:
            return 0
    
    def _is_verified_purchase(self, reviewer_type: str) -> bool:
        """Determine if review is from verified purchase"""
        reviewer_config = self.reviewer_types[reviewer_type]
        return self.faker.boolean(chance_of_getting_true=int(reviewer_config['verified_purchase_rate'] * 100))
    
    def _generate_review_timing(self) -> int:
        """Generate days between purchase and review"""
        timing_choice = self.faker.random.choices(
            list(self.review_timing.keys()),
            weights=list(self.review_timing.values())
        )[0]
        
        timing_ranges = {
            'immediate': (0, 1),
            'quick': (2, 7),
            'normal': (8, 28),
            'delayed': (29, 365)
        }
        
        min_days, max_days = timing_ranges[timing_choice]
        return self.faker.random_int(min_days, max_days)
    
    def _get_price_tier(self, category: str) -> str:
        """Get price tier for product category"""
        price_tiers = ['budget', 'economy', 'standard', 'premium', 'luxury']
        
        # Different categories have different price tier distributions
        if category == 'technology':
            weights = [0.1, 0.2, 0.4, 0.25, 0.05]
        elif category == 'healthcare':
            weights = [0.05, 0.15, 0.5, 0.25, 0.05]
        else:
            weights = [0.2, 0.3, 0.3, 0.15, 0.05]
        
        return self.faker.random.choices(price_tiers, weights=weights)[0]
    
    def _select_time_aware_category(self, timestamp: datetime) -> str:
        """Select product category based on temporal patterns"""
        month = timestamp.month
        
        # Seasonal preferences
        if month in [11, 12]:  # Holiday season
            preferred_categories = ['technology', 'retail', 'home_garden']
        elif month in [1, 2]:  # New year
            preferred_categories = ['healthcare', 'technology']
        elif month in [6, 7, 8]:  # Summer
            preferred_categories = ['retail', 'automotive']
        else:
            preferred_categories = list(self.products.keys())
        
        available_categories = [cat for cat in preferred_categories if cat in self.products]
        if not available_categories:
            available_categories = list(self.products.keys())
        
        return self.faker.random_element(available_categories)
    
    def _select_time_aware_reviewer_type(self, timestamp: datetime, volume_multiplier: float) -> str:
        """Select reviewer type based on temporal patterns"""
        
        # High volume periods attract more casual reviewers
        if volume_multiplier > 1.5:
            weights = [0.70, 0.15, 0.10, 0.05]  # More casual
        elif volume_multiplier < 0.7:
            weights = [0.45, 0.25, 0.25, 0.05]  # More critics/enthusiasts
        else:
            weights = [0.60, 0.20, 0.15, 0.05]  # Normal distribution
        
        return self.faker.random.choices(
            list(self.reviewer_types.keys()),
            weights=weights
        )[0]
    
    def _generate_temporal_rating(self, category: str, reviewer_type: str, 
                                timestamp: datetime, volume_multiplier: float) -> int:
        """Generate rating with temporal adjustments"""
        
        # Base rating generation
        rating = self._generate_realistic_rating(category, reviewer_type)
        
        # Temporal adjustments
        month = timestamp.month
        hour = timestamp.hour
        
        # Holiday season tends to be more positive
        if month in [11, 12]:
            if rating < 5 and self.faker.boolean(chance_of_getting_true=20):
                rating += 1
        
        # High volume periods might have slightly lower average ratings (more casual reviews)
        if volume_multiplier > 2.0:
            if rating > 1 and self.faker.boolean(chance_of_getting_true=15):
                rating -= 1
        
        return max(1, min(5, rating))
    
    def _generate_temporal_review_text(self, product_name: str, rating: int, 
                                     reviewer_type: str, timestamp: datetime) -> str:
        """Generate review text with temporal context"""
        
        # Base review text
        review_text = self._generate_review_text(product_name, rating, reviewer_type)
        
        # Add temporal context for certain periods
        month = timestamp.month
        
        if month in [11, 12]:  # Holiday season
            holiday_additions = [
                "Perfect for the holidays!", "Great gift idea!", "Holiday shopping success!",
                "Arrived just in time for Christmas!", "Holiday season purchase."
            ]
            if self.faker.boolean(chance_of_getting_true=30):
                review_text += " " + self.faker.random_element(holiday_additions)
        
        elif month == 1:  # New year
            new_year_additions = [
                "New year, new purchase!", "Starting the year right!", "Resolution to buy quality!",
                "Fresh start with this product!", "New year upgrade!"
            ]
            if self.faker.boolean(chance_of_getting_true=20):
                review_text += " " + self.faker.random_element(new_year_additions)
        
        return review_text
    
    def _calculate_temporal_sentiment(self, review_text: str, rating: int, timestamp: datetime) -> float:
        """Calculate sentiment with temporal adjustments"""
        
        # Base sentiment calculation
        sentiment_score = self._calculate_sentiment_score(review_text, rating)
        
        # Temporal adjustments
        month = timestamp.month
        
        # Holiday season tends to be more positive
        if month in [11, 12]:
            sentiment_score += self.faker.random.uniform(0, 5)
        
        # Post-holiday period might be slightly more negative
        elif month == 1:
            sentiment_score -= self.faker.random.uniform(0, 3)
        
        return max(0, min(100, sentiment_score))
    
    def _generate_temporal_helpful_votes(self, reviewer_type: str, rating: int, volume_multiplier: float) -> int:
        """Generate helpful votes with temporal patterns"""
        
        # Base helpful votes
        helpful_votes = self._generate_helpful_votes(reviewer_type, rating, 100)  # Assume average length
        
        # High volume periods might dilute individual review attention
        if volume_multiplier > 1.5:
            helpful_votes = int(helpful_votes * 0.8)
        elif volume_multiplier < 0.7:
            helpful_votes = int(helpful_votes * 1.2)
        
        return max(0, helpful_votes)
    
    def _apply_review_time_series_correlations(self, data: pd.DataFrame, ts_config) -> pd.DataFrame:
        """Apply realistic time-based correlations for review data"""
        if len(data) < 2:
            return data
        
        # Sort by timestamp to ensure proper time series order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply temporal correlations for rating trends
        for i in range(1, len(data)):
            prev_rating = data.iloc[i-1]['rating']
            prev_sentiment = data.iloc[i-1]['sentiment_score']
            
            # Rating momentum (similar ratings tend to cluster)
            if prev_rating >= 4:
                correlation_strength = 0.2
                if self.faker.boolean(chance_of_getting_true=int(correlation_strength * 100)):
                    if data.iloc[i]['rating'] < 4:
                        data.loc[i, 'rating'] = min(5, data.iloc[i]['rating'] + 1)
            
            # Sentiment correlation with previous reviews
            sentiment_momentum = 0.1
            sentiment_adjustment = (prev_sentiment - 50) * sentiment_momentum
            new_sentiment = data.iloc[i]['sentiment_score'] + sentiment_adjustment
            data.loc[i, 'sentiment_score'] = max(0, min(100, new_sentiment))
            
            # Update sentiment label
            data.loc[i, 'sentiment_label'] = self._get_sentiment_label(data.iloc[i]['sentiment_score'])
        
        return data
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to review data"""
        
        # Add derived fields
        if 'timestamp' in data.columns:
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.day_name()
            data['month'] = pd.to_datetime(data['timestamp']).dt.month
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['is_weekend'] = pd.to_datetime(data['timestamp']).dt.weekday >= 5
        else:
            data['day_of_week'] = pd.to_datetime(data['review_date']).dt.day_name()
            data['month'] = pd.to_datetime(data['review_date']).dt.month
            data['is_weekend'] = pd.to_datetime(data['review_date']).dt.weekday >= 5
        
        # Calculate additional metrics
        data['helpful_ratio'] = data['helpful_votes'] / data['total_votes'].replace(0, 1)
        data['review_length_category'] = pd.cut(
            data['word_count'],
            bins=[0, 20, 50, 100, 200, float('inf')],
            labels=['very_short', 'short', 'medium', 'long', 'very_long']
        )
        
        # Add review quality score (combination of factors)
        max_helpful_votes = data['helpful_votes'].max()
        max_helpful_votes = max_helpful_votes if max_helpful_votes > 0 else 1
        
        data['quality_score'] = (
            (data['word_count'] / 100) * 0.3 +
            (data['helpful_votes'] / max_helpful_votes) * 0.4 +
            (data['verified_purchase'].astype(int)) * 0.3
        ).clip(0, 1).round(2)
        
        # Add sentiment-rating correlation flag
        data['sentiment_rating_aligned'] = (
            ((data['rating'] >= 4) & (data['sentiment_score'] >= 60)) |
            ((data['rating'] <= 2) & (data['sentiment_score'] <= 40)) |
            ((data['rating'] == 3) & (data['sentiment_score'].between(35, 65)))
        )
        
        # Add reviewer credibility score
        max_helpful_votes_cred = data['helpful_votes'].max()
        max_helpful_votes_cred = max_helpful_votes_cred if max_helpful_votes_cred > 0 else 1
        
        data['reviewer_credibility'] = (
            (data['reviewer_total_reviews'] / 50).clip(0, 1) * 0.4 +
            data['verified_purchase'].astype(int) * 0.3 +
            (data['helpful_votes'] / max_helpful_votes_cred) * 0.3
        ).round(2)
        
        return data