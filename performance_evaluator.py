#!/usr/bin/env python3
"""
Comprehensive Performance Evaluation System
For AI-Powered Restaurant Recommendation System

This module provides detailed performance metrics for the entire recommendation system,
including accuracy, speed, memory usage, and business metrics.
"""

import sys
import os
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hybrid_recommender import HybridRecommender
from data.preprocessor import DataPreprocessor
from tests.test_factory import TestDataFactory

class PerformanceEvaluator:
    """Comprehensive performance evaluation for the recommendation system"""
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
        self.timing_data = {}
        
    def evaluate_entire_system(self, data_size='large'):
        """Evaluate the entire recommendation system"""
        print("🎯 Comprehensive System Performance Evaluation")
        print("=" * 60)
        
        # Load or generate data
        if data_size == 'large':
            data = self._load_large_dataset()
        else:
            data = self._generate_test_dataset()
            
        print(f"📊 Dataset Size:")
        print(f"  • Restaurants: {len(data['restaurants'])}")
        print(f"  • Users: {len(data['users'])}")
        print(f"  • Ratings: {len(data['ratings'])}")
        print(f"  • Reviews: {len(data['reviews'])}")
        
        # Split data for evaluation
        train_data, test_data = self._split_data(data)
        
        # Initialize and train model
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        recommender = HybridRecommender()
        
        print("\n🔧 Training Models...")
        train_start = time.time()
        recommender.fit(train_data)
        train_time = time.time() - train_start
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        print(f"✅ Training completed in {train_time:.2f} seconds")
        print(f"📊 Memory usage: {memory_usage:.2f} MB")
        
        # Evaluate different aspects
        accuracy_metrics = self._evaluate_accuracy(recommender, test_data)
        speed_metrics = self._evaluate_speed(recommender, test_data)
        business_metrics = self._evaluate_business_value(recommender, test_data)
        quality_metrics = self._evaluate_recommendation_quality(recommender, test_data)
        
        # Compile results
        self.results = {
            'dataset_size': {
                'restaurants': len(data['restaurants']),
                'users': len(data['users']),
                'ratings': len(data['ratings']),
                'reviews': len(data['reviews'])
            },
            'training_metrics': {
                'train_time': train_time,
                'memory_usage_mb': memory_usage
            },
            'accuracy_metrics': accuracy_metrics,
            'speed_metrics': speed_metrics,
            'business_metrics': business_metrics,
            'quality_metrics': quality_metrics
        }
        
        # Generate report
        self._generate_performance_report()
        
        return self.results
    
    def _load_large_dataset(self):
        """Load large dataset if available, otherwise generate it"""
        try:
            # Try to load large dataset
            restaurants = pd.read_csv('data/restaurants.csv')
            users = pd.read_csv('data/users.csv')
            ratings = pd.read_csv('data/ratings.csv')
            reviews = pd.read_csv('data/reviews.csv')
            
            # Create additional required data
            user_item_matrix = ratings.pivot_table(
                index='user_id', columns='restaurant_id', values='rating', fill_value=0
            )
            
            # Create feature matrix
            factory = TestDataFactory()
            restaurant_features = factory._create_feature_matrix(restaurants)
            
            return {
                'restaurants': restaurants,
                'users': users,
                'ratings': ratings,
                'reviews': reviews,
                'user_item_matrix': user_item_matrix,
                'restaurant_features': restaurant_features,
                'label_encoders': {}
            }
            
        except FileNotFoundError:
            print("⚠️  Large dataset not found, generating test data...")
            return self._generate_test_dataset(scale=10)
    
    def _generate_test_dataset(self, scale=1):
        """Generate test dataset for evaluation"""
        factory = TestDataFactory()
        
        restaurants = factory.create_restaurants(200 * scale)
        users = factory.create_users(100 * scale)
        ratings = factory.create_ratings(users, restaurants, 2000 * scale)
        reviews = factory.create_reviews(ratings.head(1000 * scale))
        
        user_item_matrix = ratings.pivot_table(
            index='user_id', columns='restaurant_id', values='rating', fill_value=0
        )
        
        restaurant_features = factory._create_feature_matrix(restaurants)
        
        return {
            'restaurants': restaurants,
            'users': users,
            'ratings': ratings,
            'reviews': reviews,
            'user_item_matrix': user_item_matrix,
            'restaurant_features': restaurant_features,
            'label_encoders': {}
        }
    
    def _split_data(self, data, test_size=0.2):
        """Split data into train and test sets"""
        ratings = data['ratings']
        
        # Split ratings
        train_ratings, test_ratings = train_test_split(
            ratings, test_size=test_size, random_state=42
        )
        
        # Create train data
        train_user_item_matrix = train_ratings.pivot_table(
            index='user_id', columns='restaurant_id', values='rating', fill_value=0
        )
        
        train_data = data.copy()
        train_data['ratings'] = train_ratings
        train_data['user_item_matrix'] = train_user_item_matrix
        
        return train_data, test_ratings
    
    def _evaluate_accuracy(self, recommender, test_data):
        """Evaluate recommendation accuracy"""
        print("\n📈 Evaluating Accuracy...")
        
        predictions = []
        actuals = []
        
        # Sample users for evaluation
        test_users = test_data['user_id'].unique()[:100]  # Test on 100 users
        
        for user_id in test_users:
            user_ratings = test_data[test_data['user_id'] == user_id]
            
            if len(user_ratings) > 0:
                # Get recommendations
                try:
                    recs = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
                    
                    # Calculate accuracy metrics
                    for _, rating_row in user_ratings.iterrows():
                        restaurant_id = rating_row['restaurant_id']
                        actual_rating = rating_row['rating']
                        
                        # Find if this restaurant was recommended
                        predicted_score = 3.0  # Default
                        for rec in recs:
                            if rec['restaurant_id'] == restaurant_id:
                                predicted_score = rec.get('score', 3.0) * 5  # Convert to 1-5 scale
                                break
                        
                        predictions.append(predicted_score)
                        actuals.append(actual_rating)
                        
                except Exception as e:
                    continue
        
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Calculate coverage (percentage of items that can be recommended)
            unique_recommended = set()
            for user_id in test_users[:20]:  # Sample for coverage
                try:
                    recs = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
                    unique_recommended.update([r['restaurant_id'] for r in recs])
                except:
                    continue
            
            total_restaurants = len(recommender.data['restaurants'])
            coverage = len(unique_recommended) / total_restaurants
            
            return {
                'rmse': rmse,
                'mae': mae,
                'coverage': coverage,
                'samples_evaluated': len(predictions)
            }
        else:
            return {
                'rmse': float('inf'),
                'mae': float('inf'),
                'coverage': 0.0,
                'samples_evaluated': 0
            }
    
    def _evaluate_speed(self, recommender, test_data):
        """Evaluate recommendation speed"""
        print("⚡ Evaluating Speed...")
        
        test_users = test_data['user_id'].unique()[:50]  # Test on 50 users
        times = []
        
        for user_id in test_users:
            start_time = time.time()
            try:
                recs = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
                end_time = time.time()
                times.append(end_time - start_time)
            except:
                continue
        
        if times:
            return {
                'avg_response_time': np.mean(times),
                'min_response_time': np.min(times),
                'max_response_time': np.max(times),
                'p95_response_time': np.percentile(times, 95),
                'throughput_recs_per_sec': 1 / np.mean(times) if np.mean(times) > 0 else 0
            }
        else:
            return {
                'avg_response_time': float('inf'),
                'min_response_time': float('inf'),
                'max_response_time': float('inf'),
                'p95_response_time': float('inf'),
                'throughput_recs_per_sec': 0
            }
    
    def _evaluate_business_value(self, recommender, test_data):
        """Evaluate business value metrics"""
        print("💰 Evaluating Business Value...")
        
        test_users = test_data['user_id'].unique()[:30]  # Test on 30 users
        
        # Calculate diversity
        all_recommendations = []
        user_satisfaction_scores = []
        
        for user_id in test_users:
            try:
                recs = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
                rec_restaurants = [r['restaurant_id'] for r in recs]
                all_recommendations.extend(rec_restaurants)
                
                # Calculate user satisfaction (based on ratings of recommended items)
                user_ratings = test_data[test_data['user_id'] == user_id]
                if len(user_ratings) > 0:
                    recommended_ratings = user_ratings[user_ratings['restaurant_id'].isin(rec_restaurants)]
                    if len(recommended_ratings) > 0:
                        avg_rating = recommended_ratings['rating'].mean()
                        user_satisfaction_scores.append(avg_rating)
                
            except:
                continue
        
        # Diversity: How many unique restaurants are recommended
        unique_recs = len(set(all_recommendations))
        total_recs = len(all_recommendations)
        diversity = unique_recs / total_recs if total_recs > 0 else 0
        
        # Average user satisfaction
        avg_satisfaction = np.mean(user_satisfaction_scores) if user_satisfaction_scores else 0
        
        return {
            'diversity_score': diversity,
            'avg_user_satisfaction': avg_satisfaction,
            'unique_restaurants_recommended': unique_recs,
            'total_recommendations_made': total_recs
        }
    
    def _evaluate_recommendation_quality(self, recommender, test_data):
        """Evaluate qualitative aspects of recommendations"""
        print("🎯 Evaluating Recommendation Quality...")
        
        # Test emotional intelligence
        emotional_contexts = [
            "I'm feeling stressed and need comfort food",
            "I'm happy and want to celebrate",
            "I'm feeling romantic and want a nice dinner",
            "I'm feeling adventurous and want to try something new"
        ]
        
        emotional_responses = 0
        total_emotional_tests = 0
        
        test_user = test_data['user_id'].iloc[0] if len(test_data) > 0 else 1
        
        for context in emotional_contexts:
            try:
                recs = recommender.get_hybrid_recommendations(
                    test_user, 
                    n_recommendations=5,
                    user_text_input=context
                )
                if len(recs) > 0:
                    emotional_responses += 1
                total_emotional_tests += 1
            except:
                total_emotional_tests += 1
                continue
        
        emotional_success_rate = emotional_responses / total_emotional_tests if total_emotional_tests > 0 else 0
        
        return {
            'emotional_intelligence_success_rate': emotional_success_rate,
            'emotional_contexts_tested': total_emotional_tests,
            'emotional_responses_generated': emotional_responses
        }
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        
        # Dataset Information
        print(f"\n📈 Dataset Information:")
        print(f"  • Restaurants: {self.results['dataset_size']['restaurants']:,}")
        print(f"  • Users: {self.results['dataset_size']['users']:,}")
        print(f"  • Ratings: {self.results['dataset_size']['ratings']:,}")
        print(f"  • Reviews: {self.results['dataset_size']['reviews']:,}")
        
        # Training Performance
        print(f"\n🔧 Training Performance:")
        print(f"  • Training Time: {self.results['training_metrics']['train_time']:.2f} seconds")
        print(f"  • Memory Usage: {self.results['training_metrics']['memory_usage_mb']:.2f} MB")
        
        # Accuracy Metrics
        acc = self.results['accuracy_metrics']
        print(f"\n📈 Accuracy Metrics:")
        print(f"  • RMSE: {acc['rmse']:.3f}")
        print(f"  • MAE: {acc['mae']:.3f}")
        print(f"  • Coverage: {acc['coverage']:.2%}")
        print(f"  • Samples Evaluated: {acc['samples_evaluated']}")
        
        # Speed Metrics
        speed = self.results['speed_metrics']
        print(f"\n⚡ Speed Metrics:")
        print(f"  • Average Response Time: {speed['avg_response_time']:.3f} seconds")
        print(f"  • 95th Percentile: {speed['p95_response_time']:.3f} seconds")
        print(f"  • Throughput: {speed['throughput_recs_per_sec']:.1f} recommendations/second")
        
        # Business Metrics
        business = self.results['business_metrics']
        print(f"\n💰 Business Value Metrics:")
        print(f"  • Diversity Score: {business['diversity_score']:.2%}")
        print(f"  • User Satisfaction: {business['avg_user_satisfaction']:.2f}/5.0")
        print(f"  • Unique Restaurants: {business['unique_restaurants_recommended']}")
        
        # Quality Metrics
        quality = self.results['quality_metrics']
        print(f"\n🎯 Quality Metrics:")
        print(f"  • Emotional Intelligence Success: {quality['emotional_intelligence_success_rate']:.2%}")
        print(f"  • Emotional Contexts Tested: {quality['emotional_contexts_tested']}")
        
        # Overall Assessment
        print(f"\n🏆 Overall Assessment:")
        overall_score = self._calculate_overall_score()
        print(f"  • Overall Performance Score: {overall_score:.1f}/100")
        print(f"  • System Status: {self._get_system_status(overall_score)}")
        
        # Recommendations
        print(f"\n🎯 Recommendations for Improvement:")
        self._generate_improvement_recommendations()
        
        print("\n" + "=" * 80)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'performance_report_{timestamp}.json', 'w') as f:
            import json
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Detailed results saved to: performance_report_{timestamp}.json")
    
    def _calculate_overall_score(self):
        """Calculate overall performance score out of 100"""
        score = 0
        
        # Accuracy (30 points)
        if self.results['accuracy_metrics']['rmse'] < 1.0:
            score += 30
        elif self.results['accuracy_metrics']['rmse'] < 1.5:
            score += 20
        elif self.results['accuracy_metrics']['rmse'] < 2.0:
            score += 10
        
        # Speed (25 points)
        if self.results['speed_metrics']['avg_response_time'] < 0.5:
            score += 25
        elif self.results['speed_metrics']['avg_response_time'] < 1.0:
            score += 20
        elif self.results['speed_metrics']['avg_response_time'] < 2.0:
            score += 15
        elif self.results['speed_metrics']['avg_response_time'] < 5.0:
            score += 10
        
        # Business Value (25 points)
        diversity = self.results['business_metrics']['diversity_score']
        satisfaction = self.results['business_metrics']['avg_user_satisfaction']
        
        if diversity > 0.8 and satisfaction > 4.0:
            score += 25
        elif diversity > 0.6 and satisfaction > 3.5:
            score += 20
        elif diversity > 0.4 and satisfaction > 3.0:
            score += 15
        elif diversity > 0.2 and satisfaction > 2.5:
            score += 10
        
        # Quality (20 points)
        emotional_success = self.results['quality_metrics']['emotional_intelligence_success_rate']
        if emotional_success > 0.8:
            score += 20
        elif emotional_success > 0.6:
            score += 15
        elif emotional_success > 0.4:
            score += 10
        elif emotional_success > 0.2:
            score += 5
        
        return score
    
    def _get_system_status(self, score):
        """Get system status based on overall score"""
        if score >= 80:
            return "🟢 Excellent - Production Ready"
        elif score >= 60:
            return "🟡 Good - Minor Optimizations Needed"
        elif score >= 40:
            return "🟠 Fair - Significant Improvements Required"
        else:
            return "🔴 Poor - Major Issues Need Addressing"
    
    def _generate_improvement_recommendations(self):
        """Generate specific improvement recommendations"""
        recommendations = []
        
        # Check accuracy
        if self.results['accuracy_metrics']['rmse'] > 1.5:
            recommendations.append("• Improve model accuracy by tuning hyperparameters")
            recommendations.append("• Consider ensemble methods or advanced algorithms")
        
        # Check speed
        if self.results['speed_metrics']['avg_response_time'] > 2.0:
            recommendations.append("• Optimize recommendation algorithms for speed")
            recommendations.append("• Implement caching for frequent requests")
            recommendations.append("• Consider asynchronous processing")
        
        # Check coverage
        if self.results['accuracy_metrics']['coverage'] < 0.5:
            recommendations.append("• Improve item coverage by enhancing content-based filtering")
            recommendations.append("• Add more diverse recommendation strategies")
        
        # Check diversity
        if self.results['business_metrics']['diversity_score'] < 0.5:
            recommendations.append("• Increase recommendation diversity")
            recommendations.append("• Implement novelty and serendipity measures")
        
        # Check emotional intelligence
        if self.results['quality_metrics']['emotional_intelligence_success_rate'] < 0.7:
            recommendations.append("• Enhance emotional intelligence model training")
            recommendations.append("• Expand emotion-cuisine mapping database")
        
        if not recommendations:
            recommendations.append("• System is performing well! Consider advanced features like:")
            recommendations.append("  - Real-time learning from user feedback")
            recommendations.append("  - Advanced personalization algorithms")
            recommendations.append("  - Multi-objective optimization")
        
        for rec in recommendations:
            print(f"  {rec}")

def main():
    """Main performance evaluation function"""
    evaluator = PerformanceEvaluator()
    results = evaluator.evaluate_entire_system(data_size='large')
    return results

if __name__ == "__main__":
    main()
