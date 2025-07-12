#!/usr/bin/env python3
"""
Comprehensive Model Performance Evaluation
Calculates accuracy, precision, recall, RMSE, and other metrics for the recommendation system
"""

import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessor import DataPreprocessor
from models.hybrid_recommender import HybridRecommender
from models.collaborative_filtering import CollaborativeFiltering
from models.content_based_filtering import ContentBasedFiltering
from models.sentiment_analyzer import SentimentAnalyzer

class ModelPerformanceEvaluator:
    """Comprehensive performance evaluation for recommendation models"""
    
    def __init__(self):
        self.results = {}
        self.preprocessor = DataPreprocessor()
        
    def load_data(self, data_dir='data'):
        """Load the large-scale dataset"""
        print("📊 Loading dataset...")
        
        self.restaurants = pd.read_csv(f'{data_dir}/restaurants.csv')
        self.users = pd.read_csv(f'{data_dir}/users.csv')
        self.ratings = pd.read_csv(f'{data_dir}/ratings.csv')
        self.reviews = pd.read_csv(f'{data_dir}/reviews.csv')
        
        print(f"✅ Loaded:")
        print(f"   • {len(self.restaurants):,} restaurants")
        print(f"   • {len(self.users):,} users")
        print(f"   • {len(self.ratings):,} ratings")
        print(f"   • {len(self.reviews):,} reviews")
        
        return True
    
    def prepare_test_data(self, test_size=0.2):
        """Split data into train/test sets"""
        print(f"🔀 Splitting data (train: {1-test_size:.0%}, test: {test_size:.0%})...")
        
        # Split ratings data
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )
        
        # Create user-item matrices
        self.train_matrix = self.train_ratings.pivot_table(
            index='user_id', columns='restaurant_id', values='rating', fill_value=0
        )
        
        self.test_matrix = self.test_ratings.pivot_table(
            index='user_id', columns='restaurant_id', values='rating', fill_value=0
        )
        
        print(f"✅ Train set: {len(self.train_ratings):,} ratings")
        print(f"✅ Test set: {len(self.test_ratings):,} ratings")
        
        return True
    
    def evaluate_collaborative_filtering(self):
        """Evaluate collaborative filtering model"""
        print("\n🤖 Evaluating Collaborative Filtering...")
        
        start_time = time.time()
        
        # Train model
        cf_model = CollaborativeFiltering()
        cf_model.fit(self.train_matrix)
        
        train_time = time.time() - start_time
        
        # Make predictions
        predictions = []
        actuals = []
        
        start_time = time.time()
        
        for _, row in self.test_ratings.head(1000).iterrows():  # Sample for speed
            try:
                pred = cf_model.predict_rating(row['user_id'], row['restaurant_id'])
                if pred is not None and not np.isnan(pred) and not np.isinf(pred):
                    predictions.append(pred)
                    actuals.append(row['rating'])
            except Exception as e:
                # Skip failed predictions
                continue
        
        prediction_time = time.time() - start_time
        
        # Calculate metrics with safety checks
        if len(predictions) > 0 and len(actuals) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
        else:
            rmse = float('inf')
            mae = float('inf')
        
        # Binary classification metrics (good vs bad rating)
        actual_binary = [1 if x >= 4.0 else 0 for x in actuals]
        pred_binary = [1 if x >= 4.0 else 0 for x in predictions]
        
        if len(actual_binary) > 0:
            accuracy = accuracy_score(actual_binary, pred_binary)
            precision = precision_score(actual_binary, pred_binary, average='weighted', zero_division=0)
            recall = recall_score(actual_binary, pred_binary, average='weighted', zero_division=0)
            f1 = f1_score(actual_binary, pred_binary, average='weighted', zero_division=0)
        else:
            accuracy = precision = recall = f1 = 0.0
        
        self.results['collaborative_filtering'] = {
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'train_time': train_time,
            'prediction_time': prediction_time,
            'predictions_per_second': len(predictions) / prediction_time if prediction_time > 0 else 0
        }
        
        print(f"✅ Collaborative Filtering Results:")
        print(f"   • RMSE: {rmse:.3f}" if rmse != float('inf') else "   • RMSE: No valid predictions")
        print(f"   • MAE: {mae:.3f}" if mae != float('inf') else "   • MAE: No valid predictions")
        print(f"   • Accuracy: {accuracy:.3f}")
        print(f"   • F1 Score: {f1:.3f}")
        print(f"   • Train Time: {train_time:.2f}s")
        print(f"   • Predictions/sec: {len(predictions)/prediction_time:.1f}" if prediction_time > 0 else "   • Predictions/sec: 0")
        print(f"   • Valid predictions: {len(predictions)}/1000")
        
        return self.results['collaborative_filtering']
    
    def evaluate_content_based_filtering(self):
        """Evaluate content-based filtering model"""
        print("\n🏷️ Evaluating Content-Based Filtering...")
        
        start_time = time.time()
        
        # Prepare restaurant features
        from sklearn.preprocessing import LabelEncoder
        
        le_cuisine = LabelEncoder()
        le_city = LabelEncoder()
        le_price = LabelEncoder()
        
        restaurant_features = np.column_stack([
            le_cuisine.fit_transform(self.restaurants['cuisine']),
            le_city.fit_transform(self.restaurants['city']),
            le_price.fit_transform(self.restaurants['price_range']),
            self.restaurants['rating'].values
        ])
        
        # Train model
        cb_model = ContentBasedFiltering()
        cb_model.fit(self.restaurants, restaurant_features)
        
        train_time = time.time() - start_time
        
        # Test recommendation quality
        start_time = time.time()
        
        similarity_scores = []
        for restaurant_id in self.restaurants['restaurant_id'].head(100):  # Sample for speed
            recommendations = cb_model.get_restaurant_recommendations(restaurant_id, 5)
            if recommendations:
                avg_similarity = np.mean([rec['similarity_score'] for rec in recommendations])
                similarity_scores.append(avg_similarity)
        
        prediction_time = time.time() - start_time
        
        self.results['content_based_filtering'] = {
            'avg_similarity_score': np.mean(similarity_scores) if similarity_scores else 0,
            'recommendation_coverage': len(similarity_scores) / 100,
            'train_time': train_time,
            'prediction_time': prediction_time,
            'recommendations_per_second': len(similarity_scores) / prediction_time if prediction_time > 0 else 0
        }
        
        print(f"✅ Content-Based Filtering Results:")
        print(f"   • Avg Similarity Score: {np.mean(similarity_scores):.3f}")
        print(f"   • Recommendation Coverage: {len(similarity_scores)/100:.1%}")
        print(f"   • Train Time: {train_time:.2f}s")
        print(f"   • Recommendations/sec: {len(similarity_scores)/prediction_time:.1f}")
        
        return self.results['content_based_filtering']
    
    def evaluate_sentiment_analysis(self):
        """Evaluate sentiment analysis accuracy"""
        print("\n😊 Evaluating Sentiment Analysis...")
        
        start_time = time.time()
        
        analyzer = SentimentAnalyzer()
        
        # Test on sample of reviews
        sample_reviews = self.reviews.head(1000)
        
        predictions = []
        actuals = []
        
        for _, review in sample_reviews.iterrows():
            result = analyzer.analyze_sentiment(review['review_text'])
            predictions.append(result['sentiment'])
            actuals.append(review['sentiment'])
        
        processing_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
        f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
        
        self.results['sentiment_analysis'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'processing_time': processing_time,
            'reviews_per_second': len(sample_reviews) / processing_time
        }
        
        print(f"✅ Sentiment Analysis Results:")
        print(f"   • Accuracy: {accuracy:.3f}")
        print(f"   • Precision: {precision:.3f}")
        print(f"   • Recall: {recall:.3f}")
        print(f"   • F1 Score: {f1:.3f}")
        print(f"   • Reviews/sec: {len(sample_reviews)/processing_time:.1f}")
        
        return self.results['sentiment_analysis']
    
    def evaluate_hybrid_model(self):
        """Evaluate the complete hybrid recommendation system"""
        print("\n🔀 Evaluating Hybrid Recommendation System...")
        
        start_time = time.time()
        
        # Prepare data for hybrid model
        data = {
            'restaurants': self.restaurants,
            'users': self.users,
            'ratings': self.train_ratings,
            'reviews': self.reviews,
            'user_item_matrix': self.train_matrix,
            'restaurant_features': self._prepare_restaurant_features(),
            'label_encoders': {}
        }
        
        # Train hybrid model
        hybrid_model = HybridRecommender()
        hybrid_model.fit(data)
        
        train_time = time.time() - start_time
        
        # Test recommendations
        start_time = time.time()
        
        recommendation_quality = []
        coverage_users = []
        
        for user_id in self.users['user_id'].head(50):  # Sample for speed
            try:
                recommendations = hybrid_model.get_hybrid_recommendations(user_id, 10)
                if recommendations:
                    # Calculate quality metrics
                    avg_score = np.mean([rec.get('score', 0) for rec in recommendations])
                    recommendation_quality.append(avg_score)
                    coverage_users.append(user_id)
            except Exception as e:
                print(f"   Warning: Failed to get recommendations for user {user_id}: {e}")
        
        prediction_time = time.time() - start_time
        
        self.results['hybrid_model'] = {
            'avg_recommendation_score': np.mean(recommendation_quality) if recommendation_quality else 0,
            'user_coverage': len(coverage_users) / 50,
            'train_time': train_time,
            'prediction_time': prediction_time,
            'recommendations_per_second': len(coverage_users) / prediction_time if prediction_time > 0 else 0
        }
        
        print(f"✅ Hybrid Model Results:")
        print(f"   • Avg Recommendation Score: {np.mean(recommendation_quality):.3f}" if recommendation_quality else "   • Avg Recommendation Score: 0.000")
        print(f"   • User Coverage: {len(coverage_users)/50:.1%}")
        print(f"   • Train Time: {train_time:.2f}s")
        print(f"   • Recommendations/sec: {len(coverage_users)/prediction_time:.1f}" if prediction_time > 0 else "   • Recommendations/sec: 0")
        print(f"   • Successful users: {len(coverage_users)}/50")
        
        return self.results['hybrid_model']
    
    def _prepare_restaurant_features(self):
        """Prepare restaurant features for models"""
        from sklearn.preprocessing import LabelEncoder
        
        le_cuisine = LabelEncoder()
        le_city = LabelEncoder()
        le_price = LabelEncoder()
        
        features = np.column_stack([
            le_cuisine.fit_transform(self.restaurants['cuisine']),
            le_city.fit_transform(self.restaurants['city']),
            le_price.fit_transform(self.restaurants['price_range']),
            self.restaurants['rating'].values
        ])
        
        return features
    
    def generate_performance_report(self, save_dir='reports'):
        """Generate comprehensive performance report"""
        print("\n📊 Generating Performance Report...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create summary
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'dataset_size': {
                'restaurants': len(self.restaurants),
                'users': len(self.users),
                'ratings': len(self.ratings),
                'reviews': len(self.reviews)
            },
            'model_performance': self.results,
            'overall_scores': self._calculate_overall_scores()
        }
        
        # Save JSON report
        with open(f'{save_dir}/performance_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create visualization
        self._create_performance_charts(save_dir)
        
        # Print summary
        print("=" * 60)
        print("📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for model, metrics in self.results.items():
            print(f"\n🔸 {model.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   • {metric.replace('_', ' ').title()}: {value:.3f}")
                else:
                    print(f"   • {metric.replace('_', ' ').title()}: {value}")
        
        overall = self._calculate_overall_scores()
        print(f"\n🏆 OVERALL SYSTEM SCORE: {overall['overall_score']:.3f}/1.000")
        print(f"🎯 RECOMMENDATION QUALITY: {overall['recommendation_quality']:.3f}/1.000")
        print(f"⚡ SYSTEM PERFORMANCE: {overall['system_performance']:.3f}/1.000")
        
        print(f"\n📁 Report saved to: {save_dir}/performance_report.json")
        
        return summary
    
    def _calculate_overall_scores(self):
        """Calculate overall system performance scores"""
        scores = {}
        
        # Recommendation Quality Score
        quality_components = []
        if 'collaborative_filtering' in self.results:
            cf_score = 1 - (self.results['collaborative_filtering']['rmse'] / 5)  # Normalize RMSE
            quality_components.append(cf_score)
        
        if 'sentiment_analysis' in self.results:
            quality_components.append(self.results['sentiment_analysis']['accuracy'])
        
        if 'hybrid_model' in self.results:
            quality_components.append(self.results['hybrid_model']['user_coverage'])
        
        scores['recommendation_quality'] = np.mean(quality_components) if quality_components else 0
        
        # System Performance Score
        performance_components = []
        for model_results in self.results.values():
            if 'predictions_per_second' in model_results:
                # Normalize prediction speed (higher is better)
                speed_score = min(1.0, model_results['predictions_per_second'] / 100)
                performance_components.append(speed_score)
        
        scores['system_performance'] = np.mean(performance_components) if performance_components else 0
        
        # Overall Score
        scores['overall_score'] = (scores['recommendation_quality'] + scores['system_performance']) / 2
        
        return scores
    
    def _create_performance_charts(self, save_dir):
        """Create performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('AI Recommendation System Performance Report', fontsize=16, fontweight='bold')
            
            # 1. Model Accuracy Comparison
            if 'collaborative_filtering' in self.results and 'sentiment_analysis' in self.results:
                models = ['Collaborative Filtering', 'Sentiment Analysis']
                accuracies = [
                    self.results['collaborative_filtering']['accuracy'],
                    self.results['sentiment_analysis']['accuracy']
                ]
                
                axes[0, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e'])
                axes[0, 0].set_title('Model Accuracy Comparison')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].set_ylim(0, 1)
                
                for i, v in enumerate(accuracies):
                    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
            # 2. Processing Speed
            models = []
            speeds = []
            for model, results in self.results.items():
                if 'predictions_per_second' in results or 'reviews_per_second' in results:
                    models.append(model.replace('_', ' ').title())
                    speed = results.get('predictions_per_second', results.get('reviews_per_second', 0))
                    speeds.append(speed)
            
            if models:
                axes[0, 1].bar(models, speeds, color='#2ca02c')
                axes[0, 1].set_title('Processing Speed (Items/Second)')
                axes[0, 1].set_ylabel('Items per Second')
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 3. Training Time Comparison
            models = []
            times = []
            for model, results in self.results.items():
                if 'train_time' in results:
                    models.append(model.replace('_', ' ').title())
                    times.append(results['train_time'])
            
            if models:
                axes[1, 0].bar(models, times, color='#d62728')
                axes[1, 0].set_title('Model Training Time')
                axes[1, 0].set_ylabel('Time (seconds)')
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 4. Overall Scores
            overall = self._calculate_overall_scores()
            score_names = ['Recommendation Quality', 'System Performance', 'Overall Score']
            score_values = [
                overall['recommendation_quality'],
                overall['system_performance'],
                overall['overall_score']
            ]
            
            colors = ['#9467bd', '#8c564b', '#e377c2']
            axes[1, 1].bar(score_names, score_values, color=colors)
            axes[1, 1].set_title('Overall System Scores')
            axes[1, 1].set_ylabel('Score (0-1)')
            axes[1, 1].set_ylim(0, 1)
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            for i, v in enumerate(score_values):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/performance_charts.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance charts saved to: {save_dir}/performance_charts.png")
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping chart generation")
    
    def run_full_evaluation(self):
        """Run complete model evaluation"""
        print("🚀 Starting Comprehensive Model Performance Evaluation")
        print("=" * 70)
        
        start_total = time.time()
        
        # Load and prepare data
        self.load_data()
        self.prepare_test_data()
        
        # Evaluate individual models
        self.evaluate_collaborative_filtering()
        self.evaluate_content_based_filtering()
        self.evaluate_sentiment_analysis()
        self.evaluate_hybrid_model()
        
        # Generate report
        report = self.generate_performance_report()
        
        total_time = time.time() - start_total
        
        print(f"\n⏱️ Total Evaluation Time: {total_time:.2f} seconds")
        print("🎉 Evaluation Complete!")
        
        return report

def main():
    """Main evaluation function"""
    evaluator = ModelPerformanceEvaluator()
    report = evaluator.run_full_evaluation()
    return report

if __name__ == "__main__":
    main()
