"""
Comprehensive test suite for the Emotional Intelligence Engine

This module contains unit tests, integration tests, and performance tests
for the emotional state-based recommendation system.
"""

import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import the modules to test
from models.emotional_intelligence import (
    EmotionalState, ContextualFactors, VADEREmotionDetector,
    TransformerEmotionDetector, ContextAnalyzer, EmotionalStateManager,
    EmotionalIntelligenceEngine
)
from config import Config

class TestEmotionalState(unittest.TestCase):
    """Test the EmotionalState data class"""
    
    def setUp(self):
        self.sample_state = EmotionalState(
            primary_emotion='happy',
            secondary_emotion='excited',
            intensity=0.8,
            confidence=0.9,
            context={'weather': 'sunny', 'time_of_day': 'morning'},
            timestamp=datetime.now()
        )
    
    def test_emotional_state_creation(self):
        """Test creating an EmotionalState instance"""
        self.assertEqual(self.sample_state.primary_emotion, 'happy')
        self.assertEqual(self.sample_state.secondary_emotion, 'excited')
        self.assertEqual(self.sample_state.intensity, 0.8)
        self.assertEqual(self.sample_state.confidence, 0.9)
        self.assertIsInstance(self.sample_state.timestamp, datetime)
    
    def test_to_dict_conversion(self):
        """Test converting EmotionalState to dictionary"""
        state_dict = self.sample_state.to_dict()
        
        self.assertIsInstance(state_dict, dict)
        self.assertEqual(state_dict['primary_emotion'], 'happy')
        self.assertIsInstance(state_dict['timestamp'], str)
    
    def test_from_dict_conversion(self):
        """Test creating EmotionalState from dictionary"""
        state_dict = self.sample_state.to_dict()
        reconstructed_state = EmotionalState.from_dict(state_dict)
        
        self.assertEqual(reconstructed_state.primary_emotion, self.sample_state.primary_emotion)
        self.assertEqual(reconstructed_state.intensity, self.sample_state.intensity)
        self.assertIsInstance(reconstructed_state.timestamp, datetime)

class TestContextualFactors(unittest.TestCase):
    """Test the ContextualFactors data class"""
    
    def test_contextual_factors_creation(self):
        """Test creating ContextualFactors instance"""
        context = ContextualFactors(
            weather='sunny',
            temperature=22.5,
            time_of_day='morning',
            day_of_week='monday',
            calendar_stress=0.6
        )
        
        self.assertEqual(context.weather, 'sunny')
        self.assertEqual(context.temperature, 22.5)
        self.assertEqual(context.calendar_stress, 0.6)
    
    def test_to_dict_conversion(self):
        """Test converting ContextualFactors to dictionary"""
        context = ContextualFactors(weather='rainy', temperature=15.0)
        context_dict = context.to_dict()
        
        self.assertIsInstance(context_dict, dict)
        self.assertEqual(context_dict['weather'], 'rainy')
        self.assertEqual(context_dict['temperature'], 15.0)

class TestVADEREmotionDetector(unittest.TestCase):
    """Test the VADER-based emotion detector"""
    
    def setUp(self):
        self.detector = VADEREmotionDetector()
    
    def test_positive_emotion_detection(self):
        """Test detection of positive emotions"""
        text = "I'm feeling absolutely amazing today! Everything is wonderful!"
        emotion = self.detector.detect_emotion(text)
        
        self.assertIn(emotion.primary_emotion, ['happy', 'excited', 'energetic'])
        self.assertGreater(emotion.intensity, 0.5)
        self.assertIsInstance(emotion.timestamp, datetime)
    
    def test_negative_emotion_detection(self):
        """Test detection of negative emotions"""
        text = "I'm really stressed and anxious about work. Nothing is going right."
        emotion = self.detector.detect_emotion(text)
        
        self.assertIn(emotion.primary_emotion, ['stressed', 'anxious', 'sad'])
        self.assertGreater(emotion.intensity, 0.3)
    
    def test_neutral_emotion_detection(self):
        """Test detection of neutral emotions"""
        text = "The weather is okay today."
        emotion = self.detector.detect_emotion(text)
        
        self.assertIn(emotion.primary_emotion, ['neutral', 'calm'])
    
    def test_empty_text_handling(self):
        """Test handling of empty text input"""
        emotion = self.detector.detect_emotion("")
        
        self.assertEqual(emotion.primary_emotion, 'neutral')
        self.assertLess(emotion.confidence, 0.5)
    
    def test_context_integration(self):
        """Test emotion detection with contextual factors"""
        context = ContextualFactors(weather='rainy', time_of_day='evening')
        text = "Feeling okay"
        emotion = self.detector.detect_emotion(text, context)
        
        self.assertIsNotNone(emotion.context)
        self.assertEqual(emotion.context['weather'], 'rainy')

class TestTransformerEmotionDetector(unittest.TestCase):
    """Test the transformer-based emotion detector"""
    
    def setUp(self):
        # Mock the transformer to avoid loading heavy models in tests
        with patch('models.emotional_intelligence.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'HAPPY', 'score': 0.8},
                {'label': 'EXCITED', 'score': 0.2}
            ]
            mock_pipeline.return_value = mock_classifier
            self.detector = TransformerEmotionDetector()
    
    def test_transformer_emotion_detection(self):
        """Test transformer-based emotion detection"""
        with patch.object(self.detector, 'classifier') as mock_classifier:
            mock_classifier.return_value = [
                {'label': 'JOY', 'score': 0.85},
                {'label': 'OPTIMISM', 'score': 0.15}
            ]
            
            text = "I'm really happy today!"
            emotion = self.detector.detect_emotion(text)
            
            self.assertEqual(emotion.primary_emotion, 'joy')
            self.assertEqual(emotion.confidence, 0.85)
    
    def test_fallback_on_error(self):
        """Test fallback to VADER when transformer fails"""
        with patch.object(self.detector, 'classifier', side_effect=Exception("Model error")):
            text = "I'm feeling great!"
            emotion = self.detector.detect_emotion(text)
            
            # Should still return a valid EmotionalState
            self.assertIsInstance(emotion, EmotionalState)
            self.assertIsNotNone(emotion.primary_emotion)

class TestContextAnalyzer(unittest.TestCase):
    """Test the context analyzer"""
    
    def setUp(self):
        self.analyzer = ContextAnalyzer()
    
    def test_time_of_day_classification(self):
        """Test time of day classification"""
        morning_time = datetime(2025, 1, 1, 8, 0)
        afternoon_time = datetime(2025, 1, 1, 14, 0)
        evening_time = datetime(2025, 1, 1, 19, 0)
        night_time = datetime(2025, 1, 1, 23, 0)
        
        self.assertEqual(self.analyzer._get_time_of_day(morning_time), 'morning')
        self.assertEqual(self.analyzer._get_time_of_day(afternoon_time), 'afternoon')
        self.assertEqual(self.analyzer._get_time_of_day(evening_time), 'evening')
        self.assertEqual(self.analyzer._get_time_of_day(night_time), 'night')
    
    def test_calendar_stress_estimation(self):
        """Test calendar stress estimation"""
        # Weekday work hours should have higher stress
        weekday_work = datetime(2025, 1, 6, 14, 0)  # Monday 2 PM
        weekend_evening = datetime(2025, 1, 4, 19, 0)  # Saturday 7 PM
        
        weekday_stress = self.analyzer._estimate_calendar_stress(weekday_work)
        weekend_stress = self.analyzer._estimate_calendar_stress(weekend_evening)
        
        self.assertGreater(weekday_stress, weekend_stress)
        self.assertGreater(weekday_stress, 0.5)
    
    @patch('requests.get')
    def test_weather_data_fetch_success(self, mock_get):
        """Test successful weather data fetching"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'weather': [{'description': 'sunny'}],
            'main': {'temp': 22.5}
        }
        mock_get.return_value = mock_response
        
        # Set API key for test
        self.analyzer.weather_api_key = 'test_key'
        
        weather_data = self.analyzer._get_weather_data('New York')
        
        self.assertEqual(weather_data['description'], 'sunny')
        self.assertEqual(weather_data['temperature'], 22.5)
    
    @patch('requests.get')
    def test_weather_data_fetch_failure(self, mock_get):
        """Test handling of weather data fetch failure"""
        mock_get.side_effect = Exception("Network error")
        
        self.analyzer.weather_api_key = 'test_key'
        weather_data = self.analyzer._get_weather_data('New York')
        
        self.assertIsNone(weather_data)
    
    def test_analyze_context(self):
        """Test complete context analysis"""
        context = self.analyzer.analyze_context()
        
        self.assertIsInstance(context, ContextualFactors)
        self.assertIsNotNone(context.time_of_day)
        self.assertIsNotNone(context.day_of_week)
        self.assertIsInstance(context.calendar_stress, float)

class TestEmotionalStateManager(unittest.TestCase):
    """Test the emotional state manager"""
    
    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = Config.EMOTIONAL_DATA_DIR
        Config.EMOTIONAL_DATA_DIR = self.test_dir
        
        # Mock the detector to avoid heavy model loading
        with patch('models.emotional_intelligence.TransformerEmotionDetector'):
            with patch('models.emotional_intelligence.VADEREmotionDetector') as mock_vader:
                mock_detector = Mock()
                mock_detector.detect_emotion.return_value = EmotionalState(
                    primary_emotion='happy',
                    secondary_emotion=None,
                    intensity=0.7,
                    confidence=0.8,
                    context={},
                    timestamp=datetime.now()
                )
                mock_vader.return_value = mock_detector
                self.manager = EmotionalStateManager()
                self.manager.detector = mock_detector
    
    def tearDown(self):
        # Cleanup temporary directory
        shutil.rmtree(self.test_dir)
        Config.EMOTIONAL_DATA_DIR = self.original_data_dir
    
    def test_detect_current_emotion(self):
        """Test current emotion detection"""
        emotion = self.manager.detect_current_emotion("user123", "I'm feeling great!")
        
        self.assertIsInstance(emotion, EmotionalState)
        self.assertEqual(emotion.primary_emotion, 'happy')
    
    def test_emotional_state_caching(self):
        """Test emotional state caching"""
        user_id = "user123"
        text = "I'm happy"
        
        # First call
        emotion1 = self.manager.detect_current_emotion(user_id, text)
        
        # Second call with same input should use cache
        emotion2 = self.manager.detect_current_emotion(user_id, text)
        
        self.assertEqual(emotion1.primary_emotion, emotion2.primary_emotion)
    
    def test_emotional_history_storage(self):
        """Test storing and retrieving emotional history"""
        user_id = "user456"
        
        # Generate some emotional states
        self.manager.detect_current_emotion(user_id, "Happy text")
        self.manager.detect_current_emotion(user_id, "Sad text")
        
        # Retrieve history
        history = self.manager.get_emotional_history(user_id)
        
        self.assertGreater(len(history), 0)
        self.assertIsInstance(history[0], EmotionalState)
    
    def test_emotional_patterns_analysis(self):
        """Test emotional patterns analysis"""
        user_id = "user789"
        
        # Create fake history file
        history_data = []
        for i in range(10):
            state = EmotionalState(
                primary_emotion='happy' if i % 2 == 0 else 'sad',
                secondary_emotion=None,
                intensity=0.5,
                confidence=0.7,
                context={},
                timestamp=datetime.now() - timedelta(days=i)
            )
            history_data.append(state.to_dict())
        
        history_file = os.path.join(self.test_dir, f"user_{user_id}_history.json")
        with open(history_file, 'w') as f:
            json.dump(history_data, f)
        
        patterns = self.manager.get_emotional_patterns(user_id)
        
        self.assertIn('happy', patterns)
        self.assertIn('sad', patterns)
        self.assertEqual(patterns['happy'], 0.5)  # 50% of the time

class TestEmotionalIntelligenceEngine(unittest.TestCase):
    """Test the main emotional intelligence engine"""
    
    def setUp(self):
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = Config.EMOTIONAL_DATA_DIR
        Config.EMOTIONAL_DATA_DIR = self.test_dir
        
        # Mock the state manager
        with patch('models.emotional_intelligence.EmotionalStateManager') as mock_manager:
            mock_state_manager = Mock()
            mock_state_manager.detect_current_emotion.return_value = EmotionalState(
                primary_emotion='happy',
                secondary_emotion=None,
                intensity=0.8,
                confidence=0.9,
                context={},
                timestamp=datetime.now()
            )
            mock_manager.return_value = mock_state_manager
            
            self.engine = EmotionalIntelligenceEngine()
            self.engine.state_manager = mock_state_manager
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        Config.EMOTIONAL_DATA_DIR = self.original_data_dir
    
    def test_emotion_cuisine_mapping_loading(self):
        """Test loading of emotion-cuisine mappings"""
        mapping = self.engine.emotion_cuisine_mapping
        
        self.assertIsInstance(mapping, dict)
        self.assertIn('happy', mapping)
        self.assertIn('sad', mapping)
        self.assertIsInstance(mapping['happy'], dict)
    
    def test_emotional_restaurant_scores(self):
        """Test calculation of emotional restaurant scores"""
        # Create sample restaurant data
        restaurants = pd.DataFrame({
            'restaurant_id': [1, 2, 3],
            'name': ['Happy Pizza', 'Comfort Diner', 'Fusion Cafe'],
            'cuisine': ['italian', 'american', 'fusion']
        })
        
        scores = self.engine.get_emotional_restaurant_scores(
            "user123", restaurants, "I'm feeling great!"
        )
        
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        
        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_emotional_explanation_generation(self):
        """Test generation of emotional explanations"""
        emotional_state = EmotionalState(
            primary_emotion='romantic',
            secondary_emotion=None,
            intensity=0.9,
            confidence=0.8,
            context={},
            timestamp=datetime.now()
        )
        
        restaurant_data = {
            'name': 'French Bistro',
            'cuisine': 'french'
        }
        
        explanation = self.engine.get_emotional_explanation(emotional_state, restaurant_data)
        
        self.assertIsInstance(explanation, str)
        self.assertIn('romantic', explanation.lower())
        self.assertIn('French Bistro', explanation)
    
    def test_restaurant_mood_score_calculation(self):
        """Test restaurant mood score calculation"""
        restaurant_data = pd.Series({
            'name': 'Test Restaurant',
            'cuisine': 'italian',
            'atmosphere': 'cozy'
        })
        
        emotional_state = EmotionalState(
            primary_emotion='calm',
            secondary_emotion=None,
            intensity=0.6,
            confidence=0.7,
            context={},
            timestamp=datetime.now()
        )
        
        mood_score = self.engine._calculate_restaurant_mood_score(restaurant_data, emotional_state)
        
        self.assertIsInstance(mood_score, float)
        self.assertGreaterEqual(mood_score, 0.0)
        self.assertLessEqual(mood_score, 1.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete emotional intelligence system"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = Config.EMOTIONAL_DATA_DIR
        Config.EMOTIONAL_DATA_DIR = self.test_dir
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        Config.EMOTIONAL_DATA_DIR = self.original_data_dir
    
    @patch('models.emotional_intelligence.TransformerEmotionDetector')
    def test_end_to_end_emotion_detection(self, mock_transformer):
        """Test complete end-to-end emotion detection and scoring"""
        # Setup mocks
        mock_detector = Mock()
        mock_detector.detect_emotion.return_value = EmotionalState(
            primary_emotion='excited',
            secondary_emotion='happy',
            intensity=0.85,
            confidence=0.9,
            context={'time_of_day': 'evening'},
            timestamp=datetime.now()
        )
        mock_transformer.return_value = mock_detector
        
        # Create engine
        engine = EmotionalIntelligenceEngine()
        engine.state_manager.detector = mock_detector
        
        # Create test restaurant data
        restaurants = pd.DataFrame({
            'restaurant_id': [1, 2, 3, 4],
            'name': ['Mexican Fiesta', 'Quiet Cafe', 'Italian Comfort', 'Fusion Adventure'],
            'cuisine': ['mexican', 'cafe', 'italian', 'fusion']
        })
        
        # Get emotional scores
        scores = engine.get_emotional_restaurant_scores(
            "test_user", restaurants, "I'm so excited for dinner tonight!"
        )
        
        # Verify results
        self.assertEqual(len(scores), 4)
        
        # Mexican should score high for excited emotion
        mexican_score = scores['1']  # Mexican Fiesta
        self.assertGreater(mexican_score, 0.5)
    
    def test_emotional_history_persistence(self):
        """Test that emotional history persists across sessions"""
        user_id = "persistence_test_user"
        
        # Create first manager instance
        with patch('models.emotional_intelligence.TransformerEmotionDetector'):
            manager1 = EmotionalStateManager()
            manager1.detector = Mock()
            manager1.detector.detect_emotion.return_value = EmotionalState(
                primary_emotion='happy',
                secondary_emotion=None,
                intensity=0.7,
                confidence=0.8,
                context={},
                timestamp=datetime.now()
            )
            
            # Detect emotion
            manager1.detect_current_emotion(user_id, "I'm happy!")
        
        # Create second manager instance (simulating new session)
        with patch('models.emotional_intelligence.TransformerEmotionDetector'):
            manager2 = EmotionalStateManager()
            
            # Retrieve history
            history = manager2.get_emotional_history(user_id)
            
            self.assertGreater(len(history), 0)
            self.assertEqual(history[0].primary_emotion, 'happy')

class TestPerformance(unittest.TestCase):
    """Performance tests for emotional intelligence system"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_data_dir = Config.EMOTIONAL_DATA_DIR
        Config.EMOTIONAL_DATA_DIR = self.test_dir
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        Config.EMOTIONAL_DATA_DIR = self.original_data_dir
    
    @patch('models.emotional_intelligence.TransformerEmotionDetector')
    def test_bulk_emotion_detection_performance(self, mock_transformer):
        """Test performance with multiple emotion detection requests"""
        import time
        
        # Setup mock
        mock_detector = Mock()
        mock_detector.detect_emotion.return_value = EmotionalState(
            primary_emotion='neutral',
            secondary_emotion=None,
            intensity=0.5,
            confidence=0.6,
            context={},
            timestamp=datetime.now()
        )
        mock_transformer.return_value = mock_detector
        
        manager = EmotionalStateManager()
        manager.detector = mock_detector
        
        # Test with multiple requests
        start_time = time.time()
        
        for i in range(100):
            manager.detect_current_emotion(f"user_{i}", f"test text {i}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process 100 requests in reasonable time (< 5 seconds)
        self.assertLess(total_time, 5.0)
        
        # Test caching effectiveness
        start_time = time.time()
        
        # Same requests should be faster due to caching
        for i in range(100):
            manager.detect_current_emotion(f"user_{i}", f"test text {i}")
        
        end_time = time.time()
        cached_time = end_time - start_time
        
        # Cached requests should be significantly faster
        self.assertLess(cached_time, total_time * 0.5)
    
    def test_large_restaurant_dataset_scoring(self):
        """Test performance with large restaurant datasets"""
        # Create large restaurant dataset
        large_restaurants = pd.DataFrame({
            'restaurant_id': range(1000),
            'name': [f'Restaurant {i}' for i in range(1000)],
            'cuisine': ['italian', 'mexican', 'chinese', 'american', 'japanese'] * 200
        })
        
        with patch('models.emotional_intelligence.EmotionalStateManager') as mock_manager:
            mock_state_manager = Mock()
            mock_state_manager.detect_current_emotion.return_value = EmotionalState(
                primary_emotion='happy',
                secondary_emotion=None,
                intensity=0.8,
                confidence=0.9,
                context={},
                timestamp=datetime.now()
            )
            mock_manager.return_value = mock_state_manager
            
            engine = EmotionalIntelligenceEngine()
            engine.state_manager = mock_state_manager
            
            import time
            start_time = time.time()
            
            scores = engine.get_emotional_restaurant_scores(
                "test_user", large_restaurants, "I'm happy!"
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process 1000 restaurants in reasonable time (< 2 seconds)
            self.assertLess(processing_time, 2.0)
            self.assertEqual(len(scores), 1000)

if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEmotionalState,
        TestContextualFactors,
        TestVADEREmotionDetector,
        TestTransformerEmotionDetector,
        TestContextAnalyzer,
        TestEmotionalStateManager,
        TestEmotionalIntelligenceEngine,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    print(f"{'='*50}")
