#!/usr/bin/env python3
"""
Minimal test to check what's causing the hang
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MinimalTest(unittest.TestCase):
    """Minimal test case"""
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        self.assertEqual(1 + 1, 2)
        print("Basic test passed")
    
    def test_imports(self):
        """Test critical imports"""
        import pandas as pd
        import numpy as np
        from tests.test_factory import TestDataFactory
        
        factory = TestDataFactory()
        users = factory.create_users(5)
        self.assertEqual(len(users), 5)
        print("Import test passed")
    
    def test_config_import(self):
        """Test config import"""
        from config import Config
        self.assertTrue(hasattr(Config, 'USE_LLM_ENHANCEMENT'))
        print("Config test passed")

if __name__ == '__main__':
    # Run with minimal verbosity
    unittest.main(argv=[''], exit=False, verbosity=2)
