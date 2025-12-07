"""
Unit tests for Email Network Analyzer
"""

import sys
import unittest
import pandas as pd
import numpy as np
import os

# Add src to path
sys.path.append('src')

from src.analyzer import EmailNetworkAnalyzer
from src.data_loader import generate_sample_data

class TestEmailNetworkAnalyzer(unittest.TestCase):
    """Test suite for EmailNetworkAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = EmailNetworkAnalyzer()
        
        # Create sample data
        self.sample_df = pd.DataFrame([
            {'sender': 'alice@company.com', 'recipients': 'bob@company.com, charlie@company.com', 'timestamp': '2023-01-01 10:00:00'},
            {'sender': 'bob@company.com', 'recipients': 'charlie@company.com', 'timestamp': '2023-01-01 11:00:00'},
            {'sender': 'charlie@company.com', 'recipients': 'alice@company.com, dave@company.com', 'timestamp': '2023-01-02 09:00:00'},
            {'sender': 'dave@company.com', 'recipients': 'alice@company.com', 'timestamp': '2023-01-02 14:00:00'},
            {'sender': 'eve@company.com', 'recipients': 'bob@company.com, dave@company.com', 'timestamp': '2023-01-03 12:00:00'}
        ])
        
        self.analyzer.load_emails(self.sample_df)
    
    def test_load_emails(self):
        """Test email loading functionality."""
        self.assertEqual(self.analyzer.graph.number_of_nodes(), 5)
        self.assertEqual(self.analyzer.graph.number_of_edges(), 7)
        self.assertIn('alice@company.com', self.analyzer.graph.nodes())
        self.assertTrue(self.analyzer.graph.has_edge('alice@company.com', 'bob@company.com'))
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = self.analyzer.calculate_metrics()
        
        self.assertIn('degree_centrality', metrics)
        self.assertIn('betweenness', metrics)
        self.assertIn('closeness', metrics)
        self.assertIn('clustering', metrics)
        self.assertIn('density', metrics)
        
        # Check that metrics have expected length
        self.assertEqual(len(metrics['degree_centrality']), 5)
        self.assertGreaterEqual(metrics['density'], 0)
    
    def test_find_key_connectors(self):
        """Test key connector identification."""
        connectors = self.analyzer.find_key_connectors(top_n=3)
        
        self.assertEqual(len(connectors), 3)
        self.assertIsInstance(connectors[0], tuple)
        self.assertEqual(len(connectors[0]), 2)  # (node, score)
        
        # Check that scores are in descending order
        scores = [score for _, score in connectors]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_predict_connections(self):
        """Test connection prediction."""
        predictions = self.analyzer.predict_connections('alice@company.com', top_n=3)
        
        # Should return list of tuples
        self.assertIsInstance(predictions, list)
        for pred in predictions:
            self.assertIsInstance(pred, tuple)
            self.assertEqual(len(pred), 2)  # (node, score)
    
    def test_get_reconnection_time(self):
        """Test reconnection time calculation."""
        # Test existing path
        time = self.analyzer.get_reconnection_time('eve@company.com', 'charlie@company.com')
        self.assertIsInstance(time, (int, float))
        
        # Test self connection (should be 0)
        self_time = self.analyzer.get_reconnection_time('alice@company.com', 'alice@company.com')
        self.assertEqual(self_time, 0)
    
    def test_get_network_summary(self):
        """Test network summary generation."""
        summary = self.analyzer.get_network_summary()
        
        expected_keys = [
            'total_nodes', 'total_edges', 'density', 'connected_components',
            'avg_clustering', 'avg_degree_centrality', 'avg_betweenness',
            'avg_closeness', 'most_connected', 'most_influential'
        ]
        
        for key in expected_keys:
            self.assertIn(key, summary)
    
    def test_model_save_load(self):
        """Test model persistence."""
        # Save model
        temp_path = 'temp_test_model.pkl'
        self.analyzer.save_model(temp_path)
        
        # Load model
        loaded_analyzer = EmailNetworkAnalyzer.load_model(temp_path)
        
        # Compare metrics
        original_metrics = self.analyzer.calculate_metrics()
        loaded_metrics = loaded_analyzer.calculate_metrics()
        
        self.assertEqual(original_metrics['density'], loaded_metrics['density'])
        self.assertEqual(self.analyzer.graph.number_of_nodes(), loaded_analyzer.graph.number_of_nodes())
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid node for prediction
        with self.assertRaises(ValueError):
            self.analyzer.predict_connections('nonexistent@company.com')
        
        # Test invalid nodes for reconnection time
        with self.assertRaises(ValueError):
            self.analyzer.get_reconnection_time('nonexistent1@company.com', 'nonexistent2@company.com')

class TestDataLoader(unittest.TestCase):
    """Test suite for data loader functions."""
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        df = generate_sample_data(100)
        
        self.assertEqual(len(df), 100)
        self.assertIn('sender', df.columns)
        self.assertIn('recipients', df.columns)
        self.assertIn('timestamp', df.columns)
        
        # Check that all emails have @ symbol
        for email in df['sender']:
            self.assertIn('@', email)
        
        for recipients in df['recipients']:
            for recipient in recipients.split(','):
                self.assertIn('@', recipient.strip())

if __name__ == '__main__':
    unittest.main()
