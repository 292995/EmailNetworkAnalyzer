"""
Email Network Analyzer Module

This module provides comprehensive analysis of email communication networks,
including centrality calculations, connection prediction, and network metrics.
"""

import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import pickle
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailNetworkAnalyzer:
    """
    Analyzes email communication networks to identify key connectors,
    predict missing connections, and optimize reconnection times.
    """
    
    def __init__(self):
        """Initialize the analyzer with an empty directed graph."""
        self.graph = nx.DiGraph()
        self.communication_log = []
        self.metrics_cache = {}
        logger.info("EmailNetworkAnalyzer initialized")
    
    def load_emails(self, df):
        """
        Load email data from DataFrame with columns: sender, recipients, timestamp
        
        Args:
            df (pd.DataFrame): DataFrame with columns 'sender', 'recipients', 'timestamp'
        """
        try:
            logger.info(f"Loading {len(df)} email records...")
            
            for idx, row in df.iterrows():
                sender = str(row['sender']).lower().strip()
                recipients = row['recipients']
                
                # Handle different recipient formats
                if isinstance(recipients, str):
                    if ',' in recipients:
                        recipient_list = [r.strip().lower() for r in recipients.split(',')]
                    else:
                        recipient_list = [recipients.strip().lower()]
                elif isinstance(recipients, list):
                    recipient_list = [str(r).lower().strip() for r in recipients]
                else:
                    recipient_list = [str(recipients).lower().strip()]
                
                # Add nodes and edges
                for recipient in recipient_list:
                    # Add nodes if they don't exist
                    self.graph.add_node(sender)
                    self.graph.add_node(recipient)
                    
                    # Add edge with weight (communication count) and timestamp
                    if self.graph.has_edge(sender, recipient):
                        self.graph[sender][recipient]['weight'] += 1
                        self.graph[sender][recipient]['timestamps'].append(row['timestamp'])
                    else:
                        self.graph.add_edge(
                            sender, 
                            recipient, 
                            weight=1,
                            timestamps=[row['timestamp']]
                        )
                
                # Log communication
                self.communication_log.append({
                    'sender': sender,
                    'recipients': recipient_list,
                    'timestamp': row['timestamp']
                })
            
            logger.info(f"Network loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error loading emails: {e}")
            raise
    
    def calculate_metrics(self, force_recalculation=False):
        """
        Calculate comprehensive network metrics.
        
        Args:
            force_recalculation (bool): Force recalculation even if cached
        
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        try:
            if not force_recalculation and self.metrics_cache:
                logger.info("Returning cached metrics")
                return self.metrics_cache
            
            logger.info("Calculating network metrics...")
            
            # Use undirected graph for some metrics
            undirected_graph = self.graph.to_undirected()
            
            metrics = {}
            
            # Centrality measures
            metrics['degree_centrality'] = nx.degree_centrality(undirected_graph)
            metrics['betweenness'] = nx.betweenness_centrality(undirected_graph)
            metrics['closeness'] = nx.closeness_centrality(undirected_graph)
            
            # Handle eigenvector centrality (can fail on disconnected graphs)
            try:
                metrics['eigenvector'] = nx.eigenvector_centrality(undirected_graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge, using zeros")
                metrics['eigenvector'] = {node: 0.0 for node in undirected_graph.nodes()}
            
            # Structural metrics
            metrics['clustering'] = nx.clustering(undirected_graph)
            metrics['density'] = nx.density(undirected_graph)
            metrics['components'] = nx.number_connected_components(undirected_graph)
            
            # Additional metrics
            metrics['pagerank'] = nx.pagerank(self.graph)
            metrics['in_degree'] = dict(self.graph.in_degree())
            metrics['out_degree'] = dict(self.graph.out_degree())
            
            self.metrics_cache = metrics
            logger.info("Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
    
    def find_key_connectors(self, top_n=10):
        """
        Identify top influencers based on multiple centrality measures.
        
        Args:
            top_n (int): Number of top connectors to return
        
        Returns:
            list: List of tuples (node, centrality_score)
        """
        try:
            metrics = self.calculate_metrics()
            degree_centrality = metrics['degree_centrality']
            
            # Sort by degree centrality
            sorted_nodes = sorted(
                degree_centrality.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            logger.info(f"Found top {top_n} key connectors")
            return sorted_nodes[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding key connectors: {e}")
            raise
    
    def predict_connections(self, node, top_n=5):
        """
        Predict missing connections using common neighbors algorithm.
        
        Args:
            node (str): Node to predict connections for
            top_n (int): Number of top predictions to return
        
        Returns:
            list: List of tuples (potential_connection, score)
        """
        try:
            if node not in self.graph.nodes():
                raise ValueError(f"Node {node} not found in network")
            
            # Use common neighbor centrality for prediction
            predictions = []
            
            for potential_node in self.graph.nodes():
                if potential_node != node and not self.graph.has_edge(node, potential_node):
                    # Calculate common neighbors score
                    node_neighbors = set(self.graph.neighbors(node))
                    potential_neighbors = set(self.graph.neighbors(potential_node))
                    common_neighbors = node_neighbors.intersection(potential_neighbors)
                    
                    if common_neighbors:
                        score = len(common_neighbors)  # Simple scoring
                        predictions.append((potential_node, score))
            
            # Sort by score and return top N
            sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
            
            logger.info(f"Predicted {len(sorted_predictions)} potential connections for {node}")
            return sorted_predictions[:top_n]
            
        except Exception as e:
            logger.error(f"Error predicting connections for {node}: {e}")
            raise
    
    def get_reconnection_time(self, person1, person2):
        """
        Calculate shortest reconnection path between two individuals.
        
        Args:
            person1 (str): First person's email
            person2 (str): Second person's email
        
        Returns:
            int: Number of steps in shortest path (inf if no path exists)
        """
        try:
            if person1 not in self.graph.nodes() or person2 not in self.graph.nodes():
                raise ValueError("One or both nodes not found in network")
            
            try:
                shortest_path = nx.shortest_path(self.graph, source=person1, target=person2)
                steps = len(shortest_path) - 1  # Number of edges in path
                logger.info(f"Reconnection path from {person1} to {person2}: {steps} steps")
                return steps
            except nx.NetworkXNoPath:
                logger.info(f"No path exists between {person1} and {person2}")
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error calculating reconnection time: {e}")
            raise
    
    def get_network_summary(self):
        """
        Get comprehensive network summary statistics.
        
        Returns:
            dict: Dictionary containing network summary statistics
        """
        try:
            metrics = self.calculate_metrics()
            
            summary = {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'density': metrics['density'],
                'connected_components': metrics['components'],
                'avg_clustering': np.mean(list(metrics['clustering'].values())),
                'avg_degree_centrality': np.mean(list(metrics['degree_centrality'].values())),
                'avg_betweenness': np.mean(list(metrics['betweenness'].values())),
                'avg_closeness': np.mean(list(metrics['closeness'].values())),
                'most_connected': max(metrics['degree_centrality'].items(), key=lambda x: x[1]),
                'most_influential': max(metrics['betweenness'].items(), key=lambda x: x[1])
            }
            
            logger.info("Network summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating network summary: {e}")
            raise
    
    def save_model(self, path):
        """
        Save the analyzer model to disk.
        
        Args:
            path (str): Path to save the model
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    @classmethod
    def load_model(cls, path):
        """
        Load a saved analyzer model from disk.
        
        Args:
            path (str): Path to load the model from
        
        Returns:
            EmailNetworkAnalyzer: Loaded analyzer instance
        """
        try:
            with open(path, 'rb') as f:
                analyzer = pickle.load(f)
            logger.info(f"Model loaded from {path}")
            return analyzer
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def main():
    """Main function for testing the analyzer."""
    print("Email Network Analyzer - Testing")
    analyzer = EmailNetworkAnalyzer()
    
    # Create sample data
    sample_data = pd.DataFrame([
        {'sender': 'alice@company.com', 'recipients': 'bob@company.com, charlie@company.com', 'timestamp': '2023-01-01 10:00:00'},
        {'sender': 'bob@company.com', 'recipients': 'charlie@company.com', 'timestamp': '2023-01-01 11:00:00'},
        {'sender': 'charlie@company.com', 'recipients': 'alice@company.com, dave@company.com', 'timestamp': '2023-01-02 09:00:00'},
        {'sender': 'dave@company.com', 'recipients': 'alice@company.com', 'timestamp': '2023-01-02 14:00:00'},
        {'sender': 'eve@company.com', 'recipients': 'bob@company.com, dave@company.com', 'timestamp': '2023-01-03 12:00:00'}
    ])
    
    analyzer.load_emails(sample_data)
    
    # Test functionality
    print("Network Summary:", analyzer.get_network_summary())
    print("Key Connectors:", analyzer.find_key_connectors(top_n=3))
    print("Predictions for alice@company.com:", analyzer.predict_connections('alice@company.com', top_n=2))
    print("Reconnection time (eve to charlie):", analyzer.get_reconnection_time('eve@company.com', 'charlie@company.com'))

if __name__ == "__main__":
    main()
