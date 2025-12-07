"""
Network Visualization Module

This module provides visualization capabilities for email network analysis,
including network graphs, centrality comparisons, and statistical plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """
    Provides visualization tools for email network analysis.
    """
    
    def __init__(self, analyzer):
        """
        Initialize visualizer with an analyzer instance.
        
        Args:
            analyzer: EmailNetworkAnalyzer instance
        """
        self.analyzer = analyzer
        self.setup_style()
        logger.info("NetworkVisualizer initialized")
    
    def setup_style(self):
        """Set up matplotlib style for consistent plots."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_network(self, top_n=20, figsize=(15, 10), save_path=None):
        """
        Plot network with node size by centrality.
        
        Args:
            top_n (int): Number of top nodes to visualize
            figsize (tuple): Figure size
            save_path (str): Path to save plot (optional)
        """
        try:
            metrics = self.analyzer.calculate_metrics()
            top_connectors = self.analyzer.find_key_connectors(top_n)
            top_nodes = [node for node, _ in top_connectors]
            
            # Create subgraph with top nodes
            subgraph = self.analyzer.graph.subgraph(top_nodes)
            
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            
            # Calculate node sizes based on degree centrality
            node_sizes = [metrics['degree_centrality'][n] * 5000 for n in subgraph.nodes()]
            
            # Draw network
            nx.draw(subgraph, pos,
                    node_color='lightblue',
                    node_size=node_sizes,
                    with_labels=True,
                    edge_color='gray',
                    arrows=True,
                    font_size=8,
                    font_weight='bold')
            
            plt.title(f"Top {top_n} Network Connectors\nNode size reflects connectivity", fontsize=14)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Network plot saved to {save_path}")
            
            plt.show()
            logger.info(f"Network visualization completed for top {top_n} nodes")
            
        except Exception as e:
            logger.error(f"Error plotting network: {e}")
            raise
    
    def plot_centrality_comparison(self, save_path=None):
        """
        Compare different centrality measures.
        
        Args:
            save_path (str): Path to save plot (optional)
        """
        try:
            metrics = self.analyzer.calculate_metrics()
            top_connectors = self.analyzer.find_key_connectors(10)
            top_nodes = [node for node, _ in top_connectors]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            measures = [
                ('degree_centrality', 'Degree Centrality'),
                ('betweenness', 'Betweenness Centrality'),
                ('closeness', 'Closeness Centrality'),
                ('eigenvector', 'Eigenvector Centrality')
            ]
            
            for idx, (measure, title) in enumerate(measures):
                ax = axes[idx // 2, idx % 2]
                values = [metrics[measure].get(n, 0) for n in top_nodes]
                
                bars = ax.bar(range(len(top_nodes)), values)
                ax.set_title(title, fontsize=12)
                ax.set_xticks(range(len(top_nodes)))
                ax.set_xticklabels([n.split('@')[0] for n in top_nodes], rotation=45, ha='right')
                ax.set_ylabel('Centrality Score')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}',
                           ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Centrality comparison plot saved to {save_path}")
            
            plt.show()
            logger.info("Centrality comparison visualization completed")
            
        except Exception as e:
            logger.error(f"Error plotting centrality comparison: {e}")
            raise
    
    def plot_network_statistics(self, save_path=None):
        """
        Plot comprehensive network statistics.
        
        Args:
            save_path (str): Path to save plot (optional)
        """
        try:
            metrics = self.analyzer.calculate_metrics()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Degree distribution
            degrees = [deg for _, deg in self.analyzer.graph.degree()]
            axes[0, 0].hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Degree Distribution')
            axes[0, 0].set_xlabel('Degree')
            axes[0, 0].set_ylabel('Frequency')
            
            # Clustering coefficient distribution
            clustering_values = list(metrics['clustering'].values())
            axes[0, 1].hist(clustering_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Clustering Coefficient Distribution')
            axes[0, 1].set_xlabel('Clustering Coefficient')
            axes[0, 1].set_ylabel('Frequency')
            
            # Centrality measures scatter
            degree_vals = list(metrics['degree_centrality'].values())
            betweenness_vals = list(metrics['betweenness'].values())
            axes[1, 0].scatter(degree_vals, betweenness_vals, alpha=0.6)
            axes[1, 0].set_title('Degree vs Betweenness Centrality')
            axes[1, 0].set_xlabel('Degree Centrality')
            axes[1, 0].set_ylabel('Betweenness Centrality')
            
            # Closeness vs Betweenness
            closeness_vals = list(metrics['closeness'].values())
            axes[1, 1].scatter(closeness_vals, betweenness_vals, alpha=0.6, color='orange')
            axes[1, 1].set_title('Closeness vs Betweenness Centrality')
            axes[1, 1].set_xlabel('Closeness Centrality')
            axes[1, 1].set_ylabel('Betweenness Centrality')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Network statistics plot saved to {save_path}")
            
            plt.show()
            logger.info("Network statistics visualization completed")
            
        except Exception as e:
            logger.error(f"Error plotting network statistics: {e}")
            raise
    
    def plot_connection_predictions(self, node, save_path=None):
        """
        Plot connection predictions for a specific node.
        
        Args:
            node (str): Node to predict connections for
            save_path (str): Path to save plot (optional)
        """
        try:
            predictions = self.analyzer.predict_connections(node, top_n=10)
            
            if not predictions:
                logger.warning(f"No predictions available for {node}")
                return
            
            nodes, scores = zip(*predictions)
            node_names = [n.split('@')[0] for n in nodes]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(nodes)), scores)
            plt.title(f'Connection Predictions for {node.split("@")[0]}')
            plt.xlabel('Potential Connections')
            plt.ylabel('Prediction Score')
            plt.xticks(range(len(nodes)), node_names, rotation=45, ha='right')
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.2f}',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Connection predictions plot saved to {save_path}")
            
            plt.show()
            logger.info(f"Connection predictions visualization completed for {node}")
            
        except Exception as e:
            logger.error(f"Error plotting connection predictions: {e}")
            raise
