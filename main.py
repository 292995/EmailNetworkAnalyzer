#!/usr/bin/env python3
"""
Email-Based Network Analysis - Main Execution Script

This script orchestrates the complete email network analysis workflow:
1. Data loading and validation
2. Network analysis and metrics calculation
3. Connection prediction
4. Visualization and reporting
5. Model persistence

Author: ---
Date: 2023
"""

import argparse
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from src.analyzer import EmailNetworkAnalyzer
from src.visualizer import NetworkVisualizer
from src.data_loader import load_email_data, generate_sample_data, save_sample_data, analyze_data_distribution

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs_to_create = ['data/processed', 'models', 'results', 'logs', 'notebooks']
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Directory setup completed")

def validate_data_files(data_path):
    """Validate required data files exist."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    logger.info(f"Validated data file: {data_path}")

def load_and_validate_dataset(data_path):
    """Load dataset and perform basic validation."""
    try:
        df = load_email_data(data_path)
        logger.info(f"Dataset loaded: {len(df)} samples")
        
        # Analyze data distribution
        analysis = analyze_data_distribution(df)
        logger.info(f"Data analysis: {analysis['total_emails']} emails, {analysis['unique_senders']} unique senders")
        
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def analyze_network_workflow(data_path, model_save_path):
    """Complete network analysis workflow."""
    logger.info("Starting network analysis workflow...")
    
    # Validate inputs
    validate_data_files(data_path)
    
    # Load dataset
    df = load_and_validate_dataset(data_path)
    
    # Initialize analyzer
    analyzer = EmailNetworkAnalyzer()
    
    # Load emails into network
    analyzer.load_emails(df)
    
    # Calculate metrics
    logger.info("Calculating network metrics...")
    metrics = analyzer.calculate_metrics()
    
    # Get network summary
    summary = analyzer.get_network_summary()
    logger.info(f"Network summary: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    
    # Find key connectors
    key_connectors = analyzer.find_key_connectors(top_n=10)
    logger.info(f"Found {len(key_connectors)} key connectors")
    
    # Save model
    analyzer.save_model(model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    results = {
        'analyzer': analyzer,
        'summary': summary,
        'key_connectors': key_connectors,
        'metrics': metrics
    }
    
    logger.info("Network analysis workflow completed successfully")
    return results

def prediction_workflow(model_path, node, top_n=5):
    """Complete prediction workflow."""
    logger.info(f"Starting prediction workflow for {node}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load analyzer
    analyzer = EmailNetworkAnalyzer.load_model(model_path)
    
    # Make predictions
    predictions = analyzer.predict_connections(node, top_n)
    logger.info(f"Generated {len(predictions)} predictions for {node}")
    
    return predictions

def visualization_workflow(model_path, top_n=20):
    """Complete visualization workflow."""
    logger.info("Starting visualization workflow...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load analyzer
    analyzer = EmailNetworkAnalyzer.load_model(model_path)
    
    # Initialize visualizer
    visualizer = NetworkVisualizer(analyzer)
    
    # Generate visualizations
    visualizer.plot_network(top_n=top_n, save_path='results/network_plot.png')
    visualizer.plot_centrality_comparison(save_path='results/centrality_comparison.png')
    visualizer.plot_network_statistics(save_path='results/network_statistics.png')
    
    logger.info("Visualization workflow completed")
    
    return visualizer

def display_results(results, mode='analysis'):
    """Display formatted results."""
    print("\n" + "="*60)
    if mode == 'analysis':
        print("EMAIL NETWORK ANALYSIS RESULTS")
        print("="*60)
        summary = results['summary']
        print(f"Total Nodes: {summary['total_nodes']}")
        print(f"Total Edges: {summary['total_edges']}")
        print(f"Network Density: {summary['density']:.4f}")
        print(f"Connected Components: {summary['connected_components']}")
        print(f"Average Clustering: {summary['avg_clustering']:.4f}")
        print(f"Most Connected: {summary['most_connected'][0]} (score: {summary['most_connected'][1]:.4f})")
        print(f"Most Influential: {summary['most_influential'][0]} (score: {summary['most_influential'][1]:.4f})")
        
        print(f"\nTop 5 Key Connectors:")
        for i, (node, score) in enumerate(results['key_connectors'][:5], 1):
            print(f"  {i}. {node}: {score:.4f}")
    
    elif mode == 'prediction':
        print("CONNECTION PREDICTIONS")
        print("="*60)
        if results:
            for i, (node, score) in enumerate(results, 1):
                print(f"  {i}. {node}: {score:.2f}")
        else:
            print("  No predictions available")
    
    print("="*60 + "\n")

def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(description='Email Network Analysis System')
    parser.add_argument('--mode', choices=['analyze', 'predict', 'visualize', 'demo', 'generate'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--data_path', type=str, default='data/processed/emails.csv',
                       help='Path to email data CSV')
    parser.add_argument('--model_path', type=str, default='models/email_network_model.pkl',
                       help='Path to saved model')
    parser.add_argument('--node', type=str, help='Node for prediction')
    parser.add_argument('--top_n', type=int, default=20, help='Number of top items to show')
    parser.add_argument('--n_emails', type=int, default=1000, help='Number of emails to generate')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    try:
        if args.mode == 'analyze':
            logger.info("Running analysis mode...")
            results = analyze_network_workflow(args.data_path, args.model_path)
            display_results(results, mode='analysis')
            
        elif args.mode == 'predict':
            if not args.node:
                raise ValueError("--node is required for prediction mode")
            logger.info("Running prediction mode...")
            results = prediction_workflow(args.model_path, args.node, args.top_n)
            display_results(results, mode='prediction')
            
        elif args.mode == 'visualize':
            logger.info("Running visualization mode...")
            visualizer = visualization_workflow(args.model_path, args.top_n)
            
        elif args.mode == 'generate':
            logger.info(f"Generating {args.n_emails} sample emails...")
            df = generate_sample_data(args.n_emails)
            save_sample_data(df, args.data_path)
            print(f"Generated sample data saved to {args.data_path}")
            
        elif args.mode == 'demo':
            logger.info("Running demo mode...")
            print(" EMAIL NETWORK ANALYSIS SYSTEM ")
            print("="*50)
            print("Project Status: Complete")
            print("Features: Centrality analysis, connection prediction, visualization")
            print("Metrics: Degree, Betweenness, Closeness, Clustering")
            print("Algorithms: Common neighbors for prediction")
            print("\nUsage examples:")
            print("  Analyze: python main.py --mode analyze --data_path data.csv")
            print("  Predict: python main.py --mode predict --node user@company.com")
            print("  Visualize: python main.py --mode visualize")
            print("  Generate sample: python main.py --mode generate --n_emails 1000")
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
