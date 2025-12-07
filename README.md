# Email-Based Network Analysis for Organizational Connectivity

## Overview
This project analyzes organizational communication patterns through email networks to identify key connectors, predict relationships, and optimize reconnection times. The system achieves comprehensive network analysis with predictive capabilities.

## Features
- Network centrality calculations (degree, betweenness, closeness)
- Connection prediction using common neighbors
- Reconnection time optimization
- Interactive visualizations
- Predictive analytics for missing connections

## Requirements
- Python 3.7+
- See `requirements.txt` for complete list

## Setup
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare email data (CSV format)
4. Run analysis: `python main.py`

## Usage
1. Train analysis: `python -m src.analyzer --mode analyze --data data/emails.csv`
2. Predict connections: `python -m src.analyzer --mode predict --node user@company.com`

## Results
- Network density analysis
- Key connector identification
- Connection prediction accuracy
- Visualization outputs
