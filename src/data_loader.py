"""
Data Loader Module

This module provides utilities for loading and preparing email data
for network analysis.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import re

logger = logging.getLogger(__name__)

def validate_email_format(email):
    """Validate email format using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def load_email_data(file_path):
    """
    Load email data from CSV file with validation.
    
    Expected CSV format:
    - sender: email address of sender
    - recipients: comma-separated email addresses
    - timestamp: datetime string
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Validated email data
    """
    try:
        logger.info(f"Loading email data from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['sender', 'recipients', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate email formats
        invalid_senders = df[~df['sender'].apply(validate_email_format)]
        if not invalid_senders.empty:
            logger.warning(f"Invalid sender emails found: {len(invalid_senders)} records")
            logger.warning(f"Sample invalid emails: {invalid_senders['sender'].head().tolist()}")
        
        # Validate recipients format (comma-separated emails)
        def validate_recipients(recipients):
            if pd.isna(recipients):
                return False
            recipient_list = [r.strip() for r in str(recipients).split(',')]
            return all(validate_email_format(r) for r in recipient_list)
        
        df['valid_recipients'] = df['recipients'].apply(validate_recipients)
        invalid_recipients = df[~df['valid_recipients']]
        if not invalid_recipients.empty:
            logger.warning(f"Invalid recipient emails found: {len(invalid_recipients)} records")
        
        # Filter valid records
        valid_df = df[df['valid_recipients'] & df['sender'].apply(validate_email_format)].copy()
        logger.info(f"Loaded {len(valid_df)} valid email records from {len(df)} total")
        
        return valid_df[['sender', 'recipients', 'timestamp']]
        
    except Exception as e:
        logger.error(f"Error loading email data: {e}")
        raise

def generate_sample_data(n_emails=1000):
    """
    Generate synthetic email data for testing and demonstration.
    
    Args:
        n_emails (int): Number of email records to generate
        
    Returns:
        pd.DataFrame: Synthetic email data
    """
    try:
        logger.info(f"Generating {n_emails} synthetic email records")
        
        # Define some realistic company domains and users
        domains = ['company.com', 'tech.org', 'startup.io', 'enterprise.co', 'business.net']
        first_names = ['alice', 'bob', 'charlie', 'dave', 'eve', 'frank', 'grace', 'heidi', 'ivan', 'judy']
        last_names = ['smith', 'jones', 'williams', 'brown', 'davis', 'miller', 'wilson', 'taylor', 'anderson', 'thomas']
        
        # Generate realistic email addresses
        employees = []
        for first in first_names:
            for last in last_names[:3]:  # Limit to create more connections
                domain = np.random.choice(domains)
                employees.append(f"{first}.{last}@{domain}")
        
        data = []
        start_date = datetime(2023, 1, 1)
        
        for i in range(n_emails):
            sender = np.random.choice(employees)
            
            # Simulate realistic communication patterns
            # More likely to send to colleagues in same domain
            same_domain_recipients = [emp for emp in employees if emp.split('@')[1] == sender.split('@')[1]]
            
            if np.random.random() < 0.7 and same_domain_recipients:  # 70% chance to send internally
                n_recipients = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
                recipients = np.random.choice(same_domain_recipients, size=min(n_recipients, len(same_domain_recipients)), replace=False)
            else:  # External communication
                n_recipients = np.random.choice([1, 2], p=[0.8, 0.2])
                recipients = np.random.choice(employees, size=min(n_recipients, len(employees)), replace=False)
            
            # Ensure sender is not in recipients
            recipients = [r for r in recipients if r != sender]
            
            if recipients:  # Only add if there are recipients
                timestamp = start_date + pd.Timedelta(days=np.random.randint(0, 365))
                
                data.append({
                    'sender': sender,
                    'recipients': ','.join(recipients),
                    'timestamp': timestamp
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} synthetic email records")
        return df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        raise

def save_sample_data(df, output_path):
    """
    Save generated sample data to CSV file.
    
    Args:
        df (pd.DataFrame): Data to save
        output_path (str): Path to save CSV file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving sample data: {e}")
        raise

def analyze_data_distribution(df):
    """
    Analyze the distribution of email data.
    
    Args:
        df (pd.DataFrame): Email data to analyze
        
    Returns:
        dict: Analysis results
    """
    try:
        analysis = {
            'total_emails': len(df),
            'unique_senders': df['sender'].nunique(),
            'unique_recipients': df['recipients'].str.split(',').explode().nunique(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'avg_recipients_per_email': df['recipients'].str.count(',').mean() + 1,
            'top_senders': df['sender'].value_counts().head(10).to_dict()
        }
        
        logger.info("Data distribution analysis completed")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing data distribution: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    print("Testing data loader...")
    
    # Generate sample data
    sample_df = generate_sample_data(100)
    print(f"Generated sample data: {len(sample_df)} records")
    
    # Analyze distribution
    analysis = analyze_data_distribution(sample_df)
    print(f"Analysis: {analysis['total_emails']} emails, {analysis['unique_senders']} unique senders")
    
    # Save sample data
    save_sample_data(sample_df, 'data/processed/sample_emails.csv')
    print("Sample data saved to data/processed/sample_emails.csv")
