# Data Directory

## Data Structure
- `raw/` - Original email data files (not in repo)
- `processed/` - Processed network data (CSV format)

## Data Format
Email data should be in CSV format with these columns:
- `sender`: Email address of sender
- `recipients`: Comma-separated email addresses of recipients
- `timestamp`: DateTime of communication

## Example Data Structure
```csv
sender,recipients,timestamp
alice@company.com,bob@company.com,charlie@company.com,2023-01-01 10:00:00
bob@company.com,dave@company.com,2023-01-01 11:00:00
