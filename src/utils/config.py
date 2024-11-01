import os
from typing import Dict

def load_database_config() -> Dict[str, str]:
    """Load database configuration from environment variables."""
    return {
        'username': os.getenv('DB_USERNAME', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'mypassword'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'disease_outbreaks')
    }

def get_database_url() -> str:
    """Get database URL from configuration."""
    config = load_database_config()
    return f"postgresql+psycopg2://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
