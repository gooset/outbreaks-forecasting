import pandas as pd
from datetime import datetime, timedelta
import os
from config import PROJECT_ROOT

from src.data_processing.generators import generate_arbovirus_data, generate_vector_sightings
from src.data_processing.processors import process_merged_data
from src.data_processing.operations import DatabaseOperations
from src.utils.config import get_database_url
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def initialize_project(cities_csv_path: str):
    """Initialize project database and load cities data."""
    db_url = get_database_url()
    db_ops = DatabaseOperations(db_url)
    
    logger.info("Initializing database...")
    db_ops.initialize_database()
    
    logger.info("Loading cities data...")
    cities_df = pd.read_csv(cities_csv_path)
    db_ops.insert_cities(cities_df)
    
    return db_ops

def generate_and_store_data(db_ops: DatabaseOperations, cities_df: pd.DataFrame,
                          start_date: datetime, end_date: datetime):
    """Generate and store synthetic data."""
    logger.info("Generating arbovirus data...")
    arbovirus_df = generate_arbovirus_data(cities_df, 
                                   start_date.strftime('%Y-%m-%d'),
                                   end_date.strftime('%Y-%m-%d'))
    
    logger.info("Generating vector sightings...")
    vector_df = generate_vector_sightings(cities_df, 
                                        start_date.year,
                                        end_date.year)
    
    logger.info("Storing generated data...")
    db_ops.insert_arbovirus_data(arbovirus_df)
    db_ops.insert_vector_sightings(vector_df)

def get_surveillance_data(start_date,
                         end_date):
    """Retrieve merged surveillance data."""
    db_ops = DatabaseOperations(get_database_url())
    return db_ops.get_merged_data(start_date, end_date)


