from datetime import datetime
import pandas as pd
from typing import Tuple

def process_merged_data(weather_df: pd.DataFrame, arbovirus_df: pd.DataFrame, 
                       vector_df: pd.DataFrame) -> pd.DataFrame:
    """Process and merge weather, arbovirus, and vector data."""
    # Convert dates to datetime
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    arbovirus_df['date'] = pd.to_datetime(arbovirus_df['date'])
    
    # Add year to weather and arbovirus data
    weather_df['year'] = weather_df['date'].dt.year
    arbovirus_df['year'] = arbovirus_df['date'].dt.year
    
    # Merge weather and arbovirus data
    merged_df = pd.merge(weather_df, arbovirus_df, on=['city', 'date'], how='left')
    merged_df['arbovirus_bool'] = merged_df['arbovirus_bool'].fillna(0)
    
    # Merge with vector data
    final_df = pd.merge(merged_df, vector_df, 
                       on=['city', 'year'], 
                       how='left')
    
    # Clean up
    final_df.drop(columns=['year'], inplace=True)
    return final_df
